"""Pallas/Triton fused AF2 transition: LayerNorm -> Linear -> relu -> Linear in one kernel.

fused_ops pairs it with a backward through the reference. The [M, factor*C] hidden activation never leaves the SM: each row
tile streams the hidden dim in chunks, accumulating the second projection in fp32.

  out = relu(LN(x) @ w1 + b1) @ w2 + b2
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from colabfold_kernels.device import compute_capability, is_rocm

_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


def ln_rows(x, scale_ref, offset_ref, eps):
  """In a kernel: the float32 rows x [block_m, C] layer-normed, scaled and offset."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  d = x - mean
  var = jnp.mean(d * d, axis=-1, keepdims=True)
  y = d * jax.lax.rsqrt(var + eps)
  return y * scale_ref[...][None, :].astype(jnp.float32) + offset_ref[...][None, :].astype(jnp.float32)


def unless_padding(keep_ref, rows, compute, padding):
  """In a kernel: compute() for a row block with a row keep_ref marks, padding() for one with none (the bucket's
  padding: its rows then skip the work and get what padding() stores). keep_ref None: always compute()."""
  if keep_ref is None:
    compute()
    return
  live = jnp.max(jnp.abs(plgpu.load(keep_ref, mask=rows, other=0).astype(jnp.float32))) > 0
  pl.when(live)(compute)
  pl.when(jnp.logical_not(live))(padding)


def sigmoid(x):
  """In a kernel: sigmoid as 0.5 + 0.5 tanh(x / 2), one MUFU.TANH on CUDA instead of an exp and an IEEE divide."""
  if is_rocm():
    return jax.nn.sigmoid(x)
  return 0.5 + 0.5 * plgpu.approx_tanh(0.5 * x)


def swish(x):
  """In a kernel: x sigmoid(x) as h + h tanh(h), h = x / 2."""
  if is_rocm():
    return jax.nn.swish(x)
  h = 0.5 * x
  return h + h * plgpu.approx_tanh(h)


def fast(activation):
  """The tanh form of a sigmoid or swish activation; any other passes through."""
  return {jax.nn.sigmoid: sigmoid, jax.nn.swish: swish}.get(activation, activation)


def ref_ln(xf, scale, offset, eps):
  """The float32 reference LayerNorm over the last axis."""
  mean = xf.mean(-1, keepdims=True)
  var = ((xf - mean) ** 2).mean(-1, keepdims=True)
  return (xf - mean) * jax.lax.rsqrt(var + eps) * scale + offset


def _kernel(x_ref, s_ref, o_ref_ln, w1_ref, b1_ref, w2_ref, b2_ref, o_ref, *, eps, m, block_m,
            chunk, add_input):
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  y = ln_rows(x, s_ref, o_ref_ln, eps)
  y = y.astype(w1_ref.dtype)

  def body(k, acc):
    ks = pl.ds(pl.multiple_of(k * chunk, chunk), chunk)
    h = _dot(y, w1_ref[:, ks]) + b1_ref[ks][None, :].astype(jnp.float32)
    h = jnp.maximum(h, 0.0).astype(w2_ref.dtype)
    return acc + _dot(h, w2_ref[ks, :])

  acc = jnp.zeros((block_m, w2_ref.shape[1]), jnp.float32)
  acc = jax.lax.fori_loop(0, w1_ref.shape[1] // chunk, body, acc)
  out = acc + b2_ref[...][None, :].astype(jnp.float32)
  if add_input:
    out = out + x
  plgpu.store(o_ref, out.astype(o_ref.dtype), mask=rows[:, None])


def supported(c, hdim):
  # 1.9x over XLA at 128 channels; at 256 (1 MB of weights) XLA's GEMM is 1.4-2.5x faster on A100 and gfx1151
  top = 128 if is_rocm() or compute_capability() == 80 else 256
  return c & (c - 1) == 0 and 16 <= c <= top and hdim % 64 == 0


@functools.partial(jax.jit, static_argnames=("eps", "add_input", "block_m", "chunk", "num_warps",
                                             "num_stages"))
def fused_transition(x, ln_scale, ln_offset, w1, b1, w2, b2, *, eps=1e-5, add_input=False, block_m=None,
                     chunk=None, num_warps=None, num_stages=None):
  """x [..., C]; ln_scale/ln_offset [C]; w1 [C, H]; b1 [H]; w2 [H, C]; b2 [C]. Returns x's shape.

  add_input returns x + transition(x), the residual update, from the tile already loaded.
  """
  c = x.shape[-1]
  hdim = w1.shape[1]
  xf = x.reshape(-1, c)
  m = xf.shape[0]
  # swept on sm_120 and gfx1151: the fp32 [block_m, C] accumulator lives in registers
  wide, rocm = c > 128, is_rocm()
  sm80_small = not rocm and compute_capability() == 80 and c == 128 and hdim % 128 == 0 and m <= 16_384
  block_m = block_m or (64 if wide or rocm else 128)
  chunk = chunk or (64 if wide else 128 if sm80_small else 32)
  num_warps = num_warps or (8 if wide or sm80_small else 4)
  num_stages = num_stages or (1 if wide or rocm or sm80_small else 2)
  assert c & (c - 1) == 0 and 16 <= c <= 256 and hdim % 64 == 0, (c, hdim)
  vec = lambda n: pl.BlockSpec((n,), lambda i: (0,))
  out = pl.pallas_call(
      functools.partial(_kernel, eps=eps, m=m, block_m=block_m, chunk=chunk, add_input=add_input),
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((block_m, c), lambda i: (i, 0)), vec(c), vec(c),
                pl.BlockSpec((c, hdim), lambda i: (0, 0)), vec(hdim),
                pl.BlockSpec((hdim, c), lambda i: (0, 0)), vec(c)],
      out_specs=pl.BlockSpec((block_m, c), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((m, c), x.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="fused_transition",
  )(xf, ln_scale, ln_offset, w1, b1, w2, b2)
  return out.reshape(x.shape)


def ref_transition(x, ln_scale, ln_offset, w1, b1, w2, b2, *, eps=1e-5, add_input=False):
  """float32 reference."""
  xf = x.astype(jnp.float32)
  y = ref_ln(xf, ln_scale, ln_offset, eps)
  h = jax.nn.relu(y @ w1.astype(jnp.float32) + b1.astype(jnp.float32))
  return h @ w2.astype(jnp.float32) + b2.astype(jnp.float32) + (xf if add_input else 0)


def _glu_kernel(x_ref, s_ref, o_ref_ln, wa_ref, wb_ref, wo_ref, *refs, eps, m, block_m, chunk, add_input,
                activation):
  keep_ref, o_ref = refs if len(refs) == 2 else (None,) + refs
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m

  def compute():
    x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
    y = ln_rows(x, s_ref, o_ref_ln, eps)
    y = y.astype(wa_ref.dtype)

    def body(k, acc):
      ks = pl.ds(pl.multiple_of(k * chunk, chunk), chunk)
      h = fast(activation)(_dot(y, wa_ref[:, ks])) * _dot(y, wb_ref[:, ks])
      return acc + _dot(h.astype(wo_ref.dtype), wo_ref[ks, :])

    acc = jnp.zeros((block_m, wo_ref.shape[1]), jnp.float32)
    acc = jax.lax.fori_loop(0, wa_ref.shape[1] // chunk, body, acc)
    if keep_ref is not None:
      acc = acc * plgpu.load(keep_ref, mask=rows, other=0.0).astype(jnp.float32)[:, None]
    if add_input:
      acc = acc + x
    plgpu.store(o_ref, acc.astype(o_ref.dtype), mask=rows[:, None])

  def padding():  # keep 0 on every row: what compute() gives them, without the transition
    out = plgpu.load(x_ref, mask=rows[:, None], other=0.0) if add_input else jnp.zeros(o_ref.shape, o_ref.dtype)
    plgpu.store(o_ref, out.astype(o_ref.dtype), mask=rows[:, None])

  unless_padding(keep_ref, rows, compute, padding)


def glu_supported(c, hdim):
  return c & (c - 1) == 0 and 16 <= c <= 512 and hdim % 64 == 0


@functools.partial(jax.jit, static_argnames=("eps", "add_input", "activation", "block_m", "chunk", "num_warps",
                                             "num_stages"))
def fused_glu_transition(x, ln_scale, ln_offset, wa, wb, wo, keep=None, *, eps=1e-5, add_input=False,
                         activation=jax.nn.swish, block_m=None, chunk=32, num_warps=None, num_stages=None):
  """AF3's transition: LN(x) -> activation(. @ wa) * (. @ wb) -> @ wo, no biases; x [..., C], wa/wb [C, H], wo [H, C].

  keep [...] scales each output row (OpenFold3's output mask); add_input returns x + that from the tile already loaded.
  A row block keep zeroes entirely skips the transition (the bucket's padding).
  """
  c, hdim = x.shape[-1], wa.shape[1]
  assert glu_supported(c, hdim), (c, hdim)
  # swept on sm_120 at 832 tokens: 128 channels 1.00 ms vs 2.06 unfused, 64 channels 0.20 vs 0.71;
  # ROCm takes the AF2 transition's tiles, three 128-channel stages would not fit gfx1151's 64 KB
  wide = c >= 128 and not is_rocm()
  if c > 128:
    # the fp32 [block_m, C] accumulator takes the registers: 32 rows x 4 warps, swept on gfx1151 at 256 tokens
    # (256 channels: 8.4 against XLA's 40 ms, 512: 91 against 148)
    block_m, num_warps, num_stages = block_m or 32, num_warps or 4, num_stages or 1
  block_m = block_m or (128 if wide else 64)
  num_warps = num_warps or (8 if wide else 4)
  num_stages = num_stages or (3 if wide else 1)
  xf = x.reshape(-1, c)
  m = xf.shape[0]
  vec = pl.BlockSpec((c,), lambda i: (0,))
  full = lambda a, b: pl.BlockSpec((a, b), lambda i: (0, 0))
  out = pl.pallas_call(
      functools.partial(_glu_kernel, eps=eps, m=m, block_m=block_m, chunk=chunk, add_input=add_input,
                        activation=activation),
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((block_m, c), lambda i: (i, 0)), vec, vec, full(c, hdim), full(c, hdim), full(hdim, c)]
           + ([] if keep is None else [pl.BlockSpec((block_m,), lambda i: (i,))]),
      out_specs=pl.BlockSpec((block_m, c), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((m, c), x.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="fused_glu_transition",
  )(xf, ln_scale, ln_offset, wa.astype(x.dtype), wb.astype(x.dtype), wo.astype(x.dtype),
    *([] if keep is None else [keep.reshape(m)]))
  return out.reshape(x.shape)


def ref_glu_transition(x, ln_scale, ln_offset, wa, wb, wo, keep=None, *, eps=1e-5, add_input=False,
                       activation=jax.nn.swish):
  """float32 reference."""
  xf = x.astype(jnp.float32)
  y = ref_ln(xf, ln_scale, ln_offset, eps)
  h = activation(y @ wa.astype(jnp.float32)) * (y @ wb.astype(jnp.float32))
  out = h @ wo.astype(jnp.float32)
  if keep is not None:
    out = out * keep.astype(jnp.float32)[..., None]
  return out + (xf if add_input else 0)

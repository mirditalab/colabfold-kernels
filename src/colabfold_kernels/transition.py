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

from colabfold_kernels.device import is_rocm

_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


def _kernel(x_ref, s_ref, o_ref_ln, w1_ref, b1_ref, w2_ref, b2_ref, o_ref, *, eps, m, block_m,
            chunk, add_input):
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  mean = jnp.mean(x, axis=-1, keepdims=True)
  d = x - mean
  var = jnp.mean(d * d, axis=-1, keepdims=True)
  y = d * jax.lax.rsqrt(var + eps)
  y = y * s_ref[...][None, :].astype(jnp.float32) + o_ref_ln[...][None, :].astype(jnp.float32)
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
  # gfx1151: 1.9x over XLA at 128 channels, no faster at 256
  return c & (c - 1) == 0 and 16 <= c <= (128 if is_rocm() else 256) and hdim % 64 == 0


@functools.partial(jax.jit, static_argnames=("eps", "add_input", "block_m", "chunk", "num_warps",
                                             "num_stages"))
def fused_transition(x, ln_scale, ln_offset, w1, b1, w2, b2, *, eps=1e-5, add_input=False, block_m=None,
                     chunk=None, num_warps=None, num_stages=None):
  """x [..., C]; ln_scale/ln_offset [C]; w1 [C, H]; b1 [H]; w2 [H, C]; b2 [C]. Returns x's shape.

  add_input returns x + transition(x), the residual update, from the tile already loaded.
  """
  c = x.shape[-1]
  hdim = w1.shape[1]
  # swept on sm_120 and gfx1151: the fp32 [block_m, C] accumulator lives in registers
  wide, rocm = c > 128, is_rocm()
  block_m = block_m or (64 if wide or rocm else 128)
  chunk = chunk or (64 if wide else 32)
  num_warps = num_warps or (8 if wide else 4)
  num_stages = num_stages or (1 if wide or rocm else 2)
  assert supported(c, hdim), (c, hdim)
  xf = x.reshape(-1, c)
  m = xf.shape[0]
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
  mean = xf.mean(-1, keepdims=True)
  var = ((xf - mean) ** 2).mean(-1, keepdims=True)
  y = (xf - mean) * jax.lax.rsqrt(var + eps) * ln_scale + ln_offset
  h = jax.nn.relu(y @ w1.astype(jnp.float32) + b1.astype(jnp.float32))
  return h @ w2.astype(jnp.float32) + b2.astype(jnp.float32) + (xf if add_input else 0)


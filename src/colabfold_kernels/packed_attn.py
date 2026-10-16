"""Pallas/Triton flash attention reading q/k/v/gate in place from one packed projection.

fused_ops pairs it with a backward through tri_flash. p is [X, Y, G, H, D] with slots q, k, v and, when G == 4, the gate,
as produced by a single [X*Y, a] x [a, G*H*D] GEMM. Rows attend along Y (batch X); columns
attend along X (batch Y), so column attention needs no transposed copy of its input or output.

  o[x, y, h] = sigmoid(g + gb_h) * softmax_k(scale * q . k + bias_{h,q,k} if kmask else -inf) . v

Output is [X, Y, H, D] in the input's layout.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from colabfold_kernels.transition import ln_rows, ref_ln
from colabfold_kernels.tri_flash import NEG, _chunks, _kernel, _pow2_tile

_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


@functools.partial(jax.jit, static_argnames=(
    "sm_scale", "column", "block_q", "block_k", "num_warps", "num_stages"))
def packed_attention(p, bias, kmask, gate_bias=None, *, sm_scale, column=False, block_q=128,
                     block_k=None, num_warps=4, num_stages=None):
  """p [X, Y, G, H, D] (G = 3, or 4 with a gate); bias [H, Sq, Sk]; kmask [batch, Sk] bool;
  gate_bias [H, D] when G == 4. Rows: batch X, sequence Y; column: batch Y, sequence X."""
  X, Y, G, H, D = p.shape
  gated = G == 4
  assert G in (3, 4) and (gate_bias is not None) == gated
  nb, s = (Y, X) if column else (X, Y)
  chunks = _chunks(D)
  dk = chunks[-1][0] + chunks[-1][1]
  # swept on sm_120 at the triangle and MSA shapes: 12-17% over block_q=64
  block_k = block_k or _pow2_tile(min(32, D))
  num_stages = num_stages or (3 if column else 2)
  # grid (q_block, head, batch): batch innermost keeps bias[h, q_block, :] hot in L2.
  grid = (pl.cdiv(s, block_q), H, nb)
  if column:
    q_spec = lambda slot: pl.BlockSpec((block_q, None, None, None, dk), lambda i, h, j: (i, j, slot, h, 0))
    kv_spec = lambda slot: pl.BlockSpec((s, None, None, None, dk), lambda i, h, j: (0, j, slot, h, 0))
    o_spec = pl.BlockSpec((block_q, None, None, dk), lambda i, h, j: (i, j, h, 0))
  else:
    q_spec = lambda slot: pl.BlockSpec((None, block_q, None, None, dk), lambda i, h, j: (j, i, slot, h, 0))
    kv_spec = lambda slot: pl.BlockSpec((None, s, None, None, dk), lambda i, h, j: (j, 0, slot, h, 0))
    o_spec = pl.BlockSpec((None, block_q, None, dk), lambda i, h, j: (j, i, h, 0))
  # tri_flash's kernel: the BlockSpecs pick q/k/v/gate out of p, the kernel never sees the packing
  in_specs = [q_spec(0), kv_spec(1), kv_spec(2), pl.BlockSpec((None, block_q, s), lambda i, h, j: (h, i, 0)),
              pl.BlockSpec((None, s), lambda i, h, j: (j, 0))]
  args = [p, p, p, bias.astype(p.dtype), kmask]
  if gated:
    in_specs += [q_spec(3), pl.BlockSpec((None, dk), lambda i, h, j: (h, 0))]
    args += [p, gate_bias]
  return pl.pallas_call(
      functools.partial(_kernel, sm_scale=sm_scale, block_q=block_q, block_k=block_k, sq=s, d=D,
                        chunks=chunks, gated=gated),
      grid=grid,
      in_specs=in_specs,
      out_specs=o_spec,
      out_shape=jax.ShapeDtypeStruct((X, Y, H, D), p.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="packed_attention",
  )(*args)


def ref_packed_attention(p, bias, kmask, gate_bias=None, *, sm_scale, column=False):
  """float32 reference."""
  p = p.astype(jnp.float32)
  if column:
    p = jnp.swapaxes(p, 0, 1)
  q, k, v = p[:, :, 0], p[:, :, 1], p[:, :, 2]
  logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) * sm_scale + bias.astype(jnp.float32)[None]
  logits = jnp.where(kmask[:, None, None, :], logits, NEG)
  o = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, axis=-1), v)
  if gate_bias is not None:
    o = o * jax.nn.sigmoid(p[:, :, 3] + gate_bias.astype(jnp.float32))
  return jnp.swapaxes(o, 0, 1) if column else o


def attention_via(p, bias, kmask, gate_bias=None, *, sm_scale, column=False, attention):
  """packed_attention through a differentiable heads-major attention(q, k, v, mask_bias, bias, scale):
  the backward's reference, so the backward runs that kernel's backward."""
  pp = jnp.swapaxes(p, 0, 1) if column else p
  q, k, v = (jnp.swapaxes(pp[:, :, i], 1, 2) for i in range(3))
  mask_bias = jnp.where(kmask, 0., -1e9).astype(p.dtype)[:, None, None, :]
  o = jnp.swapaxes(attention(q, k, v, mask_bias, bias.astype(p.dtype), sm_scale), 1, 2)
  if gate_bias is not None:
    o = (o.astype(jnp.float32) * jax.nn.sigmoid(pp[:, :, 3].astype(jnp.float32)
                                               + gate_bias.astype(jnp.float32))).astype(p.dtype)
  return jnp.swapaxes(o, 0, 1) if column else o


def projection_supported(*widths):
  """Whole-tile dots: power-of-two widths of at least 16 (Triton's dot K), at most 128 (shared memory)."""
  return all(w & (w - 1) == 0 and 16 <= w <= 128 for w in widths)


def _ln_proj_kernel(x_ref, s_ref, offset_ref, *refs, eps, m, block_m, slots, heads):
  # refs: w [slots, a, n] if slots, wb [heads, a] if heads, then the outputs in the same order
  refs = list(refs)
  w_ref = refs.pop(0) if slots else None
  wb_ref = refs.pop(0) if heads else None
  p_ref = refs.pop(0) if slots else None
  b_ref = refs.pop(0) if heads else None
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  y = ln_rows(x, s_ref, offset_ref, eps).astype(x_ref.dtype)
  for g in range(slots):
    n = w_ref.shape[2]
    plgpu.store(p_ref.at[:, pl.ds(g * n, n)], _dot(y, w_ref[g]).astype(p_ref.dtype), mask=rows[:, None])
  for h in range(heads):
    # a few heads: row reductions, a dot would pad them to 16 columns
    b = jnp.sum(y.astype(jnp.float32) * wb_ref[h][None, :].astype(jnp.float32), axis=1)
    plgpu.store(b_ref.at[h], b.astype(b_ref.dtype), mask=rows)


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "num_warps", "num_stages"))
def ln_projection(x, ln_scale, ln_offset, w=None, wb=None, *, eps=1e-5, block_m=32, num_warps=8, num_stages=2):
  """LN(x) projected by each slot of w and by wb, from one read of x [M, a].

  w [G, a, n] gives p [M, G*n], the packed projection; wb [a, h] gives the bias channel-major [h, M].
  Returns (p, bias), None for an absent weight.
  """
  m, a = x.shape
  slots = 0 if w is None else w.shape[0]
  heads = 0 if wb is None else wb.shape[1]
  vec = pl.BlockSpec((a,), lambda i: (0,))
  in_specs, args, out_specs, out_shape = [pl.BlockSpec((block_m, a), lambda i: (i, 0)), vec, vec], [], [], []
  if slots:
    n = w.shape[2]
    in_specs.append(pl.BlockSpec(w.shape, lambda i: (0, 0, 0)))
    args.append(w)
    out_specs.append(pl.BlockSpec((block_m, slots * n), lambda i: (i, 0)))
    out_shape.append(jax.ShapeDtypeStruct((m, slots * n), x.dtype))
  if heads:
    in_specs.append(pl.BlockSpec((heads, a), lambda i: (0, 0)))
    args.append(wb.T)
    out_specs.append(pl.BlockSpec((heads, block_m), lambda i: (0, i)))
    out_shape.append(jax.ShapeDtypeStruct((heads, m), x.dtype))
  # swept on the L40S at the triangle attention's 4 x 128 outputs: 30% over 64 rows x 4 warps
  out = pl.pallas_call(
      functools.partial(_ln_proj_kernel, eps=eps, m=m, block_m=block_m, slots=slots, heads=heads),
      grid=(pl.cdiv(m, block_m),),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shape,
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="ln_projection",
  )(x, ln_scale, ln_offset, *args)
  out = list(out)
  return (out.pop(0) if slots else None), (out.pop(0) if heads else None)


def ref_ln_projection(x, ln_scale, ln_offset, w=None, wb=None, *, eps=1e-5):
  y = ref_ln(x.astype(jnp.float32), ln_scale, ln_offset, eps)
  p = None if w is None else jnp.einsum('ma,gan->mgn', y, w.astype(jnp.float32)).reshape(x.shape[0], -1)
  return p, (None if wb is None else (y @ wb.astype(jnp.float32)).T)


def _out_proj_kernel(o_ref, w_ref, b_ref, *rest, m, block_m):
  r_ref, out_ref = rest if len(rest) == 2 else (None, rest[0])
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  o = plgpu.load(o_ref, mask=rows[:, None], other=0.0)
  out = _dot(o, w_ref[...]) + b_ref[...][None, :].astype(jnp.float32)
  if r_ref is not None:
    out = out + plgpu.load(r_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  plgpu.store(out_ref, out.astype(out_ref.dtype), mask=rows[:, None])


@functools.partial(jax.jit, static_argnames=("block_m", "num_warps", "num_stages"))
def output_projection(o, w, b, residual=None, *, block_m=64, num_warps=4, num_stages=2):
  """o [M, k] @ w [k, c] + b, plus residual [M, c] when given, in one pass. The result has residual's dtype, else o's."""
  m, k = o.shape
  c = w.shape[1]
  rows = pl.BlockSpec((block_m, c), lambda i: (i, 0))
  extra = [] if residual is None else [residual]
  return pl.pallas_call(
      functools.partial(_out_proj_kernel, m=m, block_m=block_m),
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((block_m, k), lambda i: (i, 0)), pl.BlockSpec((k, c), lambda i: (0, 0)),
                pl.BlockSpec((c,), lambda i: (0,))] + [rows] * len(extra),
      out_specs=rows,
      out_shape=jax.ShapeDtypeStruct((m, c), o.dtype if residual is None else residual.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="output_projection",
  )(o, w, b, *extra)


def ref_output_projection(o, w, b, residual=None):
  out = o.astype(jnp.float32) @ w.astype(jnp.float32) + b.astype(jnp.float32)
  return out if residual is None else out + residual.astype(jnp.float32)

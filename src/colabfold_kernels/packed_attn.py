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

from colabfold_kernels.tri_flash import NEG, _chunks, _kernel, _pow2_tile


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

"""Pallas/Triton flash attention for the AF2 Evoformer (MSA + triangle).

Inference (forward) only, no backward pass

Computes, per (batch n, head h):
  softmax_k( scale * Q_q . K_k + bias_{h,q,k}  if mask_{n,k} else -inf ) . V_k
Layout: q/k/v [N, H, S, D]; bias [H, Sq, Sk] (shared across N); kmask [N, Sk] bool.
"""
import functools, math
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

# jnp.dot is stable across jax versions, contrary to pl.dot/plgpu.dot
_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)

NEG = -0.7 * float(np.finfo(np.float32).max)
LOG2E = math.log2(math.e)


def _pow2_tile(d):
  """Triton tiles are powers of two, and its dot needs at least 16 columns."""
  return max(16, 1 << (int(d) - 1).bit_length())


def _chunks(d):
  """d as power-of-two dot widths: narrow dots beat one wider tile of padding."""
  out, off, rem = [], 0, int(d)
  while rem > 0:
    w = 1 << (rem.bit_length() - 1) if rem >= 16 else 16
    out.append((off, w))
    off, rem = off + w, rem - w
  tile = _pow2_tile(d)
  return tuple(out) if off < tile else ((0, tile),)


def _kernel(q_ref, k_ref, v_ref, bias_ref, kmask_ref, o_ref, *,
            sm_scale, block_q, block_k, sq, d, chunks):
  # q_ref [block_q, W]; k_ref/v_ref [Sk, W]; bias_ref [block_q, Sk]; kmask_ref [Sk],
  # W being d split into power-of-two dot widths. Sq/Sk are NOT padded to block
  # multiples; boundary blocks are handled with masked loads (offs < Sk) and a
  # masked output store (offs_q < Sq) so no jnp.pad copy of q/k/v is needed.
  sk = k_ref.shape[0]
  # only a last chunk reaching past d needs a lane mask, and those lanes read zero
  cols = [None if off + w <= d else (off + jnp.arange(w)) < d for off, w in chunks]
  qs = [q_ref[:, pl.dslice(off, w)] if c is None else
        plgpu.load(q_ref.at[:, pl.dslice(off, w)], mask=c[None, :], other=0.0)
        for (off, w), c in zip(chunks, cols)]
  m_i = jnp.full(block_q, -float('inf'), jnp.float32)
  l_i = jnp.zeros(block_q, jnp.float32)
  os = [jnp.zeros((block_q, w), jnp.float32) for _, w in chunks]

  def body(j, carry):
    o_prev, m_prev, l_prev = carry
    start = j * block_k
    kb = (start + jnp.arange(block_k)) < sk        # [block_k] in-bounds keys
    sl = pl.dslice(start, block_k)
    qk = None
    for qc, (off, w), c in zip(qs, chunks, cols):
      m = kb[:, None] if c is None else kb[:, None] & c[None, :]
      kc = plgpu.load(k_ref.at[sl, pl.dslice(off, w)], mask=m, other=0.0)
      part = _dot(qc, kc.T)                    # [block_q, block_k]
      qk = part if qk is None else qk + part
    bias = plgpu.load(bias_ref.at[:, sl], mask=kb[None, :], other=0.0)
    qk = (qk * sm_scale + bias) * LOG2E
    km = plgpu.load(kmask_ref.at[sl], mask=kb, other=False)  # OOB keys -> masked
    qk = jnp.where(km[None, :], qk, NEG)
    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    corr = jnp.exp2(m_prev - m_next)
    s = jnp.exp2(qk - m_next[:, None])
    l_next = corr * l_prev + s.sum(axis=-1)
    o_next = []
    for oc, (off, w), c in zip(o_prev, chunks, cols):
      m = kb[:, None] if c is None else kb[:, None] & c[None, :]
      vc = plgpu.load(v_ref.at[sl, pl.dslice(off, w)], mask=m, other=0.0)
      o_next.append(corr[:, None] * oc + _dot(s.astype(vc.dtype), vc))
    return o_next, m_next, l_next

  os, m_i, l_i = jax.lax.fori_loop(0, pl.cdiv(sk, block_k), body, (os, m_i, l_i))
  q_valid = (pl.program_id(0) * block_q + jnp.arange(block_q)) < sq
  for oc, (off, w), c in zip(os, chunks, cols):
    m = q_valid[:, None] if c is None else q_valid[:, None] & c[None, :]
    plgpu.store(o_ref.at[:, pl.dslice(off, w)],
                (oc / l_i[:, None]).astype(o_ref.dtype), mask=m)


@functools.partial(jax.jit, static_argnames=(
    "sm_scale", "block_q", "block_k", "num_warps", "num_stages"))
def tri_flash(q, k, v, bias, kmask, *, sm_scale, block_q=64, block_k=64,
              num_warps=4, num_stages=2):
  # q/k/v [N,H,Sq,D]; bias [H,Sq,Sk]; kmask [N,Sk] bool. No padding: Sq/Sk need
  # not be block multiples; the kernel masks the partial q-block (output) and the
  # partial k-block (loads), so the big q/k/v tensors are never copied.
  N, H, Sq, D = q.shape
  Sk = k.shape[2]
  chunks = _chunks(D)
  dk = chunks[-1][0] + chunks[-1][1]           # the tile the chunks span
  # grid (q_block, head, row): row N is innermost so the shared bias[h,q_block,:]
  # stays hot in L2 while sweeping rows (the bias is re-read for every row).
  grid = (pl.cdiv(Sq, block_q), H, N)
  bs_qo = pl.BlockSpec((None, None, block_q, dk), lambda i, h, j: (j, h, i, 0))
  bs_kv = pl.BlockSpec((None, None, Sk, dk),      lambda i, h, j: (j, h, 0, 0))
  bs_bias = pl.BlockSpec((None, block_q, Sk),    lambda i, h, j: (h, i, 0))
  bs_mask = pl.BlockSpec((None, Sk),             lambda i, h, j: (j, 0))
  return pl.pallas_call(
      functools.partial(_kernel, sm_scale=sm_scale, block_q=block_q,
                        block_k=block_k, sq=Sq, d=D, chunks=chunks),
      grid=grid,
      in_specs=[bs_qo, bs_kv, bs_kv, bs_bias, bs_mask],
      out_specs=bs_qo,
      out_shape=jax.ShapeDtypeStruct((N, H, Sq, D), q.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="tri_flash_fwd",
  )(q, k, v, bias, kmask)


def pallas_attention(q, k, v, mask_bias, nonbatched_bias, scale):
  """AF2 attention via the Pallas flash kernel.

  q/k/v heads-major [b, h, S, c]; mask_bias additive [b,1,1,S_kv] (~-1e9 = invalid);
  nonbatched_bias [h,S_qo,S_kv] shared, or None. Returns [b, h, S_qo, c].
  """
  b, h, sq, c = q.shape
  sk = k.shape[2]
  # The kernel reuses q's channel dim for v (one BlockSpec), so key_dim must equal value_dim
  # True for AF2's attention
  assert q.shape[-1] == v.shape[-1], (
      f'tri_flash needs key_dim == value_dim, got {q.shape[-1]} vs {v.shape[-1]}')
  # mask_bias is 0 or big_neg(dtype): -1e9, or -1e4 in float16. Template pointwise
  # attention shares one row across the batch, which the row grid index cannot index.
  kmask = jnp.broadcast_to(mask_bias[:, 0, 0, :] > -1e3, (b, sk))   # [b, S_kv] bool
  bias = (jnp.zeros((h, sq, sk), q.dtype)
          if nonbatched_bias is None else nonbatched_bias.astype(q.dtype))
  # Key tile matches the head: a wider tile stages smem the kernel never reads.
  block_k = _pow2_tile(min(64, c))
  # a head dim that is not a power of two, or below 16, is tiled wider and masked
  out = tri_flash(q, k, v, bias, kmask, sm_scale=float(scale),
                  block_q=64, block_k=block_k)
  return out.astype(q.dtype)


def ref_attn(q, k, v, bias, kmask, sm_scale):
  logits = jnp.einsum('nhqd,nhkd->nhqk', q, k) * sm_scale + bias[None]
  logits = jnp.where(kmask[:, None, None, :], logits, NEG)
  w = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
  return jnp.einsum('nhqk,nhkd->nhqd', w, v)



"""Pallas/Triton flash attention for the AF2 Evoformer (MSA + triangle).

Computes, per (batch n, head h):
  softmax_k( scale * Q_q . K_k + bias_{h,q,k}  if mask_{n,k} else -inf ) . V_k
Layout: q/k/v [N, H, S, D]; bias [H, Sq, Sk] (shared across N); kmask [N, Sk] bool.
"""
import functools, math
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from colabfold_kernels.device import is_rocm

# jnp.dot is stable across jax versions, contrary to pl.dot/plgpu.dot
_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)

NEG = -0.7 * float(np.finfo(np.float32).max)
# half's floor: below any f16 logit by more than f16's exp2 range, and the difference of two of them still finite
NEG16 = -30000.0
LOG2E = math.log2(math.e)
E4M3_MAX = 448.0
# softmax probabilities are at most 1: 2^8 times that, through the exponent, keeps the small ones in e4m3's range
P_SHIFT = 8.0


def e4m3(x):
  """x rounded to float8 e4m3, saturating: the format has no infinity, so an unclipped cast past 448 gives NaN."""
  return jnp.clip(x.astype(jnp.float32), -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)


def exp2(x):
  """In a kernel: 2^x as one ex2.approx.ftz.f32 on CUDA; jnp.exp2's libdevice call adds a denormal fixup around it."""
  if is_rocm() or x.dtype != jnp.float32:
    return jnp.exp2(x)
  [r] = plgpu.elementwise_inline_asm("ex2.approx.ftz.f32 $0, $1;", args=[x], constraints="=f,f", pack=1,
                                     result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)])
  return r


def sub_exp2_h(x, m):
  """In a kernel, CUDA only: 2^(x - m) for f16 x and m, two at a time (sub.f16x2, ex2.approx.f16x2)."""
  [r] = plgpu.elementwise_inline_asm("{ .reg .b32 t; sub.f16x2 t, $1, $2; ex2.approx.f16x2 $0, t; }", args=[x, m],
                                     constraints="=r,r,r", pack=2,
                                     result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, jnp.float16)])
  return r


def _and(a, b):
  """a & b, where None is no mask."""
  return b if a is None else a if b is None else a & b


def _load(ref, mask, other):
  """ref, masked where mask is given."""
  return ref[...] if mask is None else plgpu.load(ref, mask=mask, other=other)


def fp8_supported(d, block_k):
  """An 8-bit MMA's K is 32: the head dim (q.k) and the key block (p.v) both need it, in one power-of-two chunk."""
  return d & (d - 1) == 0 and d >= 32 and block_k >= 32


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


def _kernel(q_ref, k_ref, v_ref, bias_ref, kmask_ref, *rest,
            sm_scale, block_q, block_k, sq, d, chunks, gated=False, fp8=False, prefix=False, half=False, group=1):
  # gated: a gate [block_q, W] and its bias [W] come first, the output is o * sigmoid(gate + bias)
  # fp8: k and v come in e4m3 (fp8_supported), q is cast here, both dots accumulate in fp32
  # prefix: the row's count of leading key blocks with no masked key comes first; those skip the key mask
  # half (CUDA): the scores in f16, the bias (log2 domain, f16) the q.k^T accumulator, packed f16 max, subtract and
  # exp2; the row sum p.1 on the tensor cores
  pre_ref, *rest = rest if prefix else (None, *rest)
  g_ref, gb_ref, *rest = rest if gated else (None, None, *rest)
  o_ref, lse_ref = rest[0], rest[1] if len(rest) > 1 else None
  # q_ref [block_q, W]; k_ref/v_ref [Sk, W]; bias_ref [block_q, Sk]; kmask_ref [Sk],
  # W being d split into power-of-two dot widths. Sq/Sk are NOT padded to block
  # multiples; boundary blocks are handled with masked loads (offs < Sk) and a
  # masked output store (offs_q < Sq) so no jnp.pad copy of q/k/v is needed.
  sk = k_ref.shape[0]
  # only a last chunk reaching past d needs a lane mask, and those lanes read zero
  cols = [None if off + w <= d else (off + jnp.arange(w)) < d for off, w in chunks]
  # the query rows past Sq are masked on the way in as well as out: the block is block_q rows wherever Sq ends, and
  # in packed_attention's layout a row past the end of the last batch is past the end of p (column attention over
  # one MSA row reads block_q - 1 rows of [Y, G, H, D] beyond it - an illegal address on a single-sequence fold)
  # group: the grid's first axis is a q block's group of batches (packed_attention's grouped order)
  q_valid = (pl.program_id(0) // group * block_q + jnp.arange(block_q)) < sq
  qs = [plgpu.load(q_ref.at[:, pl.dslice(off, w)],
                   mask=q_valid[:, None] if c is None else q_valid[:, None] & c[None, :], other=0.0)
        for (off, w), c in zip(chunks, cols)]
  if fp8:
    qs = [e4m3(qc) for qc in qs]
  if half:
    qs = [(qc.astype(jnp.float32) * (sm_scale * LOG2E)).astype(jnp.float16) for qc in qs]
  m_i = jnp.full(block_q, -float('inf'), jnp.float16 if half else jnp.float32)
  # half: the row sum as p.1 on the tensor cores - in v's padding column d when the head is narrower than its tile,
  # else as a dot of its own, every one of whose 16 columns is the sum
  ones_col = half and chunks[-1][0] + chunks[-1][1] > d
  l_i = jnp.zeros(() if ones_col else (block_q, 16) if half else block_q, jnp.float32)
  os = [jnp.zeros((block_q, w), jnp.float32) for _, w in chunks]

  def body(j, carry, tail=False, masked=True):
    # tail: the one key block that runs past Sk; every other block is whole and needs no bounds masks
    o_prev, m_prev, l_prev = carry
    # the tail's start is traced: a static one past Sk - block_k fails the slice bounds check its mask makes safe
    start = jnp.int32(j * block_k) if tail else pl.multiple_of(j * block_k, block_k)
    kb = (start + jnp.arange(block_k)) < sk if tail else None        # [block_k] in-bounds keys
    sl = pl.dslice(start, block_k)
    if half:
      return half_body(sl, kb, masked, o_prev, m_prev, l_prev)
    qk = None
    for qc, (off, w), c in zip(qs, chunks, cols):
      m = _and(None if kb is None else kb[:, None], None if c is None else c[None, :])
      kc = _load(k_ref.at[sl, pl.dslice(off, w)], m, 0.0)
      part = _dot(qc, kc.T)                    # [block_q, block_k]
      qk = part if qk is None else qk + part
    bias = plgpu.load(bias_ref.at[:, sl], mask=_and(q_valid[:, None], None if kb is None else kb[None, :]), other=0.0)
    qk = (qk * sm_scale + bias) * LOG2E
    if masked:
      km = _load(kmask_ref.at[sl], kb, False)
      # masked keys share a finite floor; keys past Sk get none, so a fully masked row averages Sk keys
      qk = jnp.where(km[None, :], qk, NEG if kb is None else jnp.where(kb, NEG, -jnp.inf)[None, :])
    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    corr = exp2(m_prev - m_next)
    # FP8's shift goes into the row max, once a row instead of 1 FADD a score
    s = exp2(qk - (m_next - P_SHIFT if fp8 else m_next)[:, None])
    l_next = corr * l_prev + s.sum(axis=-1)
    o_next = []
    for oc, (off, w), c in zip(o_prev, chunks, cols):
      m = _and(None if kb is None else kb[:, None], None if c is None else c[None, :])
      vc = _load(v_ref.at[sl, pl.dslice(off, w)], m, 0.0)
      o_next.append(corr[:, None] * oc + _dot(s.astype(vc.dtype), vc))
    return o_next, m_next, l_next

  def half_body(sl, kb, masked, o_prev, m_prev, l_prev):
    qk = None
    for qc, (off, w), c in zip(qs, chunks, cols):
      m = _and(None if kb is None else kb[:, None], None if c is None else c[None, :])
      kc = _load(k_ref.at[sl, pl.dslice(off, w)], m, 0.0).astype(jnp.float16)
      part = jnp.dot(qc, kc.T, preferred_element_type=jnp.float16)
      qk = part if qk is None else qk + part
    # the bias comes negated and is subtracted: added, Triton folds it into the dot's accumulator, whose load the
    # pipeliner then leaves synchronous
    qk = qk - plgpu.load(bias_ref.at[:, sl], mask=_and(q_valid[:, None], None if kb is None else kb[None, :]), other=0.0)
    if masked:
      km = _load(kmask_ref.at[sl], kb, False)
      qk = jnp.where(km[None, :], qk, NEG16 if kb is None else jnp.where(kb, NEG16, -jnp.inf)[None, :].astype(qk.dtype))
    elif kb is not None:
      qk = jnp.where(kb[None, :], qk, -jnp.inf)
    m_next = jnp.maximum(m_prev, jnp.max(qk, axis=-1))
    corr = exp2(m_prev.astype(jnp.float32) - m_next.astype(jnp.float32))
    s = sub_exp2_h(qk, jnp.broadcast_to(m_next[:, None], qk.shape))
    l_next = l_prev if ones_col else corr[:, None] * l_prev + _dot(s, jnp.ones((block_k, 16), jnp.float16))
    o_next = []
    for oc, (off, w), c in zip(o_prev, chunks, cols):
      m = _and(None if kb is None else kb[:, None], None if c is None else c[None, :])
      vc = _load(v_ref.at[sl, pl.dslice(off, w)], m, 0.0).astype(jnp.float16)
      if c is not None and ones_col:
        vc = jnp.where((off + jnp.arange(w) == d)[None, :], jnp.float16(1), vc)
      o_next.append(corr[:, None] * oc + _dot(s, vc))
    return o_next, m_next, l_next

  carry = (os, m_i, l_i)
  if prefix:
    full = jnp.minimum(pre_ref[0], sk // block_k)
    carry = jax.lax.fori_loop(0, full, functools.partial(body, masked=False), carry)
    carry = jax.lax.fori_loop(full, sk // block_k, body, carry)
  else:
    carry = jax.lax.fori_loop(0, sk // block_k, body, carry)
  if sk % block_k:
    carry = body(sk // block_k, carry, tail=True)
  os, m_i, l_i = carry
  if half:
    off, w = chunks[-1]
    l_i = (jnp.sum(jnp.where((off + jnp.arange(w) == d)[None, :], os[-1], 0.0), axis=1) if ones_col
           else jnp.max(l_i, axis=1))
    m_i = m_i.astype(jnp.float32)
  if lse_ref is not None:
    # log2 domain, which is what the backward's exp2 wants
    plgpu.store(lse_ref, m_i + jnp.log2(l_i), mask=q_valid)
  for oc, (off, w), c in zip(os, chunks, cols):
    m = q_valid[:, None] if c is None else q_valid[:, None] & c[None, :]
    o = oc / l_i[:, None]
    if gated:
      g = plgpu.load(g_ref.at[:, pl.dslice(off, w)], mask=m, other=0.0).astype(jnp.float32)
      gb = (gb_ref[pl.dslice(off, w)] if c is None else
            plgpu.load(gb_ref.at[pl.dslice(off, w)], mask=c, other=0.0)).astype(jnp.float32)
      o = o * jax.nn.sigmoid(g + gb[None, :])
    plgpu.store(o_ref.at[:, pl.dslice(off, w)], o.astype(o_ref.dtype), mask=m)


@functools.partial(jax.jit, static_argnames=(
    "sm_scale", "block_q", "block_k", "num_warps", "num_stages", "seq_major",
    "want_lse"))
def tri_flash(q, k, v, bias, kmask, *, sm_scale, block_q=64, block_k=64,
              num_warps=4, num_stages=2, seq_major=False, want_lse=False):
  # q/k/v [N,H,Sq,D], or [N,Sq,H,D] when seq_major; bias [H,Sq,Sk]; kmask [N,Sk]
  # bool. No padding: Sq/Sk need not be block multiples; the kernel masks the
  # partial q-block (output) and the partial k-block (loads), so the big q/k/v
  # tensors are never copied.
  if seq_major:
    N, Sq, H, D = q.shape
    Sk = k.shape[1]
  else:
    N, H, Sq, D = q.shape
    Sk = k.shape[2]
  chunks = _chunks(D)
  dk = chunks[-1][0] + chunks[-1][1]           # the tile the chunks span
  # grid (q_block, head, row): row N is innermost so the shared bias[h,q_block,:]
  # stays hot in L2 while sweeping rows (the bias is re-read for every row).
  grid = (pl.cdiv(Sq, block_q), H, N)
  if seq_major:
    bs_qo = pl.BlockSpec((None, block_q, None, dk), lambda i, h, j: (j, i, h, 0))
    bs_kv = pl.BlockSpec((None, Sk, None, dk),      lambda i, h, j: (j, 0, h, 0))
    out_shape = jax.ShapeDtypeStruct((N, Sq, H, D), q.dtype)
  else:
    bs_qo = pl.BlockSpec((None, None, block_q, dk), lambda i, h, j: (j, h, i, 0))
    bs_kv = pl.BlockSpec((None, None, Sk, dk),      lambda i, h, j: (j, h, 0, 0))
    out_shape = jax.ShapeDtypeStruct((N, H, Sq, D), q.dtype)
  bs_bias = pl.BlockSpec((None, block_q, Sk),    lambda i, h, j: (h, i, 0))
  bs_mask = pl.BlockSpec((None, Sk),             lambda i, h, j: (j, 0))
  # lse is [N, H, Sq] whatever q's layout, since only the backward reads it
  bs_lse = pl.BlockSpec((None, None, block_q), lambda i, h, j: (j, h, i))
  out_specs = bs_qo if not want_lse else [bs_qo, bs_lse]
  if want_lse:
    out_shape = [out_shape, jax.ShapeDtypeStruct((N, H, Sq), jnp.float32)]
  return pl.pallas_call(
      functools.partial(_kernel, sm_scale=sm_scale, block_q=block_q,
                        block_k=block_k, sq=Sq, d=D, chunks=chunks),
      grid=grid,
      in_specs=[bs_qo, bs_kv, bs_kv, bs_bias, bs_mask],
      out_specs=out_specs,
      out_shape=out_shape,
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="tri_flash_fwd",
  )(q, k, v, bias, kmask)


def _bwd_kernel(q_ref, k_ref, v_ref, bias_ref, kmask_ref, do_ref, lse_ref, delta_ref,
                _zq_ref, _zb_ref, dq_ref, dk_ref, dv_ref, dbias_ref, *,
                sm_scale, block_q, block_k, sq, sk, d, chunks):
  # One k block per program, sweeping q: dK/dV accumulate in registers, while dQ
  # and dBias land from every k block and so go through atomics. The head dim is
  # split the way the forward splits it, so only the widest chunk has to fit.
  start_k = pl.program_id(0) * block_k
  offs_k = start_k + jnp.arange(block_k)
  kb = offs_k < sk
  ks = pl.dslice(start_k, block_k)
  cols = [None if off + w <= d else (off + jnp.arange(w)) < d for off, w in chunks]
  # full 2d masks: a load broadcasts a [rows, 1] mask, an atomic does not
  kvm = [jnp.broadcast_to(kb[:, None], (block_k, w)) if c is None
         else kb[:, None] & c[None, :] for (_, w), c in zip(chunks, cols)]
  kcs = [plgpu.load(k_ref.at[ks, pl.dslice(off, w)], mask=m, other=0.0)
         for (off, w), m in zip(chunks, kvm)]
  vcs = [plgpu.load(v_ref.at[ks, pl.dslice(off, w)], mask=m, other=0.0)
         for (off, w), m in zip(chunks, kvm)]
  km = plgpu.load(kmask_ref.at[ks], mask=kb, other=False)

  def body(i, carry):
    dk_acc, dv_acc = carry
    start_q = i * block_q
    offs_q = start_q + jnp.arange(block_q)
    qb = offs_q < sq
    qs = pl.dslice(start_q, block_q)
    qm = [jnp.broadcast_to(qb[:, None], (block_q, w)) if c is None
          else qb[:, None] & c[None, :] for (_, w), c in zip(chunks, cols)]
    qcs = [plgpu.load(q_ref.at[qs, pl.dslice(off, w)], mask=m, other=0.0)
           for (off, w), m in zip(chunks, qm)]
    docs = [plgpu.load(do_ref.at[qs, pl.dslice(off, w)], mask=m, other=0.0)
            for (off, w), m in zip(chunks, qm)]
    lse = plgpu.load(lse_ref.at[qs], mask=qb, other=0.0)
    delta = plgpu.load(delta_ref.at[qs], mask=qb, other=0.0)
    bias = plgpu.load(bias_ref.at[qs, ks], mask=qb[:, None] & kb[None, :], other=0.0)

    qk = None
    dp = None
    for qc, kc, doc, vc in zip(qcs, kcs, docs, vcs):
      part = _dot(qc, kc.T)
      qk = part if qk is None else qk + part
      part = _dot(doc, vc.T)
      dp = part if dp is None else dp + part
    live = qb[:, None] & (km & kb)[None, :]
    # the clamp the forward used, so an all-masked row differentiates the
    # uniform softmax it was given
    s = jnp.where(live, (qk * sm_scale + bias) * LOG2E, NEG)
    # NEG absorbs log2(l) in a fully masked row's lse, whose forward was the uniform 1/Sk
    p = jnp.where(lse[:, None] <= NEG / 2, jnp.where(kb, 1. / sk, 0.)[None, :], jnp.exp2(s - lse[:, None]))
    # a masked logit is a constant, thus nothing flows back through it
    ds = jnp.where(live, p * (dp - delta[:, None]), 0.0)

    dk_next, dv_next = [], []
    for j, (qc, doc) in enumerate(zip(qcs, docs)):
      dv_next.append(dv_acc[j] + _dot(p.astype(doc.dtype).T, doc))
      dk_next.append(dk_acc[j] + _dot(ds.astype(qc.dtype).T, qc))
    for (off, w), kc, m in zip(chunks, kcs, qm):
      plgpu.atomic_add(dq_ref, (qs, pl.dslice(off, w)),
                       (_dot(ds.astype(kc.dtype), kc) * sm_scale).astype(dq_ref.dtype),
                       mask=m)
    plgpu.atomic_add(dbias_ref, (qs, ks), ds.astype(dbias_ref.dtype),
                     mask=qb[:, None] & kb[None, :])
    return dk_next, dv_next

  zero = [jnp.zeros((block_k, w), jnp.float32) for _, w in chunks]
  dk_acc, dv_acc = jax.lax.fori_loop(0, pl.cdiv(sq, block_q), body, (zero, zero))
  for (off, w), dkc, dvc, m in zip(chunks, dk_acc, dv_acc, kvm):
    plgpu.store(dk_ref.at[ks, pl.dslice(off, w)],
                (dkc * sm_scale).astype(dk_ref.dtype), mask=m)
    plgpu.store(dv_ref.at[ks, pl.dslice(off, w)], dvc.astype(dv_ref.dtype), mask=m)


@functools.partial(jax.jit, static_argnames=(
    "sm_scale", "block_q", "block_k", "num_warps", "num_stages", "seq_major"))
def tri_flash_bwd(q, k, v, bias, kmask, out, dout, lse, *, sm_scale,
                  block_q=64, block_k=64, num_warps=4, num_stages=2,
                  seq_major=False):
  """dQ, dK, dV and dBias. lse comes from the forward with want_lse."""
  if seq_major:
    N, Sq, H, D = q.shape
    Sk = k.shape[1]
  else:
    N, H, Sq, D = q.shape
    Sk = k.shape[2]
  chunks = _chunks(D)
  tile = chunks[-1][0] + chunks[-1][1]
  # the row statistic, cheap enough in xla and fused into the surrounding graph
  delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), axis=-1)
  if seq_major:
    delta = delta.transpose(0, 2, 1)          # [N, H, Sq] like lse
  # grid (k block, head, row): row innermost keeps bias and dbias hot, as in the
  # forward, and dbias is shared across rows so its atomics want that most.
  grid = (pl.cdiv(Sk, block_k), H, N)
  if seq_major:
    bs_q = pl.BlockSpec((None, Sq, None, tile), lambda i, h, j: (j, 0, h, 0))
    bs_k = pl.BlockSpec((None, Sk, None, tile), lambda i, h, j: (j, 0, h, 0))
  else:
    bs_q = pl.BlockSpec((None, None, Sq, tile), lambda i, h, j: (j, h, 0, 0))
    bs_k = pl.BlockSpec((None, None, Sk, tile), lambda i, h, j: (j, h, 0, 0))
  bs_bias = pl.BlockSpec((None, Sq, Sk), lambda i, h, j: (h, 0, 0))
  bs_mask = pl.BlockSpec((None, Sk), lambda i, h, j: (j, 0))
  bs_row = pl.BlockSpec((None, None, Sq), lambda i, h, j: (j, h, 0))
  zeros_q = jnp.zeros(q.shape, jnp.float32)
  zeros_b = jnp.zeros((H, Sq, Sk), jnp.float32)
  dq, dk, dv, dbias = pl.pallas_call(
      functools.partial(_bwd_kernel, sm_scale=sm_scale, block_q=block_q,
                        block_k=block_k, sq=Sq, sk=Sk, d=D, chunks=chunks),
      grid=grid,
      in_specs=[bs_q, bs_k, bs_k, bs_bias, bs_mask, bs_q, bs_row, bs_row,
                bs_q, bs_bias],
      out_specs=[bs_q, bs_k, bs_k, bs_bias],
      out_shape=[jax.ShapeDtypeStruct(q.shape, jnp.float32),
                 jax.ShapeDtypeStruct(k.shape, k.dtype),
                 jax.ShapeDtypeStruct(v.shape, v.dtype),
                 jax.ShapeDtypeStruct((H, Sq, Sk), jnp.float32)],
      # dq and dbias accumulate, thus they start at zero and alias their input
      input_output_aliases={8: 0, 9: 3},
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="tri_flash_bwd",
  )(q, k, v, bias, kmask, dout, lse, delta, zeros_q, zeros_b)
  return dq.astype(q.dtype), dk, dv, dbias


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _attn_diff(q, k, v, bias, kmask, sm_scale, seq_major):
  block_k = _pow2_tile(min(64, q.shape[-1]))
  return tri_flash(q, k, v, bias, kmask, sm_scale=sm_scale, block_q=64,
                   block_k=block_k, seq_major=seq_major)


def _attn_diff_fwd(q, k, v, bias, kmask, sm_scale, seq_major):
  block_k = _pow2_tile(min(64, q.shape[-1]))
  out, lse = tri_flash(q, k, v, bias, kmask, sm_scale=sm_scale, block_q=64,
                       block_k=block_k, seq_major=seq_major, want_lse=True)
  return out, (q, k, v, bias, kmask, out, lse)


def _attn_diff_bwd(sm_scale, seq_major, res, dout):
  q, k, v, bias, kmask, out, lse = res
  d = q.shape[-1]
  # The backward holds q, dO, K and V at once where the forward holds two of
  # them, so a wide head needs a shorter block to stay inside shared memory.
  block_q = 32 if d > 64 else 64
  block_k = min(block_q, _pow2_tile(min(64, d)))
  dq, dk, dv, dbias = tri_flash_bwd(q, k, v, bias, kmask, out, dout, lse,
                                    sm_scale=sm_scale, block_q=block_q,
                                    block_k=block_k, seq_major=seq_major)
  # kmask is boolean, thus its cotangent is the empty float0
  return (dq, dk, dv, dbias.astype(bias.dtype),
          jnp.zeros(kmask.shape, jax.dtypes.float0))


_attn_diff.defvjp(_attn_diff_fwd, _attn_diff_bwd)


def pallas_attention(q, k, v, mask_bias, nonbatched_bias, scale, *, seq_major=False):
  """AF2 attention via the Pallas flash kernel.

  q/k/v heads-major [b, h, S, c], or seq-major [b, S, h, c] when seq_major.
  mask_bias additive [b,1,1,S_kv] (~-1e9 = invalid); nonbatched_bias
  [h,S_qo,S_kv] shared, or None.
  Returns q's layout.
  """
  if seq_major:
    b, sq, h, c = q.shape
    sk = k.shape[1]
  else:
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
  # A head dim that is not a power of two, or below 16, is tiled wider and masked.
  out = _attn_diff(q, k, v, bias, kmask, float(scale), seq_major)
  return out.astype(q.dtype)


def ref_attn(q, k, v, bias, kmask, sm_scale):
  logits = jnp.einsum('nhqd,nhkd->nhqk', q, k) * sm_scale + bias[None]
  logits = jnp.where(kmask[:, None, None, :], logits, NEG)
  w = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
  return jnp.einsum('nhqk,nhkd->nhqd', w, v)



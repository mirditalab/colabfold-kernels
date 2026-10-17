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
from colabfold_kernels.device import compute_capability
from colabfold_kernels.tri_flash import NEG, _chunks, _kernel, _pow2_tile, e4m3, fp8_supported

_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


@functools.partial(jax.jit, static_argnames=(
    "sm_scale", "column", "block_q", "block_k", "num_warps", "num_stages"))
def packed_attention(p, bias, kmask, gate_bias=None, kv=None, *, sm_scale, column=False, block_q=None,
                     block_k=None, num_warps=4, num_stages=None):
  """p [X, Y, G, H, D] (G = 3, or 4 with a gate); bias [H, Sq, Sk]; kmask [batch, Sk] bool;
  gate_bias [H, D] when G == 4. Rows: batch X, sequence Y; column: batch Y, sequence X.
  kv [X, Y, 2, H, D], the k and v slots in e4m3, leaves p with q and the gate: both dots then run e4m3."""
  X, Y, G, H, D = p.shape
  G += 0 if kv is None else 2
  gated = G == 4
  assert G in (3, 4) and (gate_bias is not None) == gated
  nb, s = (Y, X) if column else (X, Y)
  chunks = _chunks(D)
  dk = chunks[-1][0] + chunks[-1][1]
  # swept on sm_120 at the triangle and MSA shapes: 12-17% over block_q=64
  # ...and on an A100 (sm_80) at a 1,000-residue deep-MSA fold's: heads narrower than 32 (the extra MSA's 8) take
  # 64 x 64 tiles, the key tile otherwise only 16 wide (5,120 rows: 83.5 -> 58.0 ms), and the row attention three
  # stages like the column's (508 rows: 9.78 -> 7.11 ms, bit-identical)
  ampere = compute_capability() == 80
  narrow = ampere and D < 32
  block_q = block_q or (64 if narrow else 128)
  block_k = block_k or (64 if narrow else _pow2_tile(min(32, D)))
  fp8 = kv is not None
  assert not fp8 or fp8_supported(D, block_k), (D, block_k)
  num_stages = num_stages or (3 if column or ampere else 2)
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
  # the array and slot each of k, v and the gate is read from
  (ka, ks), (va, vs), gs = ((p, 1), (p, 2), 3) if kv is None else ((kv, 0), (kv, 1), 1)
  # every batch re-reads the bias, most of the kernel's L2 traffic: rows padded to 128 bytes keep a key block's
  # slice of it from straddling one more 32-byte sector (1,048 keys: 15% slower unpadded)
  sp = -(-s // 64) * 64
  bias = jnp.pad(bias.astype(p.dtype), ((0, 0), (0, 0), (0, sp - s)))
  in_specs = [q_spec(0), kv_spec(ks), kv_spec(vs), pl.BlockSpec((None, block_q, sp), lambda i, h, j: (h, i, 0)),
              pl.BlockSpec((None, s), lambda i, h, j: (j, 0))]
  args = [p, ka, va, bias, kmask]
  if gated:
    in_specs += [q_spec(gs), pl.BlockSpec((None, dk), lambda i, h, j: (h, 0))]
    args += [p, gate_bias]
  return pl.pallas_call(
      functools.partial(_kernel, sm_scale=sm_scale, block_q=block_q, block_k=block_k, sq=s, d=D,
                        chunks=chunks, gated=gated, fp8=fp8),
      grid=grid,
      in_specs=in_specs,
      out_specs=o_spec,
      out_shape=jax.ShapeDtypeStruct((X, Y, H, D), p.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="packed_attention",
  )(*args)


def unsplit_kv(p, kv):
  """The packed [X, Y, G, H, D] layout from p (q, then the gate) and the e4m3 k and v."""
  return jnp.concatenate([p[:, :, :1], kv.astype(p.dtype), p[:, :, 1:]], axis=2)


def ref_packed_attention(p, bias, kmask, gate_bias=None, kv=None, *, sm_scale, column=False):
  """float32 reference."""
  p = (p if kv is None else unsplit_kv(p, kv)).astype(jnp.float32)
  if column:
    p = jnp.swapaxes(p, 0, 1)
  q, k, v = p[:, :, 0], p[:, :, 1], p[:, :, 2]
  logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) * sm_scale + bias.astype(jnp.float32)[None]
  logits = jnp.where(kmask[:, None, None, :], logits, NEG)
  o = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, axis=-1), v)
  if gate_bias is not None:
    o = o * jax.nn.sigmoid(p[:, :, 3] + gate_bias.astype(jnp.float32))
  return jnp.swapaxes(o, 0, 1) if column else o


def attention_via(p, bias, kmask, gate_bias=None, kv=None, *, sm_scale, column=False, attention):
  """packed_attention through a differentiable heads-major attention(q, k, v, mask_bias, bias, scale):
  the backward's reference, so the backward runs that kernel's backward."""
  p = p if kv is None else unsplit_kv(p, kv)
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


def ln_projection_supported(a, n=None):
  """ln_projection's LayerNorm holds a whole row: a a power of two to 256; slots past 128 columns go in 128s."""
  return a & (a - 1) == 0 and 16 <= a <= 256 and (n is None or projection_supported(n) or n in (256, 384, 512))


def _ln_proj_kernel(x_ref, s_ref, offset_ref, *refs, eps, m, block_m, slots, heads, f8_slots):
  # refs: w [slots, a, n] if slots, wb [heads, a] if heads, then the outputs in the same order, the e4m3 one last
  refs = list(refs)
  w_ref = refs.pop(0) if slots else None
  wb_ref = refs.pop(0) if heads else None
  p_ref = refs.pop(0) if slots else None
  b_ref = refs.pop(0) if heads else None
  p8_ref = refs.pop(0) if f8_slots else None
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  y = ln_rows(x, s_ref, offset_ref, eps).astype(x_ref.dtype)
  half = [g for g in range(slots) if g not in f8_slots]
  n = w_ref.shape[2] if slots else 0
  # a slot past 128 columns as several dots on the same normed tile: a whole one would outgrow shared memory
  chunk = min(n, 128)
  for g in range(slots):
    for c0 in range(0, n, chunk):
      out = _dot(y, w_ref[g, :, pl.ds(c0, chunk)])
      if g in f8_slots:
        plgpu.store(p8_ref.at[:, pl.ds(f8_slots.index(g) * n + c0, chunk)], e4m3(out), mask=rows[:, None])
      else:
        plgpu.store(p_ref.at[:, pl.ds(half.index(g) * n + c0, chunk)], out.astype(p_ref.dtype), mask=rows[:, None])
  for h in range(heads):
    # a few heads: row reductions, a dot would pad them to 16 columns
    b = jnp.sum(y.astype(jnp.float32) * wb_ref[h][None, :].astype(jnp.float32), axis=1)
    plgpu.store(b_ref.at[h], b.astype(b_ref.dtype), mask=rows)


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "num_warps", "num_stages", "f8_slots"))
def ln_projection(x, ln_scale, ln_offset, w=None, wb=None, *, eps=1e-5, block_m=32, num_warps=8, num_stages=2,
                  f8_slots=()):
  """LN(x) projected by each slot of w and by wb, from one read of x [M, a].

  w [G, a, n] gives p [M, G*n], the packed projection; wb [a, h] gives the bias channel-major [h, M].
  Returns (p, bias), None for an absent weight. f8_slots, slots of w written e4m3 to a third output
  [M, len(f8_slots)*n] in that order, leave p with the others.
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
    out_specs.append(pl.BlockSpec((block_m, (slots - len(f8_slots)) * n), lambda i: (i, 0)))
    out_shape.append(jax.ShapeDtypeStruct((m, (slots - len(f8_slots)) * n), x.dtype))
  if heads:
    in_specs.append(pl.BlockSpec((heads, a), lambda i: (0, 0)))
    args.append(wb.T)
    out_specs.append(pl.BlockSpec((heads, block_m), lambda i: (0, i)))
    out_shape.append(jax.ShapeDtypeStruct((heads, m), x.dtype))
  if f8_slots:
    out_specs.append(pl.BlockSpec((block_m, len(f8_slots) * n), lambda i: (i, 0)))
    out_shape.append(jax.ShapeDtypeStruct((m, len(f8_slots) * n), jnp.float8_e4m3fn))
  # swept on the L40S at the triangle attention's 4 x 128 outputs: 30% over 64 rows x 4 warps
  out = pl.pallas_call(
      functools.partial(_ln_proj_kernel, eps=eps, m=m, block_m=block_m, slots=slots, heads=heads,
                        f8_slots=f8_slots),
      grid=(pl.cdiv(m, block_m),),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shape,
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="ln_projection",
  )(x, ln_scale, ln_offset, *args)
  out = list(out)
  p, b = (out.pop(0) if slots else None), (out.pop(0) if heads else None)
  return (p, b, out.pop(0)) if f8_slots else (p, b)


def ref_ln_projection(x, ln_scale, ln_offset, w=None, wb=None, *, eps=1e-5, f8_slots=()):
  y = ref_ln(x.astype(jnp.float32), ln_scale, ln_offset, eps)
  b = None if wb is None else (y @ wb.astype(jnp.float32)).T
  if w is None:
    return None, b
  p = jnp.einsum('ma,gan->mgn', y, w.astype(jnp.float32))
  half = [g for g in range(w.shape[0]) if g not in f8_slots]
  flat = lambda gs: p[:, list(gs)].reshape(x.shape[0], -1)
  return (flat(half), b, flat(f8_slots)) if f8_slots else (flat(half), b)


def _out_proj_kernel(o_ref, w_ref, b_ref, *rest, m, block_m, block_k):
  r_ref, out_ref = rest if len(rest) == 2 else (None, rest[0])
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m

  # the weight streamed block_k rows at a time: a whole one past 128 channels outgrows shared memory
  def body(k, acc):
    ks = pl.ds(pl.multiple_of(k * block_k, block_k), block_k)
    return acc + _dot(plgpu.load(o_ref.at[:, ks], mask=rows[:, None], other=0.0), w_ref[ks, :])

  acc = jnp.zeros((block_m, w_ref.shape[1]), jnp.float32)
  out = jax.lax.fori_loop(0, o_ref.shape[1] // block_k, body, acc) + b_ref[...][None, :].astype(jnp.float32)
  if r_ref is not None:
    out = out + plgpu.load(r_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  plgpu.store(out_ref, out.astype(out_ref.dtype), mask=rows[:, None])


def output_projection_supported(k, c):
  """k a multiple of 64 or a whole tile, c a power of two from 16 to 256 (the fp32 accumulator's registers)."""
  return (k % 64 == 0 or projection_supported(k)) and c & (c - 1) == 0 and 16 <= c <= 256


@functools.partial(jax.jit, static_argnames=("block_m", "num_warps", "num_stages"))
def output_projection(o, w, b, residual=None, *, block_m=64, num_warps=None, num_stages=2):
  """o [M, k] @ w [k, c] + b, plus residual [M, c] when given, in one pass.

  The result has residual's dtype, else o's.
  """
  m, k = o.shape
  c = w.shape[1]
  block_k = min(k, 64)
  if compute_capability() == 80 and c > 128:
    # an A100 at the MSA's 508,000 x 256, bit-identical: 1.13 -> 0.75 ms
    block_m, num_warps, num_stages = 128, num_warps or 8, 3
  num_warps = num_warps or (8 if c > 128 else 4)
  rows = pl.BlockSpec((block_m, c), lambda i: (i, 0))
  extra = [] if residual is None else [residual]
  return pl.pallas_call(
      functools.partial(_out_proj_kernel, m=m, block_m=block_m, block_k=block_k),
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

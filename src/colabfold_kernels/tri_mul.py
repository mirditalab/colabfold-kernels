"""Pallas/Triton fused gated dual projection for AF2 triangle-multiplication.

Inference (forward) only, no backward pass

Computes the masked, sigmoid-gated dual projection in one kernel so the [M, 2c]
projection/gate intermediates are never materialized. The triangle matmul itself
stays a native XLA batched GEMM.

  out[m, p] = mask[m] * (x[m]·wp[:,p] + bp[p]) * sigmoid(x[m]·wg[:,p] + bg[p])
where x = layer-normed pair [M=N*N, K=c_z], wp/wg [K, P=2*c_i].
"""
import functools
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from colabfold_kernels.device import is_rocm, multiprocessor_count, shared_memory_limit
from colabfold_kernels.transition import fast, ln_rows, ref_ln, sigmoid, unless_padding
from colabfold_kernels.tri_flash import e4m3

# jnp.dot is stable across jax versions, contrary to pl.dot/plgpu.dot
_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


def _gdp_kernel_tiled(x_ref, wp_ref, bp_ref, wg_ref, bg_ref, mask_ref, o_ref, *,
                      activation, block_k):
  # one [BM, BN] output tile; looping over K lets Triton pipeline the loads
  def body(k, acc):
    ks = pl.ds(pl.multiple_of(k * block_k, block_k), block_k)
    x = x_ref[:, ks]
    return acc[0] + _dot(x, wp_ref[ks, :]), acc[1] + _dot(x, wg_ref[ks, :])
  zero = jnp.zeros((x_ref.shape[0], wp_ref.shape[1]), jnp.float32)
  proj, gate = jax.lax.fori_loop(0, x_ref.shape[1] // block_k, body, (zero, zero))
  proj += bp_ref[...][None, :].astype(jnp.float32)
  gate += bg_ref[...][None, :].astype(jnp.float32)
  o = mask_ref[...][:, None].astype(jnp.float32) * proj * fast(activation)(gate)
  o_ref[...] = o.astype(o_ref.dtype)


def _half(x, wp, bp, wg, bg, mask, activation, in_kernel=True):
  proj = _dot(x, wp) + bp[None, :].astype(jnp.float32)
  gate = _dot(x, wg) + bg[None, :].astype(jnp.float32)
  # fast()'s tanh forms are Pallas-only on CUDA
  return mask[:, None].astype(jnp.float32) * proj * (fast(activation) if in_kernel else activation)(gate)


def _gdp_kernel_split_cm(x_ref, wpl_ref, bpl_ref, wgl_ref, bgl_ref,
                         wpr_ref, bpr_ref, wgr_ref, bgr_ref, mask_ref,
                         ol_ref, or_ref, *, activation, chunk=None):
  # _gdp_kernel_split, stored channel-major [ci, BM] for the triangle einsum.
  # chunk: the output columns stream in chunks, for widths whose whole weights do not fit
  x = x_ref[...]
  m = mask_ref[...]
  if chunk is not None:
    def body(j, carry):
      js = pl.ds(pl.multiple_of(j * chunk, chunk), chunk)
      for w_p, b_p, w_g, b_g, o in ((wpl_ref, bpl_ref, wgl_ref, bgl_ref, ol_ref),
                                    (wpr_ref, bpr_ref, wgr_ref, bgr_ref, or_ref)):
        o[js, :] = _half(x, w_p[:, js], b_p[js], w_g[:, js], b_g[js], m, activation).T.astype(o.dtype)
      return carry

    jax.lax.fori_loop(0, ol_ref.shape[0] // chunk, body, 0)
    return
  ol_ref[...] = _half(x, wpl_ref[...], bpl_ref[...], wgl_ref[...], bgl_ref[...],
                      m, activation).T.astype(ol_ref.dtype)
  or_ref[...] = _half(x, wpr_ref[...], bpr_ref[...], wgr_ref[...], bgr_ref[...],
                      m, activation).T.astype(or_ref.dtype)


def _ln_gdp_kernel(x_ref, s_ref, offset_ref, wpl_ref, bpl_ref, wgl_ref, bgl_ref,
                   wpr_ref, bpr_ref, wgr_ref, bgr_ref, mask_ref, ol_ref, or_ref, *, activation, eps, m, block_m,
                   row_axis=0, m_out=None, fp8=False, chunk=None):
  # _gdp_kernel_split_cm on LN(x), so the normed pair never leaves the SM; rows past m are masked, not padded
  # m_out: rows written, past m zeros (their mask reads 0), the padding an FP8 GEMM's 16-byte rows want
  # chunk: the output columns stream in chunks, for widths whose whole weights do not fit
  # a block whose mask is all zero (padding) writes the zeros it would compute, without reading x or the weights
  row = pl.program_id(row_axis) * block_m + jnp.arange(block_m)
  rows, out_rows = row < m, row < (m_out or m)
  # below zero: outside what the contraction reads, not written; inside a computed block it masks like zero
  mask = jnp.maximum(plgpu.load(mask_ref, mask=rows, other=0), 0)
  cast = e4m3 if fp8 else (lambda t: t.astype(ol_ref.dtype))
  halves = ((wpl_ref, bpl_ref, wgl_ref, bgl_ref, ol_ref), (wpr_ref, bpr_ref, wgr_ref, bgr_ref, or_ref))

  def padding():
    for *_, o in halves:
      plgpu.store(o, cast(jnp.zeros(o.shape, jnp.float32)), mask=out_rows[None, :])

  def compute():
    x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
    x = ln_rows(x, s_ref, offset_ref, eps).astype(x_ref.dtype)
    if chunk is None:
      for w_p, b_p, w_g, b_g, o in halves:
        plgpu.store(o, cast(_half(x, w_p[...], b_p[...], w_g[...], b_g[...], mask, activation).T),
                    mask=out_rows[None, :])
      return

    def body(j, carry):
      js = pl.ds(pl.multiple_of(j * chunk, chunk), chunk)
      for w_p, b_p, w_g, b_g, o in halves:
        plgpu.store(o.at[js, :], cast(_half(x, w_p[:, js], b_p[js], w_g[:, js], b_g[js], mask, activation).T),
                    mask=out_rows[None, :])
      return carry

    jax.lax.fori_loop(0, ol_ref.shape[0] // chunk, body, 0)

  # the rows past m up to m_out are the FP8 GEMM's zero padding: written
  unless_padding(mask_ref, rows, compute, padding, other=0)


def _ln_gdp_persistent_kernel(x_ref, wp_ref, wg_ref, fold_ref, mask_ref, ol_ref, or_ref, *,
                              activation, eps, m, block_m, width, streams):
  # _ln_gdp_kernel, persistent: program (stream, group) owns `width` output columns of one half and every streams-th
  # row block, loading its weights once; the LN is folded into them (_ln_gdp_persistent), so raw rows feed the MMA
  stream, group = pl.program_id(0), pl.program_id(1)
  cols = pl.ds(pl.multiple_of(group * width, width), width)
  wp_s, wg_s = wp_ref[:, cols], wg_ref[:, cols]
  sum_p, sum_g, bias_p, bias_g = (fold_ref[i, cols] for i in range(4))
  left = ol_ref.shape[0] // width  # groups writing the left half; the rest write the right
  out_cols = pl.ds(pl.multiple_of(group % left * width, width), width)

  def block(j, carry):
    row = (stream + j * streams) * block_m + jnp.arange(block_m)
    rows = row < m
    span = pl.ds(pl.multiple_of((stream + j * streams) * block_m, block_m), block_m)

    def store(val):
      @pl.when(group < left)
      def _():
        plgpu.store(ol_ref.at[out_cols, span], val.astype(ol_ref.dtype), mask=rows[None, :])

      @pl.when(group >= left)
      def _():
        plgpu.store(or_ref.at[out_cols, span], val.astype(or_ref.dtype), mask=rows[None, :])

    def compute():
      x = plgpu.load(x_ref.at[span, :], mask=rows[:, None], other=0.0)
      xf = x.astype(jnp.float32)
      mean = jnp.mean(xf, axis=-1, keepdims=True)
      d = xf - mean
      rstd = jax.lax.rsqrt(jnp.mean(d * d, axis=-1, keepdims=True) + eps)
      proj = rstd * (_dot(x, wp_s) - mean * sum_p[None, :]) + bias_p[None, :]
      gate = rstd * (_dot(x, wg_s) - mean * sum_g[None, :]) + bias_g[None, :]
      # below zero: outside what the contraction reads, not written; inside a computed block it masks like zero
      mask = jnp.maximum(plgpu.load(mask_ref.at[span], mask=rows, other=0), 0)
      store((mask[:, None].astype(jnp.float32) * proj * fast(activation)(gate)).T)

    unless_padding(mask_ref.at[span], rows, compute, lambda: store(jnp.zeros((width, block_m), jnp.float32)), other=0)
    return carry

  count = (pl.cdiv(m, block_m) - stream + streams - 1) // streams
  jax.lax.fori_loop(0, count, block, 0)


# swept on sm_120 at c = 128 (N = 256 and 1024): two programs per SM, 64 columns each; more of either was slower
_PERSISTENT_WIDTH, _PERSISTENT_PER_SM = 64, 2


def _persistent_supported(k, ci):
  # 64 channels ran 25-33% slower than _ln_gdp_kernel
  return k == 128 and ci == 128 and multiprocessor_count() is not None


def _ln_gdp_persistent(x, ln_scale, ln_offset, wp, bp, wg, bg, mask, *, eps, block_m):
  """ln_gated_dual_proj's flat form [M, K] -> left, right [ci, M] with _ln_gdp_persistent_kernel."""
  M = x.shape[0]
  ci = wp.shape[1] // 2
  groups = 2 * ci // _PERSISTENT_WIDTH
  streams = max(1, min(pl.cdiv(M, block_m), multiprocessor_count() * _PERSISTENT_PER_SM // groups))
  (wp_s, fp), (wg_s, fg) = _fold(wp, ln_scale, ln_offset, bp, x.dtype), _fold(wg, ln_scale, ln_offset, bg, x.dtype)
  fold = jnp.stack([fp[0], fg[0], fp[1], fg[1]])
  return pl.pallas_call(
      functools.partial(_ln_gdp_persistent_kernel, activation=jax.nn.sigmoid, eps=eps, m=M, block_m=block_m,
                        width=_PERSISTENT_WIDTH, streams=streams),
      grid=(streams, groups),
      out_shape=[jax.ShapeDtypeStruct((ci, M), x.dtype)] * 2,
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1), name="ln_gated_dual_proj",
  )(x, wp_s, wg_s, fold, mask)


def _fold(w, scale, offset, b, dtype):
  """LN(x) w + b = rstd (x w' - mean colsum(w')) + offset w + b with w' = scale w, so the raw rows feed the MMA:
  w' in dtype and [colsum(w'), offset w + b], the colsum of the rounded w' the MMA reads."""
  # behind a barrier: XLA's excess precision drops a bare cast round trip, reduce_precision flushes f16 subnormals
  ws = jax.lax.optimization_barrier((scale.astype(jnp.float32)[:, None] * w.astype(jnp.float32)).astype(dtype))
  return ws, jnp.stack([ws.astype(jnp.float32).sum(0),
                        offset.astype(jnp.float32) @ w.astype(jnp.float32) + b.astype(jnp.float32)])


def _row_stats(x, axis, eps):
  """[mean, 1 / sigma] of x's rows over `axis` in float32, [2, M]: a folded LN's statistics, one read of x in XLA, so
  the tiled kernels' K loops are only MMAs (gathered there, every column tile repeated them and ran 1.5x slower)."""
  xf = x.astype(jnp.float32)
  mean = xf.mean(axis)
  return jnp.stack([mean, jax.lax.rsqrt(jnp.mean(jnp.square(xf - jnp.expand_dims(mean, axis)), axis) + eps)])


def _tile(cols_outer):
  """In a kernel: (row tile, column tile), the grid's axes in either order."""
  return (pl.program_id(1), pl.program_id(0)) if cols_outer else (pl.program_id(0), pl.program_id(1))


def _tile_grid(rows, cols, cols_outer):
  return (cols, rows) if cols_outer else (rows, cols)


def _ln_gdp_tiled_kernel(x_ref, st_ref, wp_ref, wg_ref, fp_ref, fg_ref, mask_ref, l_ref, r_ref, *, activation, m, k,
                         ci, block_m, block_n, block_k, cols_outer):
  # _ln_gdp_kernel as a tiled GEMM for wide pairs: program (rows, columns) loops over K with the LN folded (_fold, its
  # statistics from _row_stats), so any multiple of 32 channels runs and no weight streams per chunk
  i, j = _tile(cols_outer)
  span = pl.ds(pl.multiple_of(i * block_m, block_m), block_m)
  rows = i * block_m + jnp.arange(block_m) < m
  cols = pl.ds(pl.multiple_of(j * block_n, block_n), block_n)
  left = ci // block_n  # column blocks writing the left half; the rest write the right
  out_cols = pl.ds(pl.multiple_of(j % left * block_n, block_n), block_n)

  def store(val):
    @pl.when(j < left)
    def _():
      plgpu.store(l_ref.at[out_cols, span], val.astype(l_ref.dtype), mask=rows[None, :])

    @pl.when(j >= left)
    def _():
      plgpu.store(r_ref.at[out_cols, span], val.astype(r_ref.dtype), mask=rows[None, :])

  def compute():
    proj, gate = _folded_gemms(x_ref, st_ref, wp_ref, wg_ref, fp_ref, fg_ref, span, rows, cols, k, block_k)
    # below zero: outside what the contraction reads, not written; inside a computed block it masks like zero
    mask = jnp.maximum(plgpu.load(mask_ref.at[span], mask=rows, other=0), 0).astype(jnp.float32)
    store((mask[:, None] * proj * fast(activation)(gate)).T)

  unless_padding(mask_ref.at[span], rows, compute, lambda: store(jnp.zeros((block_n, block_m), jnp.float32)), other=0)


def _folded_gemms(x_ref, st_ref, wp_ref, wg_ref, fp_ref, fg_ref, span, rows, cols, k, block_k):
  """In a kernel: LN(x) wp + bp and LN(x) wg + bg for the rows `span` and columns `cols`, as GEMMs over K on the raw x
  with the weights folded (_fold) and the rows' statistics from _row_stats."""
  def body(t, acc):
    ks = pl.ds(pl.multiple_of(t * block_k, block_k), block_k)
    x = plgpu.load(x_ref.at[span, ks], mask=rows[:, None], other=0.0)
    return acc[0] + _dot(x, wp_ref[ks, cols]), acc[1] + _dot(x, wg_ref[ks, cols])

  zero = jnp.zeros((rows.shape[0], cols.size), jnp.float32)
  acc_p, acc_g = jax.lax.fori_loop(0, k // block_k, body, (zero, zero))
  return _unfold(acc_p, st_ref, fp_ref, span, rows, cols), _unfold(acc_g, st_ref, fg_ref, span, rows, cols)


def _unfold(acc, st_ref, f_ref, span, rows, cols):
  """In a kernel: rstd (acc - mean colsum) + bias, a folded GEMM's LN applied after it (_fold, _row_stats)."""
  mean = plgpu.load(st_ref.at[0, span], mask=rows, other=0.0)[:, None]
  rstd = plgpu.load(st_ref.at[1, span], mask=rows, other=0.0)[:, None]
  return rstd * (acc - mean * f_ref[0, cols][None, :]) + f_ref[1, cols][None, :]


# swept on sm_120 at 256-512 channels, N = 384 and 768: 128 x 64 tiles
_TILED = (128, 64)


def tiled_supported(k, ci):
  """The tiled kernels' widths, to 512; not on ROCm, where they are untuned."""
  return k % 32 == 0 and ci % _TILED[1] == 0 and max(k, ci) <= 512 and not is_rocm()


def _ln_gdp_tiled(x, ln_scale, ln_offset, wp, bp, wg, bg, mask, *, eps):
  """ln_gated_dual_proj's flat form [M, K] -> left, right [ci, M] with _ln_gdp_tiled_kernel."""
  M, K = x.shape
  ci = wp.shape[1] // 2
  block_m, block_n = _TILED
  cols_outer = K > 320  # swept at N = 384: rows outer at 320 channels, columns outer at 448
  (wp_s, fp), (wg_s, fg) = _fold(wp, ln_scale, ln_offset, bp, x.dtype), _fold(wg, ln_scale, ln_offset, bg, x.dtype)
  return pl.pallas_call(
      functools.partial(_ln_gdp_tiled_kernel, activation=jax.nn.sigmoid, m=M, k=K, ci=ci, block_m=block_m,
                        block_n=block_n, block_k=32, cols_outer=cols_outer),
      grid=_tile_grid(pl.cdiv(M, block_m), 2 * ci // block_n, cols_outer),
      out_shape=[jax.ShapeDtypeStruct((ci, M), x.dtype)] * 2,
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3), name="ln_gated_dual_proj",
  )(x, _row_stats(x, 1, eps), wp_s, wg_s, fp, fg, mask)


def _gdp_kernel_split(x_ref, wpl_ref, bpl_ref, wgl_ref, bgl_ref,
                      wpr_ref, bpr_ref, wgr_ref, bgr_ref, mask_ref,
                      ol_ref, or_ref, *, activation):
  # Computes the left [BM, ci] and right [BM, ci] gated projections as two
  # separate dots (weights pre-split on the host) and writes them to two
  # contiguous outputs -- so the triangle einsum consumes them directly with no
  # post-kernel strided slice/copy of the combined [N,N,2ci] tensor.
  x = x_ref[...]
  m = mask_ref[...]
  ol_ref[...] = _half(x, wpl_ref[...], bpl_ref[...], wgl_ref[...], bgl_ref[...],
                      m, activation).astype(ol_ref.dtype)
  or_ref[...] = _half(x, wpr_ref[...], bpr_ref[...], wgr_ref[...], bgr_ref[...],
                      m, activation).astype(or_ref.dtype)


# (rows, warps, column chunk) of the streamed kernels, swept on gfx1151 at 256 and 512 channels; 128 rows run out of LDS
_STREAM = (64, 8, 32)


def stream_supported(*widths, widest=512):
  """Streamed column chunks: power-of-two widths from 16 to `widest`."""
  return all(w & (w - 1) == 0 and 16 <= w <= widest for w in widths)


def _gdp_tiled(x, wp, bp, wg, bg, mask, activation):
  M, K = x.shape
  P = wp.shape[1]
  block_k = 32 if K % 32 == 0 else K
  block_n = 64 if P % 64 == 0 else 32 if P % 32 == 0 else P
  block_m = 128
  Mp = -(-M // block_m) * block_m
  x = jnp.pad(x, [(0, Mp - M), (0, 0)])
  mask = jnp.pad(mask, [(0, Mp - M)])
  # column tiles outermost, so the row tiles sharing an x block run together
  out = pl.pallas_call(
      functools.partial(_gdp_kernel_tiled, activation=activation, block_k=block_k),
      grid=(P // block_n, Mp // block_m),
      in_specs=[
          pl.BlockSpec((block_m, K), lambda j, i: (i, 0)),
          pl.BlockSpec((K, block_n), lambda j, i: (0, j)),
          pl.BlockSpec((block_n,), lambda j, i: (j,)),
          pl.BlockSpec((K, block_n), lambda j, i: (0, j)),
          pl.BlockSpec((block_n,), lambda j, i: (j,)),
          pl.BlockSpec((block_m,), lambda j, i: (i,)),
      ],
      out_specs=pl.BlockSpec((block_m, block_n), lambda j, i: (i, j)),
      out_shape=jax.ShapeDtypeStruct((Mp, P), x.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=4),
      name="gated_dual_proj",
  )(x, wp, bp, wg, bg, mask)
  return out[:M]


@functools.partial(jax.jit,
                   static_argnames=("block_m", "split", "channel_major", "activation"))
def gated_dual_proj(x, wp, bp, wg, bg, mask, *, block_m=None, split=False,
                    channel_major=False, activation=jax.nn.sigmoid):
  # x [M,K]; wp/wg [K,P]; bp/bg [P]; mask [M].
  # split=False -> gated [M,P]; split=True -> (left [M,P/2], right [M,P/2]),
  # the two halves written contiguously so the caller needs no slice.
  if not split:
    return _gdp_tiled(x, wp, bp, wg, bg, mask, activation)
  M, K = x.shape
  P = wp.shape[1]
  ci = P // 2
  # at 512 channels the streamed projection loses to XLA (gfx1151: 18.9 against 9.9 ms at 256 tokens)
  stream = channel_major and max(K, ci) > 128 and stream_supported(K, ci, widest=256)
  if max(K, ci) > 128 and not stream:  # whole weights would not fit: XLA
    g = _half(x, wp, bp, wg, bg, mask, activation, in_kernel=False).astype(x.dtype)
    return (g[:, :ci].T, g[:, ci:].T) if channel_major else (g[:, :ci], g[:, ci:])
  # swept on sm_120: 128 rows x 8 warps is 17% faster, but only worth it without a pad copy; ROCm keeps 64 x 4
  wide = not is_rocm()
  if stream:
    block_m = _STREAM[0]
    cp = plgpu.CompilerParams(num_warps=_STREAM[1], num_stages=1)
  else:
    block_m = block_m or (128 if wide and M % 128 == 0 else 64)
    cp = (plgpu.CompilerParams(num_warps=8, num_stages=2) if wide and block_m == 128
          else plgpu.CompilerParams(num_warps=4, num_stages=1))
  Mp = -(-M // block_m) * block_m
  x = jnp.pad(x, [(0, Mp - M), (0, 0)])
  mask = jnp.pad(mask, [(0, Mp - M)])
  xspec = pl.BlockSpec((block_m, K), lambda i: (i, 0))
  mspec = pl.BlockSpec((block_m,), lambda i: (i,))
  wpl, wpr = wp[:, :ci], wp[:, ci:]
  wgl, wgr = wg[:, :ci], wg[:, ci:]
  bpl, bpr = bp[:ci], bp[ci:]
  bgl, bgr = bg[:ci], bg[ci:]
  wspec = pl.BlockSpec((K, ci), lambda i: (0, 0))
  bspec = pl.BlockSpec((ci,), lambda i: (0,))
  split_in = [xspec, wspec, bspec, wspec, bspec, wspec, bspec, wspec, bspec, mspec]
  if channel_major:
    bs = pl.BlockSpec((ci, block_m), lambda i: (0, i))
    left, right = pl.pallas_call(
        functools.partial(_gdp_kernel_split_cm, activation=activation, chunk=_STREAM[2] if stream else None),
        grid=(Mp // block_m,), in_specs=split_in, out_specs=[bs, bs],
        out_shape=[jax.ShapeDtypeStruct((ci, Mp), x.dtype),
                   jax.ShapeDtypeStruct((ci, Mp), x.dtype)],
        compiler_params=cp, name="gated_dual_proj_split_cm",
    )(x, wpl, bpl, wgl, bgl, wpr, bpr, wgr, bgr, mask)
    return left[:, :M], right[:, :M]
  bs = pl.BlockSpec((block_m, ci), lambda i: (i, 0))
  left, right = pl.pallas_call(
      functools.partial(_gdp_kernel_split, activation=activation),
      grid=(Mp // block_m,), in_specs=split_in, out_specs=[bs, bs],
      out_shape=[jax.ShapeDtypeStruct((Mp, ci), x.dtype),
                 jax.ShapeDtypeStruct((Mp, ci), x.dtype)],
      compiler_params=cp, name="gated_dual_proj_split",
  )(x, wpl, bpl, wgl, bgl, wpr, bpr, wgr, bgr, mask)
  return left[:M], right[:M]


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "fp8", "transpose", "pad"))
def ln_gated_dual_proj(x, ln_scale, ln_offset, wp, bp, wg, bg, mask, *, eps=1e-5, block_m=64, fp8=False,
                       transpose=False, pad=1):
  """gated_dual_proj(LN(x), ..., split=True, channel_major=True) in one pass over the raw input x [M, K].

  Returns left, right [P/2, M]. Past 128 channels the plain form is tiled (tiled_supported); the others stream
  power-of-two widths.
  fp8 writes them e4m3, unscaled and saturating. The pair x [A, B, K] with mask [A, B] gives left, right
  [P/2, A, B'], B' being B padded with zeros to a multiple of pad; transpose reads it transposed, [P/2, B, A'], so
  the incoming edges' operands come out contracted along their last axis too.
  """
  K = x.shape[-1]
  ci = wp.shape[1] // 2
  # past 128 channels the weights stream by output column
  chunk = None if epilogue_supported(K, ci, ci) else _STREAM[2]
  block_m, num_warps = (_STREAM[0], _STREAM[1]) if chunk else (block_m, 4)
  if chunk is None and not fp8 and (x.ndim == 2 or not transpose and pad == 1) and _persistent_supported(K, ci):
    left, right = _ln_gdp_persistent(x.reshape(-1, K), ln_scale, ln_offset, wp, bp, wg, bg, mask.reshape(-1), eps=eps,
                                     block_m=block_m)
    return left.reshape(ci, *x.shape[:-1]), right.reshape(ci, *x.shape[:-1])
  if chunk is not None and not fp8 and (x.ndim == 2 or not transpose and pad == 1) and tiled_supported(K, ci):
    left, right = _ln_gdp_tiled(x.reshape(-1, K), ln_scale, ln_offset, wp, bp, wg, bg, mask.reshape(-1), eps=eps)
    return left.reshape(ci, *x.shape[:-1]), right.reshape(ci, *x.shape[:-1])
  dtype = jnp.float8_e4m3fn if fp8 else x.dtype
  halves = [t for w, b in ((wp, bp), (wg, bg)) for t in (w[:, :ci], b[:ci])]
  halves += [t for w, b in ((wp, bp), (wg, bg)) for t in (w[:, ci:], b[ci:])]
  m_out = None
  if x.ndim == 3 and transpose:
    # grid (a block, b): each block is block_m rows of column b of the pair, stored at (b, a); b innermost, so
    # consecutive blocks read neighbouring rows of the pair
    A, B = x.shape[0], x.shape[1]
    m, m_out = A, -(-A // pad) * pad
    grid, row_axis = (pl.cdiv(m_out, block_m), B), 0
    vec = lambda n: pl.BlockSpec((n,), lambda a, b: (0,))
    wspec = pl.BlockSpec((K, ci), lambda a, b: (0, 0))
    xspec = pl.BlockSpec((block_m, None, K), lambda a, b: (a, b, 0))
    mspec = pl.BlockSpec((block_m, None), lambda a, b: (a, b))
    out = pl.BlockSpec((ci, None, block_m), lambda a, b: (0, b, a))
    out_shape = [jax.ShapeDtypeStruct((ci, B, m_out), dtype)] * 2
  elif x.ndim == 3:
    # grid (a, b block): block_m rows along row a of the pair, stored at (a, b)
    A, B = x.shape[0], x.shape[1]
    m, m_out = B, -(-B // pad) * pad
    grid, row_axis = (A, pl.cdiv(m_out, block_m)), 1
    vec = lambda n: pl.BlockSpec((n,), lambda a, b: (0,))
    wspec = pl.BlockSpec((K, ci), lambda a, b: (0, 0))
    xspec = pl.BlockSpec((None, block_m, K), lambda a, b: (a, b, 0))
    mspec = pl.BlockSpec((None, block_m), lambda a, b: (a, b))
    out = pl.BlockSpec((ci, None, block_m), lambda a, b: (0, a, b))
    out_shape = [jax.ShapeDtypeStruct((ci, A, m_out), dtype)] * 2
  else:
    M = x.shape[0]
    grid, m, row_axis = (pl.cdiv(M, block_m),), M, 0
    vec = lambda n: pl.BlockSpec((n,), lambda i: (0,))
    wspec = pl.BlockSpec((K, ci), lambda i: (0, 0))
    xspec = pl.BlockSpec((block_m, K), lambda i: (i, 0))
    mspec = pl.BlockSpec((block_m,), lambda i: (i,))
    out = pl.BlockSpec((ci, block_m), lambda i: (0, i))
    out_shape = [jax.ShapeDtypeStruct((ci, M), dtype)] * 2
  left, right = pl.pallas_call(
      functools.partial(_ln_gdp_kernel, activation=jax.nn.sigmoid, eps=eps, m=m, block_m=block_m,
                        row_axis=row_axis, m_out=m_out, fp8=fp8, chunk=chunk),
      grid=grid,
      in_specs=[xspec, vec(K), vec(K)] + [wspec, vec(ci)] * 4 + [mspec],
      out_specs=[out, out],
      out_shape=out_shape,
      # swept on the L40S: 15% faster than 128 rows x 8 warps
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1), name="ln_gated_dual_proj",
  )(x, ln_scale, ln_offset, *halves, mask)
  return left, right


def ref_ln_gdp(x, ln_scale, ln_offset, wp, bp, wg, bg, mask, *, eps=1e-5, transpose=False, pad=1, **_):
  pair = x.shape[:2] if x.ndim == 3 else None
  if pair is not None:
    x, mask = (jnp.swapaxes(x, 0, 1), mask.T) if transpose else (x, mask)
    pair = x.shape[:2]
    x, mask = x.reshape(-1, x.shape[-1]), mask.reshape(-1)
  y = ref_gdp(ref_ln(x.astype(jnp.float32), ln_scale, ln_offset, eps), wp.astype(jnp.float32),
              bp.astype(jnp.float32), wg.astype(jnp.float32), bg.astype(jnp.float32), mask.astype(jnp.float32))
  ci = wp.shape[1] // 2
  left, right = y[:, :ci].T, y[:, ci:].T
  if pair is None:
    return left, right
  rows = -(-pair[1] // pad) * pad
  shape = lambda h: jnp.pad(h.reshape(ci, *pair), ((0, 0), (0, 0), (0, rows - pair[1])))
  return shape(left), shape(right)


def _epilogue_kernel(x_ref, a_ref, s_ref, o_ref_ln, wo_ref, bo_ref, wg_ref, bg_ref, *rest,
                     eps, m, block_m, input_ln=False, add_input=False, residual=False, keep=False, chunk=None):
  # input_ln: a_ref is the raw module input and rest leads with its LayerNorm's scale and offset
  # chunk: the output columns stream in chunks (_epilogue_stream_rows), for widths whose whole weights do not fit
  rest = list(rest)
  as_ref, ao_ref = (rest.pop(0), rest.pop(0)) if input_ln else (None, None)
  r_ref = rest.pop(0) if residual else None
  keep_ref = rest.pop(0) if keep else None
  o_ref, = rest
  # x_ref [C, block_m] channel-major triangle product; a_ref [block_m, Ca] normed input
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  # the product read as zeros where keep leaves a row out: a contraction over the live length does not write it
  xrows = rows if keep_ref is None else rows & (plgpu.load(keep_ref, mask=rows, other=0) > 0)
  refs = (x_ref, a_ref, s_ref, o_ref_ln, wo_ref, bo_ref, wg_ref, bg_ref, as_ref, ao_ref, r_ref, o_ref, rows, xrows,
          eps)
  compute = (functools.partial(_epilogue_rows, *refs, input_ln, add_input) if chunk is None
             else functools.partial(_epilogue_stream_rows, *refs, chunk, input_ln, add_input))

  def padding():  # rows keep leaves out: zeros, or the residual (add_input's raw input), which the aliased output holds
    if not (add_input or r_ref is not None):
      plgpu.store(o_ref, jnp.zeros(o_ref.shape, o_ref.dtype), mask=rows[:, None])

  unless_padding(keep_ref, rows, compute, padding)


def _epilogue_rows(x_ref, a_ref, s_ref, o_ref_ln, wo_ref, bo_ref, wg_ref, bg_ref, as_ref, ao_ref, r_ref, o_ref, rows,
                   xrows, eps, input_ln, add_input):
  x = plgpu.load(x_ref, mask=xrows[None, :], other=0.0).astype(jnp.float32).T
  y = ln_rows(x, s_ref, o_ref_ln, eps)
  out = _dot(y.astype(wo_ref.dtype), wo_ref[...]) + bo_ref[...][None, :].astype(jnp.float32)
  a = plgpu.load(a_ref, mask=rows[:, None], other=0.0)
  if input_ln:
    a_raw = a.astype(jnp.float32)
    a = ln_rows(a_raw, as_ref, ao_ref, eps).astype(a_ref.dtype)
  gate = _dot(a, wg_ref[...]) + bg_ref[...][None, :].astype(jnp.float32)
  out = out * sigmoid(gate)
  if r_ref is not None:
    out = out + plgpu.load(r_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  if add_input:
    out = out + a_raw
  plgpu.store(o_ref, out.astype(o_ref.dtype), mask=rows[:, None])


def _epilogue_stream_rows(x_ref, a_ref, s_ref, o_ref_ln, wo_ref, bo_ref, wg_ref, bg_ref, as_ref, ao_ref, r_ref, o_ref,
                          rows, xrows, eps, chunk, input_ln, add_input):
  # _epilogue_rows for widths whose whole weights do not fit: the normed rows stay, the output columns stream
  x = plgpu.load(x_ref, mask=xrows[None, :], other=0.0).astype(jnp.float32).T
  y = ln_rows(x, s_ref, o_ref_ln, eps).astype(wo_ref.dtype)
  a = plgpu.load(a_ref, mask=rows[:, None], other=0.0)
  if input_ln:
    a = ln_rows(a.astype(jnp.float32), as_ref, ao_ref, eps).astype(a_ref.dtype)

  def body(j, carry):
    js = pl.ds(pl.multiple_of(j * chunk, chunk), chunk)
    out = _dot(y, wo_ref[:, js]) + bo_ref[js][None, :].astype(jnp.float32)
    out = out * sigmoid(_dot(a, wg_ref[:, js]) + bg_ref[js][None, :].astype(jnp.float32))
    if r_ref is not None:
      out = out + plgpu.load(r_ref.at[:, js], mask=rows[:, None], other=0.0).astype(jnp.float32)
    if add_input:
      out = out + plgpu.load(a_ref.at[:, js], mask=rows[:, None], other=0.0).astype(jnp.float32)
    plgpu.store(o_ref.at[:, js], out.astype(o_ref.dtype), mask=rows[:, None])
    return carry

  jax.lax.fori_loop(0, o_ref.shape[1] // chunk, body, 0)


def epilogue_supported(c, ca, o):
  """Whole-tile dots: power-of-two widths of at least 16 (Triton's dot K), at most 128 (shared memory)."""
  return all(w & (w - 1) == 0 and 16 <= w <= 128 for w in (c, ca, o))


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "num_warps", "num_stages"))
def trimul_epilogue(x_cm, act, ln_scale, ln_offset, wo, bo, wg, bg, residual=None, *, eps=1e-5,
                    block_m=64, num_warps=4, num_stages=2):
  """Fused triangle-multiplication output: sigmoid(act @ wg + bg) * (LN(x_cm.T) @ wo + bo).

  x_cm [C, M] channel-major einsum output; act [M, Ca] the normed module input;
  wo [C, O]; wg [Ca, O]. Replaces the transpose, center LayerNorm, output and gating projections.
  residual [M, O], when given, is added: the module's residual update in the same pass.
  Widths the kernel does not take run the reference in XLA. The result has the residual's dtype, else act's.
  """
  c, m = x_cm.shape
  ca, o = act.shape[1], wo.shape[1]
  dtype = act.dtype if residual is None else residual.dtype
  if not (epilogue_supported(c, ca, o) or stream_supported(c, ca, o)):
    return ref_trimul_epilogue(x_cm, act, ln_scale, ln_offset, wo, bo, wg, bg, residual, eps=eps).astype(dtype)
  extra = [] if residual is None else [residual]
  return _epilogue(x_cm, act, [ln_scale, ln_offset, wo, bo, wg, bg], extra, dtype, eps=eps, block_m=block_m,
                   num_warps=num_warps, num_stages=num_stages)


def _epilogue(x_cm, act, params, extra, dtype, *, eps, block_m, num_warps, num_stages, input_ln=False,
              add_input=False, keep=None):
  c, m = x_cm.shape
  ca, o = act.shape[1], params[2].shape[1]
  stream = not epilogue_supported(c, ca, o)
  if stream:
    (block_m, num_warps, chunk), num_stages = _STREAM, 1
    # Triton on NVIDIA stages ~5 bytes a row and channel in shared memory (64 rows x 512: 160 KB): rows to fit the card
    smem = None if is_rocm() else shared_memory_limit()
    while smem and block_m > 16 and block_m * max(c, ca, o) * 5 > smem:
      block_m //= 2
  kernel = functools.partial(_epilogue_kernel, eps=eps, m=m, block_m=block_m, input_ln=input_ln, add_input=add_input,
                             residual=bool(extra), keep=keep is not None, chunk=chunk if stream else None)
  vec = lambda n: pl.BlockSpec((n,), lambda i: (0,))
  rows = pl.BlockSpec((block_m, o), lambda i: (i, 0))
  return pl.pallas_call(
      kernel,
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((c, block_m), lambda i: (0, i)),
                pl.BlockSpec((block_m, ca), lambda i: (i, 0)), vec(c), vec(c),
                pl.BlockSpec((c, o), lambda i: (0, 0)), vec(o),
                pl.BlockSpec((ca, o), lambda i: (0, 0)), vec(o)]
               + [vec(ca)] * (2 if input_ln else 0) + [rows] * len(extra)
               + ([] if keep is None else [pl.BlockSpec((block_m,), lambda i: (i,))]),
      out_specs=rows,
      out_shape=jax.ShapeDtypeStruct((m, o), dtype),
      # the output is the residual's buffer: add_input's raw input, else the residual
      input_output_aliases={1: 0} if add_input else ({2 + len(params): 0} if extra else {}),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="trimul_epilogue",
  )(x_cm, act, *params, *extra, *([] if keep is None else [keep.reshape(m)]))


def _epilogue_tiled_kernel(x_ref, a_ref, sx_ref, sa_ref, wo_ref, wg_ref, fo_ref, fg_ref, keep_ref, o_ref, *, m, c, ca,
                          block_m, block_n, block_k, add_input):
  # trimul_epilogue_ln as two tiled GEMMs for wide pairs (output projection over c, gate over ca), LNs folded (_fold).
  # Not aliased: a row block's column tiles would overwrite input another tile still reads, so its padding copies
  i, j = _tile(True)
  span = pl.ds(pl.multiple_of(i * block_m, block_m), block_m)
  rows = i * block_m + jnp.arange(block_m) < m
  cols = pl.ds(pl.multiple_of(j * block_n, block_n), block_n)
  keep = plgpu.load(keep_ref.at[span], mask=rows, other=0)
  # the product read as zeros where keep leaves a row out: a contraction over the live length does not write it
  xrows = rows & (keep > 0)

  def gemm(ref, w_ref, n, channel_major):
    def body(t, acc):
      ks = pl.ds(pl.multiple_of(t * block_k, block_k), block_k)
      v = (plgpu.load(ref.at[ks, span], mask=xrows[None, :], other=0.0).T if channel_major
           else plgpu.load(ref.at[span, ks], mask=rows[:, None], other=0.0))
      return acc + _dot(v.astype(w_ref.dtype), w_ref[ks, cols])

    return jax.lax.fori_loop(0, n // block_k, body, jnp.zeros((block_m, block_n), jnp.float32))

  @pl.when(jnp.max(keep.astype(jnp.float32)) > 0)
  def _():
    out = _unfold(gemm(x_ref, wo_ref, c, True), sx_ref, fo_ref, span, rows, cols)
    out = out * sigmoid(_unfold(gemm(a_ref, wg_ref, ca, False), sa_ref, fg_ref, span, rows, cols))
    if add_input:
      out = out + plgpu.load(a_ref.at[span, cols], mask=rows[:, None], other=0.0).astype(jnp.float32)
    plgpu.store(o_ref.at[span, cols], out.astype(o_ref.dtype), mask=rows[:, None])

  @pl.when(jnp.max(keep.astype(jnp.float32)) <= 0)
  def _():
    pad = (plgpu.load(a_ref.at[span, cols], mask=rows[:, None], other=0.0) if add_input
           else jnp.zeros((block_m, block_n), o_ref.dtype))
    plgpu.store(o_ref.at[span, cols], pad.astype(o_ref.dtype), mask=rows[:, None])


def _epilogue_tiled_supported(c, ca, o):
  # at 256 channels and N = 768 the streamed epilogue is faster (1.5 against 1.75 ms); untuned on ROCm
  return (all(w % 64 == 0 for w in (c, ca, o)) and not all(w & (w - 1) == 0 and w <= 256 for w in (c, ca, o))
          and not is_rocm())


def _epilogue_tiled(x_cm, x, in_scale, in_offset, ln_scale, ln_offset, wo, bo, wg, bg, keep, *, eps, add_input):
  c, m = x_cm.shape
  ca, o = x.shape[1], wo.shape[1]
  assert not add_input or o == ca, (o, ca)  # the residual is read over the output's columns
  (wo_s, fo), (wg_s, fg) = _fold(wo, ln_scale, ln_offset, bo, x.dtype), _fold(wg, in_scale, in_offset, bg, x.dtype)
  keep = jnp.ones((m,), x.dtype) if keep is None else keep.reshape(m)
  # the product's statistics over what the kernel reads: rows keep leaves out as zeros
  sx = _row_stats(jnp.where(keep[None, :] > 0, x_cm, 0), 0, eps)
  return pl.pallas_call(
      functools.partial(_epilogue_tiled_kernel, m=m, c=c, ca=ca, block_m=_TILED[0], block_n=_TILED[1], block_k=64,
                        add_input=add_input),
      grid=_tile_grid(pl.cdiv(m, _TILED[0]), o // _TILED[1], True),
      out_shape=jax.ShapeDtypeStruct((m, o), x.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3), name="trimul_epilogue",
  )(x_cm, x, sx, _row_stats(x, 1, eps), wo_s, wg_s, fo, fg, keep)


@functools.partial(jax.jit, static_argnames=("eps", "add_input", "block_m", "num_warps", "num_stages"))
def trimul_epilogue_ln(x_cm, x, in_scale, in_offset, ln_scale, ln_offset, wo, bo, wg, bg, keep=None, *, eps=1e-5,
                       add_input=False, block_m=64, num_warps=4, num_stages=2):
  """trimul_epilogue gated by LN(x), recomputed from the raw module input x [M, Ca] with in_scale/in_offset.

  add_input adds x, the module's residual update, from the tile already loaded. keep [M]: the rows that matter; a row
  block with none (the bucket's padding) writes x unchanged with add_input, else zeros, without the work.
  """
  c, ca, o = x_cm.shape[0], x.shape[1], wo.shape[1]
  if not (epilogue_supported(c, ca, o) or stream_supported(c, ca, o) or _epilogue_tiled_supported(c, ca, o)):
    return ref_trimul_epilogue_ln(x_cm, x, in_scale, in_offset, ln_scale, ln_offset, wo, bo, wg, bg, keep, eps=eps,
                                  add_input=add_input).astype(x.dtype)
  if not epilogue_supported(c, ca, o) and _epilogue_tiled_supported(c, ca, o):
    return _epilogue_tiled(x_cm, x, in_scale, in_offset, ln_scale, ln_offset, wo, bo, wg, bg, keep, eps=eps,
                           add_input=add_input)
  return _epilogue(x_cm, x, [ln_scale, ln_offset, wo, bo, wg, bg, in_scale, in_offset], [], x.dtype, eps=eps,
                   block_m=block_m, num_warps=num_warps, num_stages=num_stages, input_ln=True, add_input=add_input,
                   keep=keep)


def ref_trimul_epilogue_ln(x_cm, x, in_scale, in_offset, ln_scale, ln_offset, wo, bo, wg, bg, keep=None, *, eps=1e-5,
                           add_input=False):
  if keep is not None:  # the kernels read the product as zeros where keep leaves a row out
    x_cm = jnp.where(keep.reshape(1, -1) > 0, x_cm, 0)
  act = ref_ln(x.astype(jnp.float32), in_scale, in_offset, eps)
  return ref_trimul_epilogue(x_cm, act, ln_scale, ln_offset, wo, bo, wg, bg, x if add_input else None, eps=eps)


def ref_trimul_epilogue(x_cm, act, ln_scale, ln_offset, wo, bo, wg, bg, residual=None, *, eps=1e-5):
  y = ref_ln(x_cm.T.astype(jnp.float32), ln_scale, ln_offset, eps)
  out = y @ wo.astype(jnp.float32) + bo.astype(jnp.float32)
  out = out * jax.nn.sigmoid(act.astype(jnp.float32) @ wg.astype(jnp.float32) + bg.astype(jnp.float32))
  return out if residual is None else out + residual.astype(jnp.float32)


def ref_gdp(x, wp, bp, wg, bg, mask):
  proj = x @ wp + bp
  gate = x @ wg + bg
  return (mask[:, None] * proj * jax.nn.sigmoid(gate))


# Fused LayerNorm: bf16 in/out, fp32 internal. One kernel replaces the
# bf16->fp32 upcast + layernorm + fp32->bf16 downcast (kills the convert ops
# and never materializes the fp32 tensor), while keeping fp32 accuracy.
def _ln_kernel(x_ref, scale_ref, offset_ref, o_ref, *, eps, m, c, block_m):
  bc = x_ref.shape[-1]
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m     # [block_m]
  if bc == c:
    x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    d = x - mean
    var = jnp.mean(d * d, axis=-1, keepdims=True)
    scale, offset = scale_ref[...], offset_ref[...]
    valid = rows[:, None]
  else:
    # c is not a power of two: tile to the next one, mask the lanes past c
    cols = jnp.arange(bc) < c
    valid = rows[:, None] & cols[None, :]
    x = plgpu.load(x_ref, mask=valid, other=0.0).astype(jnp.float32)
    mean = jnp.sum(x, axis=-1, keepdims=True) / c
    d = jnp.where(cols[None, :], x - mean, 0.0)
    var = jnp.sum(d * d, axis=-1, keepdims=True) / c
    scale = plgpu.load(scale_ref, mask=cols, other=0.0)
    offset = plgpu.load(offset_ref, mask=cols, other=0.0)
  y = d * jax.lax.rsqrt(var + eps)
  y = y * scale[None, :].astype(jnp.float32) + offset[None, :].astype(jnp.float32)
  plgpu.store(o_ref, y.astype(o_ref.dtype), mask=valid)


# Triton keeps the tile in registers, so a wider one only spills and compiles slower
_TILE_BYTES = 32 * 1024


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "out_dtype"))
def pallas_layer_norm(x, scale, offset, *, eps=1e-5, block_m=64, out_dtype=None):
  # x [..., c]; normalize over last axis; scale/offset [c]. Returns x's shape, in out_dtype or x's dtype.
  c = x.shape[-1]
  bc = max(16, 1 << (c - 1).bit_length())        # Triton tiles are powers of two
  xf = x.reshape(-1, c)
  m = xf.shape[0]
  rows = max(1, _TILE_BYTES // (bc * x.dtype.itemsize))
  block_m = min(block_m, 1 << (rows.bit_length() - 1))
  out = pl.pallas_call(
      functools.partial(_ln_kernel, eps=eps, m=m, c=c, block_m=block_m),
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((block_m, bc), lambda i: (i, 0)),
                pl.BlockSpec((bc,), lambda i: (0,)),
                pl.BlockSpec((bc,), lambda i: (0,))],
      out_specs=pl.BlockSpec((block_m, bc), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((m, c), out_dtype or x.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
      name="pallas_layer_norm",
  )(xf, scale, offset)
  return out.reshape(x.shape)


def ref_layer_norm(x, scale, offset, eps=1e-5):
  xf = x.astype(jnp.float32)
  mean = jnp.mean(xf, -1, keepdims=True)
  var = jnp.mean((xf - mean) ** 2, -1, keepdims=True)
  y = (xf - mean) * jax.lax.rsqrt(var + eps) * scale + offset
  return y.astype(x.dtype)


if __name__ == "__main__":
  import time
  key = jax.random.PRNGKey(0)
  M, K, P = 1000 * 1000, 128, 256
  x = (jax.random.normal(key, (M, K)) * 0.3).astype(jnp.bfloat16)
  wp = (jax.random.normal(key, (K, P)) * 0.1).astype(jnp.bfloat16)
  wg = (jax.random.normal(key, (K, P)) * 0.1).astype(jnp.bfloat16)
  bp = (jax.random.normal(key, (P,)) * 0.1).astype(jnp.bfloat16)
  bg = (jax.random.normal(key, (P,)) * 0.1).astype(jnp.bfloat16)
  mask = (jax.random.uniform(key, (M,)) > 0.1).astype(jnp.bfloat16)
  o = gated_dual_proj(x, wp, bp, wg, bg, mask)
  r = ref_gdp(x.astype(jnp.float32), wp.astype(jnp.float32), bp.astype(jnp.float32),
              wg.astype(jnp.float32), bg.astype(jnp.float32), mask.astype(jnp.float32))
  err = float(jnp.max(jnp.abs(o.astype(jnp.float32) - r)))
  print(f"gated_dual_proj bf16 err={err:.3e}")
  f = jax.jit(gated_dual_proj)
  o = f(x, wp, bp, wg, bg, mask); o.block_until_ready()
  t = time.time()
  for _ in range(50): o = f(x, wp, bp, wg, bg, mask)
  o.block_until_ready()
  print(f"per-call {(time.time()-t)/50*1000:.2f} ms")


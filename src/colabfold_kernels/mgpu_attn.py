"""Mosaic GPU packed attention, packed_attn.packed_attention's contract: C warpgroups share a head's bias tile, one TMA loader."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from colabfold_kernels.tri_flash import LOG2E, NEG

L = plgpu.Layout


def supported(head_dim, dtype, seq_len):
  """A swizzle-tile head dim and a sequence the (8, w) TMA tiles divide; fused_ops picks it on sm_12x only."""
  return head_dim in (16, 32, 64) and jnp.dtype(dtype) in (jnp.bfloat16, jnp.float16) and seq_len % 8 == 0


# with qk8, q.k^T runs e4m3 (q pre-scaled by sm_scale * log2(e)): 7-10% faster on sm_120; e4m3 p.v measured no faster
F8 = jnp.float8_e4m3fn


def _acc_to_lhs(x, dtype):
  """The FA2 register trick: an accumulator's two adjacent n8 tiles are one k16 LHS tile."""
  import jax.experimental.mosaic.gpu as mgpu

  @plgpu.inline_mgpu(arg_types=(L.MMA_ACC(dtype),),
                     return_type=plgpu.ShapeDtypeStruct(x.shape, dtype, layout=L.MMA_LHS(dtype)))
  def f(ctx, a):
    r = a.registers  # acc (1, n/8, 1, 1, 2, ...) -> lhs (1, n/16, 1, 2, 2, ...)
    pairs = r.reshape(r.shape[0], r.shape[1] // 2, 2, *r.shape[2:])
    r = np.moveaxis(pairs, 2, 3).reshape(r.shape[0], r.shape[1] // 2, r.shape[2], 2, r.shape[4], *r.shape[5:])
    return mgpu.FragmentedArray(_registers=r, _layout=mgpu.MMALayouts(mgpu.utils.dtype_to_ir_type(dtype)).lhs,
                                _is_signed=None)
  return f(x)


def _tiles(cols, itemsize):
  w = min(cols, 128 // itemsize)
  return plgpu.TilingTransform((8, w)), plgpu.SwizzleTransform(w * itemsize)


@functools.partial(jax.jit, static_argnames=("sm_scale", "column", "C", "block_q", "block_k", "stages", "group", "trim"))
def packed_attention(p, bias, kmask, gate_bias=None, qk8=None, *, sm_scale, column=False, C=None, block_q=128,
                     block_k=32, stages=None, group=None, trim=False):
  """p [X, Y, G, H, D], bias [H, S, S], kmask [batch, S], gate_bias [H, D] iff G == 4; rows attend along Y.

  qk8 [X, Y, 2, H, D] holds q, already scaled by sm_scale * log2(e), and k in e4m3; p keeps v and the gate.
  group batch groups run each query block back to back, so they read its bias tile from L2 once.
  trim: packed_attn.packed_attention's; a batch group runs the key blocks its longest row needs.
  """
  X, Y, G, H, D = p.shape
  fp8 = qk8 is not None
  G += 2 if fp8 else 0
  gated, dt = G == 4, p.dtype
  # p's slots for v and the gate
  vs, gs = (0, 1) if fp8 else (2, 3)
  assert G in (3, 4) and (gate_bias is not None) == gated
  nb, s = (Y, X) if column else (X, Y)
  # per-warpgroup MMA rows; a key-mask copy of C x block_k floats is a multiple of 128 bytes
  assert block_q % 64 == 0 and block_k % 32 == 0
  # 3 warpgroups fit the smem and registers at D <= 32, 2 above
  C = C or (3 if D <= 32 else 2)
  stages = stages or (3 if D <= 32 else 2)
  bq, bk = block_q, block_k
  nbg, nq, nk = -(-nb // C), -(-s // bq), -(-s // bk)
  # sweep on sm_120 (4 heads of 32, 256 rows): 2 is level to 1,048 queries and best at 3,000, 8 best from 4,000
  # (182 -> 240 TF/s there), as localfold switches at 3,500
  group = min(group or (8 if s >= 3500 else 2), nbg)
  nsg = -(-nbg // group)
  bias = jnp.pad(bias.astype(dt), [(0, 0), (0, nq * bq - s), (0, nk * bk - s)])
  # masked keys share NEG, keys past s get -inf: a fully masked row averages its s keys, as in tri_flash
  kadd = jnp.pad(jnp.where(kmask, 0., NEG).astype(jnp.float32), [(0, nbg * C - nb), (0, nk * bk - s)],
                 constant_values=-jnp.inf)
  # a TMA copy moves a multiple of 128 bytes
  gb = jnp.pad((gate_bias if gated else jnp.zeros((H, D))).astype(jnp.float32), [(0, 0), (0, max(0, 32 - D))])
  ACC, LHS, RHS = L.MMA_ACC(dt), L.MMA_LHS(dt), L.MMA_RHS(dt)
  qkt = F8 if fp8 else dt
  QK_LHS, QK_RHS = L.MMA_LHS(qkt), L.MMA_RHS(qkt)
  # 65536 registers: the memory warpgroup keeps 24 per thread, compute warpgroups split the rest
  regs = min(240, (65536 // 128 - 24) // C // 8 * 8)
  if trim:
    from colabfold_kernels.packed_attn import trim_limits
    ends, lim = trim_limits(kmask, bk)
    # a batch group's key blocks, its longest row's, then the query limit, then each row's own
    ends = jnp.pad(ends, (0, nbg * C - nb))
    lims = jnp.concatenate([ends.reshape(nbg, C).max(axis=1), lim, ends])
  else:
    lims = jnp.full((nbg + 1,), nk, jnp.int32).at[nbg].set(s)

  def rows_of(ref, b, *idx):  # the [s, ...] sequence of batch b
    return ref.at[(slice(None), b, *idx) if column else (b, slice(None), *idx)]

  def kernel(p_ref, bias_ref, kadd_ref, gb_ref, qk_ref, lims_ref, o_ref, buf, *scratch):
    # an e4m3 q has a buffer of its own; a bf16 one shares the gate's and output's
    q_smem, scratch = (scratch[0], scratch[1:]) if fp8 else (buf, scratch)
    b_smem, k_smem, v_smem, ka_smem, gb_smem, q_bar, g_bar, produced, consumed = scratch
    # grid (batch supergroup x head, query block x batch group): a query block's batch groups run adjacent and share
    # its bias tile in L2, a supergroup's K/V stay there across its query blocks
    sh, qg, wg = lax.axis_index("sh"), lax.axis_index("qg"), lax.axis_index("wg")
    h, i = lax.rem(sh, H), lax.div(qg, group)
    bg = lax.div(sh, H) * group + lax.rem(qg, group)
    b0 = bg * C
    qrows = pl.ds(i * bq, bq)

    def key_blocks():  # this CTA's key blocks: none past the query limit
      return jnp.where(i * bq < lims_ref[nbg], lims_ref[bg], 0) if trim else nk

    # a supergroup past the batch groups has nothing to do
    @pl.when((wg < C) & (bg < nbg))
    def _compute():
      plgpu.set_max_registers(regs, action="increase")
      b = jnp.minimum(b0 + wg, nb - 1)
      plgpu.copy_gmem_to_smem(rows_of(qk_ref, b, 0, h).at[qrows], q_smem.at[wg], q_bar.at[wg])
      if gated:
        plgpu.copy_gmem_to_smem(gb_ref.at[h], gb_smem.at[wg], q_bar.at[wg])
      plgpu.barrier_wait(q_bar.at[wg])
      q = plgpu.load(q_smem.at[wg], layout=QK_LHS)
      if not fp8:
        q = (q.astype(jnp.float32) * (sm_scale * LOG2E)).astype(dt)
      # the gate reuses q's buffer: the copy's warpgroup sync orders it after every warp's q load
      if gated:
        plgpu.copy_gmem_to_smem(rows_of(p_ref, b, gs, h).at[qrows], buf.at[wg], g_bar.at[wg])

      def step(j, carry):
        o, m, l = carry
        slot = lax.rem(j, stages)
        plgpu.barrier_wait(produced.at[slot])
        kt = plgpu.load(plgpu.transpose_ref(k_smem.at[slot, wg], (1, 0)), layout=QK_RHS)
        ka = plgpu.load(ka_smem.at[slot, wg], layout=ACC.reduce(0))
        # scores in log2 units, accumulated onto bias + key mask
        acc = plgpu.load(b_smem.at[slot], layout=ACC).astype(jnp.float32) * LOG2E + lax.broadcast_in_dim(ka, (bq, bk), [1])
        sc = plgpu.mma(acc, q, kt)
        m_new = jnp.maximum(m, sc.max(axis=1))
        corr = jnp.exp2(m - m_new)
        pr = jnp.exp2(sc - lax.broadcast_in_dim(m_new, sc.shape, [0]))
        l = l * corr + pr.sum(axis=1)
        vv = plgpu.load(v_smem.at[slot, wg], layout=RHS)
        o = plgpu.mma(o * lax.broadcast_in_dim(corr, o.shape, [0]), _acc_to_lhs(pr.astype(dt), dt), vv)
        plgpu.barrier_arrive(consumed.at[slot])  # every thread arrives (128 C), after its own loads
        return o, m_new, l

      o0 = plgpu.layout_cast(jnp.zeros((bq, D), jnp.float32), ACC)
      m0 = plgpu.layout_cast(jnp.full((bq,), -jnp.inf, jnp.float32), ACC.reduce(1))
      l0 = plgpu.layout_cast(jnp.zeros((bq,), jnp.float32), ACC.reduce(1))
      if trim:
        # a group no padding trims runs the static loop, the others the traced bound two steps at a time: a plain
        # traced loop cost 11% at no padding, this pair of loops nothing
        n = key_blocks()

        def trimmed():
          carry = lax.fori_loop(0, n // 2, lambda j, c: step(2 * j + 1, step(2 * j, c)), (o0, m0, l0))
          return lax.cond(n % 2 == 1, lambda: step(n - 1, carry), lambda: carry)

        o, _, l = lax.cond(n == nk, lambda: lax.fori_loop(0, nk, step, (o0, m0, l0)), trimmed)
      else:
        o, _, l = lax.fori_loop(0, nk, step, (o0, m0, l0))
      if trim:
        # no key ran, or none of this row's (a padded row in a group that has keys): zeros, not 0 / 0 or an average
        l = jnp.where(l > 0, l, jnp.inf)
        o = o * jnp.where(lims_ref[nbg + 1 + b0 + wg] > 0, 1.0, 0.0)
      o = o / lax.broadcast_in_dim(l, o.shape, [0])
      if gated:
        plgpu.barrier_wait(g_bar.at[wg])
        g = plgpu.load(buf.at[wg], layout=ACC).astype(jnp.float32)
        g = g + lax.broadcast_in_dim(plgpu.load(gb_smem.at[wg, pl.ds(0, D)], layout=ACC.reduce(0)), o.shape, [1])
        o = o / (1. + jnp.exp2(-g * LOG2E))
      buf.at[wg][...] = o.astype(dt)
      plgpu.commit_smem()

      # a clamped row past nb computed NaN from its -inf key mask; it is never stored
      @pl.when(b0 + wg < nb)
      def _store():
        plgpu.copy_smem_to_gmem(buf.at[wg], rows_of(o_ref, b, h).at[qrows])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when((wg == C) & (bg < nbg))
    def _memory():
      plgpu.set_max_registers(24, action="decrease")

      def fetch(j):
        slot, keys = lax.rem(j, stages), pl.ds(j * bk, bk)
        plgpu.copy_gmem_to_smem(bias_ref.at[h, qrows, keys], b_smem.at[slot], produced.at[slot])
        plgpu.copy_gmem_to_smem(kadd_ref.at[pl.ds(b0, C), keys], ka_smem.at[slot], produced.at[slot])
        for w in range(C):
          b = jnp.minimum(b0 + w, nb - 1)
          plgpu.copy_gmem_to_smem(rows_of(qk_ref, b, 1, h).at[keys], k_smem.at[slot, w], produced.at[slot])
          plgpu.copy_gmem_to_smem(rows_of(p_ref, b, vs, h).at[keys], v_smem.at[slot, w], produced.at[slot])

      # one warp issues the copies: a warpgroup-wide copy syncs all 128 threads per copy
      @plgpu.warp_map
      def _per_warp(warp):
        @pl.when(warp == 0)
        def _issue():
          # traced key offsets: a tail tile past s is TMA zero fill, not a static bounds error
          n = key_blocks()
          pl.loop(0, jnp.minimum(stages, n) if trim else min(stages, nk))(fetch)

          @pl.loop(stages, jnp.maximum(n, stages) if trim else max(nk, stages))
          def _(j):
            plgpu.barrier_wait(consumed.at[lax.rem(j, stages)])
            fetch(j)

  def entry(*refs):
    pl.run_scoped(
        lambda *scratch: kernel(*refs, *scratch),
        plgpu.SMEM((C, bq, D), dt, transforms=_tiles(D, dt.itemsize)),
        *([plgpu.SMEM((C, bq, D), F8, transforms=_tiles(D, 1))] if fp8 else []),
        plgpu.SMEM((stages, bq, bk), dt, transforms=_tiles(bk, dt.itemsize)),
        plgpu.SMEM((stages, C, bk, D), qkt, transforms=_tiles(D, jnp.dtype(qkt).itemsize)),
        plgpu.SMEM((stages, C, bk, D), dt, transforms=_tiles(D, dt.itemsize)),
        plgpu.SMEM((stages, C, bk), jnp.float32),
        plgpu.SMEM((C, gb.shape[1]), jnp.float32),
        plgpu.Barrier(num_arrivals=1 + gated, num_barriers=C),
        plgpu.Barrier(num_barriers=C),
        plgpu.Barrier(num_arrivals=2 + 2 * C, num_barriers=stages),
        plgpu.Barrier(num_arrivals=C, num_barriers=stages),
        collective_axes="wg",
    )

  return plgpu.kernel(
      entry, out_type=jax.ShapeDtypeStruct((X, Y, H, D), dt), grid=(nsg * H, nq * group), grid_names=("sh", "qg"),
      num_threads=C + 1, thread_name="wg",
      compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane, approx_math=True),
  )(p, bias, kadd, gb, p if qk8 is None else qk8, lims)

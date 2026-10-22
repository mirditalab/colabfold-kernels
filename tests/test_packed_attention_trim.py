"""packed attention's trim: on a padded pair (tokens past n masked), the real rows match the untrimmed kernel and the
padded ones are zero or untrimmed; on a mask with holes, every row the trim does not zero matches. Both the Triton kernel and,
on sm_12x, the Mosaic GPU one; rows and columns, gated, bf16 and the e4m3 k/v / q/k slots. poison: p NaN past the live
block rounded to 128, what ln_projection leaves unwritten there; trimmed rows must still come out zero."""
import sys

import jax
import jax.numpy as jnp

from colabfold_kernels import fused_ops
from colabfold_kernels import packed_attn as pa
from colabfold_kernels.tri_flash import LOG2E, e4m3

H, D = 4, 32
fails = []


def check(name, o, base, ref, keep):
  """real rows: the untrimmed kernel's to a bf16 rounding (codegen of the shorter loop may reorder an FMA); padded
  rows: zero, or the untrimmed value where a live query block runs past the last token; all finite"""
  o, base = o.astype(jnp.float32), base.astype(jnp.float32)
  keep = jnp.broadcast_to(keep[:, :, None, None], o.shape)
  rel = float(jnp.max(jnp.where(keep, jnp.abs(o - base) / (jnp.abs(base) + 1e-3), 0)))
  near = jnp.abs(o - base) <= 2 ** -7 * (jnp.abs(base) + 1e-3)
  stray = float(jnp.max(jnp.where(keep | (o == 0) | near, 0, jnp.abs(o))))
  err = float(jnp.abs(jnp.where(keep, o - ref, 0)).max())
  zeros = float(jnp.mean(jnp.where(keep, 0, (o == 0)).astype(jnp.float32)) / max(float(jnp.mean(~keep)), 1e-9))
  finite = bool(jnp.isfinite(o).all())
  ok = rel <= 2 ** -7 and stray == 0 and err < 0.05 and finite
  print(f"{'ok  ' if ok else 'BAD '}{name}: real rows vs untrimmed rel {rel:.3g}, padded neither 0 nor untrimmed "
        f"{stray:.3g} ({zeros:.0%} zero), vs f32 {err:.3g}")
  if not ok:
    fails.append(name)


def case(S, n, column, mosaic, fp8, holes=False, poison=False):
  k = jax.random.split(jax.random.PRNGKey(S * 7 + n), 5)
  p = jax.random.normal(k[0], (S, S, 4, H, D)).astype(jnp.bfloat16)
  bias = jax.random.normal(k[1], (H, S, S)).astype(jnp.bfloat16)
  gb = jax.random.normal(k[2], (H, D)).astype(jnp.bfloat16)
  tok = jnp.arange(S) < n
  if holes:
    tok = tok & (jax.random.uniform(k[3], (S,)) > 0.2)
  kmask = tok[:, None] & tok[None, :]
  kw = dict(sm_scale=D ** -0.5, column=column)
  extra, pp = [], p
  if fp8:
    if mosaic:  # q pre-scaled and k in e4m3, p keeps v and the gate
      extra = [jnp.stack([e4m3(p[:, :, 0].astype(jnp.float32) * D ** -0.5 * LOG2E), e4m3(p[:, :, 1])], axis=2)]
      pp = p[:, :, 2:]
    else:  # k and v in e4m3, p keeps q and the gate
      extra = [jnp.stack([e4m3(p[:, :, 1]), e4m3(p[:, :, 2])], axis=2)]
      pp = p[:, :, jnp.array([0, 3])]
  f = fused_ops._mgpu_attn().packed_attention if mosaic else pa.packed_attention
  base = f(pp, bias, kmask, gb, *extra, **kw)
  if poison:
    out = jnp.arange(S) >= -(-n // 128) * 128
    pp = jnp.where((out[:, None] | out[None, :])[:, :, None, None, None], jnp.nan, pp).astype(pp.dtype)
  o = f(pp, bias, kmask, gb, *extra, trim=True, **kw)
  ref = pa.ref_packed_attention(p, bias, kmask, gb, **kw)
  # rows trim keeps: batch rows with a key, and sequence positions before the last token
  last = jnp.max(jnp.where(tok, jnp.arange(S) + 1, 0))
  rows = kmask.any(axis=1)[:, None] & (jnp.arange(S) < last)[None, :]
  keep = rows.T if column else rows
  name = f"{'mosaic' if mosaic else 'triton'} S={S} n={n} {'col' if column else 'row'}{' fp8' if fp8 else ''}" \
         f"{' holes' if holes else ''}{' poison' if poison else ''}"
  # e4m3 slots: the f32 reference is not the yardstick, the untrimmed kernel is
  check(name, o, base, base.astype(jnp.float32) if fp8 else ref, keep)


mosaic_ok = fused_ops._mgpu_attn() is not None
fp8_ok = fused_ops.fp8_attention_slots({'fp8': True}, None, D, 0) is not None or mosaic_ok
for mosaic in ([False, True] if mosaic_ok else [False]):
  for S, n in [(256, 256), (256, 255), (256, 129), (256, 128), (256, 127), (256, 31), (384, 257), (384, 1)]:
    for column in (False, True):
      case(S, n, column, mosaic, fp8=False)
  case(256, 200, False, mosaic, fp8=False, holes=True)
  case(256, 200, True, mosaic, fp8=False, holes=True)
  for S, n in [(384, 1), (384, 129), (256, 100)]:
    for column in (False, True):
      case(S, n, column, mosaic, fp8=False, poison=True)
  if fp8_ok:
    for column in (False, True):
      case(384, 300, column, mosaic, fp8=True)
print("PASS 0" if not fails else f"FAIL {len(fails)}")
sys.exit(bool(fails))

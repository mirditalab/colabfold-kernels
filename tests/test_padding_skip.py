"""Kernels that skip all-padding row blocks of a padded pair (n real tokens in a bucket of S, mask zero past n).
ln_gated_dual_proj (flat, pair, transposed, e4m3 with 16-row padding, streamed past 128 channels): outputs on real rows
match the float32 reference as before, padded ones are exactly zero, all finite. ln_projection, output_projection,
trimul_epilogue_ln and fused_glu_transition with keep: real rows bit-identical to the call without keep, padded rows
either that or what the skipped block writes (zeros, or the residual unchanged), all finite."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from colabfold_kernels import packed_attn as pa
from colabfold_kernels import transition as tr
from colabfold_kernels import tri_mul as tm
from colabfold_kernels.device import compute_capability, is_rocm

fails = []
rng = np.random.default_rng(0)
dt = jnp.bfloat16


def f(*shape, s=1.0):
  return jnp.asarray(rng.normal(size=shape) * s, dt)


def check(name, outs, refs, tol):
  worst = 0.0
  ok = True
  for o, r in zip(outs, refs):
    o, r = np.asarray(o.astype(jnp.float32)), np.asarray(r)
    pad = r == 0
    err = float(np.abs(o - r).max() / (np.abs(r).max() + 1e-6))
    ok &= bool(np.isfinite(o).all()) and err < tol and float(np.abs(o[pad]).max(initial=0)) == 0
    worst = max(worst, err)
  print(f"{'ok  ' if ok else 'BAD '}{name}: max err {worst:.3g}")
  if not ok:
    fails.append(name)


fp8 = compute_capability() is not None and compute_capability() >= 89 and not is_rocm()
for S, n in [(256, 256), (256, 200), (256, 129), (256, 63), (192, 65)]:
  for cz, ci in [(128, 128), (128, 256)]:
    tok = jnp.arange(S) < n
    mask = (tok[:, None] & tok[None, :]).astype(dt)
    x = f(S, S, cz, s=2.)
    g = (f(cz, s=.1) + 1, f(cz, s=.1), f(cz, 2 * ci, s=.1), f(2 * ci, s=.1), f(cz, 2 * ci, s=.1), f(2 * ci, s=.1))
    if not (tm.epilogue_supported(cz, ci, ci) or tm.stream_supported(cz, ci, widest=256)):
      continue
    out = tm.ln_gated_dual_proj(x.reshape(-1, cz), *g, mask.reshape(-1))
    check(f"ln_gdp flat S={S} n={n} {cz}->{ci}", out, tm.ref_ln_gdp(x.reshape(-1, cz), *g, mask.reshape(-1)), 0.02)
    for transpose in (False, True):
      out = tm.ln_gated_dual_proj(x, *g, mask, transpose=transpose)
      check(f"ln_gdp pair S={S} n={n} {cz}->{ci} {'T' if transpose else ''}", out,
            tm.ref_ln_gdp(x, *g, mask, transpose=transpose), 0.02)
      if fp8 and ci == 128:
        out = tm.ln_gated_dual_proj(x, *g, mask, fp8=True, transpose=transpose, pad=16)
        check(f"ln_gdp e4m3 S={S} n={n} {'T' if transpose else ''}", out,
              tm.ref_ln_gdp(x, *g, mask, transpose=transpose, pad=16), 0.08)


def check_keep(name, with_keep, without, keep, pad_value):
  ok = True
  for o, b, pv in zip(with_keep, without, pad_value):
    o, b = np.asarray(o.astype(jnp.float32)), np.asarray(b.astype(jnp.float32))
    pv = np.broadcast_to(np.asarray(jnp.asarray(pv).astype(jnp.float32)), o.shape)
    k = np.asarray(keep).astype(bool).reshape((-1,) + (1,) * (o.ndim - 1)) if o.shape[0] == keep.size else \
        np.asarray(keep).astype(bool).reshape((1, -1) + (1,) * (o.ndim - 2))
    k = np.broadcast_to(k, o.shape)
    real = float(np.abs(np.where(k, o - b, 0)).max())
    padded = np.where(k, True, (o == b) | (o == pv))
    ok &= real == 0 and bool(padded.all()) and bool(np.isfinite(o).all())
    skipped = float(np.mean(np.where(k, False, (o == pv) & (o != b))) / max(np.mean(~k), 1e-9))
  print(f"{'ok  ' if ok else 'BAD '}{name}: real rows identical {ok}, padded rows skipped {skipped:.0%}")
  if not ok:
    fails.append(name)


for S, n in [(256, 256), (256, 200), (256, 129), (192, 65)]:
  tok = jnp.arange(S) < n
  keep = (tok[:, None] & tok[None, :]).reshape(-1).astype(dt)
  m, c, h, d = S * S, 128, 4, 32
  x = f(m, c, s=2.)
  sc, of = f(c, s=.1) + 1, f(c, s=.1)
  w, wb = f(4, c, h * d, s=.1), f(c, h, s=.1)
  zero = jnp.zeros(())
  out = pa.ln_projection(x, sc, of, w, wb, keep)
  base = pa.ln_projection(x, sc, of, w, wb)
  check_keep(f"ln_projection S={S} n={n}", out, base, keep, [zero, zero])
  if fp8:
    out = pa.ln_projection(x, sc, of, w, wb, keep, f8_slots=(1, 2))
    base = pa.ln_projection(x, sc, of, w, wb, f8_slots=(1, 2))
    check_keep(f"ln_projection e4m3 S={S} n={n}", out, base, keep, [zero, zero, zero])
  o, wo, bo = f(m, h * d), f(h * d, c, s=.1), f(c, s=.1)
  check_keep(f"output_projection S={S} n={n}", [pa.output_projection(o, wo, bo, x, keep)],
             [pa.output_projection(o, wo, bo, x)], keep, [x])
  check_keep(f"output_projection no residual S={S} n={n}", [pa.output_projection(o, wo, bo, None, keep)],
             [pa.output_projection(o, wo, bo)], keep, [zero])
  for ci in (128, 256):
    tri = f(ci, m)
    g = (sc, of, f(ci, s=.1) + 1, f(ci, s=.1), f(ci, c, s=.1), f(c, s=.1), f(c, c, s=.1), f(c, s=.1))
    for add_input in (False, True):
      check_keep(f"trimul_epilogue_ln {ci} add_input={add_input} S={S} n={n}",
                 [tm.trimul_epilogue_ln(tri, x, *g, keep, add_input=add_input)],
                 [tm.trimul_epilogue_ln(tri, x, *g, add_input=add_input)], keep, [x if add_input else zero])
  wa, wb2, wo2 = f(c, 4 * c, s=.1), f(c, 4 * c, s=.1), f(4 * c, c, s=.1)
  for add_input in (False, True):
    check_keep(f"fused_glu_transition add_input={add_input} S={S} n={n}",
               [tr.fused_glu_transition(x, sc, of, wa, wb2, wo2, keep, add_input=add_input)],
               [tr.fused_glu_transition(x, sc, of, wa, wb2, wo2, add_input=add_input)], keep, [x if add_input else zero])
print("PASS 0" if not fails else f"FAIL {len(fails)}")
sys.exit(bool(fails))

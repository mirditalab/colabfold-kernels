"""Kernels that skip all-padding row blocks of a padded pair (n real tokens in a bucket of S, mask zero past n): their
outputs on real rows match the float32 reference as before, padded ones are exactly zero, all finite. ln_gated_dual_proj
(flat, pair, transposed, e4m3 with 16-row padding, streamed past 128 channels)."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

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
print("PASS 0" if not fails else f"FAIL {len(fails)}")
sys.exit(bool(fails))

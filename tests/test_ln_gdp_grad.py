"""layer_norm and gated_dual_proj gradients, against jax.grad of a jnp reference."""
import os, sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import fused_ops

def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all(): return float('inf')
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))

def ref_ln(x, scale, offset, *, eps=1e-5):
  xf = x.astype(jnp.float32)
  d = xf - jnp.mean(xf, -1, keepdims=True)
  y = d * jax.lax.rsqrt(jnp.mean(d * d, -1, keepdims=True) + eps) * scale + offset
  return y.astype(x.dtype)

def ref_gdp(x, wp, bp, wg, bg, mask, *, split=False, channel_major=False):
  xf = x.astype(jnp.float32)
  y = mask.astype(jnp.float32)[:, None] * (xf @ wp + bp) * jax.nn.sigmoid(xf @ wg + bg)
  y = y.astype(x.dtype)
  if not split:
    return y
  ci = y.shape[-1] // 2
  l, r = y[:, :ci], y[:, ci:]
  return (l.T, r.T) if channel_major else (l, r)

def dot(y, do):
  if isinstance(y, tuple):
    return sum(jnp.sum(u.astype(jnp.float32) * d.astype(jnp.float32)) for u, d in zip(y, do))
  return jnp.sum(y.astype(jnp.float32) * do.astype(jnp.float32))

bad = 0
rng = np.random.default_rng(0)
gc = {"use_pallas": True, "kernel_backend": "pallas"}
dt = jnp.bfloat16
f = lambda *sh: jnp.asarray(rng.standard_normal(sh) * .5, dt)

ln = fused_ops.layer_norm(gc, dt)
for M, C in ((4096, 128), (777, 64), (1024, 256)):
  x = f(M, C)
  sc = jnp.asarray(rng.standard_normal(C), jnp.float32)
  of = jnp.asarray(rng.standard_normal(C), jnp.float32)
  do = f(M, C)
  mk = lambda fn: jax.jit(jax.grad(lambda *a: dot(fn(*a, eps=1e-5), do), argnums=(0, 1, 2)))
  # the backward is derived from the reference, so pin the forward to it too
  fwd_err = rel(ln(x, sc, of, eps=1e-5), ref_ln(x, sc, of, eps=1e-5))
  errs = [rel(a, b) for a, b in zip(mk(ln)(x, sc, of), mk(ref_ln)(x, sc, of))]
  ok = max(errs) < 0.02 and fwd_err < 0.02
  bad += not ok
  print(f"  layer_norm M={M:5d} C={C:3d}   fwd {fwd_err:.1e}  dx {errs[0]:.1e}"
        f"  dscale {errs[1]:.1e}  doffset {errs[2]:.1e}   {'OK' if ok else 'FAIL'}")

gdp = fused_ops.gated_dual_proj(gc, dt)
for M, K, P, split, cm in ((2048, 128, 256, False, False), (2048, 128, 256, True, False),
                           (2048, 128, 256, True, True), (999, 64, 128, True, True)):
  x, wp, bp, wg, bg = f(M, K), f(K, P), f(P), f(K, P), f(P)
  mask = jnp.asarray(rng.random((M,)) > .1, dt)
  ci = P // 2
  do = ((f(ci, M), f(ci, M)) if cm else (f(M, ci), f(M, ci))) if split else f(M, P)
  def mk(fn):
    def loss(x, wp, bp, wg, bg):
      y = (fn(x, wp, bp, wg, bg, mask, split=True, channel_major=cm) if split
           else fn(x, wp, bp, wg, bg, mask, split=False))
      return dot(y, do)
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))
  kw = dict(split=True, channel_major=cm) if split else dict(split=False)
  yk, yr = gdp(x, wp, bp, wg, bg, mask, **kw), ref_gdp(x, wp, bp, wg, bg, mask, **kw)
  fwd_err = max(rel(a, b) for a, b in zip(yk, yr)) if split else rel(yk, yr)
  errs = [rel(a, b) for a, b in zip(mk(gdp)(x, wp, bp, wg, bg),
                                    mk(ref_gdp)(x, wp, bp, wg, bg))]
  ok = max(errs) < 0.03 and fwd_err < 0.03
  bad += not ok
  print(f"  gdp M={M:5d} K={K:3d} P={P:3d} split={int(split)} cm={int(cm)}  fwd {fwd_err:.1e}  " +
        "  ".join(f"{n} {e:.1e}" for n, e in zip(("dx", "dwp", "dbp", "dwg", "dbg"), errs)) +
        f"   {'OK' if ok else 'FAIL'}")
print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)

"""The CUDA gated dual projection as fused_ops selects it, forward and gradient.

  CK_SRC=<repo>/src KERNEL_DIR=<dir with libvolta_*.so> CC=75 python tests/test_legacy_gdp.py

CC picks the family: 75 takes the mma kernel, 70 the wmma one. The kernels run
on any card at or above the arch they were built for.
"""
import os, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import fused_ops, volta as V

KD = os.environ["KERNEL_DIR"]
V._package_path = lambda kernel, cc: os.path.join(KD, {
    "layer_norm": "libvolta_ops.so",
    "gated_dual_proj": "libvolta_ops.so" if cc >= 75 else "libvolta_gdp_wmma.so"}[kernel])
CC = int(os.environ.get("CC", "75"))
gc = {"use_pallas": True, "kernel_backend": "cuda_legacy", "compute_capability": CC}
dt = jnp.float16
rng = np.random.default_rng(0)
bad = 0


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all():
    return float("inf")
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))


def ref(x, wp, bp, wg, bg, mask, act, *, split=False, channel_major=False):
  xf = x.astype(jnp.float32)
  y = mask.astype(jnp.float32)[:, None] * (xf @ wp + bp) * act(xf @ wg + bg)
  y = y.astype(dt)
  if not split:
    return y
  ci = y.shape[-1] // 2
  l, r = y[:, :ci], y[:, ci:]
  return (l.T, r.T) if channel_major else (l, r)


def dot(y, d):
  if isinstance(y, tuple):
    return sum(jnp.sum(u.astype(jnp.float32) * v.astype(jnp.float32)) for u, v in zip(y, d))
  return jnp.sum(y.astype(jnp.float32) * d.astype(jnp.float32))


f = lambda *sh: jnp.asarray(rng.standard_normal(sh) * .5, dt)
for act, name in ((None, "default"), (jax.nn.sigmoid, "sigmoid"), (jax.nn.swish, "swish")):
  gdp = fused_ops.gated_dual_proj(gc, dt, act)
  if gdp is None:
    print(f"  {name:8s} no kernel selected   FAIL")
    bad += 1
    continue
  a = jax.nn.sigmoid if act is None else act
  for M, K, P, split, cm in ((640, 64, 128, True, True), (2048, 128, 256, True, False),
                             (999, 64, 128, False, False)):
    x, wp, bp, wg, bg = f(M, K), f(K, P), f(P), f(K, P), f(P)
    mask = jnp.asarray(rng.random((M,)) > .1, dt)
    kw = dict(split=True, channel_major=cm) if split else dict(split=False)
    ci = P // 2
    do = ((f(ci, M), f(ci, M)) if cm else (f(M, ci), f(M, ci))) if split else f(M, P)
    yk = gdp(x, wp, bp, wg, bg, mask, **kw)
    yr = ref(x, wp, bp, wg, bg, mask, a, **kw)
    fwd = max(rel(u, v) for u, v in zip(yk, yr)) if split else rel(yk, yr)
    mk = lambda fn: jax.jit(jax.grad(
        lambda *t: dot(fn(*t, mask, **kw), do), argnums=(0, 1, 2, 3, 4)))
    errs = [rel(u, v) for u, v in zip(mk(gdp)(x, wp, bp, wg, bg),
                                      mk(lambda *t, **k: ref(*t, a, **k))(x, wp, bp, wg, bg))]
    ok = fwd < 0.03 and max(errs) < 0.03
    bad += not ok
    print(f"  {name:8s} M={M:5d} K={K:3d} P={P:3d} split={int(split)} cm={int(cm)}  "
          f"fwd {fwd:.1e}  " + "  ".join(f"{n} {e:.1e}" for n, e in
                                         zip(("dx", "dwp", "dbp", "dwg", "dbg"), errs)) +
          f"   {'OK' if ok else 'FAIL'}")

# af3 hands the kernel a gated linear unit: no bias, no mask, one flat [M, K] block
K, P, M = 128, 256, 4096
for act, name in ((jax.nn.sigmoid, "sigmoid"), (jax.nn.swish, "swish")):
  gdp = fused_ops.gated_dual_proj(gc, dt, act)
  x, wg, wp = f(M, K), f(K, P), f(K, P)
  zero, keep = jnp.zeros((P,), jnp.float32), jnp.ones((M,), dt)
  got = gdp(x, wp, zero, wg, zero, keep, split=False)
  want = act(x.astype(jnp.float32) @ wg) * (x.astype(jnp.float32) @ wp)
  e = rel(got, want)
  ok = e < 0.03
  bad += not ok
  print(f"  glu {name:7s} M={M} K={K} P={P}  rel {e:.1e}   {'OK' if ok else 'FAIL'}")

# swish is unbounded where sigmoid is not, so check the float16 output range
gdp = fused_ops.gated_dual_proj(gc, dt, jax.nn.swish)
M, K, P = 256, 64, 128
for amp in (1, 4, 16, 64, 256):
  x = jnp.asarray(rng.standard_normal((M, K)) * .5 * amp, dt)
  wp, bp, wg, bg = f(K, P), f(P), f(K, P), f(P)
  mask = jnp.ones((M,), dt)
  y = np.asarray(gdp(x, wp, bp, wg, bg, mask, split=False), np.float32)
  r = np.asarray(ref(x, wp, bp, wg, bg, mask, jax.nn.swish), np.float32)
  fin = bool(np.isfinite(y).all())
  same = np.isfinite(y) == np.isfinite(r)
  ok = same.all()
  bad += not ok
  print(f"  swish amp={amp:4d}  |y|max {np.abs(y[np.isfinite(y)]).max():9.1f}  finite {fin}  "
        f"matches reference {bool(ok)}   {'OK' if ok else 'FAIL'}")

# anything else must decline rather than gate with a sigmoid behind the caller's back
if fused_ops.gated_dual_proj(gc, dt, jax.nn.relu) is not None:
  print("  relu took a kernel   FAIL")
  bad += 1
else:
  print("  relu declines to the caller's own path   OK")

print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)

"""Differentiate through fused_ops.attention, the way the model calls it.

  CK_SRC=<repo>/src KERNEL_DIR=<dir with libvolta_*.so> python tests/test_fused_ops_grad.py

Going through fused_ops matters: the kernels can each be differentiable on their
own while the dispatch path is not, which is how the backward library came to be
never registered.
"""
import os, sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import fused_ops, volta as V
KD = os.environ.get("KERNEL_DIR")
if KD:  # let a locally built .so set stand in for the installed wheel
  V._package_path = lambda kernel, cc: os.path.join(KD, {
      "attention": "libvolta_mma.so", "attention_bwd": "libvolta_mma_bwd.so"}[kernel])

b, h, s, c = 2, 4, 96, 32
rng = np.random.default_rng(0)
def mk(dt):
  f = lambda *sh: jnp.asarray(rng.standard_normal(sh) * .5, dt)
  q, k, v, nb, do = f(b,h,s,c), f(b,h,s,c), f(b,h,s,c), f(h,s,s), f(b,h,s,c)
  neg = -1e4 if dt == jnp.float16 else -1e9
  mb = jnp.asarray(np.where(rng.random((b,1,1,s)) > .15, 0.0, neg), dt)
  return q, k, v, mb, nb, do

bad = 0
for backend, dt, cc in (("pallas", jnp.bfloat16, None), ("cuda_legacy", jnp.float16, 75)):
  gc = {"use_pallas": True, "kernel_backend": backend, "compute_capability": cc}
  fn = fused_ops.attention(gc, dt, key_dim=c, value_dim=c, layout="heads")
  print(f"{backend:12s} -> {getattr(fn, 'func', fn).__name__ if fn else None}")
  if fn is None:
    print("    no kernel selected"); bad += 1; continue
  q, k, v, mb, nb, do = mk(dt)
  scale = float(c) ** -0.5
  try:
    out = jax.jit(lambda *a: fn(*a, scale))(q, k, v, mb, nb)
    print(f"    forward   ok   {np.asarray(out, np.float32).sum():.4f}")
  except Exception as e:
    print(f"    forward   FAILED  {type(e).__name__}: {str(e).splitlines()[0][:80]}")
    bad += 1
  try:
    g = jax.jit(jax.grad(lambda *a: jnp.sum(fn(*a, scale).astype(jnp.float32) * do.astype(jnp.float32)),
                         argnums=(0, 4)))(q, k, v, mb, nb)
    print(f"    backward  ok   dq {np.abs(np.asarray(g[0], np.float32)).max():.4f}"
          f"  dbias {np.abs(np.asarray(g[1], np.float32)).max():.4f}")
  except Exception as e:
    print(f"    backward  FAILED  {type(e).__name__}: {str(e).splitlines()[0][:100]}")
    bad += 1
print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)

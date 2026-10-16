"""packed attention, fused transition and the tri-mul epilogue: forward and gradient vs float32 references."""
import os, sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import fused_ops
from colabfold_kernels.packed_attn import ref_packed_attention
from colabfold_kernels.transition import ref_glu_transition, ref_transition
from colabfold_kernels.tri_mul import ref_trimul_epilogue

def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all(): return float('inf')
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))

bad = 0
rng = np.random.default_rng(0)
gc = {"use_pallas": True, "kernel_backend": "pallas"}
dt = jnp.bfloat16
f = lambda *sh, s=.5: jnp.asarray(rng.standard_normal(sh) * s, dt)
f32 = lambda *sh, s=.5: jnp.asarray(rng.standard_normal(sh) * s, jnp.float32)

def check(name, fn, ref, args, diff, tol=0.03):
  """forward and the gradient of <out, do> w.r.t. args[diff] against the float32 reference."""
  global bad
  out = fn(*args)
  fwd = rel(out, ref(*args))
  do = f(*out.shape)
  g = lambda h: jax.grad(lambda *d: jnp.sum(h(*[d[diff.index(i)] if i in diff else a
                                                   for i, a in enumerate(args)]).astype(jnp.float32)
                                            * do.astype(jnp.float32)), argnums=tuple(range(len(diff))))
  ga = jax.jit(g(fn))(*[args[i] for i in diff])
  gr = jax.jit(g(lambda *a: ref(*[x.astype(jnp.float32) if x is not None and x.dtype in (dt, jnp.float16) else x
                                  for x in a])))(*[args[i] for i in diff])
  errs = [rel(a, b) for a, b in zip(ga, gr)]
  ok = fwd < tol and max(errs) < tol
  bad += not ok
  print(f"{'ok ' if ok else 'BAD'} {name:<34} fwd {fwd:.1e}  grad {max(errs):.1e}", flush=True)

# CK_LEGACY=1 also runs the sm_70 (wmma) and sm_75 (mma) packed kernels, in float16
configs = [("triton", {**gc, "mosaic_gpu": False}, dt), ("pallas", gc, dt)] + [(f"sm{cc}", {"use_pallas": True, "kernel_backend": "cuda_legacy",
                                             "compute_capability": cc}, jnp.float16)
                                  for cc in ((70, 75) if os.environ.get("CK_LEGACY") else ())]
for tag, cfg, adt in configs:
  for X, Y, H, D, gated, column in ((64, 96, 4, 32, True, False), (96, 64, 4, 32, True, True),
                                    (37, 53, 8, 8, True, False), (50, 50, 4, 32, False, True),
                                    (40, 24, 8, 8, True, True), (33, 48, 4, 16, True, True), (5, 1, 4, 32, True, False),
                                    # Mosaic GPU on sm_12x: partial query/key tiles and batch groups, head dims 16 and 64
                                    (7, 136, 4, 32, True, False), (136, 5, 2, 64, False, True), (10, 40, 8, 16, True, False),
                                    (40, 6, 4, 16, True, True)):
    pa = fused_ops.packed_attention(cfg, adt, D, D)
    G = 4 if gated else 3
    nb, s = (Y, X) if column else (X, Y)
    p, bias = f(X, Y, G, H, D).astype(adt), f(H, s, s).astype(adt)
    kmask = jnp.asarray(rng.random((nb, s)) > .1).at[nb // 2].set(False)   # a fully masked (padding) row
    gb = f(H, D).astype(adt) if gated else None
    call = lambda p, bias, kmask, gb: pa(p, bias, kmask, gb, sm_scale=D ** -.5, column=column)
    ref = lambda p, bias, kmask, gb: ref_packed_attention(p, bias, kmask, gb, sm_scale=D ** -.5, column=column)
    check(f"{tag} packed_attention {X}x{Y} {H}x{D} {'col' if column else 'row'}{' gated' if gated else ''}",
          call, ref, (p, bias, kmask, gb), [0, 1] + ([3] if gated else []))

# tiny row counts leave only a partial block
for M, C, F in ((4096, 128, 4), (777, 64, 4), (1000, 256, 4), (300, 64, 2), (1, 128, 4), (64, 128, 4), (127, 128, 4)):
  tr = fused_ops.transition(gc, dt, C, F * C)
  if tr is None:   # ROCm keeps XLA above 128 channels
    continue
  args = (f(M, C), 1 + f32(C, s=.1), f32(C, s=.1), f(C, F * C, s=C ** -.5), f(F * C, s=.1),
          f(F * C, C, s=(F * C) ** -.5), f(C, s=.1))
  for add in (False, True):
    check(f"transition {M}x{C}{' +x' if add else ''}", lambda *a, add=add: tr(*a, add_input=add),
          lambda *a, add=add: ref_transition(*a, add_input=add), args, list(range(7)))

# AF3's swish-GLU transition, with OpenFold3's row mask and the residual
for M, C in ((4096, 128), (777, 64), (1, 128), (127, 128), (64, 64), (777, 256), (127, 512)):
  glu = fused_ops.glu_transition(gc, dt, C, 4 * C)
  args = (f(M, C), 1 + f32(C, s=.1), f32(C, s=.1), f(C, 4 * C, s=C ** -.5), f(C, 4 * C, s=C ** -.5),
          f(4 * C, C, s=(4 * C) ** -.5))
  keep = jnp.asarray(rng.random(M) > .2, dt)
  for masked in (False, True):
    for add in (False, True):
      a = args + ((keep,) if masked else ())
      check(f"glu_transition {M}x{C}{' keep' if masked else ''}{' +x' if add else ''}",
            lambda *a, add=add: glu(*a, add_input=add), lambda *a, add=add: ref_glu_transition(*a, add_input=add),
            a, list(range(6)))

# AF2's fp16 path hands the epilogue a float32 product; a width the kernel does not take (96) runs XLA, 256 and 512 stream
ep = fused_ops.trimul_epilogue(gc, dt)
for M, C, xdt in ((4096, 128, dt), (999, 128, dt), (500, 64, dt), (1, 128, dt), (64, 128, dt), (127, 128, dt),
                  (999, 128, jnp.float32), (300, 96, dt), (999, 256, dt), (127, 512, dt)):
  args = (f(C, M).astype(xdt), f(M, C), 1 + f32(C, s=.1), f32(C, s=.1), f(C, C, s=C ** -.5), f(C, s=.1),
          f(C, C, s=C ** -.5), f(C, s=.1))
  tag = f"{M}x{C}{' f32 x' if xdt == jnp.float32 else ''}"
  check(f"trimul_epilogue {tag}", ep, ref_trimul_epilogue, args, list(range(8)))
  for res in (f(M, C), f32(M, C)):
    check(f"trimul_epilogue {tag} +{res.dtype} residual", ep, ref_trimul_epilogue, args + (res,), list(range(9)))

print("FAIL" if bad else "PASS", bad)
sys.exit(bool(bad))

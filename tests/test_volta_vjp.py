"""volta_attention as a differentiable op, against jax.grad of an fp32 reference.

  CK_SRC=<repo>/src KERNEL_DIR=<dir with libvolta_*.so> CC=75 python tests/test_volta_vjp.py

CC picks which family to exercise, and KERNEL_DIR lets a locally built .so set
stand in for the installed wheel -- the kernels run on any card at or above the
arch they were built for, so sm_75 ones can be checked on anything newer.
"""
import os, sys, ctypes
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get("CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import volta as V

# point the loader at a locally built .so set instead of the installed wheel
KD = os.environ["KERNEL_DIR"]
def _fake_path(kernel, cc):
    return os.path.join(KD, {"attention": "libvolta_mma.so" if cc >= 75 else "libvolta_wmma.so",
                             "attention_bwd": "libvolta_mma_bwd.so" if cc >= 75 else "libvolta_wmma_bwd.so"}[kernel])
V._package_path = _fake_path
CC = int(os.environ.get("CC", "75"))
print("available:", V.available(CC), " bwd_available:", V.bwd_available(CC))

def reference(q, k, v, mask_bias, nb, scale):
    logits = scale * jnp.einsum('bhqd,bhkd->bhqk', q, k) + nb[None]
    logits = jnp.where(mask_bias[:, 0, 0, :][:, None, None, :] > -1e3, logits, -1e4)
    return jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(logits, -1), v)

def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not np.isfinite(a).all(): return float('inf')
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))

bad = 0
for b, h, sq, sk, c in ((2, 4, 96, 96, 32), (2, 4, 70, 83, 16), (1, 8, 64, 64, 48), (2, 2, 96, 96, 8)):
    rng = np.random.default_rng(0)
    f16 = lambda x: jnp.asarray(x, jnp.float16)
    q = f16(rng.standard_normal((b, h, sq, c)) * .5); k = f16(rng.standard_normal((b, h, sk, c)) * .5)
    v = f16(rng.standard_normal((b, h, sk, c)) * .5); nb = f16(rng.standard_normal((h, sq, sk)) * .5)
    mb = jnp.asarray(np.where(rng.random((b, 1, 1, sk)) > .15, 0.0, -1e4), jnp.float16)
    do = f16(rng.standard_normal((b, h, sq, c)) * .5)
    scale = float(c) ** -0.5
    loss = lambda fn, *a: jnp.sum(fn(*a).astype(jnp.float32) * do.astype(jnp.float32))
    ours = jax.jit(lambda *a: V.volta_attention(*a, scale, CC, 64, 32))
    o_out = ours(q, k, v, mb, nb)
    g = jax.grad(lambda *a: loss(lambda *x: V.volta_attention(*x, scale, CC, 64, 32), *a),
                 argnums=(0, 1, 2, 4))(q, k, v, mb, nb)
    qf, kf, vf, nbf = [x.astype(jnp.float32) for x in (q, k, v, nb)]
    r_out = reference(qf, kf, vf, mb, nbf, scale)
    rg = jax.grad(lambda a, bb, cc_, nn: loss(lambda *x: reference(x[0], x[1], x[2], mb, x[3], scale), a, bb, cc_, nn),
                  argnums=(0, 1, 2, 3))(qf, kf, vf, nbf)
    errs = [("out", o_out, r_out)] + [(n, x, y) for n, x, y in zip(("dq", "dk", "dv", "dbias"), g, rg)]
    line = "  ".join(f"{n} {rel(x, y):.1e}" for n, x, y in errs)
    worst = max(rel(x, y) for _, x, y in errs)
    ok = worst < 0.05
    bad += not ok
    print(f"  b={b} h={h} sq={sq} sk={sk} c={c:2d}   {line}   {'OK' if ok else 'FAIL'}")
print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)

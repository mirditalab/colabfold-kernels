"""pallas_attention as a differentiable op, against jax.grad of an fp32 reference.

  CK_SRC=<repo>/src python tests/test_pallas_vjp.py

Head dims cover af2's 8/16/32 and af3's 48 and 96, the last two being the ones
the head-dim chunking has to carry.
"""
import os, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import tri_flash as TF

def reference(q, k, v, mb, nb, scale):
    lg = scale * jnp.einsum('bhqd,bhkd->bhqk', q, k) + nb[None]
    lg = jnp.where(mb[:, 0, 0, :][:, None, None, :] > -1e3, lg, TF.NEG)
    return jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(lg, -1), v)

def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not np.isfinite(a).all(): return float('inf')
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))

bad = 0
# head 4 and 8 are below triton's dot minimum of 16 for 16-bit inputs, which
# _chunks pads around; unpadded they are silently wrong (rel err ~1.2)
for b, h, sq, sk, c in ((2, 4, 96, 96, 32), (2, 4, 70, 83, 16), (1, 8, 64, 64, 48),
                        (2, 2, 128, 128, 8), (2, 2, 96, 96, 4), (1, 4, 96, 96, 96)):
    rng = np.random.default_rng(0)
    f = lambda *sh: jnp.asarray(rng.standard_normal(sh) * .5, jnp.float32)
    q, k, v = f(b, h, sq, c), f(b, h, sk, c), f(b, h, sk, c)
    nb = f(h, sq, sk)
    mb = jnp.asarray(np.where(rng.random((b, 1, 1, sk)) > .15, 0.0, -1e9), jnp.float32)
    do = f(b, h, sq, c)
    scale = float(c) ** -0.5
    loss = lambda fn, *a: jnp.sum(fn(*a) * do)
    ours_f = lambda *a: TF.pallas_attention(a[0], a[1], a[2], mb, a[3], scale)
    ref_f = lambda *a: reference(a[0], a[1], a[2], mb, a[3], scale)
    o = jax.jit(ours_f)(q, k, v, nb)
    g = jax.jit(jax.grad(lambda *a: loss(ours_f, *a), argnums=(0, 1, 2, 3)))(q, k, v, nb)
    r = ref_f(q, k, v, nb)
    rg = jax.jit(jax.grad(lambda *a: loss(ref_f, *a), argnums=(0, 1, 2, 3)))(q, k, v, nb)
    # asking for lse must not perturb the output the inference path returns
    o2, _ = TF.tri_flash(q, k, v, nb, mb[:, 0, 0, :] > -1e3, sm_scale=scale,
                         block_q=64, block_k=TF._pow2_tile(min(64, c)),
                         want_lse=True)
    rows = [("out", o, r), ("out|lse", o2, np.asarray(o))] + list(
        zip(("dq", "dk", "dv", "dbias"), g, rg))
    worst = max(rel(x, y) for _, x, y in rows)
    bad += worst >= 0.02
    print(f"  b={b} h={h} sq={sq} sk={sk} c={c:2d}   " +
          "  ".join(f"{n} {rel(x,y):.1e}" for n, x, y in rows) +
          f"   {'OK' if worst < 0.02 else 'FAIL'}")
print("RESULT:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)

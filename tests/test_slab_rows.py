"""Slab row map of ln_projection / output_projection (rows read and written in place) vs gathering the slab first:
row and column slabs, a remainder slab, keep; rows outside the slab stay untouched."""
import os, sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import packed_attn as pa

bad = 0
N, a, G, n = 600, 128, 4, 128
ks = jax.random.split(jax.random.key(0), 6)
x = jax.random.normal(ks[0], (N * N, a), jnp.bfloat16)
sc, of = 1 + 0.1 * jax.random.normal(ks[1], (a,)), 0.1 * jax.random.normal(ks[2], (a,))
w = (jax.random.normal(ks[3], (G, a, n)) * a ** -0.5).astype(jnp.bfloat16)
wo = (jax.random.normal(ks[4], (n, a)) * n ** -0.5).astype(jnp.bfloat16)
bo = 0.1 * jax.random.normal(ks[5], (a,), jnp.bfloat16)
keep = (jnp.arange(N) < 560).astype(jnp.bfloat16)
keep2 = keep[:, None] * keep[None, :]
f = lambda v: np.asarray(v, np.float32)
for col in (False, True):
  for j0, s in ((256, 256), (512, 88)):
    xx, yy = (N, s) if col else (s, N)
    slab = (xx * yy, s, N) if col else (xx * yy,) * 3
    st = jnp.array([j0 if col else j0 * N], jnp.int32)
    kp = jax.lax.dynamic_slice_in_dim(keep2, j0, s, 1 if col else 0).reshape(-1)
    p, _ = pa.ln_projection(x, sc, of, w, None, kp, st, slab=slab)
    p0, _ = pa.ln_projection(pa.slab_rows(x, st, slab), sc, of, w, None, kp)
    o = p[:, :n]
    r = pa.output_projection(o, wo, bo, x, kp, st, slab=slab)
    i = pa.slab_index(st, slab)
    r0 = x.at[i].set(pa.output_projection(o, wo, bo, x[i], kp))
    errs = (float(np.abs(f(p) - f(p0)).max()), float(np.abs(f(r) - f(r0)).max()),
            float(np.abs(f(jnp.delete(r, i, 0)) - f(jnp.delete(x, i, 0))).max()))
    ok = max(errs) == 0
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {'column' if col else 'row'} slab {s}: projection {errs[0]:.2e} "
          f"output {errs[1]:.2e} untouched {errs[2]:.2e}")

print("FAILED" if bad else "all ok")
sys.exit(bool(bad))

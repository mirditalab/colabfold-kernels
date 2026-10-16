"""LayerNorm-fused projections, the LN-recomputing tri-mul epilogue and the output projection vs float32 references."""
import os, sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import fused_ops
from colabfold_kernels import packed_attn as pa, tri_mul as tm

def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all(): return float('inf')
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))

bad = 0
rng = np.random.default_rng(0)
gc = {"use_pallas": True, "kernel_backend": "pallas"}
dt = jnp.bfloat16
f = lambda *sh, s=.5: jnp.asarray(rng.standard_normal(sh) * s, dt)
f32 = lambda *sh, s=.5: jnp.asarray(rng.standard_normal(sh) * s + (1 if len(sh) == 1 else 0), jnp.float32)

def check(name, fn, ref, args, diff, tol=0.03):
  """forward and the gradient of the summed outputs against the float32 reference, at args[diff]."""
  global bad
  outs = [o for o in jax.tree.leaves(fn(*args))]
  refs = [o for o in jax.tree.leaves(ref(*args))]
  errs = [rel(o, r) for o, r in zip(outs, refs)]
  cots = [f(*o.shape) for o in outs]
  def loss(h):
    def l(*d):
      full = [d[diff.index(i)] if i in diff else a for i, a in enumerate(args)]
      return sum(jnp.sum(o.astype(jnp.float32) * c.astype(jnp.float32))
                 for o, c in zip(jax.tree.leaves(h(*full)), cots))
    return jax.grad(l, argnums=tuple(range(len(diff))))(*[args[i] for i in diff])
  gerr = max(rel(a, b) for a, b in zip(loss(fn), loss(ref)))
  ok = len(outs) == len(refs) and max(errs) < tol and gerr < tol
  bad += not ok
  print(f"{'ok ' if ok else 'BAD'} {name}: fwd {max(errs):.2e} grad {gerr:.2e}")

for m in (1, 59, 4101, 260 * 260):
  a, ci, h, d = 128, 128, 4, 32
  x, mask = f(m, a, s=2.), (rng.uniform(size=m) > .1).astype(np.float32)
  sc, of = f32(a), f32(a)
  wp, wg, bp, bg = f(a, 2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(2 * ci, s=.1)
  check(f"ln_gdp m={m}", fused_ops.ln_gated_dual_proj(gc, dt, a, ci), tm.ref_ln_gdp,
        [x, sc, of, wp, bp, wg, bg, jnp.asarray(mask, dt)], [0, 1, 3])
  x_cm = f(ci, m, s=2.)
  ep = [f32(ci), f32(ci), f(ci, a, s=.1), f(a, s=.1), f(a, a, s=.1), f(a, s=.1)]
  ep_ln = fused_ops.trimul_epilogue_ln(gc, dt)
  for add in (False, True):
    check(f"epilogue_ln m={m} add_input={add}", lambda *z, add=add: ep_ln(*z, add_input=add),
          lambda *z, add=add: tm.ref_trimul_epilogue_ln(*z, add_input=add), [x_cm, x, sc, of, *ep], [0, 1, 2, 6])
  w, wb = f(4, a, h * d, s=.1), f(a, h, s=.1)
  lp = fused_ops.ln_projection(gc, dt, a, h * d)
  check(f"ln_projection m={m}", lp, pa.ref_ln_projection, [x, sc, of, w, wb], [0, 1, 3, 4])
  check(f"ln_projection bias only m={m}", lambda x, s, o, wb: lp(x, s, o, None, wb),
        lambda x, s, o, wb: pa.ref_ln_projection(x, s, o, None, wb), [x, sc, of, wb], [0, 3])
  check(f"ln_projection 3 slots, 8 heads m={m}", lp, pa.ref_ln_projection,
        [x, sc, of, f(3, a, h * d, s=.1), f(a, 8, s=.1)], [0, 3])
  op = fused_ops.output_projection(gc, dt, h * d, a)
  o, wo, bo = f(m, h * d), f(h * d, a, s=.1), f(a, s=.1)
  check(f"output_projection m={m}", op, pa.ref_output_projection, [o, wo, bo, x], [0, 1, 3])
  check(f"output_projection no residual m={m}", op, pa.ref_output_projection, [o, wo, bo], [0, 1])

# the unfused path is exactly what the fused one replaces: LN, then the existing kernels
m = 4101
x, sc, of = f(m, 128, s=2.), f32(128), f32(128)
mask = jnp.ones((m,), dt)
wp, wg, bp, bg = f(128, 256, s=.1), f(128, 256, s=.1), f(256, s=.1), f(256, s=.1)
y = tm.pallas_layer_norm(x, sc, of)
old = tm.gated_dual_proj(y, wp, bp, wg, bg, mask, split=True, channel_major=True)
new = tm.ln_gated_dual_proj(x, sc, of, wp, bp, wg, bg, mask)
same = all(np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32)) for a, b in zip(old, new))
diff = max(rel(a, b) for a, b in zip(new, old))
print(f"{'ok ' if diff < 1e-2 else 'BAD'} ln_gdp vs layer_norm + gdp: {'bitwise' if same else f'{diff:.2e}'}")
bad += diff >= 1e-2

print("FAILED" if bad else "all ok")
sys.exit(bool(bad))

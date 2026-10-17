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

# the narrowest widths the kernels take (Triton's dot K is 16) and the template stack's 64 channels, 4 heads of 16
for a, ci, n in ((16, 16, 16), (64, 64, 64)):
  m = 4101
  x, sc, of = f(m, a, s=2.), f32(a), f32(a)
  mask = jnp.asarray((rng.uniform(size=m) > .1).astype(np.float32), dt)
  check(f"ln_gdp a={a} ci={ci}", fused_ops.ln_gated_dual_proj(gc, dt, a, ci), tm.ref_ln_gdp,
        [x, sc, of, f(a, 2 * ci, s=.1), f(2 * ci, s=.1), f(a, 2 * ci, s=.1), f(2 * ci, s=.1), mask], [0, 3])
  ep = [f32(ci), f32(ci), f(ci, a, s=.1), f(a, s=.1), f(a, a, s=.1), f(a, s=.1)]
  check(f"epilogue_ln a={a} ci={ci}", lambda *z: fused_ops.trimul_epilogue_ln(gc, dt)(*z, add_input=True),
        lambda *z: tm.ref_trimul_epilogue_ln(*z, add_input=True), [f(ci, m, s=2.), x, sc, of, *ep], [0, 1])
  check(f"ln_projection a={a} n={n}", fused_ops.ln_projection(gc, dt, a, n), pa.ref_ln_projection,
        [x, sc, of, f(4, a, n, s=.1), f(a, 4, s=.1)], [0, 3])
  check(f"output_projection k={n} c={a}", fused_ops.output_projection(gc, dt, n, a), pa.ref_output_projection,
        [f(m, n), f(n, a, s=.1), f(a, s=.1), x], [0, 1])
assert fused_ops.ln_projection(gc, dt, 8, 64) is None and fused_ops.output_projection(gc, dt, 64, 8) is None

# the MSA stack's 256 channels: the weight streamed in 64-row blocks
for k, c in ((256, 256), (64, 256), (256, 128)):
  m = 4101
  check(f"output_projection k={k} c={c}", fused_ops.output_projection(gc, dt, k, c), pa.ref_output_projection,
        [f(m, k), f(k, c, s=.06), f(c, s=.1), f(m, c)], [0, 1])
assert fused_ops.output_projection(gc, dt, 96, 256) is None and fused_ops.output_projection(gc, dt, 256, 512) is None
# and the MSA attention's query LayerNorm with its 256-wide slots, in 128-column dots
for a, n, slots in ((256, 256, 4), (256, 256, 3), (256, 128, 4)):
  m = 4101
  x, sc, of = f(m, a, s=2.), f32(a), f32(a)
  check(f"ln_projection a={a} n={n} slots={slots}", fused_ops.ln_projection(gc, dt, a, n), pa.ref_ln_projection,
        [x, sc, of, f(slots, a, n, s=.06), None], [0, 3])
assert fused_ops.ln_projection(gc, dt, 512, 128) is None and fused_ops.ln_projection(gc, dt, 256, 192) is None

# the shallow outer product mean's update: K is 32 per MSA row, norm 0 on padding (only eps divides then)
from colabfold_kernels import opm
for b, rows in ((59, 1), (259, 1), (259, 16)):
  kk = 32 * rows
  lhs, t, ob = f(b, kk), f(kk, b, 128, s=.2), f(128, s=.1)
  norm = jnp.asarray((rng.uniform(size=(b, b)) > .1).astype(np.float32) * rows, dt)
  up = fused_ops.opm_update(gc, dt, kk, 128)
  check(f"opm_update b={b} rows={rows}", up, opm.ref_opm_update, [lhs, t, ob, norm, f(b, b, 128)], [0, 1, 4])
  check(f"opm_update b={b} rows={rows} no residual", up, opm.ref_opm_update, [lhs, t, ob, norm], [0, 1])
assert fused_ops.opm_update(gc, dt, 48, 128) is None and fused_ops.opm_update(gc, dt, 32, 96) is None

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

# e4m3, on cards with 8-bit MMAs only: errors at its 3-bit mantissa
def rms(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2))) if np.isfinite(a).all() else float('inf')

def expect(name, ok, detail):
  global bad
  bad += not ok
  print(f"{'ok ' if ok else 'BAD'} {name}: {detail}")

from colabfold_kernels.device import compute_capability, is_rocm
if (compute_capability() or 0) >= 89 and not is_rocm():
  from colabfold_kernels.tri_flash import e4m3
  for n, d in ((259, 32), (130, 64)):
    p, bias, gb = f(n, n, 4, 4, d, s=1.), f(4, n, n), f(4, d)
    kmask = jnp.asarray(rng.uniform(size=(n, n)) > .1)
    for column in (False, True):
      kw = dict(sm_scale=d ** -0.5, column=column)
      o8 = pa.packed_attention(p[:, :, jnp.array([0, 3])], bias, kmask, gb, e4m3(p[:, :, 1:3]), **kw)
      # e4m3 rounds q and k by ~3% each, so the logits and their softmax move by about that much
      e = rms(o8, pa.ref_packed_attention(p, bias, kmask, gb, **kw))
      expect(f"fp8 attention n={n} d={d} column={column}", e < 0.08, f"rms {e:.2e}")
  # an 8-bit MMA's K is 32: narrower heads are never offered e4m3
  expect("fp8 attention refused at head dims 8 and 16",
         not any(fused_ops.fp8_attention_slots({"fp8": True}, dt, d, 64) or fused_ops.fp8_attention({"fp8": True}, d)
                 for d in (8, 16)), "")
  # where the Mosaic kernel runs it takes q (pre-scaled) and k in e4m3, and p keeps v and the gate
  from colabfold_kernels import mgpu_attn
  from colabfold_kernels.tri_flash import LOG2E
  if fused_ops._mgpu_attn() is not None:
    expect("fp8 attention takes q and k where the Mosaic kernel runs",
           fused_ops.fp8_attention_slots({"fp8": True}, dt, 32, 264)[0] == (0, 1)
           and not fused_ops.fp8_attention({"fp8": True}, 32), "")
    for n, d in ((264, 32), (136, 64)):
      p, bias, gb = f(n, n, 4, 4, d, s=1.), f(4, n, n), f(4, d)
      kmask = jnp.asarray(rng.uniform(size=(n, n)) > .1)
      qk8 = e4m3(p[:, :, :2].astype(jnp.float32) * jnp.asarray([d ** -0.5 * LOG2E, 1.])[:, None, None])
      for column in (False, True):
        kw = dict(sm_scale=d ** -0.5, column=column)
        o8 = mgpu_attn.packed_attention(p[:, :, 2:], bias, kmask, gb, qk8, **kw)
        e = rms(o8, pa.ref_packed_attention(p, bias, kmask, gb, **kw))
        expect(f"fp8 Mosaic attention n={n} d={d} column={column}", e < 0.08, f"rms {e:.2e}")
  m = 4101
  x, sc, of = f(m, 128, s=2.), f32(128), f32(128)
  w4, wb = f(4, 128, 128, s=.1), f(128, 4, s=.1)
  p16, b16 = pa.ln_projection(x, sc, of, w4, wb)
  ph, b8, p8 = pa.ln_projection(x, sc, of, w4, wb, f8_slots=(1, 2))
  p16 = np.asarray(p16, np.float32).reshape(m, 4, 128)
  # another compile of the same dots: a few elements move by one bf16 ulp
  e = max(rel(np.asarray(ph, np.float32).reshape(m, 2, 128), p16[:, [0, 3]]), rel(b8, b16))
  expect("ln_projection f8_slots: bf16 slots and bias", e < 1e-2, f"max rel {e:.2e}")
  e = rms(np.asarray(p8, np.float32).reshape(m, 2, 128), p16[:, 1:3])
  expect("ln_projection f8_slots: e4m3 slots", e < 0.05, f"rms {e:.2e}")
  _, _, p8s = pa.ln_projection(x, sc, of, w4, wb, f8_slots=(0, 1), f8_scales=(0.25, 1.0))
  e = rms(np.asarray(p8s, np.float32).reshape(m, 2, 128), p16[:, 0:2] * np.array([0.25, 1.0])[None, :, None])
  expect("ln_projection f8_scales scale each e4m3 slot", e < 0.05, f"rms {e:.2e}")
  # the triangle contraction's e4m3 operands, padded to 16 along the contracted axis, plain and transposed
  n, cz = 259, 128
  x, mask = f(n, n, cz, s=2.), jnp.asarray((rng.uniform(size=(n, n)) > .1).astype(np.float32), dt)
  g = [f32(cz), f32(cz), f(cz, 256, s=.1), f(256, s=.1), f(cz, 256, s=.1), f(256, s=.1)]
  for transpose in (False, True):
    out = tm.ln_gated_dual_proj(x, *g, mask, fp8=True, transpose=transpose, pad=16)
    ref = tm.ref_ln_gdp(x, *g, mask, transpose=transpose, pad=16)
    e = max(rms(a, b) for a, b in zip(out, ref))
    zero = all(not np.asarray(o[:, :, n:], np.float32).any() for o in out)
    expect(f"ln_gdp e4m3 3D transpose={transpose}", e < 0.05 and out[0].shape == (128, n, 272) and zero,
           f"rms {e:.2e}, shape {out[0].shape}, padding zero {zero}")
  from colabfold_kernels import contract
  if contract.available():
    a8, b8 = e4m3(f(128, n, 272, s=2.)), e4m3(f(128, 264, 272, s=2.))
    e = rel(contract.fp8_contract(a8, b8), contract.ref_fp8_contract(a8, b8))
    expect("fp8_contract vs e4m3-rounded inputs", e < 1e-2, f"max rel {e:.2e}")
    assert fused_ops.fp8_contraction({"fp8": True, **gc}, dt, cz, 128, n) is None
    n = 264
    x, mask = f(n, n, cz, s=2.), jnp.asarray((rng.uniform(size=(n, n)) > .1).astype(np.float32), dt)
    proj, con = fused_ops.fp8_contraction({"fp8": True, **gc}, dt, cz, 128, n)
    for eq, transpose in (('ikc,jkc->ijc', False), ('kjc,kic->ijc', True)):
      l, r = proj(x, *g, mask, transpose=transpose)
      tri = con(*((r, l) if transpose else (l, r)))
      l32, r32 = tm.ref_ln_gdp(x.reshape(-1, cz), *g, mask.reshape(-1))
      ref = jnp.einsum(eq, *(t.T.reshape(n, n, 128) for t in (l32, r32))).transpose(2, 0, 1)
      e = rms(tri, ref)
      expect(f"fp8 triangle contraction {eq}", e < 0.05, f"rms {e:.2e}")
  else:
    print("no colabfold-cuda-kernels FP8 contraction here: not tested")
else:
  print("no 8-bit MMAs here: e4m3 paths not tested")

# the half softmax (CUDA): f16 scores, from a projection writing its packed slots in f16
if compute_capability() is not None and not is_rocm():
  for n, h, d, valid in ((264, 8, 8, 261), (130, 4, 32, 130), (100, 8, 8, 37), (264, 8, 32, 200)):
    p, bias, gb = f(n, n, 4, h, d, s=1.), f(h, n, n, s=1.), f(h, d)
    kmask = jnp.asarray(rng.uniform(size=(n, n)) > .1) & (jnp.arange(n) < valid)[None, :]
    for column in (False, True):
      kw = dict(sm_scale=d ** -0.5, column=column)
      o = pa.packed_attention(p.astype(jnp.float16), bias, kmask, gb, out_dtype=dt, **kw)
      e = rms(o, pa.ref_packed_attention(p, bias, kmask, gb, **kw))
      expect(f"half attention n={n} h={h} d={d} column={column}", o.dtype == dt and e < 5e-3, f"rms {e:.2e}")
      # the grouped grid order (a last group past the batches included) gives what the plain one does
      for half in (False, True):
        pp = p.astype(jnp.float16 if half else dt)
        o0, o1 = (pa.packed_attention(pp, bias, kmask, gb, out_dtype=dt, group=g, **kw) for g in (0, 16))
        expect(f"grouped grid n={n} d={d} column={column} half={half}", bool(jnp.all(o0 == o1)), "")
  x, sc, of = f(259, 128, s=2.), f32(128), f32(128)
  w4, wb = f(4, 128, 128, s=.1), f(128, 4, s=.1)
  (ph, bh), (pr, br) = pa.ln_projection(x, sc, of, w4, wb, p_dtype=jnp.float16), pa.ref_ln_projection(x, sc, of, w4, wb)
  expect("ln_projection p_dtype f16", ph.dtype == jnp.float16 and bh.dtype == dt and rel(ph, pr) < 1e-2,
         f"max rel {rel(ph, pr):.2e}")

print("FAILED" if bad else "all ok")
sys.exit(bool(bad))

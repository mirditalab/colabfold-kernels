"""The triangle contraction over a bucketed fold's live length (contract.live_contract), and the kernels around it.

live_contract on bf16 and f16, both edge directions, at live lengths across a
bucket: within round_up(live, 32) it matches the float32 reference, with the operands NaN past that (never read). The
trimul chain on a padded pair: ln_gated_dual_proj writing only the live region, live_contract, then trimul_epilogue_ln
masking the product by keep, against the full-length chain on the live rows. opm_update with keep: kept blocks
bit-identical to the call without it, the others the residual unchanged.
CCK_LIVE_MMA=70 or 75: f16 only, through the sm_70 / sm_75 MMA on this card.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from colabfold_kernels import contract, opm, tri_mul as tm

fails = []
rng = np.random.default_rng(0)


def f(*shape, s=1.0, dt=jnp.bfloat16):
  return jnp.asarray(rng.normal(size=shape) * s, dt)


def report(name, ok, detail):
  print(f"{'ok  ' if ok else 'BAD '}{name}: {detail}")
  if not ok:
    fails.append(name)


def region(n, e):
  inside = jnp.arange(n) < e
  return inside[:, None] & inside[None, :]


if not contract.live_available():
  print("live_contract: no colabfold-cuda-kernels library here (NVIDIA, 7.0 on), skipped")
else:
  C, S = 32, 640
  legacy = bool(os.environ.get("CCK_LIVE_MMA"))
  for dt in (jnp.float16,) if legacy else (jnp.bfloat16, jnp.float16):
    for incoming in (False, True):
      for live in (1, 31, 32, 33, 513, 522, 608, 640):
        e = min(-(-live // contract.LIVE_ROUND) * contract.LIVE_ROUND, S)
        live_rows, read = jnp.arange(S) < live, jnp.arange(S) < e
        # zeros from live up to e (the projection writes them), NaN past e: read there, the result would show it
        def operand():
          x = f(C, S, S, dt=jnp.float32)
          x = jnp.where(live_rows[:, None] & live_rows[None, :], x, 0.0)
          return jnp.where(read[:, None] & read[None, :], x, jnp.nan).astype(dt)
        a, b = operand(), operand()
        out = contract.live_contract(a, b, jnp.int32(live), incoming=incoming)
        ref = contract.ref_live_contract(jnp.nan_to_num(a.astype(jnp.float32)),
                                         jnp.nan_to_num(b.astype(jnp.float32)), jnp.int32(live), incoming=incoming)
        o, r = np.asarray(out[:, :e, :e].astype(jnp.float32)), np.asarray(ref[:, :e, :e])
        err = float(np.abs(o - r).max() / (np.abs(r).max() + 1e-6))
        report(f"live_contract {jnp.dtype(dt).name} {'in' if incoming else 'out'} live={live}",
               bool(np.isfinite(o).all()) and err < 0.01, f"max err {err:.3g} over {e} rows")

  # the trimul chain on a padded pair: live region only vs the whole length (bf16, so not under CCK_LIVE_MMA)
  cz, ci, S, n = 128, 128, 384, 290
  tok = jnp.arange(S) < n
  mask = (tok[:, None] & tok[None, :]).astype(jnp.bfloat16)
  x = f(S, S, cz, s=2.)
  g = (f(cz, s=.1) + 1, f(cz, s=.1), f(cz, 2 * ci, s=.1), f(2 * ci, s=.1), f(cz, 2 * ci, s=.1), f(2 * ci, s=.1))
  e = -(-n // contract.LIVE_ROUND) * contract.LIVE_ROUND
  keep = jnp.where(region(S, e), mask, -1).astype(mask.dtype)
  ln = (f(ci, s=.1) + 1, f(ci, s=.1))
  out_p, gate_p = (f(ci, cz, s=.1), f(cz, s=.1)), (f(cz, cz, s=.1), f(cz, s=.1))
  for incoming in () if legacy else (False, True):
    def chain(proj_mask, contract_fn):
      left, right = tm.ln_gated_dual_proj(x.reshape(-1, cz), *g, proj_mask.reshape(-1))
      left, right = left.reshape(ci, S, S), right.reshape(ci, S, S)
      tri = contract_fn(left, right)
      return tm.trimul_epilogue_ln(tri.reshape(ci, -1), x.reshape(-1, cz), *g[:2], *ln, *out_p, *gate_p,
                                   mask.reshape(-1)).reshape(S, S, cz)
    eq = 'ckj,cki->cij' if incoming else 'cik,cjk->cij'
    full = chain(mask, lambda l, r: jnp.einsum(eq, l, r))
    lv = chain(keep, lambda l, r: (contract.live_contract(r, l, jnp.int32(n), incoming=True) if incoming
                                   else contract.live_contract(l, r, jnp.int32(n))))
    o, r = np.asarray(lv[:n, :n].astype(jnp.float32)), np.asarray(full[:n, :n].astype(jnp.float32))
    err = float(np.abs(o - r).max() / (np.abs(r).max() + 1e-6))
    report(f"trimul chain {'in' if incoming else 'out'} S={S} n={n}",
           bool(np.isfinite(np.asarray(lv.astype(jnp.float32))).all()) and err < 0.02, f"max err {err:.3g}")

# opm_update with keep: the skipped blocks keep the residual
B, K, F, n = 300, 32, 128, 211
tok = jnp.arange(B) < n
keep = (tok[:, None] & tok[None, :]).astype(jnp.bfloat16)
lhs, t, bias, res = f(B, K), f(K, B, F, s=.2), f(F, s=.1), f(B, B, F)
norm = keep.astype(jnp.float32)
plain = opm.opm_update(lhs, t, bias, norm, res + 0)
kept = opm.opm_update(lhs, t, bias, norm, res + 0, keep)
kp, pp, rr = (np.asarray(v.astype(jnp.float32)) for v in (kept, plain, res))
# a block, 64 rows of one column, computes when any of its rows is kept
nb = -(-B // 64)
rows = np.zeros((nb * 64, B))
rows[:B] = np.asarray(keep.astype(jnp.float32))
blocks = np.repeat(rows.reshape(nb, 64, B).max(axis=1) > 0, 64, axis=0)[:B]
ok = bool((kp[blocks] == pp[blocks]).all() and (kp[~blocks] == rr[~blocks]).all())
report(f"opm_update keep B={B} n={n}", ok, f"{int(blocks.sum())} of {B * B} pairs computed")

if fails:
  print("FAILED:", fails)
  sys.exit(1)
print("all ok")

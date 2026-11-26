"""sigmoid and swish in the kernels keep relative accuracy deep in the negative tail, where IntelliFold-v2 runs its
triangle-attention gates (-39..-6.5) and pair-transition swish inputs (-26..-8): glu_transition and gated Triton packed
attention against float64 with the swish inputs and the gate shifted there."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from colabfold_kernels import packed_attn as pa
from colabfold_kernels import transition as tr

fails = []


def rel(o, ref):
  o = np.asarray(jnp.asarray(o).astype(jnp.float32), np.float64)
  return np.sqrt(((o - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean())


def report(name, err):
  ok = err < 5e-3
  print(f"{'ok  ' if ok else 'BAD '}{name}: rel {err:.3g}")
  if not ok:
    fails.append(name)


k = jax.random.split(jax.random.PRNGKey(0), 6)
M, C, H = 4096, 128, 512
x = jax.random.normal(k[0], (M, C)).astype(jnp.bfloat16)
# LN scale 0 / offset 1 on channel 0 makes it a constant 1, so row 0 of wa shifts every swish input
s, o = jnp.ones((C,)).at[0].set(0.0), jnp.zeros((C,)).at[0].set(1.0)
wa = (jax.random.normal(k[1], (C, H)) * 0.05).astype(jnp.bfloat16)
wb = (jax.random.normal(k[2], (C, H)) * 0.1).astype(jnp.bfloat16)
wo = (jax.random.normal(k[3], (H, C)) * 0.1).astype(jnp.bfloat16)
y = np.asarray(tr.ref_ln(x.astype(jnp.float32), s, o, 1e-5).astype(jnp.bfloat16).astype(jnp.float32), np.float64)
for shift in (0.0, -8.0, -16.0):
  wa_s = wa.at[0].add(jnp.bfloat16(shift))
  a = y @ np.asarray(wa_s, np.float64)
  ref = (a / (1 + np.exp(-a)) * (y @ np.asarray(wb, np.float64))) @ np.asarray(wo, np.float64)
  report(f"glu_transition swish inputs {shift:+.0f}", rel(tr.fused_glu_transition(x, s, o, wa_s, wb, wo), ref))

X, Hh, D = 128, 4, 64
p = jax.random.normal(k[4], (X, X, 4, Hh, D)).astype(jnp.bfloat16)
bias, kmask = jnp.zeros((Hh, X, X), jnp.bfloat16), jnp.ones((X, X), bool)
for shift in (0.0, -10.0, -20.0):
  gb = jnp.full((Hh, D), shift, jnp.bfloat16)
  with jax.default_matmul_precision('highest'):
    ref = np.asarray(pa.ref_packed_attention(p.astype(jnp.float32), bias, kmask, gb, sm_scale=D ** -0.5), np.float64)
  report(f"packed_attention gate {shift:+.0f}", rel(pa.packed_attention(p, bias, kmask, gb, sm_scale=D ** -0.5), ref))
print("PASS 0" if not fails else f"FAIL {len(fails)}")
sys.exit(bool(fails))

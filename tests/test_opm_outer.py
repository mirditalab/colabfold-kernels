"""opm_outer: AF2's outer product mean in its own order on 16 rows of an MSA whose further rows are zero. Against XLA's
two GEMMs at the full depth (AF2's compute_chunk): bit-identical where XLA does not autotune (XLA_FLAGS
--xla_gpu_autotune_level=0), else within bfloat16 rounding of each other; residue counts that are and are not a
multiple of the block."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from colabfold_kernels import opm

exact = "--xla_gpu_autotune_level=0" in os.environ.get("XLA_FLAGS", "")
fails = []


@jax.jit
def compute_chunk(left, right_t, w):
  act = jnp.einsum('abc,aed->bced', left, right_t)
  act = act.reshape(act.shape[0], -1, act.shape[-1])
  return jnp.einsum('bkd,bkf->bdf', act, jnp.broadcast_to(w, (act.shape[0],) + w.shape))


for n, depth, live in ((256, 128, 12), (520, 512, 16), (136, 64, 1)):
  k = jax.random.split(jax.random.PRNGKey(n), 3)
  rows = (jnp.arange(depth) < live)[:, None, None]
  left = jnp.where(rows, jax.random.normal(k[0], (depth, n, 32)), 0).astype(jnp.bfloat16)
  right = jnp.where(rows, jax.random.normal(k[1], (depth, n, 32)), 0).astype(jnp.bfloat16)
  w = (jax.random.normal(k[2], (32 * 32, 128)) * 0.03).astype(jnp.bfloat16)
  ref = np.asarray(compute_chunk(left, jnp.transpose(right, (0, 2, 1)), w), np.float32)
  out = np.asarray(opm.opm_outer(left[:16], right[:16], w), np.float32)
  diff = np.abs(out - ref)
  ok = bool(np.isfinite(out).all()) and (diff.max() == 0 if exact else diff.max() <= 1e-2 * np.abs(ref).max())
  print(f"{'ok  ' if ok else 'BAD '}n {n} depth {depth} live {live}: {int((diff > 0).sum())} of {diff.size} differ, "
        f"max {diff.max():.3g}{' (exact required)' if exact else ''}")
  if not ok:
    fails.append(n)

sys.exit(1 if fails else 0)

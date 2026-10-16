"""packed attention's query block must not read past p: a single-sequence fold's MSA column attention (one MSA row,
so a query block of block_q rows over a sequence of 1) read block_q - 1 rows of [Y, G, H, D] beyond the end of p and
crashed ColabFold with CUDA_ERROR_ILLEGAL_ADDRESS. The read only faults when nothing is mapped past p, so the call runs
in a fresh process with one allocation per array (XLA_PYTHON_CLIENT_ALLOCATOR=platform), where it faults every time."""
import os
import subprocess
import sys

CALL = r"""
import jax, jax.numpy as jnp
from colabfold_kernels.packed_attn import packed_attention, ref_packed_attention
X, Y, G, H, D = 1, 261, 4, 8, 32
k = jax.random.split(jax.random.PRNGKey(0), 3)
p = jax.random.normal(k[0], (X, Y, G, H, D)).astype(jnp.float16)
bias = jax.random.normal(k[1], (H, X, X)).astype(jnp.float16)
gb = jax.random.normal(k[2], (H, D)).astype(jnp.float16)
kmask = jnp.ones((Y, X), bool)
o = packed_attention(p, bias, kmask, gb, sm_scale=D ** -.5, column=True)
r = ref_packed_attention(p, bias, kmask, gb, sm_scale=D ** -.5, column=True)
err = float(jnp.abs(o.astype(jnp.float32) - r).max())
print("max err", err)
raise SystemExit(err > 0.03)
"""

run = subprocess.run([sys.executable, "-c", CALL], capture_output=True, text=True,
                     env={**os.environ, "XLA_PYTHON_CLIENT_ALLOCATOR": "platform"})
out = (run.stdout + run.stderr).strip().splitlines()
ok = run.returncode == 0
print(("ok  " if ok else "BAD ") + "packed_attention 1x261 8x32 col gated, one allocation per array: "
      + (out[-1] if out else f"exit {run.returncode}")[:160])
print("PASS 0" if ok else "FAIL 1")
sys.exit(not ok)

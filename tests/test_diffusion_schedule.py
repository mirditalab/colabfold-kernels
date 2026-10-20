"""The kernel's noise schedule ends as alphafold3's does for each schedule_tail, and which models realign."""
import os, sys, types
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
import numpy as np
from colabfold_kernels import diffusion


def cfg(tail, steps=200, max_sigma=0.0):
  return types.SimpleNamespace(steps=steps, sigma_max=160.0, sigma_min=4e-4, rho=7.0, schedule_tail=tail,
                               max_sigma=max_sigma)

ok = True
base = diffusion.schedule(cfg(""), "openfold3")
ok &= len(base) == 201 and base[-1] > 0
rep = diffusion.schedule(cfg("replace"), "protenix2")             # protenix / opendde: last point is 0
ok &= len(rep) == 201 and rep[-1] == 0 and np.array_equal(rep[:-1], base[:-1])
ext = diffusion.schedule(cfg("extend"), "intellifold2")           # intellifold2: 0 appended to the 201 points
ok &= len(ext) == 202 and ext[-1] == 0 and np.array_equal(ext[:-1], base)
app = diffusion.schedule(cfg("append", steps=14, max_sigma=256.0), "esmfold2")   # 14 points, 0 appended, capped
ok &= app[0] == 256.0 and app[-1] == 0 and np.all(app[1:] <= 256.0)
chai = diffusion.schedule(cfg("append"), "chai1")                 # chai's midpoints take no tail
ok &= len(chai) == 200 and chai[-1] > 0
ok &= set(diffusion.REALIGN) == set(diffusion.ESM_ATOMS) | {"boltz2"}   # alphafold3 REALIGN_SAMPLER
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

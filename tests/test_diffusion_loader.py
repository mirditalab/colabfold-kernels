"""The diffusion megakernel loads from colabfold-cuda-kernels and registers its FFI target."""
import os, sys
sys.path.insert(0, os.environ.get(
    "CK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from colabfold_kernels import diffusion

cc = diffusion._gpu_cc()
if cc is None or cc < 80 or diffusion._package() is None:
  print(f"SKIP: no colabfold-cuda-kernels library for this GPU (cc {cc})")
  sys.exit(0)

import colabfold_cuda_kernels as cck
ok = diffusion.available() and diffusion._STATE["target"] == cck.symbol("diffusion", cc)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

"""The triangle multiplication's contraction in e4m3, through colabfold-cuda-kernels' cuBLASLt FP8 GEMM.

  out[c] = a[c] b[c]^T, a [C, I, K] and b [C, J, K] e4m3, contracted along their last axis; fp32 accumulation, bf16 out.

Neither XLA nor Pallas Triton reaches cuBLASLt's FP8 GEMM here: XLA upcasts the operands, Triton is 1.7x behind
cuBLAS bf16 at this shape.
"""
import ctypes
import functools
import os

import jax
import jax.numpy as jnp

TARGET = "CckFp8Contract"
# cuBLASLt picks its algorithm for at most this much workspace
_WORKSPACE = 32 << 20


@functools.lru_cache(maxsize=None)
def available():
  """True once the library is loaded and its FFI target registered: NVIDIA, compute capability 8.9 on."""
  from colabfold_kernels.device import compute_capability, is_rocm
  cc = compute_capability()
  if cc is None or cc < 89 or is_rocm():
    return False
  # an explicit library wins, so a local build can stand in for the package
  path = os.environ.get("COLABFOLD_CUDA_KERNELS_LIB")
  if not path:
    try:
      import colabfold_cuda_kernels as cck
      path = cck.library_path("fp8_contract", cc)
    except (ImportError, FileNotFoundError, KeyError):
      return False
  try:
    lib = ctypes.cdll.LoadLibrary(path)
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(getattr(lib, TARGET)), platform="CUDA")
  except (OSError, AttributeError):
    return False
  return True


def supported(k, j):
  """cuBLASLt's FP8 GEMM wants 16-byte leading dimensions: K a multiple of 16, J of 8."""
  return k % 16 == 0 and j % 8 == 0


def fp8_contract(a, b):
  C, I, _ = a.shape
  out, _ = jax.ffi.ffi_call(TARGET, (jax.ShapeDtypeStruct((C, I, b.shape[1]), jnp.bfloat16),
                                     jax.ShapeDtypeStruct((_WORKSPACE,), jnp.uint8)),
                            vmap_method="sequential")(a, b)
  return out


def ref_fp8_contract(a, b):
  return jnp.einsum('cik,cjk->cij', a.astype(jnp.float32), b.astype(jnp.float32))

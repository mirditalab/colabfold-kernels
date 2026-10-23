"""The triangle multiplication's contraction in e4m3, through colabfold-cuda-kernels' cuBLASLt FP8 GEMM.

  out[c] = a[c] b[c]^T, a [C, I, K] and b [C, J, K] e4m3, contracted along their last axis; fp32 accumulation, bf16 out.

live_contract: the contraction over a bucketed fold's live length only (a CUTLASS grouped GEMM that reads it on the
device), bf16 or f16.

Neither XLA nor Pallas Triton reaches cuBLASLt's FP8 GEMM here: XLA upcasts the operands, Triton is 1.7x behind
cuBLAS bf16 at this shape.
"""
import ctypes
import functools
import os

import jax
import jax.numpy as jnp

TARGET = "CckFp8Contract"
LIVE_TARGET = "CckLiveContract"
# the depth, rows and columns live_contract computes: the live length rounded up to its K tile
LIVE_ROUND = 32
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
    except (ImportError, FileNotFoundError):
      return False
  try:
    lib = ctypes.cdll.LoadLibrary(path)
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(getattr(lib, TARGET)), platform="CUDA")
  except OSError:
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


@functools.lru_cache(maxsize=None)
def live_available():
  """True once the library is loaded and CckLiveContract registered: NVIDIA, compute capability 7.0 on."""
  from colabfold_kernels.device import compute_capability, is_rocm
  cc = compute_capability()
  if cc is None or cc < 70 or is_rocm():
    return False
  path = os.environ.get("COLABFOLD_CUDA_KERNELS_LIB")
  if not path:
    try:
      import colabfold_cuda_kernels as cck
      path = cck.library_path("live_contract", cc)
    except (ImportError, FileNotFoundError, KeyError):
      return False
  try:
    lib = ctypes.cdll.LoadLibrary(path)
    jax.ffi.register_ffi_target(LIVE_TARGET, jax.ffi.pycapsule(getattr(lib, LIVE_TARGET)), platform="CUDA")
  except (OSError, AttributeError):
    return False
  return True


def _live_out(a, b, incoming):
  c = a.shape[0]
  i, j = (a.shape[2], b.shape[2]) if incoming else (a.shape[1], b.shape[1])
  # f16 (the sm_70/75 kernels' operands) comes out f32, wide_einsum's: its sums of N products overflow f16
  return c, i, j, jnp.float32 if a.dtype == jnp.float16 else a.dtype


def live_contract(a, b, live, *, incoming=False):
  """out [C, I, J] = a[c] b[c]^T (a [C, I, K], b [C, J, K]) or, incoming, a[c]^T b[c] (a [C, K, I], b [C, K, J]),
  over the leading round_up(live, LIVE_ROUND) rows, columns and depth; out past them is not written. bf16 (out bf16) or
  f16 (out f32), every contiguous extent a multiple of 8. live: int32 [1] or a scalar. The operands must be zero from
  live up to that rounding."""
  c, i, j, dt = _live_out(a, b, incoming)
  out, _ = jax.ffi.ffi_call(LIVE_TARGET, (jax.ShapeDtypeStruct((c, i, j), dt),
                                          jax.ShapeDtypeStruct((64 * c + 256,), jnp.uint8)),
                            vmap_method="sequential")(a, b, jnp.reshape(live, (1,)).astype(jnp.int32),
                                                      incoming=incoming)
  return out


def ref_live_contract(a, b, live, *, incoming=False):
  """live_contract with zeros where it writes nothing."""
  c, i, j, dt = _live_out(a, b, incoming)
  e = -(-jnp.reshape(live, ()) // LIVE_ROUND) * LIVE_ROUND
  eq = 'cki,ckj->cij' if incoming else 'cik,cjk->cij'
  out = jnp.einsum(eq, a.astype(jnp.float32), b.astype(jnp.float32))
  inside = (jnp.arange(i) < e)[:, None] & (jnp.arange(j) < e)[None, :]
  return jnp.where(inside[None], out, 0).astype(dt)

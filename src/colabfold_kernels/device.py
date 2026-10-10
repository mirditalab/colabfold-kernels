"""What the GPUs here can run, so a caller need not work it out itself."""
import functools


def compute_capability():
  """The NVIDIA compute capability as an integer, e.g. 75, or None."""
  try:
    import jax
    cc = str(jax.devices()[0].compute_capability)
    return int(round(float(cc) * 10)) if "." in cc else int(cc)
  except Exception:
    return None


@functools.lru_cache(maxsize=None)
def is_rocm():
  """True on an AMD GPU, where Triton tiles want other sizes than on NVIDIA."""
  import jax
  return "rocm" in (jax.default_backend() + jax.devices()[0].client.platform_version).lower()


def shared_memory_limit():
  """The smallest shared memory budget across the GPUs here, or None."""
  import jax
  limits = [getattr(d, "shared_memory_per_block_optin", 0) for d in jax.local_devices()
            if d.platform == "gpu"]
  limits = [limit for limit in limits if limit]
  return min(limits) if limits else None


def dispatch(backend="auto"):
  """The mapping the fused_ops selectors read. 'auto' picks what this GPU supports."""
  cc = compute_capability()
  if backend == "auto":
    # XLA gates Pallas/Triton to sm_80+, so older NVIDIA parts take the CUDA kernels
    backend = "cuda_legacy" if cc is not None and cc < 80 else "pallas"
  return {"use_pallas": True, "kernel_backend": backend, "compute_capability": cc}

"""Fused Pallas/Triton kernels for ColabFold's AlphaFold2 and AlphaFold3 backends."""

from colabfold_kernels.device import compute_capability, dispatch, shared_memory_limit
from colabfold_kernels.fused_ops import attention, gated_dual_proj, layer_norm

__all__ = ["attention", "compute_capability", "dispatch", "gated_dual_proj",
           "layer_norm", "shared_memory_limit", "__version__"]

__version__ = "0.6.0"

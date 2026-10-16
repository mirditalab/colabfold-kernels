# colabfold-kernels

Fast fused kernels for ColabFold, written in
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) with native CUDA-fallback
for Volta- and Turing-generation GPUs.

```python
import jax.numpy as jnp
from colabfold_kernels import attention, dispatch

kernel = attention(dispatch(), jnp.bfloat16, key_dim=96, value_dim=96, layout="seq")
if kernel is not None:
    out = kernel(q, k, v, mask_bias, nonbatched_bias, scale)
```

`attention`, `layer_norm` and `gated_dual_proj` each return `None` if not supported, as do the other
`fused_ops` selectors.
`diffusion.install()` samples AF3-family diffusion in the
[colabfold-cuda-kernels](https://github.com/mirditalab/colabfold-cuda-kernels) megakernel.

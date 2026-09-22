"""Pick the fused kernel (volta, Pallas, or None for XLA) for each op."""
import functools


def _legacy_cc(global_config):
  """Give the compute capability if the CUDA kernels were asked for, else None."""
  if global_config.get('kernel_backend', 'pallas') != 'cuda_legacy':
    return None
  return global_config.get('compute_capability')


def _volta():
  from colabfold_kernels import volta
  return volta


def attention(global_config, dtype, key_dim, value_dim, layout='heads'):
  """-> f(q, k, v, mask_bias, nonbatched_bias, scale) or None.

  layout 'heads' takes q/k/v as [b, h, S, c], 'seq' as [b, S, h, c].
  """
  import jax.numpy as jnp
  if not global_config.get('use_pallas', False):
    return None
  cc = _legacy_cc(global_config)
  if dtype == jnp.float16 and cc is not None and layout == 'heads':
    va = _volta()
    # The kernels only handle a fixed set of head dims, and need key == value.
    if (va.available(cc) and key_dim == value_dim
        and va.supports(int(key_dim), cc)):
      return functools.partial(va.volta_attention, cc=cc)
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype in (jnp.bfloat16, jnp.float16)):
    from colabfold_kernels.tri_flash import pallas_attention
    if layout == 'seq':
      return functools.partial(pallas_attention, seq_major=True)
    return pallas_attention
  return None


def layer_norm(global_config, dtype):
  """-> f(x, scale, offset, *, eps) or None. Last-axis, scale+offset only."""
  import jax.numpy as jnp
  if not global_config.get('use_pallas', False):
    return None
  from colabfold_kernels.grads import differentiable_layer_norm
  cc = _legacy_cc(global_config)
  if dtype == jnp.float16 and cc is not None:
    va = _volta()
    if va.ops_available(cc):
      return differentiable_layer_norm(va.volta_layer_norm)
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype in (jnp.bfloat16, jnp.float32)):
    from colabfold_kernels.tri_mul import pallas_layer_norm
    return differentiable_layer_norm(pallas_layer_norm)
  return None


def gated_dual_proj(global_config, dtype, activation=None):
  """-> f(x, wp, bp, wg, bg, mask, *, split) or None. activation defaults to sigmoid."""
  import jax
  import jax.numpy as jnp
  if not global_config.get('use_pallas', False):
    return None
  from colabfold_kernels.grads import differentiable_gated_dual_proj
  cc = _legacy_cc(global_config)
  act = jax.nn.sigmoid if activation is None else activation
  # the prebuilt kernels gate with a sigmoid and nothing else
  if (dtype == jnp.float16 and cc is not None
      and activation in (None, jax.nn.sigmoid)):
    va = _volta()
    if va.ops_available(cc):
      return differentiable_gated_dual_proj(
          functools.partial(va.volta_gated_dual_proj, cc=cc), act)
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype in (jnp.bfloat16, jnp.float16)):
    from colabfold_kernels.tri_mul import gated_dual_proj as gdp
    fwd = gdp if activation is None else functools.partial(gdp, activation=activation)
    return differentiable_gated_dual_proj(fwd, act)
  return None

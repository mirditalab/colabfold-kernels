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


def _pallas_half(global_config, dtype):
  """The Pallas backend is on and dtype is a half type (the forward-only fused kernels)."""
  import jax.numpy as jnp
  return (global_config.get('use_pallas', False) and _legacy_cc(global_config) is None
          and global_config.get('kernel_backend', 'pallas') == 'pallas'
          and dtype in (jnp.bfloat16, jnp.float16))


@functools.lru_cache(maxsize=None)
def _mgpu_attn():
  """mgpu_attn on sm_12x, None elsewhere or when this jax's Mosaic GPU API does not match."""
  from colabfold_kernels.device import compute_capability
  cc = compute_capability()
  if cc is None or not 120 <= cc < 130:
    return None
  try:
    import jax
    import jax.numpy as jnp
    from colabfold_kernels import mgpu_attn
    # a Mosaic GPU API this kernel does not match fails only once lowered, so compile a small call here
    sds = jax.ShapeDtypeStruct
    jax.jit(functools.partial(mgpu_attn.packed_attention, sm_scale=1.0)).lower(
        sds((8, 64, 4, 2, 32), jnp.bfloat16), sds((2, 64, 64), jnp.bfloat16), sds((8, 64), jnp.bool_),
        sds((2, 32), jnp.bfloat16)).compile()
    return mgpu_attn
  except Exception:
    return None


def _mosaic_or_triton(p, bias, kmask, gate_bias=None, *, sm_scale, column=False):
  """The Mosaic GPU kernel where it runs, else Triton's."""
  from colabfold_kernels import packed_attn as pa
  mgpu_attn = _mgpu_attn()
  if mgpu_attn is not None and mgpu_attn.supported(p.shape[-1], p.dtype, p.shape[0 if column else 1]):
    return mgpu_attn.packed_attention(p, bias, kmask, gate_bias, sm_scale=sm_scale, column=column)
  return pa.packed_attention(p, bias, kmask, gate_bias, sm_scale=sm_scale, column=column)


def packed_attention(global_config, dtype, key_dim, value_dim):
  """-> f(p, bias, kmask, gate_bias, *, sm_scale, column) or None; p is the packed [X, Y, G, H, D] projection."""
  import jax.numpy as jnp
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import packed_attn as pa
  heads = attention(global_config, dtype, key_dim, value_dim)
  if heads is None or key_dim != value_dim:
    return None
  ref = functools.partial(pa.attention_via, attention=heads)
  cc = _legacy_cc(global_config)
  if cc is not None and dtype == jnp.float16:
    va = _volta()
    if not va.packed_available(cc, key_dim):
      return None
    return differentiable_via(functools.partial(va.volta_packed_attention, cc=cc), ref,
                              nondiff=('sm_scale', 'column'))
  if not _pallas_half(global_config, dtype):
    return None
  fwd = _mosaic_or_triton if global_config.get('mosaic_gpu', True) else pa.packed_attention
  return differentiable_via(fwd, ref, nondiff=('sm_scale', 'column'))


def transition(global_config, dtype, num_channel, num_intermediate):
  """-> f(x, ln_scale, ln_offset, w1, b1, w2, b2, *, eps) or None: the whole LN-MLP transition."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels import transition as t
  from colabfold_kernels.grads import differentiable_via
  if not t.supported(num_channel, num_intermediate):
    return None
  return differentiable_via(t.fused_transition, t.ref_transition, nondiff=('eps', 'add_input'))


def glu_transition(global_config, dtype, num_channel, num_intermediate):
  """-> f(x, ln_scale, ln_offset, wa, wb, wo, keep=None, *, eps, add_input) or None: AF3's whole swish-GLU transition."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels import transition as t
  from colabfold_kernels.grads import differentiable_via
  if not t.glu_supported(num_channel, num_intermediate):
    return None
  return differentiable_via(t.fused_glu_transition, t.ref_glu_transition, nondiff=('eps', 'add_input'))


def trimul_epilogue(global_config, dtype):
  """-> f(x_cm, act, ln_scale, ln_offset, wo, bo, wg, bg, *, eps) or None."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import tri_mul
  return differentiable_via(tri_mul.trimul_epilogue, tri_mul.ref_trimul_epilogue, nondiff=('eps',))


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
  if dtype == jnp.float16 and cc is not None:
    va = _volta()
    # the prebuilt kernels gate with sigmoid or swish, nothing else
    if act in va.GDP_ACTIVATIONS and va.ops_available(cc):
      return differentiable_gated_dual_proj(
          functools.partial(va.volta_gated_dual_proj, cc=cc, activation=act), act)
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype in (jnp.bfloat16, jnp.float16)):
    from colabfold_kernels.tri_mul import gated_dual_proj as gdp
    fwd = gdp if activation is None else functools.partial(gdp, activation=activation)
    return differentiable_gated_dual_proj(fwd, act)
  return None

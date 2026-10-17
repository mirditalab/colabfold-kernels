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


def fp8_attention(global_config, key_dim):
  """The projection may write k and v in e4m3 for Triton's packed_attention (0.6.0's interface, AF2 calls it): not
  where the Mosaic GPU kernel runs, which wants q and k (fp8_attention_slots)."""
  if global_config.get('mosaic_gpu', True) and _mgpu_attn() is not None:
    return False
  return fp8_attention_slots(global_config, None, key_dim, 0) is not None


def fp8_attention_slots(global_config, dtype, key_dim, seq_len):
  """(f8_slots, f8_scales) for the projection to write in e4m3 for packed_attention, or None: asked for
  (global_config.fp8), on a card with 8-bit MMAs (NVIDIA, compute capability 8.9 on), at a head dim of a power of two
  from their K of 32. Triton's kernel takes k and v; the Mosaic GPU one q, pre-scaled, and k (its e4m3 p.v is no faster).
  """
  if not global_config.get('fp8', False) or key_dim & (key_dim - 1) or key_dim < 32:
    return None
  from colabfold_kernels.device import compute_capability, is_rocm
  from colabfold_kernels.tri_flash import LOG2E
  cc = compute_capability()
  if cc is None or cc < 89 or is_rocm():
    return None
  if dtype is not None and global_config.get('mosaic_gpu', True) and _mosaic_runs(key_dim, dtype, seq_len):
    return (0, 1), (key_dim ** -0.5 * LOG2E, 1.0)
  return (1, 2), (1.0, 1.0)


def _mosaic_runs(head_dim, dtype, seq_len):
  """The Mosaic GPU kernel runs here at this shape: _mosaic_or_triton and fp8_attention must agree on it."""
  mgpu_attn = _mgpu_attn()
  return mgpu_attn is not None and mgpu_attn.supported(head_dim, dtype, seq_len)


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


def _mosaic_or_triton(p, bias, kmask, gate_bias=None, p8=None, *, sm_scale, column=False):
  """The Mosaic GPU kernel where it runs, else Triton's. p8 holds the e4m3 slots fp8_attention chose for that kernel."""
  from colabfold_kernels import packed_attn as pa
  if _mosaic_runs(p.shape[-1], p.dtype, p.shape[0 if column else 1]):
    return _mgpu_attn().packed_attention(p, bias, kmask, gate_bias, p8, sm_scale=sm_scale, column=column)
  return pa.packed_attention(p, bias, kmask, gate_bias, p8, sm_scale=sm_scale, column=column)


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


def trimul_epilogue_ln(global_config, dtype):
  """-> f(x_cm, x, in_scale, in_offset, ln_scale, ln_offset, wo, bo, wg, bg, *, eps, add_input) or None."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import tri_mul
  return differentiable_via(tri_mul.trimul_epilogue_ln, tri_mul.ref_trimul_epilogue_ln, nondiff=('eps', 'add_input'))


def ln_gated_dual_proj(global_config, dtype, num_channel, num_intermediate):
  """-> f(x, ln_scale, ln_offset, wp, bp, wg, bg, mask, *, eps) -> (left, right) channel major, or None."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import tri_mul
  if not tri_mul.epilogue_supported(num_channel, num_intermediate, num_intermediate):
    return None
  return differentiable_via(tri_mul.ln_gated_dual_proj, tri_mul.ref_ln_gdp, nondiff=('eps',))


def fp8_contraction(global_config, dtype, num_channel, num_intermediate, num_res):
  """-> (projection, contract) or None: ln_gated_dual_proj writing its halves e4m3, padded to 16, and their
  contraction through cuBLASLt's FP8 GEMM; global_config.fp8, on compute capability 8.9 on."""
  if not global_config.get('fp8', False) or ln_gated_dual_proj(global_config, dtype, num_channel,
                                                                num_intermediate) is None:
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import contract, tri_mul
  # the product's rows are num_res wide: cuBLASLt wants them 16 bytes (ColabFold pads the length to 8)
  if not contract.supported(16, num_res) or not contract.available():
    return None
  proj = differentiable_via(functools.partial(tri_mul.ln_gated_dual_proj, fp8=True, pad=16),
                            functools.partial(tri_mul.ref_ln_gdp, pad=16), nondiff=('eps', 'transpose'))
  return proj, differentiable_via(contract.fp8_contract, contract.ref_fp8_contract)


def ln_projection(global_config, dtype, num_channel, num_output=None):
  """-> f(x, ln_scale, ln_offset, w, wb, *, eps) -> (p, bias) or None: LayerNorm then the packed and bias projections.

  num_output is the width of each packed slot, None when only the bias is projected.
  """
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import packed_attn as pa
  if not pa.ln_projection_supported(num_channel, num_output):
    return None
  return differentiable_via(pa.ln_projection, pa.ref_ln_projection, nondiff=('eps', 'f8_slots', 'f8_scales'))


def output_projection(global_config, dtype, num_input, num_output):
  """-> f(o, w, b, residual=None) or None: the attention output projection with its residual update."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import packed_attn as pa
  if not pa.output_projection_supported(num_input, num_output):
    return None
  return differentiable_via(pa.output_projection, pa.ref_output_projection)


def opm_update(global_config, dtype, num_inner, num_output):
  """-> f(lhs, t, bias, norm, residual=None, *, eps) or None: the shallow outer product mean's whole pair update."""
  if not _pallas_half(global_config, dtype):
    return None
  from colabfold_kernels.grads import differentiable_via
  from colabfold_kernels import opm
  if not opm.supported(num_inner, num_output):
    return None
  return differentiable_via(opm.opm_update, opm.ref_opm_update, nondiff=('eps',))


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

"""Pallas/Triton outer product mean update on a shallow MSA, after AF2 folds its output weights into the right side.

One pass writes the pair update: the GEMM over (rows, channels), its bias, the mask norm and the residual, where XLA
writes the GEMM's [B, D*F] product and reads it back.

  out[b, d] = residual[b, d] + (lhs[b] . t[:, d] + bias) / (eps + norm[b, d])
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


def _kernel(a_ref, t_ref, b_ref, n_ref, *rest, m, block_m, block_k, eps):
  r_ref, o_ref = rest if len(rest) == 2 else (None, rest[0])
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m

  def body(k, acc):
    ks = pl.ds(pl.multiple_of(k * block_k, block_k), block_k)
    a = plgpu.load(a_ref.at[:, ks], mask=rows[:, None], other=0.0)
    return acc + _dot(a, t_ref[ks, :])

  acc = jnp.zeros((block_m, t_ref.shape[1]), jnp.float32)
  out = jax.lax.fori_loop(0, a_ref.shape[1] // block_k, body, acc) + b_ref[...][None, :].astype(jnp.float32)
  out = out / (eps + plgpu.load(n_ref, mask=rows, other=1.0).astype(jnp.float32))[:, None]
  if r_ref is not None:
    out = out + plgpu.load(r_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  plgpu.store(o_ref, out.astype(o_ref.dtype), mask=rows[:, None])


def supported(k, f):
  """K a multiple of 32 (rows times the outer channels), F a whole-tile power of two from 16 to 128."""
  return k % 32 == 0 and f & (f - 1) == 0 and 16 <= f <= 128


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "num_warps", "num_stages"))
def opm_update(lhs, t, bias, norm, residual=None, *, eps=1e-3, block_m=64, num_warps=4, num_stages=2):
  """lhs [B, K]; t [K, D, F]; bias [F]; norm [B, D]; residual [B, D, F].

  Returns [B, D, F], residual's dtype, else lhs's.
  """
  B, K = lhs.shape
  D, F = t.shape[1:]
  block_k = 64 if K % 64 == 0 else 32
  rows = pl.BlockSpec((block_m, None, F), lambda i, d: (i, d, 0))
  extra = [] if residual is None else [residual]
  # d innermost: a block of lhs is reused across the row of pair blocks, and all of t stays in L2
  return pl.pallas_call(
      functools.partial(_kernel, m=B, block_m=block_m, block_k=block_k, eps=eps),
      grid=(pl.cdiv(B, block_m), D),
      in_specs=[pl.BlockSpec((block_m, K), lambda i, d: (i, 0)), pl.BlockSpec((K, None, F), lambda i, d: (0, d, 0)),
                pl.BlockSpec((F,), lambda i, d: (0,)), pl.BlockSpec((block_m, None), lambda i, d: (i, d))]
               + [rows] * len(extra),
      out_specs=rows,
      out_shape=jax.ShapeDtypeStruct((B, D, F), lhs.dtype if residual is None else residual.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="opm_update",
  )(lhs, t, bias, norm, *extra)


def ref_opm_update(lhs, t, bias, norm, residual=None, *, eps=1e-3):
  out = jnp.einsum('bk,kdf->bdf', lhs.astype(jnp.float32), t.astype(jnp.float32)) + bias.astype(jnp.float32)
  out = out / (eps + norm.astype(jnp.float32))[..., None]
  return out if residual is None else out + residual.astype(jnp.float32)

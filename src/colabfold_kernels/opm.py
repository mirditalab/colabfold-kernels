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

from colabfold_kernels.device import is_rocm
from colabfold_kernels.transition import ln_rows, ref_ln

_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)


def _proj_kernel(x_ref, s_ref, o_ref, lw_ref, lb_ref, rw_ref, rb_ref, m_ref, l_out, r_out, *, eps, m, block_m):
  rows = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m
  x = plgpu.load(x_ref, mask=rows[:, None], other=0.0).astype(jnp.float32)
  y = ln_rows(x, s_ref, o_ref, eps).astype(x_ref.dtype)
  mv = plgpu.load(m_ref, mask=rows, other=0.0).astype(jnp.float32)[:, None]
  la = mv * (_dot(y, lw_ref[...]) + lb_ref[...][None, :].astype(jnp.float32))
  ra = mv * (_dot(y, rw_ref[...]) + rb_ref[...][None, :].astype(jnp.float32))
  plgpu.store(l_out, la.astype(l_out.dtype), mask=rows[:, None])
  plgpu.store(r_out, ra.astype(r_out.dtype), mask=rows[:, None])


def proj_supported(c, k):
  return c & (c - 1) == 0 and 16 <= c <= 256 and k & (k - 1) == 0 and 16 <= k <= 64


@functools.partial(jax.jit, static_argnames=("eps", "block_m", "num_warps", "num_stages"))
def opm_projection(x, ln_scale, ln_offset, lw, lb, rw, rb, mask, *, eps=1e-5, block_m=None, num_warps=None,
                   num_stages=1):
  c, k = x.shape[-1], lw.shape[1]
  xf = x.reshape(-1, c)
  m = xf.shape[0]
  block_m = block_m or (64 if c > 128 or is_rocm() else 128)
  num_warps = num_warps or (4 if block_m == 64 else 8)
  vec = lambda n: pl.BlockSpec((n,), lambda i: (0,))
  out_s = pl.BlockSpec((block_m, k), lambda i: (i, 0))
  la, ra = pl.pallas_call(
      functools.partial(_proj_kernel, eps=eps, m=m, block_m=block_m),
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((block_m, c), lambda i: (i, 0)), vec(c), vec(c),
                pl.BlockSpec((c, k), lambda i: (0, 0)), vec(k),
                pl.BlockSpec((c, k), lambda i: (0, 0)), vec(k),
                pl.BlockSpec((block_m,), lambda i: (i,))],
      out_specs=[out_s, out_s],
      out_shape=[jax.ShapeDtypeStruct((m, k), x.dtype)] * 2,
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="opm_projection",
  )(xf, ln_scale, ln_offset, lw, lb, rw, rb, mask.reshape(m))
  shape = x.shape[:-1] + (k,)
  return la.reshape(shape), ra.reshape(shape)


def ref_opm_projection(x, ln_scale, ln_offset, lw, lb, rw, rb, mask, *, eps=1e-5):
  y = ref_ln(x.astype(jnp.float32), ln_scale, ln_offset, eps)
  mv = mask.astype(jnp.float32)[..., None]
  return (mv * (y @ lw.astype(jnp.float32) + lb.astype(jnp.float32)),
          mv * (y @ rw.astype(jnp.float32) + rb.astype(jnp.float32)))


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

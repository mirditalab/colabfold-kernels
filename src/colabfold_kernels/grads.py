"""Backward passes for layer_norm and gdp."""
import functools

import jax
import jax.numpy as jnp


def _zero_like_cotangent(x):
  """An integral input takes the empty float0 cotangent."""
  if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact):
    return jnp.zeros_like(x)
  return jnp.zeros(jnp.shape(x), jax.dtypes.float0)


# ---- layer norm -------------------------------------------------------------
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 4))
def _layer_norm(fwd, x, scale, offset, eps):
  return fwd(x, scale, offset, eps=eps)


def _layer_norm_fwd(fwd, x, scale, offset, eps):
  return _layer_norm(fwd, x, scale, offset, eps), (x, scale, offset)


def _layer_norm_bwd(fwd, eps, res, dy):
  x, scale, offset = res
  xf = x.astype(jnp.float32)
  dyf = dy.astype(jnp.float32)
  mean = jnp.mean(xf, axis=-1, keepdims=True)
  d = xf - mean
  rstd = jax.lax.rsqrt(jnp.mean(d * d, axis=-1, keepdims=True) + eps)
  xhat = d * rstd
  axes = tuple(range(x.ndim - 1))
  dscale = jnp.sum(dyf * xhat, axis=axes)
  doffset = jnp.sum(dyf, axis=axes)
  dxhat = dyf * scale.astype(jnp.float32)
  dx = rstd * (dxhat - jnp.mean(dxhat, axis=-1, keepdims=True)
               - xhat * jnp.mean(dxhat * xhat, axis=-1, keepdims=True))
  return dx.astype(x.dtype), dscale.astype(scale.dtype), doffset.astype(offset.dtype)


_layer_norm.defvjp(_layer_norm_fwd, _layer_norm_bwd)


def differentiable_layer_norm(fwd):
  """Wrap a fused layer norm so a gradient can pass through it."""
  @functools.wraps(fwd)
  def call(x, scale, offset, *, eps=1e-5):
    return _layer_norm(fwd, x, scale, offset, eps)
  return call


# ---- gated dual projection --------------------------------------------------
# y = mask * (x @ wp + bp) * activation(x @ wg + bg), the two projections
# recomputed in the backward because the forward never writes them out.
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 7, 8, 9))
def _gdp(fwd, x, wp, bp, wg, bg, mask, split, channel_major, activation):
  if split:
    return fwd(x, wp, bp, wg, bg, mask, split=True, channel_major=channel_major)
  return fwd(x, wp, bp, wg, bg, mask, split=False)


def _gdp_fwd(fwd, x, wp, bp, wg, bg, mask, split, channel_major, activation):
  y = _gdp(fwd, x, wp, bp, wg, bg, mask, split, channel_major, activation)
  return y, (x, wp, wg, bp, bg, mask)


def _gdp_bwd(fwd, split, channel_major, activation, res, dy):
  x, wp, wg, bp, bg, mask = res
  if split:
    # the halves are the two contiguous channel blocks of one [M, P] output,
    # transposed to [P/2, M] when the caller asked for channel major
    left, right = (t.T for t in dy) if channel_major else dy
    dy = jnp.concatenate([left, right], axis=-1)
  xf = x.astype(jnp.float32)
  dyf = dy.astype(jnp.float32)
  proj = xf @ wp.astype(jnp.float32) + bp.astype(jnp.float32)
  gate = xf @ wg.astype(jnp.float32) + bg.astype(jnp.float32)
  act, act_vjp = jax.vjp(activation, gate)
  m = mask.astype(jnp.float32)[:, None]
  dproj = dyf * m * act
  dgate = act_vjp(dyf * m * proj)[0]
  dx = dproj @ wp.astype(jnp.float32).T + dgate @ wg.astype(jnp.float32).T
  if jnp.issubdtype(mask.dtype, jnp.inexact):
    dmask = jnp.sum(dyf * proj * act, axis=-1).astype(mask.dtype)
  else:
    dmask = _zero_like_cotangent(mask)
  return (dx.astype(x.dtype),
          (xf.T @ dproj).astype(wp.dtype), jnp.sum(dproj, axis=0).astype(bp.dtype),
          (xf.T @ dgate).astype(wg.dtype), jnp.sum(dgate, axis=0).astype(bg.dtype),
          dmask)


_gdp.defvjp(_gdp_fwd, _gdp_bwd)


def differentiable_gated_dual_proj(fwd, activation):
  """Wrap a fused gated dual projection so a gradient can pass through it."""
  @functools.wraps(fwd)
  def call(x, wp, bp, wg, bg, mask, *, split=False, channel_major=False):
    return _gdp(fwd, x, wp, bp, wg, bg, mask, split, channel_major, activation)
  return call


# ---- forward-only kernels, backward through a differentiable reference -------
def differentiable_via(fwd, ref, nondiff=()):
  """fwd's forward with the gradient of ref, which recomputes in the backward.

  fwd and ref take the same arguments; the keyword arguments named in nondiff are static and
  integral positional inputs get the empty float0 cotangent.
  """
  def call(*args, **kwargs):
    if set(kwargs) - set(nondiff):
      raise TypeError(f"{getattr(fwd, '__name__', fwd)}: pass {sorted(set(kwargs) - set(nondiff))} positionally")
    static = tuple(sorted(kwargs.items()))
    return _via(fwd, ref, static, *args)
  return functools.wraps(fwd)(call)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _via(fwd, ref, static, *args):
  return fwd(*args, **dict(static))


def _via_fwd(fwd, ref, static, *args):
  return _via(fwd, ref, static, *args), args


def _via_bwd(fwd, ref, static, args, dy):
  diff = [i for i, a in enumerate(args)
          if a is not None and jnp.issubdtype(jnp.asarray(a).dtype, jnp.inexact)]
  def f(*xs):
    full = list(args)
    for i, x in zip(diff, xs):
      full[i] = x
    return jax.tree.map(lambda o, d: o.astype(d.dtype), ref(*full, **dict(static)), dy)
  _, vjp = jax.vjp(f, *[args[i] for i in diff])
  grads = dict(zip(diff, vjp(dy)))
  return tuple(None if a is None else grads[i] if i in grads else _zero_like_cotangent(a)
               for i, a in enumerate(args))


_via.defvjp(_via_fwd, _via_bwd)

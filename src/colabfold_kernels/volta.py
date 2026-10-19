"""sm_70/75 CUDA kernels (colabfold-legacy-kernels) used instead of Pallas."""
import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

# Head dims each kernel instantiates: mma.sync (sm_75+) and wmma (sm_70).
# 24/48/96 are af3's: single 384/16, diffusion token 768/16, protenix2's 96.
# wmma has no 24 because its fragment is 16 wide, so 24 pads to 32.
_HEAD_DIMS_MMA = (8, 16, 24, 32, 48, 64, 96)
_HEAD_DIMS_WMMA = (8, 16, 32, 48, 64, 96)

_LOADED = {}


def _package_path(kernel, cc):
    """Give the library path from colabfold_legacy_kernels, or None."""
    try:
        import colabfold_legacy_kernels as clk
    except ImportError:
        return None
    try:
        return clk.library_path(kernel, cc)
    except (FileNotFoundError, KeyError):
        return None


def _load(kernel, cc, symbols):
    """dlopen the library for one kernel and register its FFI targets."""
    key = (kernel, cc, tuple(symbols))
    if key in _LOADED:
        return _LOADED[key]
    path = _package_path(kernel, cc)
    if not path or not os.path.exists(path):
        _LOADED[key] = False
        return False
    lib = ctypes.cdll.LoadLibrary(path)
    for sym in symbols:
        jax.ffi.register_ffi_target(
            sym, jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
    _LOADED[key] = True
    return True


def _attn_symbol(cc):
    """sm_75+ gets the CUTLASS kernel; sm_70 the wmma one."""
    return "VoltaMma" if cc >= 75 else "VoltaWmma"


def _attn_bwd_symbol(cc):
    return "VoltaMmaBwd" if cc >= 75 else "VoltaWmmaBwd"


def bwd_available(cc):
    """True if the backward library for this device loads."""
    return _load("attention_bwd", cc, (_attn_bwd_symbol(cc),))


def available(cc):
    """True if the attention library for this device loads."""
    return _load("attention", cc, (_attn_symbol(cc),))


def _native_dims(cc):
    return _HEAD_DIMS_MMA if cc >= 75 else _HEAD_DIMS_WMMA


def _pad_target(head_dim, cc):
    """Smallest supported head dim >= head_dim, or None."""
    for d in sorted(_native_dims(cc)):
        if d >= head_dim:
            return d
    return None


def supports(head_dim, cc):
    if head_dim in _native_dims(cc):
        return True
    # wmma: zero-pad to the next supported head dim (exact, beats XLA on V100).
    if cc < 75:
        return _pad_target(head_dim, cc) is not None
    return False


def ops_available(cc):
    """LayerNorm is Volta-capable; the CUTLASS gdp needs sm_75+."""
    syms = ("VoltaLayerNorm", "VoltaGdp") if cc >= 75 else ("VoltaLayerNorm",)
    ok = _load("layer_norm", cc, syms)
    if ok and cc < 75:
        ok = _load("gated_dual_proj", cc, ("VoltaGdpWmma",))
    return ok


@functools.partial(jax.jit,
                   static_argnames=("sym", "sm_scale", "block_q", "block_k", "want_lse"))
def _attn_call(q, k, v, bias, kmask, *, sym, sm_scale, block_q, block_k, want_lse):
    n, h, sq, d = q.shape
    # lse is the softmax statistic the backward rebuilds P from. Inference does
    # not want it, and the kernel then leaves the buffer alone, so ask for one
    # element rather than [n, h, sq].
    lse_shape = (n, h, sq) if want_lse else (1,)
    # sequential: the template stack vmaps over templates, and the handlers take a fixed rank.
    out, lse = jax.ffi.ffi_call(sym, (jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
                                      jax.ShapeDtypeStruct(lse_shape, jnp.float32)),
                                vmap_method="sequential")(
        q, k, v, bias, kmask,
        scale=np.float32(sm_scale),
        block_q=np.int64(block_q), block_k=np.int64(block_k), want_lse=want_lse)
    return (out, lse) if want_lse else out


@functools.partial(jax.jit, static_argnames=("sym", "sm_scale", "want_dbias"))
def _attn_bwd_call(q, k, v, bias, kmask, out, dout, lse, *, sym, sm_scale, want_dbias):
    """dQ/dK/dV and, when asked, dBias. The kernel takes out, not the row
    statistic, and makes that itself."""
    n, h, sq, d = q.shape
    sk = k.shape[2]
    return jax.ffi.ffi_call(sym, (jax.ShapeDtypeStruct(q.shape, jnp.float16),
                                  jax.ShapeDtypeStruct(k.shape, jnp.float16),
                                  jax.ShapeDtypeStruct(v.shape, jnp.float16),
                                  jax.ShapeDtypeStruct((h, sq, sk), jnp.float32)),
                            vmap_method="sequential")(
        q, k, v, bias, kmask, out, dout, lse,
        scale=np.float32(sm_scale), want_dbias=want_dbias)


# The custom_vjp covers the kernel call only. Padding the head dim and slicing
# the output are ordinary jnp ops, so their transposes come for free.
@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _attn_diff(q, k, v, bias, kmask, cc, sm_scale, block_q, block_k):
    return _attn_call(q, k, v, bias, kmask, sym=_attn_symbol(cc), sm_scale=sm_scale,
                      block_q=block_q, block_k=block_k, want_lse=False)


def _attn_diff_fwd(q, k, v, bias, kmask, cc, sm_scale, block_q, block_k):
    out, lse = _attn_call(q, k, v, bias, kmask, sym=_attn_symbol(cc), sm_scale=sm_scale,
                          block_q=block_q, block_k=block_k, want_lse=True)
    return out, (q, k, v, bias, kmask, out, lse)


def _attn_diff_bwd(cc, sm_scale, block_q, block_k, res, dout):
    q, k, v, bias, kmask, out, lse = res
    # the forward registers only its own symbol
    _load("attention_bwd", cc, (_attn_bwd_symbol(cc),))
    dq, dk, dv, dbias = _attn_bwd_call(
        q, k, v, bias, kmask, out, dout.astype(jnp.float16), lse,
        sym=_attn_bwd_symbol(cc), sm_scale=sm_scale, want_dbias=True)
    # kmask is integral, thus its cotangent is the empty float0.
    return (dq, dk, dv, dbias.astype(bias.dtype),
            jnp.zeros(kmask.shape, jax.dtypes.float0))


_attn_diff.defvjp(_attn_diff_fwd, _attn_diff_bwd)


def volta_attention(q, k, v, mask_bias, nonbatched_bias, scale, cc,
                    block_q=64, block_k=32):
    """Drop-in for tri_flash.pallas_attention. q/k/v [b,h,S,c] float16.

    Differentiable; bwd_available(cc) says whether the backward library is
    there to do it.
    """
    in_dtype = q.dtype
    b, h, sq, c = q.shape
    sk = k.shape[2]
    assert c == v.shape[-1], "kernel needs key_dim == value_dim"
    pad_to = None
    if c not in _native_dims(cc):
        pad_to = _pad_target(c, cc)
        assert pad_to is not None, f"unsupported head_dim {c}"
        pad = [(0, 0)] * 3 + [(0, pad_to - c)]
        q, k, v = (jnp.pad(t.astype(jnp.float16), pad) for t in (q, k, v))
    if cc >= 75 and (pad_to or c) == 64 and (block_q, block_k) == (64, 32):
        block_k = 64    # the mma kernel has no (64, 64, 32) instantiation
    # Template pointwise attention shares one mask row across the batch; the kernel
    # indexes the mask per row, so broadcast instead of reading past its end.
    kmask = jnp.broadcast_to(mask_bias[:, 0, 0, :] > -1e3, (b, sk)).astype(jnp.uint8)
    bias = (jnp.zeros((h, sq, sk), jnp.float16) if nonbatched_bias is None
            else nonbatched_bias.astype(jnp.float16))
    out = _attn_diff(q.astype(jnp.float16), k.astype(jnp.float16),
                     v.astype(jnp.float16), bias, kmask,
                     int(cc), float(scale), block_q, block_k)
    if pad_to is not None:
        out = out[..., :c]
    return out.astype(in_dtype)


def packed_available(cc, head_dim):
    """True if this device's attention library reads packed q/k/v (packed_attn.packed_attention)."""
    return head_dim in _native_dims(cc) and _load("attention", cc, (_attn_symbol(cc) + "Packed",))


def packed_trim_available(cc):
    """True if this device's attention library also takes trim limits (colabfold-legacy-kernels 0.6)."""
    try:
        return _load("attention", cc, (_attn_symbol(cc) + "PackedTrim",))
    except AttributeError:
        return False


@functools.partial(jax.jit, static_argnames=("sm_scale", "column", "cc", "block_q", "block_k", "trim"))
def volta_packed_attention(p, bias, kmask, gate_bias=None, *, sm_scale, column=False, cc,
                           block_q=64, block_k=32, trim=False):
    """packed_attn.packed_attention for sm_70/75: p [X, Y, G, H, D] read in place, out [X, Y, H, D].

    trim: packed_attn.packed_attention's, where the library has it (else every key and query run)."""
    X, Y, _, H, D = p.shape
    if cc >= 75 and D == 64 and (block_q, block_k) == (64, 32):
        block_k = 64    # the mma kernel has no (64, 64, 32) instantiation
    f16 = lambda t: t.astype(jnp.float16)
    gb = jnp.zeros((H, D), jnp.float16) if gate_bias is None else f16(gate_bias)
    lims = []
    if trim and packed_trim_available(cc):
        from colabfold_kernels.packed_attn import trim_limits
        lims = [jnp.concatenate(trim_limits(kmask, 1))]
    out = jax.ffi.ffi_call(_attn_symbol(cc) + ("PackedTrim" if lims else "Packed"),
                           jax.ShapeDtypeStruct((X, Y, H, D), jnp.float16),
                           vmap_method="sequential")(
        f16(p), f16(bias), kmask.astype(jnp.uint8), gb, *lims, scale=np.float32(sm_scale),
        block_q=np.int64(block_q), block_k=np.int64(block_k), column=column)
    return out.astype(p.dtype)


@functools.partial(jax.jit, static_argnames=("eps",))
def _ln_call(x, scale, offset, *, eps):
    m, c = x.shape
    return jax.ffi.ffi_call(
        "VoltaLayerNorm", jax.ShapeDtypeStruct((m, c), jnp.float16),
        vmap_method="sequential")(
            x, scale, offset, eps=np.float32(eps))


def volta_layer_norm(x, scale, offset, *, eps=1e-5):
    """Drop-in for tri_mul.pallas_layer_norm. Normalises the last axis."""
    c = x.shape[-1]
    out = _ln_call(x.reshape(-1, c).astype(jnp.float16),
                   scale.astype(jnp.float32), offset.astype(jnp.float32), eps=eps)
    return out.reshape(x.shape).astype(x.dtype)


# What the kernels gate with. jax.nn.swish is jax.nn.silu, one object.
GDP_ACTIVATIONS = {jax.nn.sigmoid: 0, jax.nn.swish: 1}


@functools.partial(jax.jit, static_argnames=("sym", "activation"))
def _gdp_call(x, wp, bp, wg, bg, mask, *, sym, activation):
    m = x.shape[0]
    n = wp.shape[1]
    return jax.ffi.ffi_call(sym, jax.ShapeDtypeStruct((m, n), jnp.float16),
                            vmap_method="sequential")(
        x, wp, bp, wg, bg, mask, activation=np.int64(activation))


def volta_gated_dual_proj(x, wp, bp, wg, bg, mask, cc, *, split=False,
                          channel_major=False, activation=jax.nn.sigmoid):
    """Drop-in for tri_mul.gated_dual_proj. x [M,K]; wp/wg [K,P]; mask [M]."""
    sym = "VoltaGdp" if cc >= 75 else "VoltaGdpWmma"
    if activation not in GDP_ACTIVATIONS:
        raise ValueError("gated dual projection gates with sigmoid or swish, not %s"
                         % getattr(activation, "__name__", activation))
    act = GDP_ACTIVATIONS[activation]
    f16 = lambda t: t.astype(jnp.float16)
    if not split:
        return _gdp_call(f16(x), f16(wp), f16(bp), f16(wg), f16(bg),
                         f16(mask), sym=sym, activation=act).astype(x.dtype)
    ci = wp.shape[1] // 2
    halves = []
    for lo, hi in ((0, ci), (ci, 2 * ci)):
        halves.append(_gdp_call(f16(x), f16(wp[:, lo:hi]), f16(bp[lo:hi]),
                                f16(wg[:, lo:hi]), f16(bg[lo:hi]), f16(mask),
                                sym=sym, activation=act).astype(x.dtype))
    if channel_major:
        # XLA would transpose row-major halves for the einsum anyway.
        halves = [h.T for h in halves]
    return halves[0], halves[1]

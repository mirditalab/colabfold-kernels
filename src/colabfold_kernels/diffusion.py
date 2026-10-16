"""The AF3-family diffusion sampler as one persistent kernel via XLA FFI, from colabfold-cuda-kernels or $COLABFOLD_MEGAKERNEL_LIB."""
import ctypes
import logging
import os

import numpy as np

from colabfold_kernels.device import compute_capability, is_rocm

logger = logging.getLogger(__name__)

H = "diffuser/~/diffusion_head/"
TOK = H + "transformer/__layer_stack_no_per_layer/__layer_stack_no_per_layer/transformer"
TOK_SHARED = H + "transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer"
ENC = H + "diffusion_atom_transformer_encoder/"
DEC = H + "diffusion_atom_transformer_decoder/"
NUM_AF3_RESTYPES = 31

# model conventions that reach the per-step computation (alphafold3/model/model_config.py)
MODELS = ("openfold3", "openbind0", "protenix1", "protenix2", "opendde", "boltz2", "intellifold2",
          "rosettafold3", "chai1", "esmfold2", "esmfold2_fast", "esmfold2_lm600m", "esmfold2_lm300m")
OR_KEY_MASK = ("protenix1", "protenix2", "opendde", "boltz2", "intellifold2", "rosettafold3")
CHAINED_KEY_ADALN = ("protenix1", "protenix2", "opendde")
MASK_TRANSITIONS = ("openfold3", "openbind0")
PRE_TRUNK_QUERY = ("boltz2", "rosettafold3", "chai1")
PARALLEL_BLOCKS = ("rosettafold3", "chai1")
ESM_ATOMS = ("esmfold2", "esmfold2_fast", "esmfold2_lm600m", "esmfold2_lm300m")
PADDED_KEYS_EXCLUDED = OR_KEY_MASK + ("chai1",) + ESM_ATOMS

# the kernel's inputs, in its In order
INPUTS = [
    "SIGMA", "SC_BASE", "FOURIER_W", "FOURIER_B", "NOISE_LN", "WN",
    "ST0_S", "ST0_O", "ST0_W1", "ST0_W2", "ST1_S", "ST1_O", "ST1_W1", "ST1_W2",
    "WM1", "BM1", "WM2", "BM2", "WSE",
    "WQKVG", "BQ", "WO", "W1", "W2", "PC", "TMASK", "WTB",
    "WPOS", "SCQ", "QMASK", "KMASK", "Q2K", "TOK2Q", "Q2TA", "ATOM_MASK", "TAM", "WAGGR", "WUPD",
    "ENC_AMODS", "ENC_APL", "ENC_WQG", "ENC_BQ", "ENC_WKV", "ENC_WO", "ENC_W1", "ENC_W2",
    "DEC_AMODS", "DEC_APL", "DEC_WQG", "DEC_BQ", "DEC_WKV", "DEC_WO", "DEC_W1", "DEC_W2",
    "NOISE_LNO", "SE_B", "TB_B", "UPD_B", "SCF", "ROPE",
    "PCS", "PLW", "PLB", "PLWS", "PLX",
    "TQKN", "EQKN", "DQKN", "CHIR_W", "CHIR_C", "CHIR_A", "CHIR_PTR", "CHIR_IDX",
    "MISC",
]
BF16 = {"ST0_W1", "ST0_W2", "ST1_W1", "ST1_W2", "WM1", "WM2", "WSE", "WQKVG", "WO", "W1", "W2", "PC", "WTB",
        "WAGGR", "PLW", "PLX", "ENC_WQG", "ENC_WKV", "ENC_WO", "ENC_W1", "ENC_W2",
        "DEC_WQG", "DEC_WKV", "DEC_WO", "DEC_W1", "DEC_W2"}
# the input contract this module builds: the atom modulations and pair logits in f32, as AF3 runs the atom transformers
CONTRACT = 2
INT32 = {"Q2K", "TOK2Q", "Q2TA", "CHIR_C", "CHIR_PTR", "CHIR_IDX"}

_STATE = {}


def arch():
    """the accumulator layout the kernel was built for"""
    return "amd" if is_rocm() else "nvidia"


def _gpu_cc():
    # ROCm reports its gfx target as the compute capability; the library is built for one, so there is nothing to gate
    return None if is_rocm() else compute_capability()


def _package():
    """(library path, FFI target) from colabfold_cuda_kernels for this GPU, or None"""
    cc = _gpu_cc()
    if cc is None:
        return None
    try:
        import colabfold_cuda_kernels as cck
        return cck.library_path("diffusion", cc), cck.symbol("diffusion", cc)
    except (ImportError, FileNotFoundError, KeyError):
        return None


def available():
    """True once the library is loaded and its FFI target registered"""
    if "ok" in _STATE:
        return _STATE["ok"]
    # an explicit library wins, so a local build (or a ROCm one) can stand in for the package
    path, target = os.environ.get("COLABFOLD_MEGAKERNEL_LIB"), "MkSample"
    if not path:
        path, target = _package() or (None, None)
    ok = False
    if path and os.path.exists(path):
        try:
            import jax
            lib = ctypes.cdll.LoadLibrary(path)
            contract = lib.mk_version() if hasattr(lib, "mk_version") else 1
            if contract != CONTRACT:
                raise OSError(f"input contract {contract}, this colabfold-kernels builds {CONTRACT}")
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, target)),
                                        platform="ROCM" if is_rocm() else "CUDA")
            lib.mk_workspace_bytes.restype = ctypes.c_size_t
            lib.mk_workspace_bytes.argtypes = [ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.c_size_t)]
            _STATE["lib"], _STATE["target"] = lib, target
            ok = True
        except (OSError, AttributeError) as e:
            logger.warning(f"diffusion megakernel {path} did not load: {e}")
    _STATE["ok"] = ok
    return ok


def supports(model, sample_config, gpu_cc=None):
    """Whether the kernel can sample this model with this configuration."""
    if model not in MODELS or getattr(sample_config, "return_trajectory", False):
        return False
    if gpu_cc is not None and gpu_cc < 80:  # bf16 mma.sync needs Ampere
        return False
    return available()


# ---------------------------------------------------------------------------------------- helpers
def _ln(x, scale=None, offset=None, eps=1e-5):
    import jax.numpy as jnp
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    y = (x - m) / jnp.sqrt(v + eps)
    if scale is not None:
        y = y * scale
    if offset is not None:
        y = y + offset
    return y


def _sigmoid(x):
    import jax
    return jax.nn.sigmoid(x)


def _silu(x):
    return x * _sigmoid(x)


def _bf16(x):
    import jax.numpy as jnp
    return jnp.asarray(x, jnp.float32).astype(jnp.bfloat16)


def _acc_index(arch_):
    """(lane, slot) -> (row, column) of a 16 x 16 accumulator tile, as [32, 8] arrays"""
    lane, i = np.meshgrid(np.arange(32), np.arange(8), indexing="ij")
    if arch_ == "amd":
        return 2 * i + (lane >> 4), lane & 15
    return (lane >> 2) + 8 * ((i >> 1) & 1), 8 * (i >> 2) + 2 * (lane & 3) + (i & 1)


def _swizzle_tiles(src, arch_):
    """[..., Q, K] -> [..., Q/16, K/16, 32 lanes, 8]: each lane reads its accumulator's 8 values as one vector"""
    import jax.numpy as jnp
    q, k = src.shape[-2:]
    r, c = _acc_index(arch_)
    idx = ((np.arange(q // 16)[:, None, None, None] * 16 + r) * k + np.arange(k // 16)[None, :, None, None] * 16 + c)
    return jnp.take(src.reshape(src.shape[:-2] + (q * k,)), jnp.asarray(idx), axis=-1)


def _glu_nt(w, w_ab=None):
    """(K, 2n) GLU weights (and boltz2's a_to_b gate) -> rows interleaved in blocks of 32 per gate"""
    import jax.numpy as jnp
    k, two_n = w.shape
    n = two_n // 2
    parts = [w[:, :n].T, w[:, n:].T] + ([w_ab.T] if w_ab is not None else [])
    return jnp.stack([p_.reshape(n // 32, 32, k) for p_ in parts], 1).reshape(len(parts) * n, k)


def schedule(cfg, model):
    """the sampler's noise levels, as diffusion_head.sample builds them"""
    if model == "chai1":  # midpoints of 2N+1
        times = np.linspace(0.0, 1.0, 2 * cfg.steps + 1, dtype=np.float32)[1::2]
    else:
        times = np.linspace(0, 1, cfg.steps + 1, dtype=np.float32)
    smax, smin, rho = cfg.sigma_max, cfg.sigma_min, cfg.rho
    sig = np.asarray(16.0 * (smax ** (1 / rho) + times * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho, np.float32)
    if getattr(cfg, "max_sigma", 0.0):
        sig = np.concatenate([[cfg.max_sigma], sig[sig <= cfg.max_sigma]]).astype(np.float32)
    return sig


class _Params:
    """the diffusion head's parameters, from haiku's {module: {name: array}}, as float32"""

    def __init__(self, params, model):
        import jax.numpy as jnp
        self.p = {f"{m}/{n}": v for m, d in params.items() if m.startswith(H[:-1]) for n, v in d.items()}
        self.model = model
        self.f32 = lambda a: jnp.asarray(a, jnp.float32)

    def has(self, k):
        return k in self.p

    def __getitem__(self, k):
        return self.f32(self.p[k])

    def get(self, k):
        return self[k] if k in self.p else None

    def tok(self, name, b):
        for base in (TOK, TOK_SHARED):
            for key in (base + name, base[:-len("transformer")] + name):
                if key in self.p:
                    a = self.p[key]
                    return self.f32(a.reshape((-1,) + a.shape[2:])[b])
        raise KeyError(name)

    def tok_opt(self, name, b):
        try:
            return self.tok(name, b)
        except KeyError:
            return None

    @property
    def num_blocks(self):
        for base in (TOK, TOK_SHARED):
            key = base + "q_projection/weights"
            if key in self.p:
                return self.p[key].shape[0] * self.p[key].shape[1]

    def pair_logits_weights(self, b):
        shared = H + "transformer/pair_input_layer_norm/scale"
        if shared in self.p:
            w = self[H + "transformer/__layer_stack_with_per_layer/pair_logits_projection/weights"]
            return self[shared][:, None] * w[b // 4][:, b % 4, :]
        return self.tok("pair_input_layer_norm/scale", b)[:, None] * self.tok("pair_logits_projection/weights", b)

    def pair_logits_bias(self, b):
        o = self.tok_opt("pair_input_layer_norm/offset", b)
        return 0.0 if o is None else o @ self.tok("pair_logits_projection/weights", b)

    def atom(self, prefix, name, b):
        for kind in ("with", "no"):
            key = prefix + f"__layer_stack_{kind}_per_layer/" + prefix.split("/")[-2] + name
            if key in self.p:
                return self.f32(self.p[key][b])
        raise KeyError(prefix + name)

    def atom_opt(self, prefix, name, b):
        try:
            return self.atom(prefix, name, b)
        except KeyError:
            return None

    def atom_pair_logits(self, prefix, pair_act):
        """(block, subset, head, query, key) atom pair logits, or None (esmfold2 has none)"""
        import jax.numpy as jnp
        if (prefix + "pair_input_layer_norm/scale" not in self.p
                and prefix + "__layer_stack_no_per_layer/pair_input_layer_norm/scale" not in self.p):
            return None
        pa = _ln(pair_act)
        if prefix + "pair_input_layer_norm/scale" in self.p:
            w0 = self[prefix + "pair_logits_projection/weights"].reshape(16, 12)
            o = self.get(prefix + "pair_input_layer_norm/offset")
            pl = pa @ (self[prefix + "pair_input_layer_norm/scale"][:, None] * w0) + (0.0 if o is None else o @ w0)
            pl = jnp.moveaxis(pl.reshape(pa.shape[:-1] + (3, 4)), -2, 0)
        else:
            stack = prefix + "__layer_stack_no_per_layer/"
            sc, w = self[stack + "pair_input_layer_norm/scale"], self[stack + "pair_logits_projection/weights"]
            pl = jnp.stack([pa @ (sc[b][:, None] * w[b]) for b in range(3)])
        return pl.transpose(0, 1, 4, 2, 3)


def _features_1d(single, tf, p):
    import jax.numpy as jnp
    if p.model in ESM_ATOMS:
        # no single track: its s_inputs alone, restype and profile widened to 33 classes
        n = NUM_AF3_RESTYPES
        zero = jnp.zeros_like(tf[..., :1])
        widen = lambda b: jnp.concatenate([zero, b[..., 21:22], b[..., :21], b[..., 22:n], zero], -1)
        return jnp.concatenate([widen(tf[..., :n]), widen(tf[..., n:2 * n]), tf[..., 2 * n:]], -1)
    f = jnp.concatenate([single, tf], -1)
    if f.shape[-1] == p.p[H + "single_cond_initial_norm/scale"].shape[0]:
        return f
    c = single.shape[-1]
    aatype_end = c + NUM_AF3_RESTYPES
    profile_end = aatype_end + NUM_AF3_RESTYPES
    pad = jnp.zeros_like(f[..., :1])
    return jnp.concatenate([f[..., :aatype_end], pad, f[..., aatype_end:profile_end], pad, f[..., profile_end:]], -1)


def _zero_gate(p, sc, get, pre):
    if p.model in ESM_ATOMS:
        return _silu(sc) @ get(pre + "adaptive_zero_cond/weights")
    return _sigmoid(sc @ get(pre + "adaptive_zero_cond/weights") + get(pre + "adaptive_zero_cond/bias"))


def _offset_bias(p, norm, w):
    import jax.numpy as jnp
    o = p.get(H + norm + "/offset")
    return jnp.zeros(w.shape[-1], jnp.float32) if o is None else o @ w


def _gidx(g):
    import jax.numpy as jnp
    return jnp.where(g.gather_mask, g.gather_idxs, -1).astype(jnp.int32).reshape(-1)


# ---------------------------------------------------------------------------------------- prep
def _prep(params, model, batch, embeddings, pair_cond, atom_cond, cfg, arch_):
    """(inputs in In order, int32 extent record without the seed, flags)"""
    import jax.numpy as jnp
    p = _Params(params, model)
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    chai, esm = model == "chai1", model in ESM_ATOMS
    tm = f32(batch.token_features.mask)
    am_ = f32(batch.predicted_structure_info.atom_mask)
    Nb = tm.shape[0]
    ac = lambda n: f32(atom_cond[n])
    NSb = atom_cond["queries_single_cond"].shape[0]
    Ab = NSb * 32
    aca = batch.atom_cross_att
    d = {}
    d["SIGMA"] = jnp.asarray(schedule(cfg, model))
    single = f32(embeddings["single"])
    tfeat = f32(embeddings["target_feat"])
    x = _ln(_features_1d(single, tfeat, p), p[H + "single_cond_initial_norm/scale"],
            p.get(H + "single_cond_initial_norm/offset")) @ p[H + "single_cond_initial_projection/weights"]
    bias = p.get(H + "single_cond_initial_projection/bias")
    d["SC_BASE"] = x if bias is None else x + bias
    d["FOURIER_W"] = p[H + "fourier_embedding_weight"]
    d["FOURIER_B"] = p[H + "fourier_embedding_bias"]
    d["NOISE_LN"] = p[H + "noise_embedding_initial_norm/scale"]
    lno = p.get(H + "noise_embedding_initial_norm/offset")
    d["NOISE_LNO"] = jnp.zeros(256, jnp.float32) if lno is None else lno
    d["WN"] = p[H + "noise_embedding_initial_projection/weights"]
    for i in range(2):
        pre = H + f"single_transition_{i}"
        d[f"ST{i}_S"] = p[pre + "ffw_layer_norm/scale"]
        d[f"ST{i}_O"] = p[pre + "ffw_layer_norm/offset"]
        d[f"ST{i}_W1"] = _glu_nt(p[pre + "ffw_transition1/weights"])
        d[f"ST{i}_W2"] = p[pre + "ffw_transition2/weights"].T
    # token pair logits: the kernel builds them from the raw pair conditioning and these
    pc = f32(pair_cond)
    mu = pc.mean(-1)
    d["PC"] = pc
    d["PCS"] = jnp.stack([mu, 1.0 / jnp.sqrt(((pc - mu[..., None]) ** 2).mean(-1) + 1e-5)], -1)
    extra = embeddings.get("structural_pair_attn_bias")
    d["PLX"] = f32(extra) if extra is not None else jnp.zeros(1, jnp.float32)
    zeros, ones = jnp.zeros(768, jnp.float32), jnp.ones(768, jnp.float32)
    wm1, bm1, wm2, bm2, wqkvg, bq, wo, w1, w2, plw, plb = ([] for _ in range(11))
    NB = p.num_blocks
    for b in range(NB):
        t = lambda n: p.tok(n, b)
        if chai:  # adaLN on the raw conditioning, scale (s + 1)
            wm1 += [t("single_cond_scale/weights").T, t("single_cond_bias/weights").T,
                    t("ffw_single_cond_scale/weights").T, t("ffw_single_cond_bias/weights").T]
            bm1 += [ones, zeros, ones, zeros]
        else:
            s1, sf = t("single_cond_layer_norm/scale")[:, None], t("ffw_single_cond_layer_norm/scale")[:, None]
            wm1 += [(s1 * t("single_cond_scale/weights")).T, (s1 * t("single_cond_bias/weights")).T,
                    (sf * t("ffw_single_cond_scale/weights")).T, (sf * t("ffw_single_cond_bias/weights")).T]
            bm1 += [t("single_cond_scale/bias"), zeros, t("ffw_single_cond_scale/bias"), zeros]
        gq = p.tok_opt("gating_query/weights", b)  # none (chai1): a zero gate is 1/2, undone by doubling WO
        wm2 += [t("adaptive_zero_cond/weights").T, t("ffw_adaptive_zero_cond/weights").T]
        bm2 += [t("adaptive_zero_cond/bias"), t("ffw_adaptive_zero_cond/bias")]
        wqkvg += [t("q_projection/weights").reshape(768, 768).T, t("k_projection/weights").reshape(768, 768).T,
                  t("v_projection/weights").reshape(768, 768).T, jnp.zeros((768, 768), jnp.float32) if gq is None else gq.T]
        bq.append(t("q_projection/bias").reshape(768))
        wo.append(t("transition2/weights").T * (2.0 if gq is None else 1.0))
        w1.append(_glu_nt(t("ffw_transition1/weights"), p.tok_opt("ffw_a_to_b/weights", b)))
        w2.append(t("ffw_transition2/weights").T)
        plw.append(p.pair_logits_weights(b).T)
        plb.append(jnp.broadcast_to(p.pair_logits_bias(b), (16,)))
    d["WM1"] = jnp.concatenate(wm1)
    d["BM1"] = jnp.concatenate(bm1)
    d["WM2"] = jnp.concatenate(wm2)
    d["BM2"] = jnp.concatenate(bm2)
    d["SE_B"] = _offset_bias(p, "single_cond_embedding_norm", p[H + "single_cond_embedding_projection/weights"])
    d["TB_B"] = _offset_bias(p, "output_norm", p[H + "diffusion_project_token_features_for_broadcast/weights"])
    d["UPD_B"] = jnp.append(_offset_bias(p, "diffusion_atom_features_layer_norm",
                                         p[H + "diffusion_atom_features_to_position_update/weights"]), 0.0)
    if chai:  # no LayerNorm before the embedding projection
        d["WSE"] = p[H + "single_cond_embedding_projection/weights"].T
    else:
        d["WSE"] = (p[H + "single_cond_embedding_norm/scale"][:, None] * p[H + "single_cond_embedding_projection/weights"]).T
    scf = p.get(H + "single_cond_final_norm/scale")
    d["SCF"] = jnp.concatenate([scf, p[H + "single_cond_final_norm/offset"]]) if scf is not None else jnp.zeros(1)
    d["WQKVG"] = jnp.concatenate(wqkvg)
    d["BQ"] = jnp.concatenate(bq)
    d["WO"] = jnp.stack(wo)
    W1 = jnp.stack(w1)
    if arch_ == "amd" and W1.shape[1] == 4608:  # AMD's tiles: each 96-row unit padded to 128 rows
        W1 = W1.reshape(-1, 48, 96, 768)
        W1 = jnp.concatenate([W1, jnp.zeros_like(W1[:, :, :32])], 2).reshape(W1.shape[0], -1, 768)
    d["W1"] = W1
    d["W2"] = jnp.stack(w2)
    PLW = jnp.stack(plw)
    d["PLW"] = PLW
    d["PLB"] = jnp.stack(plb)
    d["PLWS"] = PLW.astype(jnp.bfloat16).astype(jnp.float32).sum(-1)
    d["TMASK"] = tm
    d["WTB"] = (p[H + "output_norm/scale"][:, None] * p[H + "diffusion_project_token_features_for_broadcast/weights"]).T
    d["WPOS"] = p[H + "diffusion_atom_positions_to_features/weights"]
    scq = ac("queries_single_cond").reshape(Ab, 128)
    d["SCQ"] = (ac("q_base").reshape(Ab, 128) if model in PRE_TRUNK_QUERY else scq)
    qmask = ac("queries_mask")
    kmask = ac("keys_mask")
    d["QMASK"] = qmask.reshape(Ab)
    d["KMASK"] = kmask.reshape(-1)
    NK = kmask.shape[-1]
    q2k = _gidx(aca.queries_to_keys)
    if model in PADDED_KEYS_EXCLUDED:
        q2k = jnp.where(d["KMASK"] > 0, q2k, -1)
    d["Q2K"] = q2k
    t2q = _gidx(aca.token_atoms_to_queries)
    q2ta = _gidx(aca.queries_to_token_atoms)
    d["Q2TA"] = q2ta
    d["ATOM_MASK"] = am_.reshape(-1)
    d["TAM"] = ac("token_atoms_mask").reshape(-1)
    d["WAGGR"] = p[H + "diffusion_project_atom_features_for_aggr/weights"].T
    d["WUPD"] = (p[H + "diffusion_atom_features_layer_norm/scale"][:, None]
                 * p[H + "diffusion_atom_features_to_position_update/weights"]).T
    pair_act = ac("pair_act")
    for side, pre in (("ENC", ENC), ("DEC", DEC)):
        sc_q = scq
        if side == "DEC" and p.has(H + "diffusion_post_atom_cond_layer_norm/scale"):  # chai1
            sc_q = _ln(scq, p[H + "diffusion_post_atom_cond_layer_norm/scale"], p[H + "diffusion_post_atom_cond_layer_norm/offset"])
        pw = p.atom_pair_logits(pre, pair_act)
        if pw is None:
            pw = jnp.zeros((3, NSb, 4, 32, NK), jnp.float32)
        if model in OR_KEY_MASK:
            pw = jnp.where(kmask[None, :, None, None, :] > 0, pw, -1e9)
        if chai:  # atoms attend only to real atoms of their own token
            pw = jnp.where(ac("same_token_mask")[None, :, None] > 0, pw, -1e9)
        if esm:  # a +-64 window by rank among real atoms
            pw = jnp.where(ac("swa_mask")[None, :, None] > 0, pw, -1e9)
        am, wqg, bqa, wkv, woa, w1a, w2a = ([] for _ in range(7))
        for b in range(3):
            a = lambda n: p.atom(pre, n, b)
            ao = lambda n: p.atom_opt(pre, n, b)
            gq, wout = ao("gating_query/weights"), ao("transition2/weights")

            def mod(prefix):
                if esm:
                    c = _silu(sc_q)
                    return c @ a(prefix + "single_cond_scale/weights") + 1.0, c @ a(prefix + "single_cond_bias/weights")
                if chai:
                    return sc_q @ a(prefix + "single_cond_scale/weights") + 1.0, sc_q @ a(prefix + "single_cond_bias/weights")
                scn = _ln(sc_q, a(prefix + "single_cond_layer_norm/scale"))
                return (_sigmoid(scn @ a(prefix + "single_cond_scale/weights") + a(prefix + "single_cond_scale/bias")),
                        scn @ a(prefix + "single_cond_bias/weights"))
            qs, qb = mod("q")
            ks, kb = mod("k")
            fs, fb = mod("ffw_")
            za, zf = _zero_gate(p, sc_q, ao, ""), _zero_gate(p, sc_q, ao, "ffw_")
            am.append(jnp.stack([qs, qb, ks, kb, fs, fb, za, zf], 1))
            wqg.append(jnp.concatenate([a("q_projection/weights").reshape(128, 128).T,
                                        jnp.zeros((128, 128), jnp.float32) if gq is None else gq.T]))
            bqa.append(a("q_projection/bias").reshape(128))
            wkv.append(jnp.concatenate([a("k_projection/weights").reshape(128, 128).T,
                                        a("v_projection/weights").reshape(128, 128).T]))
            woa.append((jnp.eye(128, dtype=jnp.float32) if wout is None else wout.T) * (2.0 if gq is None else 1.0))
            w1a.append(_glu_nt(a("ffw_transition1/weights"), ao("ffw_a_to_b/weights")))
            w2a.append(a("ffw_transition2/weights").T)
        d[f"{side}_AMODS"] = jnp.stack(am)
        d[f"{side}_APL"] = _swizzle_tiles(jnp.asarray(pw, jnp.float32), arch_)
        d[f"{side}_WQG"] = jnp.stack(wqg)
        d[f"{side}_BQ"] = jnp.stack(bqa)
        d[f"{side}_WKV"] = jnp.stack(wkv)
        d[f"{side}_WO"] = jnp.stack(woa)
        d[f"{side}_W1"] = jnp.stack(w1a)
        d[f"{side}_W2"] = jnp.stack(w2a)
    # rosettafold3: q/k LayerNorms, [block][q scale, q offset, k scale, k offset][channels]
    qkn = p.tok_opt("query_layer_norm/scale", 0) is not None
    none = jnp.zeros(1, jnp.float32)
    d["TQKN"] = jnp.stack([jnp.stack([p.tok(f"{w}_layer_norm/{v}", b).reshape(-1) for w in ("query", "key")
                                      for v in ("scale", "offset")]) for b in range(NB)]) if qkn else none
    for side, pre in (("E", ENC), ("D", DEC)):
        d[f"{side}QKN"] = jnp.stack([jnp.stack([p.atom(pre, f"{w}_layer_norm/{v}", b).reshape(-1) for w in ("query", "key")
                                                for v in ("scale", "offset")]) for b in range(3)]) if qkn else none
    # rosettafold3: chiral centres, and per atom slot the (centre, role) entries, as a CSR over slots
    chir = batch.chirals if model == "rosettafold3" else None
    chiral = chir is not None and chir.centers.shape[0] > 0
    if chiral:
        cen = jnp.asarray(chir.centers, jnp.int32)
        ang = f32(chir.angles)
        M = cen.shape[0]
        slot = cen.reshape(-1)
        valid = jnp.repeat(ang != 0, 4)
        key = jnp.where(valid, slot, Nb * 24)
        order = jnp.argsort(key, stable=True)
        d["CHIR_IDX"] = jnp.arange(4 * M, dtype=jnp.int32)[order]
        counts = jnp.zeros(Nb * 24 + 1, jnp.int32).at[key].add(1)[:Nb * 24]
        d["CHIR_PTR"] = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts).astype(jnp.int32)])
        d["CHIR_C"] = cen.reshape(-1)
        d["CHIR_A"] = ang
        d["CHIR_W"] = p[H + "diffusion_atom_chiral_to_features/weights"]
    else:
        d["CHIR_IDX"] = jnp.zeros(1, jnp.int32)
        d["CHIR_PTR"] = jnp.zeros(Nb * 24 + 1, jnp.int32)
        d["CHIR_C"] = jnp.zeros(1, jnp.int32)
        d["CHIR_A"] = none
        d["CHIR_W"] = none
    # esmfold2: per atom the rotary (cos, sin) over 16 frequencies; a key's are its atom's
    if esm:
        cos, sin = atom_cond["rope_q"]
        d["ROPE"] = jnp.concatenate([f32(cos).reshape(Ab, 16), f32(sin).reshape(Ab, 16)], -1)
    else:
        d["ROPE"] = none
    flags = (1 * (model in CHAINED_KEY_ADALN) + 2 * (p.tok_opt("ffw_a_to_b/weights", 0) is not None) + 4 * qkn
             + 8 * (model in PARALLEL_BLOCKS) + 16 * chiral + 32 * chai + 64 * (scf is not None)
             + 128 * chai + 256 * esm + 512 * esm)
    d["MISC"] = jnp.asarray([cfg.gamma_0, cfg.gamma_min, cfg.noise_scale, cfg.step_scale, flags,
                             min(80.0 / cfg.steps, 2 ** 0.5 - 1)], jnp.float32)

    # the real extent (a prefix of tokens and of atom subsets) and whether trimming to it is exact
    ar = jnp.arange(Nb)
    n = tm.sum().astype(jnp.int32)
    qany = (qmask > 0).any(1)
    ns = jnp.where(qany.any(), NSb - jnp.argmax(qany[::-1]), 1).astype(jnp.int32)
    sub_lt = jnp.arange(NSb) < ns
    real_atoms = (am_ > 0) & (ar < n)[:, None]
    q2ta2 = q2ta.reshape(Nb, 24)
    keys = q2k.reshape(NSb, NK)
    consider = sub_lt[:, None] & ((kmask > 0) if model in PADDED_KEYS_EXCLUDED else True)
    ok = (jnp.all((tm > 0) == (ar < n))
          & ~jnp.any((am_ > 0) & (ar >= n)[:, None])
          & jnp.all(jnp.where((qmask > 0) & sub_lt[:, None], t2q.reshape(NSb, 32) < n * 24, True))
          & jnp.all(jnp.where(real_atoms, (q2ta2 >= 0) & (q2ta2 < ns * 32), True))
          & jnp.all(jnp.where(consider, keys < ns * 32, True)))
    # padded atoms of the last real subset point past the real tokens; their inputs are masked anyway
    d["TOK2Q"] = jnp.where(t2q >= jnp.where(ok, n, Nb) * 24, -1, t2q)
    out = []
    for k in INPUTS:
        dt = jnp.bfloat16 if k in BF16 else (jnp.int32 if k in INT32 else jnp.float32)
        out.append(jnp.asarray(d[k]).astype(dt))
    return out, jnp.stack([n, ns, ok.astype(jnp.int32)]), flags


def _dump(path, params, model, batch, embeddings):
    """debugging: the sampler's inputs, as capture.py --from-dump reads them (one file per call)"""
    import jax
    arrays = {}
    for k, v in batch.as_data_dict().items():
        if hasattr(v, "shape"):
            arrays[f"example/{k}"] = v
    for k, v in embeddings.items():
        if hasattr(v, "shape"):
            arrays[f"emb/{k}"] = v
    for m, d in params.items():
        if m.startswith(H[:-1]):
            for n, v in d.items():
                arrays[f"params/{m}/{n}"] = v
    names = sorted(arrays)

    def save(*vals):
        i = _STATE.get("dumps", 0)
        _STATE["dumps"] = i + 1
        np.savez(f"{path}.{i}.npz", model=np.array(model),
                 **{n: np.asarray(v, np.float32) if v.dtype.name == "bfloat16" else np.asarray(v) for n, v in zip(names, vals)})
    jax.debug.callback(save, *[arrays[n] for n in names])


def sample(params, model, batch, embeddings, pair_cond, atom_cond, cfg, key):
    """diffusion_head.sample's result, from one kernel launch"""
    import jax
    import jax.numpy as jnp
    if not available():
        raise RuntimeError("the diffusion megakernel is not available: install colabfold-cuda-kernels or set $COLABFOLD_MEGAKERNEL_LIB")
    if os.environ.get("COLABFOLD_MEGAKERNEL_DUMP"):
        _dump(os.environ["COLABFOLD_MEGAKERNEL_DUMP"], params, model, batch, embeddings)
    with jax.default_matmul_precision("highest"):  # not TF32: the tables are rounded to bf16 once, here
        inputs, ext, flags = _prep(params, model, batch, embeddings, pair_cond, atom_cond, cfg, arch())
    seed = jax.lax.bitcast_convert_type(jax.random.bits(key, (2,), jnp.uint32), jnp.int32)
    ext = jnp.concatenate([ext, seed])
    nb = batch.token_features.mask.shape[0]
    S = cfg.num_samples
    nsteps = len(schedule(cfg, model)) - 1
    # the kernel's workspace is XLA's: its size depends only on the bucket's shapes
    sizes = (ctypes.c_size_t * len(inputs))(*[int(np.prod(a.shape)) * a.dtype.itemsize for a in inputs])
    nsb = inputs[INPUTS.index("QMASK")].shape[0] // 32
    ws = _STATE["lib"].mk_workspace_bytes(S, nb, nsb, nsteps, flags, sizes)
    out = (jax.ShapeDtypeStruct((S, nb, 24, 3), jnp.float32), jax.ShapeDtypeStruct((ws,), jnp.uint8))
    pos, _ = jax.ffi.ffi_call(_STATE["target"], out, vmap_method="sequential")(
        ext, *inputs, samples=np.int32(S), nsteps=np.int32(nsteps), flags=np.int32(flags))
    mask = batch.predicted_structure_info.atom_mask
    return {"atom_positions": pos, "mask": jnp.tile(mask[None], (S, 1, 1))}


def override(model, cfg):
    """alphafold3's SAMPLER_OVERRIDE: this kernel's sampler for the model, or None to decline"""
    if not supports(model, cfg, _gpu_cc()):
        return None
    return lambda params, batch, embeddings, pair_cond, atom_cond, key: sample(
        params, model, batch, embeddings, pair_cond, atom_cond, cfg, key)


def install():
    """Route alphafold3's diffusion sampling through the kernel wherever it supports the model."""
    if not available():
        return False
    from alphafold3.model.network import diffusion_head
    if not hasattr(diffusion_head, "SAMPLER_OVERRIDE"):
        logger.warning("this alphafold3 has no SAMPLER_OVERRIDE; the diffusion megakernel stays off")
        return False
    diffusion_head.SAMPLER_OVERRIDE = override
    return True

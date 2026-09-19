#!/usr/bin/env python3
"""k2_engine.py — K2-Horizon（k2-horizon 架构，MoVA）的 dengine 驱动 v1。

架构（官方 modeling_k2_horizon.py 为语义基准，k2_numpy 为对账参照）：
  · 48 层，H=2560，32/8 头 GQA，head_dim=128，rope theta 1e7（rotate_half 约定）
  · 层 0-2 稠密；层 3-47：MoVA 注意力（V 投影 64 选 4，专家输出 silu，
    softplus(beta=ln2) 输出门）+ MoE（100 选 8 + 1 共享，bias 仅参与选择，norm_topk）
  · 所有 norm = 2 组分组 RMSNorm（T5 式）
v1 策略：m5 gemv 内核 + numpy 编排（正确性优先）。权重零拷贝 mmap。
"""
import os
import ctypes as ct
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf")
MAXT = int(os.environ.get("MAXT", "1024"))

print("[k2] 加载 GGUF...", flush=True)
import gguf_fast  # noqa: E402
R = gguf_fast.FastGGUF(MODEL)
T = {t.name: t for t in R.tensors}

# ★ OMP 环境必须在首次 dlopen 前设好
try:
    import mcfg as _mcfg  # noqa: E402
    import autotune as _at  # noqa: E402
    _CFG = _mcfg.load_cfg(R, T)
    _plan = _at.plan(_CFG, sum(t.n_bytes for t in R.tensors))
except Exception:
    _plan = {"OMP_NUM_THREADS": 4, "OMP_WAIT_POLICY": "active", "GOMP_SPINCOUNT": "5000"}
for _k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
    os.environ.setdefault(_k, str(_plan[_k]))
print(f"[autotune] OMP={os.environ['OMP_NUM_THREADS']} WAIT={os.environ['OMP_WAIT_POLICY']}", flush=True)

ARCH = R.fields["general.architecture"].value.decode() if isinstance(
    R.fields["general.architecture"].value, bytes) else R.fields["general.architecture"].value


def kv(name, default=None):
    f = R.fields.get(f"{ARCH}.{name}")
    return f.value if f is not None else default


def fint(name, default):
    v = kv(name)
    return int(v) if v is not None else default


H = fint("embedding_length", 2560)
NL = fint("block_count", 48)
NH = fint("attention.head_count", 32)
NKV = fint("attention.head_count_kv", 8)
HD = fint("attention.key_length", 128)
EPS = float(kv("attention.layer_norm_rms_epsilon") or 1e-6)
THETA = float(kv("rope.freq_base") or 1e7)
NEXP = fint("expert_count", 100)
NUSED = fint("expert_used_count", 8)
MEXP = fint("mova.expert_count", 64)
MUSED = fint("mova.expert_used_count", 4)
MOE_INTER = fint("expert_feed_forward_length", 768)
VOUT = NKV * HD
NGROUP = 2
DENSE_LAYERS = [0, 1, 2]

CODE = {"F32": 9, "F16": 7, "Q8_0": 0, "Q5_0": 8, "Q4_K": 5,
        "Q5_K": 3, "Q6_K": 4, "IQ4_NL": 1, "IQ3_S": 2, "IQ4_XS": 6}

# ── m5 内核直连 ──
import m5sel  # noqa: E402
BASE_M5 = os.path.join(BASE, "m5")
CODE_LIB = {}
for p, codes in (("m5_kern6.so", (0, 1, 2)), ("m5_kern9.so", (3,)), ("m5_kern8.so", (4,)),
                 ("m5_kern7.so", (5, 6)), ("m5_kernF.so", (7,)), ("m5_kern10.so", (8,))):
    try:
        lib = ct.CDLL(os.path.join(BASE_M5, p))
        lib.m5_gemv.restype = ct.c_int
        lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                                ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
        for c in codes:
            CODE_LIB[c] = lib
    except OSError:
        pass


def pf(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_float))


def p8(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_uint8))


def tcode(name):
    tt = T[name].tensor_type
    return CODE[tt if isinstance(tt, str) else str(tt).split(".")[-1]]


def wf32(name):
    import gguf.quants as Q
    t = T[name]
    return np.asarray(Q.dequantize(t.data, t.tensor_type), np.float32).reshape(-1)


def wview(name):
    """量化权重的零拷贝 uint8 视图。"""
    return np.frombuffer(T[name].data, np.uint8)


STATES = []

# ── 专家预取（SS-MoE 思路：按上次路由预读本 token 的专家权重，内核异步 I/O 与计算重叠）──
# ★ 默认关：A/B 实测（140-170ms/tok 区间）无收益——内核默认 readahead 已在做类似的事，
#   而我们的专家访问是随机块（每 token 路由不同），fadvise WILLNEED 的成功率不高。
#   真正的 I/O 杠杆是「模型能驻留内存」（更小量化）或「显式专家 LRU 缓存」（SS-MoE 的 ExpertCache）。
_PREFETCH = os.environ.get("K2_PREFETCH", "0") == "1"
try:
    _FD = R._fd.fileno() if hasattr(R, "_fd") and hasattr(R._fd, "fileno") else None
except Exception:
    _FD = None
_LAST_SEL = {}   # li -> [sel_v, sel_moe]


def _advise(name, off, length):
    if _FD is None:
        return
    t = T.get(name)
    if t is None:
        return
    try:
        os.posix_fadvise(_FD, t.data_offset + off, length, os.POSIX_FADV_WILLNEED)
    except OSError:
        pass


def prefetch_experts(li, sel_v, sel_moe):
    """按上次 token 的路由预取本 token 可能用到的专家（86% GoodPrefetch 见 SS-MoE）。"""
    if not _PREFETCH:
        return
    L = LAYERS[li]
    if L.sparse:
        for e in sel_v:
            _advise(f"blk.{li}.attn_v_exps.weight", int(e) * L.ve_per, L.ve_per)
    for e in sel_moe:
        _advise(f"blk.{li}.ffn_gate_exps.weight", int(e) * L.ex_per, L.ex_per)
        _advise(f"blk.{li}.ffn_up_exps.weight", int(e) * L.ux_per, L.ux_per)
        _advise(f"blk.{li}.ffn_down_exps.weight", int(e) * L.dx_per, L.dx_per)


class Layer:
    def __init__(self, li):
        self.li = li
        self.sparse = li not in DENSE_LAYERS
        p = f"blk.{li}."
        self.norm_a = wf32(p + "attn_norm.weight")
        self.norm_f = wf32(p + "ffn_norm.weight")
        self.qb, self.qc = wview(p + "attn_q.weight"), tcode(p + "attn_q.weight")
        self.kb, self.kc = wview(p + "attn_k.weight"), tcode(p + "attn_k.weight")
        self.ob, self.oc = wview(p + "attn_output.weight"), tcode(p + "attn_output.weight")
        self.gb, self.gc = wview(p + "attn_gate.weight"), tcode(p + "attn_gate.weight")
        if self.sparse:
            self.vgb = wview(p + "attn_v_gate.weight")
            self.vgc = tcode(p + "attn_v_gate.weight")
            self.vgate_b = wf32(p + "attn_v_gate.bias")
            self.veb = wview(p + "attn_v_exps.weight")
            self.vec = tcode(p + "attn_v_exps.weight")
            self.ve_per = self.veb.nbytes // MEXP
            self.gate_inp = wf32(p + "ffn_gate_inp.weight").reshape(NEXP, H)
            self.probs_b = wf32(p + "exp_probs_b.bias")
            self.exb = wview(p + "ffn_gate_exps.weight")
            self.exc = tcode(p + "ffn_gate_exps.weight")
            self.ex_per = self.exb.nbytes // NEXP
            self.uxb = wview(p + "ffn_up_exps.weight")
            self.uxc = tcode(p + "ffn_up_exps.weight")
            self.ux_per = self.uxb.nbytes // NEXP
            self.dxb = wview(p + "ffn_down_exps.weight")
            self.dxc = tcode(p + "ffn_down_exps.weight")
            self.dx_per = self.dxb.nbytes // NEXP
            self.sgb, self.sgc = wview(p + "ffn_gate_shexp.weight"), tcode(p + "ffn_gate_shexp.weight")
            self.sub, self.suc = wview(p + "ffn_up_shexp.weight"), tcode(p + "ffn_up_shexp.weight")
            self.sdb, self.sdc = wview(p + "ffn_down_shexp.weight"), tcode(p + "ffn_down_shexp.weight")
        else:
            self.vb, self.vc = wview(p + "attn_v.weight"), tcode(p + "attn_v.weight")
            self.g1b, self.g1c = wview(p + "ffn_gate.weight"), tcode(p + "ffn_gate.weight")
            self.u1b, self.u1c = wview(p + "ffn_up.weight"), tcode(p + "ffn_up.weight")
            self.d1b, self.d1c = wview(p + "ffn_down.weight"), tcode(p + "ffn_down.weight")
        self.K = np.zeros((MAXT, NKV, HD), np.float32)
        self.V = np.zeros((MAXT, NKV, HD), np.float32)
        STATES.extend([self.K, self.V])


print("[k2] 装载层权重...", flush=True)
LOGITS = np.zeros(1, np.float32)
LAYERS = [Layer(li) for li in range(NL)]
ONORM = wf32("output_norm.weight")


def grouped_rms1(x, wv):
    g = x.reshape(NGROUP, -1)
    var = (g * g).mean(-1, keepdims=True)
    return (g / np.sqrt(var + np.float32(EPS))).reshape(-1) * wv


def softplus_ln2(x):
    lb = np.float32(np.log(2.0))   # ★ np.float64 标量会把整条 f32 链升 f64（C 侧按 f32 读=NaN）
    return np.logaddexp(np.float32(0.0), lb * x) / lb


def softmax(x):
    """全数组归一化（仅用于 1D：路由分数）。"""
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def softmax_last(x):
    """沿最后一维归一化（attention 权重必须用这个：[NH, 1, kv_n] 逐头对 kv 位置归一）。"""
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


_INV = (1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))).astype(np.float32)


def rope1(vec, pos):
    """rotate_half rope，vec [..., hd]（可带批量头维）。"""
    ang = pos * _INV
    cos = np.cos(ang)
    sin = np.sin(ang)
    h = vec.shape[-1] // 2
    a, b = vec[..., :h], vec[..., h:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


def gemv1(code, wview_, n_out, n_in, x):
    lib = CODE_LIB.get(code)
    if lib is None:
        raise RuntimeError(f"code {code} 无内核")
    y = np.empty(n_out, np.float32)
    rc = lib.m5_gemv(code, pf(x), p8(wview_), n_out, n_in, pf(y))
    if rc != 0:
        raise RuntimeError(f"m5_gemv rc={rc}")
    return y


def silu(x):
    return x / (1.0 + np.exp(-x))


def attn_ffn_common(x, L, pos):
    """注意力（MoVA 或稠密）+ 残差。返回层注意力输出（未加残差）。"""
    h = grouped_rms1(x, L.norm_a)
    q = gemv1(L.qc, L.qb, NH * HD, H, h).reshape(NH, HD)
    k = rope1(gemv1(L.kc, L.kb, NKV * HD, H, h).reshape(NKV, HD), pos)
    L.K[pos] = k
    if L.sparse:
        # MoVA：路由 → top-MUSED 专家 silu 加权（v_gate 量化 → gemv）
        vlogits = gemv1(L.vgc, L.vgb, MEXP, H, h)
        scores = softmax(vlogits) + L.vgate_b
        sel = np.argsort(-scores, kind="stable")[:MUSED]
        _LAST_SEL.setdefault(L.li, [None, None])[0] = sel
        wts = softmax(vlogits)[sel]
        wts = wts / wts.sum()
        v = np.zeros(VOUT, np.float32)
        for k_i, e in enumerate(sel):
            ve = gemv1(L.vec, L.veb[e * L.ve_per:], VOUT, H, h)
            v += silu(ve) * wts[k_i]
        L.V[pos] = v.reshape(NKV, HD)
    else:
        v = gemv1(L.vc, L.vb, VOUT, H, h)
        L.V[pos] = v.reshape(NKV, HD)
    # 注意力（GQA，因果，T=1 单步）
    q = rope1(q.reshape(NH, HD), pos)
    kv_n = pos + 1
    Kc = np.repeat(L.K[:kv_n].transpose(1, 0, 2), NH // NKV, axis=0)   # [NH, kv_n, HD]
    Vc = np.repeat(L.V[:kv_n].transpose(1, 0, 2), NH // NKV, axis=0)
    att = softmax_last((q[:, None, :] @ Kc.transpose(0, 2, 1)) * np.float32(HD ** -0.5))  # [NH, 1, kv_n]
    out = (att @ Vc)[:, 0, :].reshape(NH * HD)
    gate = softplus_ln2(gemv1(L.gc, L.gb, NH * HD, H, h))
    out = out * gate
    o = gemv1(L.oc, L.ob, H, NH * HD, out)
    if os.environ.get("K2DBG"):
        e = lambda a: ("OK" if np.isfinite(np.asarray(a)).all() else "NAN") + "/" + str(np.asarray(a).dtype)
        print(f"[dbg] h={e(h)} q={e(q)} k={e(k)} v={e(v)} att={e(att)} out={e(out)} "
              f"gate={e(gate)} o={e(o)} | out_max={np.abs(out).max():.3f} h_max={np.abs(h).max():.3f}",
              flush=True)
    return o


def moe_ffn(h2, L):
    logits = L.gate_inp @ h2
    scores = softmax(logits)
    sel = np.argsort(-(scores + L.probs_b), kind="stable")[:NUSED]
    _LAST_SEL.setdefault(L.li, [None, None])[1] = sel
    rw = scores[sel]
    rw = rw / rw.sum()
    out = np.zeros(H, np.float32)
    for k, e in enumerate(sel):
        g = gemv1(L.exc, L.exb[e * L.ex_per:], MOE_INTER, H, h2)
        u = gemv1(L.uxc, L.uxb[e * L.ux_per:], MOE_INTER, H, h2)
        d = gemv1(L.dxc, L.dxb[e * L.dx_per:], H, MOE_INTER, silu(g) * u)
        out += d * rw[k]
    g = gemv1(L.sgc, L.sgb, MOE_INTER, H, h2)
    u = gemv1(L.suc, L.sub, MOE_INTER, H, h2)
    d = gemv1(L.sdc, L.sdb, H, MOE_INTER, silu(g) * u)
    return out + d


def dense_ffn(h2, L):
    g = gemv1(L.g1c, L.g1b, 6144, H, h2)
    u = gemv1(L.u1c, L.u1b, 6144, H, h2)
    return gemv1(L.d1c, L.d1b, H, 6144, silu(g) * u)


import gguf.quants as Q  # noqa: E402


LAYER_SNAP = {}   # li → 该层输出（仅 K2DBG 模式）

def forward(tid, pos, layer_hook=None):
    t = T["token_embd.weight"]
    x = np.asarray(Q.dequantize(t.data[tid], t.tensor_type), np.float32)
    for li, L in enumerate(LAYERS):
        _ls = _LAST_SEL.get(li)
        if _ls is not None:
            prefetch_experts(li, _ls[0] if _ls[0] is not None else [], _ls[1] if _ls[1] is not None else [])
        attn = attn_ffn_common(x, L, pos)
        x = x + attn
        h2 = grouped_rms1(x, L.norm_f)
        if L.sparse:
            x = x + moe_ffn(h2, L)
        else:
            x = x + dense_ffn(h2, L)
        if layer_hook is not None:
            layer_hook(li, x.copy())
    xf = grouped_rms1(x, ONORM)
    global LOGITS
    LOGITS = gemv1(tcode("output.weight"), wview("output.weight"), VOCAB, H, xf)
    return LOGITS


VOCAB = T["output.weight"].shape[1] if len(T["output.weight"].shape) > 1 else int(
    kv("vocab_size") or 0)
# gguf-py shape 反转：output.weight 形状 [H, vocab] ⇒ vocab = shape[1]
if VOCAB <= 0:
    VOCAB = fint("vocab_size", 250624)


def logits_of_x():
    return LOGITS


def reset():
    for b in STATES:
        b[...] = 0

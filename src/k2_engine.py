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
import time
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
# 真实 36B config: router_score_func="sigmoid"、router_scaling_factor=2.5
# （GGUF 元数据 expert_gating_func=2 即 sigmoid、expert_weights_scale=2.5）
GATING_FUNC = fint("expert_gating_func", 2)          # 1=softmax 2=sigmoid（llama.cpp 枚举）
R_SCALING = float(kv("expert_weights_scale") or 2.5)


def sigmoid1(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))

CODE = {"F32": 9, "F16": 7, "Q8_0": 0, "Q5_0": 8, "Q4_K": 5,
        "Q5_K": 3, "Q6_K": 4, "IQ4_NL": 1, "IQ3_S": 2, "IQ4_XS": 6, "IQ2_S": 13}

# ── m5 内核直连 ──
import m5sel  # noqa: E402
BASE_M5 = os.path.join(BASE, "m5")
CODE_LIB = {}
for p, codes in (("m5_kern6.so", (0, 1, 2)), ("m5_kern9.so", (3,)), ("m5_kern8.so", (4,)),
                 ("m5_kern7.so", (5, 6)), ("m5_kernF.so", (7,)), ("m5_kern10.so", (8,)),
                 ("m5_kern13.so", (13,))):
    try:
        lib = ct.CDLL(os.path.join(BASE_M5, p))
        lib.m5_gemv.restype = ct.c_int
        lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                                ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
        for c in codes:
            CODE_LIB[c] = lib
    except OSError:
        pass


# ── K2 批量专家内核（gate/up 共享 x；down 逐专家 x）──
BATCH_ON = os.environ.get("K2_BATCH", "0") == "1"
LIB_B = None
try:
    LIB_B = ct.CDLL(os.path.join(BASE_M5, "m5_batch8_k.so"))
    for _fn in ("m5_gemvn_q4k", "m5_gemvn_q6k"):
        _f = getattr(LIB_B, _fn)
        _f.restype = ct.c_int
        _f.argtypes = [ct.POINTER(ct.c_float), ct.c_long, ct.POINTER(ct.c_uint8),
                       ct.POINTER(ct.c_int), ct.c_long, ct.c_int, ct.c_int, ct.c_int,
                       ct.POINTER(ct.c_float)]
except OSError:
    LIB_B = None
BATCH_ON = BATCH_ON and LIB_B is not None

# ── 段融合 gemv（仅 IQ2_S=13）：同输入多矩阵一次 OMP 区 ──
K2_FUSE = os.environ.get("K2_FUSE", "0") == "1"
LIB13 = CODE_LIB.get(13)
class _Seg(ct.Structure):
    _fields_ = [("W", ct.c_void_p), ("n_out", ct.c_int), ("y_off", ct.c_int)]
if LIB13 is not None:
    try:
        LIB13.m5_gemv13_segs.restype = ct.c_int
        LIB13.m5_gemv13_segs.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(_Seg), ct.c_int,
                                         ct.c_int, ct.POINTER(ct.c_float)]
        _SEG_ARR = lambda n: (_Seg * n)()
    except AttributeError:
        K2_FUSE = False
else:
    K2_FUSE = False
K2_FUSE = K2_FUSE and LIB13 is not None

# ── F1c：iGPU 驻留（K2_VK=1）。IQ2_S 张量上传 GPU，六个相位走 libvkrun ──
K2_VK = os.environ.get("K2_VK", "0") == "1"
VK = None
VK_SKIP = set(x for x in os.environ.get("K2_VK_SKIP", "").split(",") if x)


def _seg_ptr(wb):
    return wb.ctypes.data_as(ct.c_void_p)


def gemv_fused4(h, qb, gb, kb, vgb, n_q, n_g, n_k, n_v):
    """q/gate/k/v_router 共享输入 h，一次调用。返回 (q, gate_in, k, vlogits)。"""
    y = np.empty(n_q + n_g + n_k + n_v, np.float32)
    arr = (_Seg * 4)()
    arr[0].W, arr[0].n_out, arr[0].y_off = _seg_ptr(qb), n_q, 0
    arr[1].W, arr[1].n_out, arr[1].y_off = _seg_ptr(gb), n_g, n_q
    arr[2].W, arr[2].n_out, arr[2].y_off = _seg_ptr(kb), n_k, n_q + n_g
    arr[3].W, arr[3].n_out, arr[3].y_off = _seg_ptr(vgb), n_v, n_q + n_g + n_k
    rc = LIB13.m5_gemv13_segs(pf(h), arr, 4, H, pf(y))
    if rc != 0:
        raise RuntimeError(f"segs rc={rc}")
    return y[:n_q], y[n_q:n_q+n_g], y[n_q+n_g:n_q+n_g+n_k], y[n_q+n_g+n_k:]


def gemv_fused_gu(h2, exb, uxb, per, uper, sel, n_inter):
    """MoE 8 专家 gate+up 共享 h2：16 段一次调用。返回 G[8,ni], U[8,ni]。"""
    ns = len(sel)
    y = np.empty(ns * 2 * n_inter, np.float32)
    arr = (_Seg * (ns * 2))()
    for i, e in enumerate(sel):
        arr[2*i].W,   arr[2*i].n_out,   arr[2*i].y_off   = _seg_ptr(exb[e*per:]), n_inter, i*2*n_inter
        arr[2*i+1].W, arr[2*i+1].n_out, arr[2*i+1].y_off = _seg_ptr(uxb[e*uper:]), n_inter, i*2*n_inter + n_inter
    rc = LIB13.m5_gemv13_segs(pf(np.ascontiguousarray(h2)), arr, ns*2, H, pf(y))
    if rc != 0:
        raise RuntimeError(f"segs gu rc={rc}")
    yv = y.reshape(ns, 2, n_inter)
    return yv[:, 0, :], yv[:, 1, :]


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

    def attach_vk(self, VK):
        """K2_VK: 登记 GPU 常驻矩阵 ((G, n_out) 或 G)；None = 该张量留 CPU。"""
        self.qg = self.kg = self.gg = self.vgg = None
        self.veg = self.exg = self.uxg = self.dxg = None
        self.sgg = self.sug = self.sdg = None
        self.og = None
        if not self.sparse:
            return
        p = f"blk.{self.li}."

        def g(name, n_out=None):
            gg = VK.reg.get(name)
            return (gg, n_out) if (gg is not None and n_out is not None) else gg

        self.og = g(p + "attn_output.weight")
        self.qg = g(p + "attn_q.weight", NH * HD)
        self.kg = g(p + "attn_k.weight", NKV * HD)
        self.gg = g(p + "attn_gate.weight", NH * HD)
        self.vgg = g(p + "attn_v_gate.weight", MEXP)
        self.veg = g(p + "attn_v_exps.weight")
        self.exg = g(p + "ffn_gate_exps.weight")
        self.uxg = g(p + "ffn_up_exps.weight")
        self.dxg = g(p + "ffn_down_exps.weight")
        self.sgg = g(p + "ffn_gate_shexp.weight")
        self.sug = g(p + "ffn_up_shexp.weight")
        self.sdg = g(p + "ffn_down_shexp.weight")


print("[k2] 装载层权重...", flush=True)
LOGITS = np.zeros(1, np.float32)
LAYERS = [Layer(li) for li in range(NL)]
ONORM = wf32("output_norm.weight")

if K2_VK:
    try:
        import k2_vk as _k2vk

        def _is13(name):
            return tcode(name) == 13

        _wanted = []
        for L in LAYERS:
            if not L.sparse:
                continue
            p = f"blk.{L.li}."
            for s in ("attn_q.weight", "attn_k.weight", "attn_gate.weight",
                      "attn_v_gate.weight", "attn_v_exps.weight",
                      "ffn_gate_exps.weight", "ffn_up_exps.weight",
                      "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                      "ffn_down_shexp.weight"):
                if _is13(p + s):
                    _wanted.append(p + s)
            if _is13(p + "ffn_down_exps.weight"):
                _wanted.append(p + "ffn_down_exps.weight")
            if tcode(p + "attn_output.weight") == 2:      # IQ3_S o-proj 上 GPU（pipe1）
                _wanted.append(p + "attn_output.weight")
        # F2：shexp(g/u/d) 并入 gate_up/down 同一 submit（多 dispatch 免费）；
        # 仅单层混量化（IQ3_S）的缺角张量走 CPU 分支。
        VK = _k2vk.VKCtx(T, _wanted).arm(upload=not os.environ.get("K2_VK_NOUPLOAD"))
        for L in LAYERS:
            L.attach_vk(VK)
        if not os.environ.get("K2_VK_NOUPLOAD"):
            VK.drop_cache()
        print("[vk] 引擎已切 iGPU 驻留", flush=True)
    except Exception as e:
        print(f"[vk] 初始化失败，回落 CPU：{e!r}", flush=True)
        VK = None


# ── 闸门强制路由：GPU 与 CPU 走同一组专家，消除 fp 序噪声的 argsort 翻转级联 ──
FORCE = None
FORCE_POS = -1
if os.environ.get("K2_FORCE_SEL"):
    import json as _json
    _f = _json.load(open(os.environ["K2_FORCE_SEL"]))
    FORCE = _f["last_sels"]
    FORCE_POS = int(_f.get("pos", os.environ.get("K2_FORCE_POS", -1)))
    print(f"[k2] 强制路由生效 pos={FORCE_POS}（闸门模式）", flush=True)


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


_PROF = {}
_PROF_ON = os.environ.get("K2_PROF") == "1"


def _tick(key, t0):
    if _PROF_ON:
        _PROF[key] = _PROF.get(key, 0.0) + (time.perf_counter() - t0)
        return time.perf_counter()
    return t0


def attn_ffn_common(x, L, pos):
    """注意力（MoVA 或稠密）+ 残差。返回层注意力输出（未加残差）。"""
    tp = time.perf_counter if _PROF_ON else None
    t0 = tp() if tp else 0
    h = grouped_rms1(x, L.norm_a)
    t0 = _tick("rms", t0)
    if VK is not None and "fused4" not in VK_SKIP and L.sparse and L.qg and L.kg and L.gg and L.vgg:
        qf, gate_in, kf, vlogits = VK.fused4(h, L.qg, L.gg, L.kg, L.vgg)
    elif K2_FUSE and L.qc == 13 and L.gc == 13 and L.kc == 13 and (not L.sparse or L.vgc == 13):
        qf, gate_in, kf, vlogits = gemv_fused4(h, L.qb, L.gb, L.kb, L.vgb if L.sparse else L.qb,
                                               NH * HD, NH * HD, NKV * HD, MEXP if L.sparse else 1)
        if not L.sparse:
            vlogits = None
    else:
        qf = gemv1(L.qc, L.qb, NH * HD, H, h)
        kf = gemv1(L.kc, L.kb, NKV * HD, H, h)
        gate_in = None
        vlogits = gemv1(L.vgc, L.vgb, MEXP, H, h) if L.sparse else None
    q = qf.reshape(NH, HD)
    k = rope1(kf.reshape(NKV, HD), pos)
    L.K[pos] = k
    t0 = _tick("qkv", t0)
    if L.sparse:
        # MoVA：sigmoid 路由（bias 只参与选择）→ top-MUSED 专家 silu 加权，权重归一 ×scaling
        pass
        sc = sigmoid1(vlogits) if GATING_FUNC == 2 else softmax(vlogits)
        if FORCE is not None and pos == FORCE_POS and FORCE[L.li][0]:
            sel = np.array(FORCE[L.li][0])
        else:
            sel = np.argsort(-(sc + L.vgate_b), kind="stable")[:MUSED]
        _LAST_SEL.setdefault(L.li, [None, None])[0] = sel
        wts = sc[sel]
        wts = wts / wts.sum() * R_SCALING
        if VK is not None and "mova" not in VK_SKIP and L.veg is not None:
            V4 = VK.mova4(h, L.veg, sel, L.ve_per, VOUT)
            v = (silu(V4) * wts[:, None]).sum(0)
        elif BATCH_ON:
            sel_a = np.ascontiguousarray(sel, np.int32)
            V4 = np.empty(MUSED * VOUT, np.float32)
            rc = LIB_B.m5_gemvn_q4k(pf(h), 0, p8(L.veb), sel_a.ctypes.data_as(ct.POINTER(ct.c_int)),
                                    L.ve_per, VOUT, H, MUSED, pf(V4))
            assert rc == 0, f"batch q4k rc={rc}"
            v = (silu(V4.reshape(MUSED, VOUT)) * wts[:, None]).sum(0)
        else:
            v = np.zeros(VOUT, np.float32)
            for k_i, e in enumerate(sel):
                ve = gemv1(L.vec, L.veb[e * L.ve_per:], VOUT, H, h)
                v += silu(ve) * wts[k_i]
        L.V[pos] = v.reshape(NKV, HD)
    else:
        v = gemv1(L.vc, L.vb, VOUT, H, h)
        L.V[pos] = v.reshape(NKV, HD)
    t0 = _tick("mova_v", t0)
    # 注意力（GQA，因果，T=1 单步）
    q = rope1(q.reshape(NH, HD), pos)
    kv_n = pos + 1
    Kc = np.repeat(L.K[:kv_n].transpose(1, 0, 2), NH // NKV, axis=0)   # [NH, kv_n, HD]
    Vc = np.repeat(L.V[:kv_n].transpose(1, 0, 2), NH // NKV, axis=0)
    att = softmax_last((q[:, None, :] @ Kc.transpose(0, 2, 1)) * np.float32(HD ** -0.5))  # [NH, 1, kv_n]
    out = (att @ Vc)[:, 0, :].reshape(NH * HD)
    t0 = _tick("attention", t0)
    gate = softplus_ln2(gate_in if gate_in is not None else gemv1(L.gc, L.gb, NH * HD, H, h))
    out = out * gate
    if VK is not None and "oproj" not in VK_SKIP and L.og is not None:
        o = VK.oproj(out, L.og, H, NH * HD)
    else:
        o = gemv1(L.oc, L.ob, H, NH * HD, out)
    t0 = _tick("gate_o", t0)
    if os.environ.get("K2DBG"):
        e = lambda a: ("OK" if np.isfinite(np.asarray(a)).all() else "NAN") + "/" + str(np.asarray(a).dtype)
        print(f"[dbg] h={e(h)} q={e(q)} k={e(k)} v={e(v)} att={e(att)} out={e(out)} "
              f"gate={e(gate)} o={e(o)} | out_max={np.abs(out).max():.3f} h_max={np.abs(h).max():.3f}",
              flush=True)
    return o


def moe_ffn(h2, L, pos=-1):
    t0 = time.perf_counter() if _PROF_ON else 0
    logits = L.gate_inp @ h2
    sc = sigmoid1(logits) if GATING_FUNC == 2 else softmax(logits)
    if FORCE is not None and pos == FORCE_POS and FORCE[L.li][1]:
        sel = np.array(FORCE[L.li][1])
    else:
        sel = np.argsort(-(sc + L.probs_b), kind="stable")[:NUSED]
    _LAST_SEL.setdefault(L.li, [None, None])[1] = sel
    rw = sc[sel]
    rw = rw / rw.sum() * R_SCALING
    t0 = _tick("moe_route", t0)
    shared_done = False
    if VK is not None and "moe" not in VK_SKIP and L.exg and L.uxg and L.sgg and L.sug:
        # F2：gate_up(16) + shared sg/su(2) 一次 submit
        G8, U8, sg_, su_ = VK.gate_up9(h2, L.exg, L.uxg, sel, L.ex_per, L.ux_per,
                                       L.sgg, L.sug, MOE_INTER)
        gu8 = silu(G8) * U8
        su_in = silu(sg_) * su_
        if L.dxg is not None and L.sdg is not None:
            D8, d9 = VK.down9(gu8, su_in, L.dxg, sel, L.dx_per, L.sdg, H, MOE_INTER)
            out = (D8 * rw[:, None]).sum(0)   # d9 由末尾 return out + d 统一加（勿双计）
            shared_done = True
        elif L.dxg is not None:
            D8 = VK.down(gu8, L.dxg, sel, L.dx_per, H, MOE_INTER)
            out = (D8 * rw[:, None]).sum(0)
            d9 = gemv1(L.sdc, L.sdb, H, MOE_INTER, su_in)
            shared_done = True
        else:
            out = np.zeros(H, np.float32)
            for k, e in enumerate(sel):
                out += gemv1(L.dxc, L.dxb[e * L.dx_per:], H, MOE_INTER, gu8[k]) * rw[k]
            if L.sdg is not None:
                d9 = VK.shared_d(su_in, L.sdg, H, MOE_INTER)
                shared_done = True
            else:
                d9 = gemv1(L.sdc, L.sdb, H, MOE_INTER, su_in)
                shared_done = True
    elif VK is not None and "moe" not in VK_SKIP and L.exg and L.uxg:
        G8, U8 = VK.gate_up(h2, L.exg, L.uxg, sel, L.ex_per, L.ux_per, MOE_INTER)
        gu8 = silu(G8) * U8
        if L.dxg is not None:
            D8 = VK.down(gu8, L.dxg, sel, L.dx_per, H, MOE_INTER)
            out = (D8 * rw[:, None]).sum(0)
        else:
            out = np.zeros(H, np.float32)
            for k, e in enumerate(sel):
                out += gemv1(L.dxc, L.dxb[e * L.dx_per:], H, MOE_INTER, gu8[k]) * rw[k]
    elif BATCH_ON:
        sel_a = np.ascontiguousarray(sel, np.int32)
        pi_ = ct.POINTER(ct.c_int)
        G = np.empty(NUSED * MOE_INTER, np.float32)
        U = np.empty(NUSED * MOE_INTER, np.float32)
        rc = LIB_B.m5_gemvn_q4k(pf(h2), 0, p8(L.exb), sel_a.ctypes.data_as(pi_),
                                L.ex_per, MOE_INTER, H, NUSED, pf(G))
        rc += LIB_B.m5_gemvn_q4k(pf(h2), 0, p8(L.uxb), sel_a.ctypes.data_as(pi_),
                                 L.ux_per, MOE_INTER, H, NUSED, pf(U))
        assert rc == 0, f"batch q4k rc={rc}"
        gu = (silu(G) * U).reshape(NUSED, MOE_INTER)
        D = np.empty(NUSED * H, np.float32)
        rc = LIB_B.m5_gemvn_q6k(pf(np.ascontiguousarray(gu.ravel())), MOE_INTER, p8(L.dxb),
                                sel_a.ctypes.data_as(pi_), L.dx_per, H, MOE_INTER, NUSED, pf(D))
        assert rc == 0, f"batch q6k rc={rc}"
        out = (D.reshape(NUSED, H) * rw[:, None]).sum(0)
    elif K2_FUSE and L.exc == 13 and L.uxc == 13 and L.dxc == 13:
        # IQ2_S：8 专家 gate+up 16 段一次调用；down 逐专家（输入互不相同）
        G, U = gemv_fused_gu(h2, L.exb, L.uxb, L.ex_per, L.ux_per, sel, MOE_INTER)
        gu = silu(G) * U
        out = np.zeros(H, np.float32)
        for k, e in enumerate(sel):
            out += gemv1(L.dxc, L.dxb[e * L.dx_per:], H, MOE_INTER, gu[k]) * rw[k]
    else:
        out = np.zeros(H, np.float32)
        for k, e in enumerate(sel):
            g = gemv1(L.exc, L.exb[e * L.ex_per:], MOE_INTER, H, h2)
            u = gemv1(L.uxc, L.uxb[e * L.ux_per:], MOE_INTER, H, h2)
            d = gemv1(L.dxc, L.dxb[e * L.dx_per:], H, MOE_INTER, silu(g) * u)
            out += d * rw[k]
    t0 = _tick("moe_experts", t0)
    if not shared_done:
        if VK is not None and L.sgg and L.sug:
            g, u = VK.shared_gu(h2, L.sgg, L.sug, MOE_INTER)
        else:
            g = gemv1(L.sgc, L.sgb, MOE_INTER, H, h2)
            u = gemv1(L.suc, L.sub, MOE_INTER, H, h2)
        su = silu(g) * u
        if VK is not None and L.sdg is not None:
            d = VK.shared_d(su, L.sdg, H, MOE_INTER)
        else:
            d = gemv1(L.sdc, L.sdb, H, MOE_INTER, su)
    else:
        d = d9
    t0 = _tick("moe_shared", t0)
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
            x = x + moe_ffn(h2, L, pos)
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

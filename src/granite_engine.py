#!/usr/bin/env python3
"""granite_engine —— granite-hybrid（Mamba-1/S4D × 36 + GQA 注意力 × 4 + MoE×40）的 Darco 驱动器。

结构（2026-09-15 与 llama.cpp granite-hybrid.cpp/mamba-base.cpp 逐行核对）：
  每层: rms(x) → 分支：
    SSM 层(36): in_proj(1536→6448) → 切 [z 3072 | xBC 3328 | dt 48]
                → conv1d(3328ch, 4tap, **只作用 xBC**, bias, silu)
                → 再切 [x 3072 | B 128 | C 128]
                → softplus(dt+bias)·A（A={1,48} ⇒ 标量 dA）→ 状态递推 → 读出
                → y += x·ssm_d（按头）→ swiglu(z, y) → ssm_norm → ssm_out
    attn 层(4): q/k/v(o) GQA（n_head=12? 由权重形状推）+ rope(128 维, base 10000)
  残差（granite 特有）: attn 分支输出 ×residual_scale(0.22) 后加回；ffn 输出 ×0.22 后加回
  MoE(40 层都有): softmax 路由 → top-6 → 权重 softmax 归一（norm_w, 无 scale、无分组、无偏置）
                  + 共享专家 (1024 中间维)
  最终: output_norm；logits ×1/logit_scale(6.0)

对账目标：与 llama.cpp（本机 build）的 logits/逐字输出一致。
状态：S4D 段先在 numpy（本文件），对账过了再下沉 C —— 与 Ling 的路径同一策略。
"""
import ctypes as ct
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf  # noqa: E402
import m5sel
import gguf_fast  # noqa: E402

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf")
MAXT = int(os.environ.get("MAXT", "1024"))

print("[granite] 加载 GGUF...", flush=True)
R = gguf_fast.FastGGUF(MODEL)
T = {t.name: t for t in R.fields and R.tensors}


def kv(name, default=None):
    f = R.fields.get(f"granitehybrid.{name}")
    return f.value if f is not None else default


H = int(kv("embedding_length"))
NL = int(kv("block_count"))
D_STATE = int(kv("ssm.state_size"))
D_CONV = int(kv("ssm.conv_kernel"))
DT_RANK = int(kv("ssm.time_step_rank"))
N_GROUP = int(kv("ssm.group_count"))
D_INNER = int(kv("ssm.inner_size"))
N_EXP = int(kv("expert_count"))
N_USED = int(kv("expert_used_count"))
RES_SCALE = float(kv("residual_scale", 0.0) or 0.0)
EMB_SCALE = float(kv("embedding_scale", 0.0) or 0.0)
LOGIT_SCALE = float(kv("logit_scale", 0.0) or 0.0)
ATTN_SCALE = float(kv("attention.scale", 0.0) or 0.0)
ROPE_DIM = int(kv("rope.dimension_count"))
ROPE_BASE = float(kv("rope.freq_base"))
EPS = float(kv("attention.layer_norm_rms_epsilon", 1e-5))

# 注意力头：从有 KV 的层权重形状推（attn_q 1536×1536；k 1536×512）
ATTN_L = [il for il in range(NL) if f"blk.{il}.attn_q.weight" in T]
tq = T[f"blk.{ATTN_L[0]}.attn_q.weight"]
tk = T[f"blk.{ATTN_L[0]}.attn_k.weight"]
AH = int(tq.shape[1]) // 128          # q 投影 1536 ⇒ 12 头 ×128（ne=(1536,1536) 反排 [out,in]）
AKV = int(tk.shape[1]) // 128         # k 投影 512 ⇒ 4 kv 头
HDIM = 128
print(f"[CFG] granitehybrid: 层={NL} SSM={NL - len(ATTN_L)} attn={len(ATTN_L)}{ATTN_L} H={H} "
      f"attn头={AH}/{AKV} d_state={D_STATE} conv={D_CONV} dt_rank={DT_RANK} "
      f"MoE={N_EXP}x{N_USED} res_scale={RES_SCALE} emb_scale={EMB_SCALE}", flush=True)
ATTN_SET = set(ATTN_L)

# ★★ 必须在**首次 dlopen m6_engine.so 之前**设好 OMP 环境变量：libgomp 在库初始化时就把
#    OMP_NUM_THREADS 读走了，之后再改无效（autotune 的探测因此要起子进程做）。
#    实测（granite-h-tiny Q4_K_M，本机 20 逻辑核 / 4 大核 + 6 小核）：
#    默认（不设）→ **1.06 tok/s**，OMP=4 → 4.34，OMP=8 → 7.82，OMP=16 → 6.48。
#    默认值不是"慢一点"而是**灾难性退化**：SMT 兄弟互抢缓存，且每层十几次 fork/join 的开销被放大。
#    Ling/smol 的驱动一直靠 autotune 设这个，granite 一开始漏了 ⇒ 差 7.4 倍。
import mcfg as _mcfg      # noqa: E402
import autotune as _at    # noqa: E402

CFG = _mcfg.load_cfg(R, T)
_plan = _at.plan(CFG, sum(t.n_bytes for t in R.tensors))
for _k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
    os.environ.setdefault(_k, str(_plan[_k]))
print(f"[autotune] OMP={os.environ['OMP_NUM_THREADS']} ({_plan.get('OMP_src', '?')}) "
      f"WAIT={os.environ['OMP_WAIT_POLICY']} SPIN={os.environ['GOMP_SPINCOUNT']} | "
      f"{_plan.get('machine', '')}", flush=True)

KEEP = []
M6 = ct.CDLL(os.path.join(BASE, "m6_engine.so"))
M6.m6_init_dl.argtypes = [ct.c_char_p] * 5
M6.m6_init_dl.restype = ct.c_int
_rc = M6.m6_init_dl(*[(os.path.join(BASE, "m5") + "/" + n).encode()
                      for n in m5sel.paths()])
assert _rc == 0, _rc
M6.m6_init_extra_dl.argtypes = [ct.c_char_p]
M6.m6_init_extra_dl.restype = ct.c_int
assert M6.m6_init_extra_dl((os.path.join(BASE, "m5/m5_kern11.so")).encode()) == 0
M6E = M6
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
# ⚠️ 不 import smol_engine —— 那有加载 smol 模型的副作用；granite 的 attn/MoE 描述符
#    在本文件内自建（ granite_attn 用 m6_llama_attn_op、MoE 用 m6_moe_batch4，都要在本层装载）。


# ═══════════════ S4D 层原型（numpy；对账通过后再决定下沉 C）═══════════════════
# 语义全部来自 llama.cpp 逐行核对（mamba-base.cpp + ggml-cpu/ops.cpp），见 ROADMAP C3。
def _deq(name):
    t = T[name]
    arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32)
    return arr.reshape(tuple(int(v) for v in t.shape)[::-1])

_CONV_W = {}
_CONV_B = {}
_SSM_A = {}
_SSM_D = {}
_SSM_DTB = {}

NHEAD_OR_DT = DT_RANK
HDIM_S4 = D_INNER // DT_RANK

class S4DState:
    """S4D 层的递推状态（conv 的 d_conv-1 帧 + ssm 状态）。每层一个。"""
    def __init__(self):
        self.hist = np.zeros((D_CONV - 1, D_INNER + 2 * N_GROUP * D_STATE), np.float32)
        self.hst = np.zeros((DT_RANK, HDIM_S4, D_STATE), np.float32)


def ssm_step(x_in, weights, st):
    """单步 S4D 层。语义已与 llama.cpp 在 pos 0..3 × 全部 36 个 SSM 层逐层对账
    （s4d_reconcile2.py：链头 cos 中位 0.99999、最小 0.99915）。

      conv: 内核按 out[t]=Σ_tap w[tap]·s[t+tap]（s = 状态帧在前、当前帧最后）
            ⇒ w[:,-1] 乘当前帧、w[:,0] 乘最老那帧；conv 后加 bias 再 silu
      scan: dt = softplus(dt_pre + dt_bias)（无 clamp）；A={1,n_head} ⇒ 每头标量衰减；
            state = dA·state + (x·dt)·B；y = Σ_s state·C（读出的是**更新后**的状态）
      门:   y = silu(z) · y（swiglu_split(z, y)）
      D:    y[h,k] += x[h,k]·ssm_d[h]（按头）
    """
    DI = D_INNER
    XBC = DI + 2 * N_GROUP * D_STATE
    zall = x_in @ weights["in"].T
    zseg = zall[:DI]
    xBC = zall[DI:DI + XBC]
    sdt = _softplus(zall[DI + XBC:] + weights["dt_b"])
    s = np.vstack([st.hist, xBC[None, :]])                   # [d_conv, XBC]，当前帧在最后
    conv_out = _silu((weights["conv"].T * s).sum(axis=0) + weights["conv_b"])
    st.hist = s[1:]                                          # 丢掉最老的一帧
    x = conv_out[:DI]
    B = conv_out[DI:DI + D_STATE]
    Cseg = conv_out[DI + D_STATE:DI + 2 * D_STATE]
    xh = x.reshape(DT_RANK, HDIM_S4)
    st.hst = st.hst * np.exp(sdt * weights["a"])[:, None, None] \
        + (xh * sdt[:, None])[:, :, None] * B[None, None, :]
    y_hk = np.tensordot(st.hst, Cseg, axes=([2], [0])) + xh * weights["d"][:, None]
    # ★ 展开必须 h 主序（y_hk 是 [head, dim]，内存序 ne0=dim 最快 ⇒ flat[h*64+k]）。
    #   写成 y_hk.T.reshape(-1) 会变成 flat[k*48+h]：实测链尾 cos 从 0.9998 掉到 -0.044。
    return _silu(zseg) * y_hk.reshape(-1)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)



# ═══════════════════════════════════════════════════════════════════════════════
# C 驱动（每 token 一次 m6_granite_forward_token）
# ═══════════════════════════════════════════════════════════════════════════════
CODE = {'Q8_0': 0, 'IQ4_NL': 1, 'IQ3_S': 2, 'Q5_K': 3, 'Q6_K': 4, 'Q4_K': 5, 'IQ4_XS': 6,
        'F16': 7, 'Q5_0': 8}

M6E = ct.CDLL(os.environ.get("M6_SO", os.path.join(BASE, "m6_engine.so")))
M6E.m6_init_dl.argtypes = [ct.c_char_p] * 5
M6E.m6_init_dl.restype = ct.c_int
_rc = M6E.m6_init_dl(*[(os.path.join(BASE, "m5") + "/" + n).encode() for n in m5sel.paths()])
assert _rc == 0, f"m6_init_dl rc={_rc}"
for _n, _a, _r in (("m6_granite_forward_token",
                    [ct.POINTER(ct.c_float), ct.c_int, ct.c_float, ct.c_int, ct.c_float,
                     ct.c_void_p, ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float)], ct.c_int),
                   ("m6_head_op",
                    [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int, ct.c_float,
                     ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)], ct.c_int)):
    getattr(M6E, _n).argtypes = _a
    getattr(M6E, _n).restype = _r
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
KEEP = []
STATES = []      # 跨 token 的状态本体（S4D 的 conv/ssm 态、注意力 KV 缓存）—— reset() 用


class S4dP(ct.Structure):
    _fields_ = [("win", ct.c_void_p), ("wout", ct.c_void_p), ("cin", ct.c_int), ("cout", ct.c_int),
                ("conv_w", ct.POINTER(ct.c_float)), ("conv_b", ct.POINTER(ct.c_float)),
                ("a", ct.POINTER(ct.c_float)), ("d", ct.POINTER(ct.c_float)),
                ("dt_b", ct.POINTER(ct.c_float)), ("norm", ct.POINTER(ct.c_float)),
                ("hist", ct.POINTER(ct.c_float)), ("hst", ct.POINTER(ct.c_float)),
                ("work", ct.POINTER(ct.c_float)),
                ("d_inner", ct.c_int), ("d_state", ct.c_int), ("dt_rank", ct.c_int),
                ("n_group", ct.c_int), ("d_conv", ct.c_int), ("hidden", ct.c_int),
                ("eps", ct.c_float)]


class LlamaAttnP(ct.Structure):
    _fields_ = [("wq", ct.c_void_p), ("wk", ct.c_void_p), ("wv", ct.c_void_p), ("wo", ct.c_void_p),
                ("cq", ct.c_int), ("ck", ct.c_int), ("cv", ct.c_int), ("co", ct.c_int),
                ("n_head", ct.c_int), ("n_kv", ct.c_int), ("head_dim", ct.c_int),
                ("hidden", ct.c_int), ("rot", ct.c_int), ("rope_base", ct.c_float),
                ("kcache", ct.POINTER(ct.c_float)), ("vcache", ct.POINTER(ct.c_float)),
                ("tlen", ct.POINTER(ct.c_int)), ("max_t", ct.c_int),
                ("work", ct.POINTER(ct.c_float))]


class GLayer(ct.Structure):
    _fields_ = [("is_attn", ct.c_int), ("branch_p", ct.c_void_p),
                ("attn_norm", ct.POINTER(ct.c_float)), ("ffn_norm", ct.POINTER(ct.c_float)),
                ("exp_ptrs", ct.POINTER(ct.c_uint64)), ("exp_codes", ct.POINTER(ct.c_int)),
                ("gate_inp", ct.POINTER(ct.c_float)),
                ("sh_ptrs", ct.POINTER(ct.c_uint64)), ("sh_codes", ct.POINTER(ct.c_int)),
                ("n_exp", ct.c_int), ("n_used", ct.c_int), ("inter", ct.c_int),
                ("sh_inter", ct.c_int)]


def qb(name):
    """量化权重缓冲（uint8 视图）+ 类型码。★ 必须持有数组本体，否则指针悬空。"""
    buf = np.frombuffer(bytes(T[name].data), np.uint8)
    KEEP.append(buf)
    return buf, CODE[T[name].tensor_type.name]


def fa(name):
    t = T[name]
    arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(-1)
    KEEP.append(arr)
    return arr


MAXT = int(os.environ.get("MAXT", "1024"))
XBC = D_INNER + 2 * N_GROUP * D_STATE
DIP = 2 * D_INNER + 2 * N_GROUP * D_STATE + DT_RANK
VOCAB = int(kv("vocab_size"))
print(f"[granite/C] attn 层={ATTN_L} 头={AH}/{AKV} emb_scale={EMB_SCALE} res_scale={RES_SCALE:.4f} "
      f"logit_scale={LOGIT_SCALE} vocab={VOCAB} MAXT={MAXT}", flush=True)

layers = (GLayer * NL)()
for il in range(NL):
    p = f"blk.{il}."
    L = layers[il]
    an, fn = fa(p + "attn_norm.weight"), fa(p + "ffn_norm.weight")
    L.attn_norm, L.ffn_norm = pf(an), pf(fn)
    if il in ATTN_SET:
        a = LlamaAttnP()
        a.wq, a.cq = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_q.weight"))
        a.wk, a.ck = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_k.weight"))
        a.wv, a.cv = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_v.weight"))
        a.wo, a.co = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_output.weight"))
        a.n_head, a.n_kv, a.head_dim, a.hidden = AH, AKV, HDIM, H
        a.rot = 0                      # ★ granite 是 NoPE：rot=0 ⇒ rope 循环 0 次，天然跳过
        a.rope_base = 0.0
        kc = np.zeros(MAXT * AKV * HDIM, np.float32)
        vc = np.zeros(MAXT * AKV * HDIM, np.float32)
        tl = np.zeros(1, np.int32)
        wk = np.zeros(AH * HDIM + 2 * AKV * HDIM + AH * MAXT + 64, np.float32)
        a.kcache, a.vcache = pf(kc), pf(vc)
        a.tlen = tl.ctypes.data_as(ct.POINTER(ct.c_int))
        a.max_t, a.work = MAXT, pf(wk)
        STATES.extend([kc, vc, np.frombuffer(tl.data, np.int32)])
        KEEP.extend([kc, vc, tl, wk, a])
        L.branch_p = ct.cast(ct.pointer(a), ct.c_void_p)
        L.is_attn = 1
    else:
        s = S4dP()
        s.win, s.cin = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ssm_in.weight"))
        s.wout, s.cout = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ssm_out.weight"))
        cw = np.frombuffer(bytes(T[p + "ssm_conv1d.weight"].data), np.float32).reshape(XBC, D_CONV).copy()
        s.conv_w = pf(cw)
        s.conv_b = pf(fa(p + "ssm_conv1d.bias"))
        s.a = pf(fa(p + "ssm_a"))
        s.d = pf(fa(p + "ssm_d"))
        s.dt_b = pf(fa(p + "ssm_dt.bias"))
        s.norm = pf(fa(p + "ssm_norm.weight"))
        hist = np.zeros((D_CONV - 1) * XBC, np.float32)
        hst = np.zeros(DT_RANK * HDIM_S4 * D_STATE, np.float32)
        wk = np.zeros(DIP + XBC + DT_RANK + D_INNER, np.float32)
        s.hist, s.hst, s.work = pf(hist), pf(hst), pf(wk)
        STATES.extend([hist, hst])
        s.d_inner, s.d_state, s.dt_rank = D_INNER, D_STATE, DT_RANK
        s.n_group, s.d_conv, s.hidden = N_GROUP, D_CONV, H
        s.eps = EPS
        KEEP.extend([cw, hist, hst, wk, s])
        L.branch_p = ct.cast(ct.pointer(s), ct.c_void_p)
        L.is_attn = 0
    # MoE：专家指针 = 张量基址 + e*per（三块 exps 的 ne 都是 [*, *, n_expert]，按专家连续）
    ep = np.zeros(3 * N_EXP, np.uint64)
    ec = np.zeros(3 * N_EXP, np.int32)
    for role, idx, nm in (("gate", 0, "ffn_gate_exps.weight"), ("up", 1, "ffn_up_exps.weight"),
                          ("down", 2, "ffn_down_exps.weight")):
        t = T[p + nm]
        buf = np.frombuffer(bytes(t.data), np.uint8)
        KEEP.append(buf)
        per = buf.nbytes // N_EXP
        for e in range(N_EXP):
            ep[3 * e + idx] = buf.ctypes.data + e * per
            ec[3 * e + idx] = CODE[t.tensor_type.name]
    gi = np.frombuffer(bytes(T[p + "ffn_gate_inp.weight"].data), np.float32).reshape(N_EXP, H).copy()
    KEEP.append(gi)
    shp = np.zeros(3, np.uint64)
    shc = np.zeros(3, np.int32)
    for idx, nm in enumerate(("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")):
        e = qb(p + nm)
        shp[idx], shc[idx] = e[0].ctypes.data, e[1]
    KEEP.extend([ep, ec, gi, shp, shc])
    L.exp_ptrs = ep.ctypes.data_as(ct.POINTER(ct.c_uint64))
    L.exp_codes = ec.ctypes.data_as(ct.POINTER(ct.c_int))
    L.gate_inp = pf(gi)
    L.sh_ptrs = shp.ctypes.data_as(ct.POINTER(ct.c_uint64))
    L.sh_codes = shc.ctypes.data_as(ct.POINTER(ct.c_int))
    L.n_exp, L.n_used, L.inter, L.sh_inter = N_EXP, N_USED, int(kv("feed_forward_length")), \
        int(kv("expert_shared_feed_forward_length"))

# 共享缓冲
X = np.zeros(H, np.float32)
TMP = np.zeros(2 * H + 2 * layers[0].sh_inter, np.float32)
MSO = np.zeros(3 * N_USED * layers[0].inter + N_USED * H, np.float32)   # 见 m6_engine.c 的式子
LOGITS = np.zeros(VOCAB, np.float32)
HSC = np.zeros(H, np.float32)
ONORM = fa("output_norm.weight")
_EMB = T["token_embd.weight"]
_EMB_ROWB = _EMB.data.shape[1]      # 每行的字节数（Q6_K: 1260）
_HEAD, _HEAD_CODE = qb("token_embd.weight")     # granite 没有 output.weight ⇒ lm_head = 词嵌入
KEEP.extend([X, TMP, MSO, LOGITS, HSC])

# ★ 只读抓手（对账用）：S4D 层 0 的输出、以及每层的分支输出，值与我们 dump 的锚点可比
PROBE_ON = os.environ.get("GPROBE", "0") == "1"
PROBE = {"layer_out": np.zeros((NL, H), np.float32)}


def forward(tid, pos):
    # ★ 只取**一行**（零拷贝）：原来写成 `bytes(_EMB.data)[a:b]` —— bytes() 会把**整个 125MB
    #   词嵌入复制一遍**再按字节偏移切片，实测每个 token 白花 50~90ms（占端到端的约 58%）。
    #   gguf_fast 的 .data 是 mmap 上的只读 uint8 **二维**视图 (n_tokens, bytes_per_row)
    #   ⇒ 直接 `[tid]` 拿一行；按字节偏移切是错的（那是切「行」维度，会切出空的）。
    row = _EMB.data[int(tid)]
    X[:] = np.asarray(gguf.quants.dequantize(row, _EMB.tensor_type),
                      np.float32) * EMB_SCALE
    M6E.m6_granite_forward_token(pf(X), H, EPS, pos, RES_SCALE, layers, NL, pf(TMP), pf(MSO),
                                 pf(PROBE["layer_out"]) if PROBE_ON else None)
    M6E.m6_head_op(pf(X), pf(ONORM), H, EPS, _HEAD.ctypes.data, _HEAD_CODE, VOCAB, pf(LOGITS), pf(HSC))
    np.divide(LOGITS, LOGIT_SCALE, out=LOGITS)   # granite-hybrid.cpp: scale(logits, 1/f_logit_scale)
    # ★ 不能写 LOGITS /= x：那样 LOGITS 会变成函数局部名，上面 pf(LOGITS) 直接 UnboundLocalError
    return LOGITS


def logits_of_x():
    return LOGITS


def reset():
    """清零跨 token 状态：每请求全量重 prefill 的语义靠它（与其它 dengine 适配器一致）。

    ★ 状态本体是 Python 侧的 numpy 数组，C 结构体里只存指针 ⇒ 清 Python 这边就够，
      不需要改 C。忘清 = 第二个请求会看到上一个会话的上下文。
    """
    for b in STATES:
        b[...] = 0


if __name__ == "__main__":
    import time
    ids = [int(v) for v in os.environ.get("TOKS", "100,101,102,103").split(",")]
    t0 = time.time()
    for pos, tid in enumerate(ids):
        forward(tid, pos)
    print(f"[PREFILL] {len(ids)} tok {time.time()-t0:.2f}s  argmax={int(LOGITS.argmax())}", flush=True)
    gen = [int(LOGITS.argmax())]
    N = int(os.environ.get("NSTEPS", "8"))
    t0 = time.time()
    for s in range(N):
        forward(gen[-1], len(ids) + s)
        gen.append(int(LOGITS.argmax()))
    print(f"[GEN] {N} 步 {time.time()-t0:.2f}s → {N/max(1e-9, time.time()-t0):.2f} tok/s")
    print(f"[GEN] 贪心 token: {gen}")

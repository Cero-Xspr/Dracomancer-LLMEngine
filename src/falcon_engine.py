#!/usr/bin/env python3
"""falcon_engine —— Falcon-H1（每层 注意力∥Mamba-2 并行 + dense FFN）的 Darco 驱动器。

结构（2026-09-15 与 llama.cpp falcon-h1.cpp 逐行核对）：
  每层: xn = rms(x, attn_norm)
        attn_out = NEOX-rope GQA(xn)          ← m6_falcon_attn_op（8/2 头 × 64，freq_base≈1e11）
        ssm_out  = Mamba-2(xn)                ← m6_granite_s4d_op **原样复用**
                （A={1,n_heads} 标量衰减 ⇒ 与 granite 同一条内核分支；形状 1536/24/128/g1/conv4）
        x += attn_out + ssm_out               ← 无 residual_scale
        x += SwiGLU-dense(rms(x, ffn_norm))   ← m6_dense_op，无 bias
  最终: rms(output_norm) → output 投影（有独立 output.weight，非 tied）；**无 logit_scale**
  注意坑：ffn_norm 的张量名**没有 .weight 后缀**（`blk.N.ffn_norm`）。
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

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/falcon-h1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf")
MAXT = int(os.environ.get("MAXT", "1024"))

print("[falcon] 加载 GGUF...", flush=True)
R = gguf_fast.FastGGUF(MODEL)
T = {t.name: t for t in R.tensors}

# ★ 必须在首次 dlopen m6_engine.so 之前设 OMP 环境变量（libgomp 初始化时读走；见 granite 教训）
import mcfg as _mcfg      # noqa: E402
import autotune as _at    # noqa: E402

CFG = _mcfg.load_cfg(R, T)
_plan = _at.plan(CFG, sum(t.n_bytes for t in R.tensors))
for _k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
    os.environ.setdefault(_k, str(_plan[_k]))
print(f"[autotune] OMP={os.environ['OMP_NUM_THREADS']} ({_plan.get('OMP_src', '?')}) "
      f"WAIT={os.environ['OMP_WAIT_POLICY']} | {_plan.get('machine', '')}", flush=True)


def kv(name, default=None):
    f = R.fields.get(f"falcon-h1.{name}")
    return f.value if f is not None else default


H = int(kv("embedding_length"))
NL = int(kv("block_count"))
NH, NKV = int(kv("attention.head_count")), int(kv("attention.head_count_kv"))
HD = int(kv("attention.key_length"))
ROT = HD                                   # 无 rope.dimension_count 键 ⇒ n_rot = head_dim
ROPE_BASE = float(kv("rope.freq_base"))
EPS = float(kv("attention.layer_norm_rms_epsilon", 1e-5))
FF = int(kv("feed_forward_length"))
D_INNER = int(kv("ssm.inner_size"))
D_STATE = int(kv("ssm.state_size"))
DT_RANK = int(kv("ssm.time_step_rank"))
N_GROUP = int(kv("ssm.group_count"))
D_CONV = int(kv("ssm.conv_kernel"))
VOCAB = int(kv("vocab_size"))
XBC = D_INNER + 2 * N_GROUP * D_STATE
DIP = 2 * D_INNER + 2 * N_GROUP * D_STATE + DT_RANK
print(f"[CFG] falcon-h1: 层={NL} H={H} 头={NH}/{NKV}×{HD} rope_base={ROPE_BASE:.3e} ffn={FF} "
      f"SSM d_inner={D_INNER} heads={DT_RANK} state={D_STATE} vocab={VOCAB}", flush=True)

CODE = {'Q8_0': 0, 'IQ4_NL': 1, 'IQ3_S': 2, 'Q5_K': 3, 'Q6_K': 4, 'Q4_K': 5, 'IQ4_XS': 6,
        'F16': 7, 'Q5_0': 8}

M6E = ct.CDLL(os.environ.get("M6_SO", os.path.join(BASE, "m6_engine.so")))
M6E.m6_init_dl.argtypes = [ct.c_char_p] * 5
M6E.m6_init_dl.restype = ct.c_int
_rc = M6E.m6_init_dl(*[(os.path.join(BASE, "m5") + "/" + n).encode() for n in m5sel.paths()])
assert _rc == 0, f"m6_init_dl rc={_rc}"
for _n, _a, _r in (("m6_falcon_forward_token",
                    [ct.POINTER(ct.c_float), ct.c_int, ct.c_float, ct.c_int,
                     ct.c_void_p, ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)],
                    ct.c_int),
                   ("m6_head_op",
                    [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int, ct.c_float,
                     ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)],
                    ct.c_int)):
    getattr(M6E, _n).argtypes = _a
    getattr(M6E, _n).restype = _r
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
KEEP = []
STATES = []      # 跨 token 状态（attn KV/tlen + ssm hist/hst）—— reset() 用


class AttnP(ct.Structure):
    """与 C 的 M6LlamaAttnP 同布局（falcon 不加字段，rope 风格由算子决定）。"""
    _fields_ = [("wq", ct.c_void_p), ("wk", ct.c_void_p), ("wv", ct.c_void_p), ("wo", ct.c_void_p),
                ("cq", ct.c_int), ("ck", ct.c_int), ("cv", ct.c_int), ("co", ct.c_int),
                ("n_head", ct.c_int), ("n_kv", ct.c_int), ("head_dim", ct.c_int),
                ("hidden", ct.c_int), ("rot", ct.c_int), ("rope_base", ct.c_float),
                ("kcache", ct.POINTER(ct.c_float)), ("vcache", ct.POINTER(ct.c_float)),
                ("tlen", ct.POINTER(ct.c_int)), ("max_t", ct.c_int),
                ("work", ct.POINTER(ct.c_float))]


class S4dP(ct.Structure):
    """与 C 的 M6GraniteS4dP 同布局（语义同 granite ⇒ 结构体共用）。"""
    _fields_ = [("win", ct.c_void_p), ("wout", ct.c_void_p), ("cin", ct.c_int), ("cout", ct.c_int),
                ("conv_w", ct.POINTER(ct.c_float)), ("conv_b", ct.POINTER(ct.c_float)),
                ("a", ct.POINTER(ct.c_float)), ("d", ct.POINTER(ct.c_float)),
                ("dt_b", ct.POINTER(ct.c_float)), ("norm", ct.POINTER(ct.c_float)),
                ("hist", ct.POINTER(ct.c_float)), ("hst", ct.POINTER(ct.c_float)),
                ("work", ct.POINTER(ct.c_float)),
                ("d_inner", ct.c_int), ("d_state", ct.c_int), ("dt_rank", ct.c_int),
                ("n_group", ct.c_int), ("d_conv", ct.c_int), ("hidden", ct.c_int),
                ("eps", ct.c_float)]


class FLayer(ct.Structure):
    _fields_ = [("attn_p", ct.c_void_p), ("ssm_p", ct.c_void_p),
                ("attn_norm", ct.POINTER(ct.c_float)), ("ffn_norm", ct.POINTER(ct.c_float)),
                ("g", ct.c_void_p), ("u", ct.c_void_p), ("d", ct.c_void_p),
                ("cg", ct.c_int), ("cu", ct.c_int), ("cd", ct.c_int), ("n_ff", ct.c_int)]


def qb(name):
    buf = np.frombuffer(bytes(T[name].data), np.uint8)
    KEEP.append(buf)
    return buf, CODE[T[name].tensor_type.name]


def fa(name):
    t = T[name]
    arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(-1)
    KEEP.append(arr)
    return arr


layers = (FLayer * NL)()
for il in range(NL):
    p = f"blk.{il}."
    L = layers[il]
    an = fa(p + "attn_norm.weight")
    fn = fa(p + "ffn_norm")            # ★ 无 .weight 后缀（falcon 特有）
    L.attn_norm, L.ffn_norm = pf(an), pf(fn)
    # 注意力（NEOX 由 C 侧算子决定；结构同 llama）
    a = AttnP()
    a.wq, a.cq = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_q.weight"))
    a.wk, a.ck = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_k.weight"))
    a.wv, a.cv = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_v.weight"))
    a.wo, a.co = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_output.weight"))
    a.n_head, a.n_kv, a.head_dim, a.hidden, a.rot = NH, NKV, HD, H, ROT
    a.rope_base = ROPE_BASE
    kc = np.zeros(MAXT * NKV * HD, np.float32)
    vc = np.zeros(MAXT * NKV * HD, np.float32)
    tl = np.zeros(1, np.int32)
    wk = np.zeros(NH * HD + 2 * NKV * HD + NH * MAXT + 64, np.float32)
    a.kcache, a.vcache = pf(kc), pf(vc)
    a.tlen = tl.ctypes.data_as(ct.POINTER(ct.c_int))
    a.max_t, a.work = MAXT, pf(wk)
    KEEP.extend([kc, vc, tl, wk, a])
    STATES.extend([kc, vc, np.frombuffer(tl.data, np.int32)])
    L.attn_p = ct.cast(ct.pointer(a), ct.c_void_p)
    # SSM（granite 同构 ⇒ 同一 C 算子与结构体）
    s = S4dP()
    s.win, s.cin = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ssm_in.weight"))
    s.wout, s.cout = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ssm_out.weight"))
    cw = np.frombuffer(bytes(T[p + "ssm_conv1d.weight"].data), np.float32).reshape(XBC, D_CONV).copy()
    s.conv_w = pf(cw)
    s.conv_b = pf(fa(p + "ssm_conv1d.bias"))
    s.a = pf(fa(p + "ssm_a"))
    s.d = pf(fa(p + "ssm_d"))
    s.dt_b = pf(fa(p + "ssm_dt.bias"))
    t = T.get(p + "ssm_norm.weight") or T.get(p + "ssm_norm")
    if t is None:
        s.norm = None                    # ★ falcon-h1 没有 ssm_norm ⇒ C 侧跳过归一化
    else:
        nm = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(-1)
        KEEP.append(nm)
        s.norm = pf(nm)
    hist = np.zeros((D_CONV - 1) * XBC, np.float32)
    hst = np.zeros(DT_RANK * (D_INNER // DT_RANK) * D_STATE, np.float32)
    wk2 = np.zeros(DIP + XBC + DT_RANK + D_INNER, np.float32)
    s.hist, s.hst, s.work = pf(hist), pf(hst), pf(wk2)
    s.d_inner, s.d_state, s.dt_rank = D_INNER, D_STATE, DT_RANK
    s.n_group, s.d_conv, s.hidden = N_GROUP, D_CONV, H
    s.eps = EPS
    KEEP.extend([cw, hist, hst, wk2, s])
    STATES.extend([hist, hst])
    L.ssm_p = ct.cast(ct.pointer(s), ct.c_void_p)
    # dense FFN（无 bias）
    for fld, nm2 in (("g", "ffn_gate.weight"), ("u", "ffn_up.weight"), ("d", "ffn_down.weight")):
        e = qb(p + nm2)
        setattr(L, fld, e[0].ctypes.data)
        setattr(L, "c" + fld, e[1])
    L.n_ff = FF

X = np.zeros(H, np.float32)
TMP = np.zeros(3 * H, np.float32)
LOGITS = np.zeros(VOCAB, np.float32)
HSC = np.zeros(H, np.float32)
ONORM = fa("output_norm.weight")
_HEAD, _HEAD_CODE = qb("output.weight")     # falcon 有独立 output.weight（非 tied）
KEEP.extend([X, TMP, LOGITS, HSC])

# 对账探针（GPROBE=1 时**每个位置**的每层输出都留一份，与 dump 的 l_out-{il} 逐 (pos,层) 比。
#   第一版只存最后一次 forward ⇒ 拿 pos3 的探针比 pos0 的 dump，全红——假象。）
PROBE_ON = os.environ.get("GPROBE", "0") == "1"
PROBE = {"by_pos": {}}


def forward(tid, pos):
    row = T["token_embd.weight"].data[int(tid)]
    X[:] = np.asarray(gguf.quants.dequantize(row, T["token_embd.weight"].tensor_type),
                      np.float32)
    _buf = np.zeros((NL, H), np.float32) if PROBE_ON else None
    M6E.m6_falcon_forward_token(pf(X), H, EPS, pos, layers, NL, pf(TMP),
                                pf(_buf) if PROBE_ON else None)
    if PROBE_ON:
        PROBE["by_pos"][pos] = _buf
    M6E.m6_head_op(pf(X), pf(ONORM), H, EPS, _HEAD.ctypes.data, _HEAD_CODE, VOCAB,
                   pf(LOGITS), pf(HSC))
    return LOGITS


def logits_of_x():
    return LOGITS


def reset():
    for b in STATES:
        b[...] = 0


if __name__ == "__main__":
    import time
    ids = [int(v) for v in os.environ.get("TOKS", "17,1243,7889,813,10513,860").split(",")]
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

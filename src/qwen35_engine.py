#!/usr/bin/env python3
"""qwen35_engine —— Qwen3.5（19 GDN + 6 全注意力，每 4 层一个）的 Darco 驱动器。

结构（2026-09-16 与 llama.cpp qwen35.cpp 逐行核对）：
  每层: rms(attn_norm) → 分支：
    GDN 层(19): m6_gdn_attn 整层驱动（conv→silu→[q|k|v]→l2norm→delta 递推→门控 rms(z)→ssm_out）
    全注意力层(6): m6_full_attn 整层驱动（joint QG 投影[每头 512=q256|gate256]→q/k 各自
                  RMSNorm→成对 rope@1e7→GQA 8/2×256→⊙sigmoid(gate)→wo）
        → x += 分支输出（attn_residual）→ rms(post_attention_norm) → dense FFN → x += ffn
  最终: rms(output_norm) → head = 词嵌入绑定（无 output.weight，F16）；无 logit scale
  ★ 第 25 块是 MTP（nextn），不进主图 ⇒ 只装载前 24 层。
  ★ 硬编码对齐：m6_gdn_attn / m6_full_attn 的 2048/16 头/切点恰为 2B 形状；EPS=1e-6 与 GGUF 一致。
"""
import ctypes as ct
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf  # noqa: E402
import gguf_fast  # noqa: E402

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/Qwen3.5-2B-f16.gguf")
MAXT = int(os.environ.get("MAXT", "1024"))

print("[qwen35] 加载 GGUF...", flush=True)
R = gguf_fast.FastGGUF(MODEL)
T = {t.name: t for t in R.tensors}

# ★ OMP 环境必须在首次 dlopen 前设好（libgomp 初始化读走；三次踩坑后的硬规矩）
import mcfg as _mcfg      # noqa: E402
import autotune as _at    # noqa: E402

CFG = _mcfg.load_cfg(R, T)
_plan = _at.plan(CFG, sum(t.n_bytes for t in R.tensors))
for _k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
    os.environ.setdefault(_k, str(_plan[_k]))
print(f"[autotune] OMP={os.environ['OMP_NUM_THREADS']} ({_plan.get('OMP_src', '?')}) "
      f"WAIT={os.environ['OMP_WAIT_POLICY']} | {_plan.get('machine', '')}", flush=True)


ARCH = R.fields["general.architecture"].value.decode() if isinstance(
    R.fields["general.architecture"].value, bytes) else R.fields["general.architecture"].value


def kv(name, default=None):
    f = R.fields.get(f"{ARCH}.{name}")
    return f.value if f is not None else default


H = int(kv("embedding_length"))
FF = int(kv("feed_forward_length") or 0)   # qwen35moe 没有 dense FFN 键（全是专家）
NL_ALL = int(kv("block_count"))
NEXTN = int(kv("nextn_predict_layers") or 0)
NL = NL_ALL - NEXTN                                   # 主图层 24；第 25 块是 MTP
_rc_arr = kv("attention.recurrent_layers")
if _rc_arr is None:                       # qwen35moe 没有 recurrent_layers 键 ⇒ 按 interval 推
    _iv = int(kv("full_attention_interval") or 4)
    RECR = [(i + 1) % _iv != 0 for i in range(NL)]     # 与 llama.cpp fallback 同式
else:
    RECR = [bool(int(v)) for v in _rc_arr][:NL]
NH = int(kv("attention.head_count"))
NKV = int(kv("attention.head_count_kv"))
HD_A = int(kv("attention.key_length"))                # 256
ROT = int(kv("rope.dimension_count"))                 # 64
ROPE_BASE = float(kv("rope.freq_base"))               # 1e7
EPS = float(kv("attention.layer_norm_rms_epsilon"))   # 1e-6（与 C 侧 #define 一致）
DI = int(kv("ssm.inner_size"))                        # 2048
NVH = int(kv("ssm.time_step_rank"))                   # 16 v 头
NKH = int(kv("ssm.group_count"))                      # 16 k 头
DS = int(kv("ssm.state_size"))                        # 128
DC = int(kv("ssm.conv_kernel"))                       # 4
NQKV = 2 * NKH * DS + DI                              # 6144
SV = DS
N_EXP = int(kv("expert_count") or 0)
N_USED = int(kv("expert_used_count") or 0)
EXP_FFN = int(kv("expert_feed_forward_length") or 0)
SHEXP_FFN = int(kv("expert_shared_feed_forward_length") or 0)
HAS_SHGATE = any(n.endswith("ffn_gate_inp_shexp.weight") for n in T)
VOCAB = len(R.fields["tokenizer.ggml.tokens"].value)
print(f"[CFG] {ARCH}: 主图层={NL}(+MTP {NEXTN}) H={H} 头={NH}/{NKV}×{HD_A} rope={ROT}@{ROPE_BASE:.0e} "
      f"FF={FF} GDN层={sum(RECR)} 全注意力层={[i for i in range(NL) if not RECR[i]]} "
      f"MoE={N_EXP}x{N_USED}(ffn={EXP_FFN},shexp={SHEXP_FFN},门控共享={HAS_SHGATE}) vocab={VOCAB}", flush=True)

CODE = {'Q8_0': 0, 'IQ4_NL': 1, 'IQ3_S': 2, 'Q5_K': 3, 'Q6_K': 4, 'Q4_K': 5, 'IQ4_XS': 6,
        'F16': 7, 'Q5_0': 8, 'F32': 9}

M6E = ct.CDLL(os.environ.get("M6_SO", os.path.join(BASE, "m6_engine.so")))
M6E.m6_init_dl.argtypes = [ct.c_char_p] * 5
M6E.m6_init_dl.restype = ct.c_int
_rc = M6E.m6_init_dl(*[(os.path.join(BASE, "m5") + "/" + n).encode() for n in
                       ("m5_kern6.so", "m5_kern9.so", "m5_kern8.so", "m5_kern7.so", "m5_kernF.so")])
assert _rc == 0, f"m6_init_dl rc={_rc}"
for _n, _a, _r in (("m6_gdn_attn",
                    [ct.POINTER(ct.c_float), ct.c_void_p, ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float)], None),
                   ("m6_full_attn",
                    [ct.POINTER(ct.c_float), ct.c_void_p, ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float), ct.c_int, ct.c_int, ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)], None),
                   ("m6_dense_op", [ct.POINTER(ct.c_float), ct.c_void_p, ct.POINTER(ct.c_float)], None),
                   ("m6_granite_shexp",
                    [ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_int),
                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                     ct.POINTER(ct.c_float), ct.c_int, ct.c_int], None),
                   ("m6_granite_moe",
                    [ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_int),
                     ct.POINTER(ct.c_float), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)], None),
                   ("m6_head_op",
                    [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int, ct.c_float,
                     ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)],
                    ct.c_int)):
    getattr(M6E, _n).argtypes = _a
    getattr(M6E, _n).restype = _r
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
KEEP = []
STATES = []


class GdnW(ct.Structure):
    _fields_ = [("attn_norm", ct.POINTER(ct.c_float)),
                ("qkv_buf", ct.c_void_p), ("qkv_code", ct.c_int),
                ("gate_buf", ct.c_void_p), ("gate_code", ct.c_int),
                ("beta_wt", ct.POINTER(ct.c_float)), ("alpha_wt", ct.POINTER(ct.c_float)),
                ("ssm_a", ct.POINTER(ct.c_float)), ("ssm_dt", ct.POINTER(ct.c_float)),
                ("conv", ct.POINTER(ct.c_float)), ("ssm_norm", ct.POINTER(ct.c_float)),
                ("ssm_out_buf", ct.c_void_p), ("ssm_out_code", ct.c_int),
                ("post_norm", ct.POINTER(ct.c_float)),
                ("n_qkv", ct.c_int), ("n_vh", ct.c_int), ("d_in", ct.c_int)]


class FullW(ct.Structure):
    _fields_ = [("attn_norm", ct.POINTER(ct.c_float)),
                ("q_buf", ct.c_void_p), ("q_code", ct.c_int),
                ("k_buf", ct.c_void_p), ("k_code", ct.c_int),
                ("v_buf", ct.c_void_p), ("v_code", ct.c_int),
                ("q_norm", ct.POINTER(ct.c_float)), ("k_norm", ct.POINTER(ct.c_float)),
                ("o_buf", ct.c_void_p), ("o_code", ct.c_int),
                ("post_norm", ct.POINTER(ct.c_float)), ("rope_inv", ct.POINTER(ct.c_float)),
                ("nh", ct.c_int)]


class DenseP(ct.Structure):
    _fields_ = [("g", ct.c_void_p), ("u", ct.c_void_p), ("d", ct.c_void_p),
                ("cg", ct.c_int), ("cu", ct.c_int), ("cd", ct.c_int),
                ("n_ff", ct.c_int), ("h", ct.c_int)]


def qb(name):
    buf = np.frombuffer(bytes(T[name].data), np.uint8)
    KEEP.append(buf)
    return buf, CODE[T[name].tensor_type.name]


def fa(name):
    t = T[name]
    arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(-1)
    KEEP.append(arr)
    return arr


rope_inv = np.array([ROPE_BASE ** (-2.0 * t / ROT) for t in range(ROT // 2)], np.float32)
KEEP.append(rope_inv)

LAYERS = []          # per layer dict: kind, gw/fw ptr, dense dp, buffers
for il in range(NL):
    p = f"blk.{il}."
    if RECR[il]:
        w = GdnW()
        w.attn_norm = pf(fa(p + "attn_norm.weight"))
        w.qkv_buf, w.qkv_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_qkv.weight"))
        w.gate_buf, w.gate_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_gate.weight"))
        # ★ beta/alpha 必须预转置成 [H][NVH] 连续 float（C 侧按 t*NVH+i 读）。
        #   GGUF 的 ne=[2048,16]，ne0 最快 ⇒ flat[t + e*2048] ⇒ 先 reshape(NVH,H)（行=头）
        #   再转置才是 (t,e)；直接 reshape(H,NVH) 是错位的（cos 0.99 级的静默偏差，靠对账抓）。
        bw = np.ascontiguousarray(np.asarray(
            gguf.quants.dequantize(T[p + "ssm_beta.weight"].data, T[p + "ssm_beta.weight"].tensor_type),
            np.float32).reshape(NVH, H).T)
        aw = np.ascontiguousarray(np.asarray(
            gguf.quants.dequantize(T[p + "ssm_alpha.weight"].data, T[p + "ssm_alpha.weight"].tensor_type),
            np.float32).reshape(NVH, H).T)
        KEEP += [bw, aw]
        w.beta_wt, w.alpha_wt = pf(bw), pf(aw)
        w.ssm_a = pf(fa(p + "ssm_a"))
        w.ssm_dt = pf(fa(p + "ssm_dt.bias"))
        cw = np.frombuffer(bytes(T[p + "ssm_conv1d.weight"].data), np.float32).copy()   # [NQKV][4] tap 最快
        KEEP.append(cw)
        w.conv = pf(cw)
        w.ssm_norm = pf(fa(p + "ssm_norm.weight"))
        w.ssm_out_buf, w.ssm_out_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ssm_out.weight"))
        w.post_norm = pf(fa(p + "post_attention_norm.weight"))
        w.n_qkv, w.n_vh, w.d_in = NQKV, NVH, DI
        ssm = np.zeros(NVH * SV * SV, np.float32)
        tail = np.zeros(NQKV * (DC - 1), np.float32)
        buf = np.zeros(40000, np.float32)
        KEEP += [ssm, tail, buf, w]
        STATES += [ssm, tail]
        LAYERS.append(dict(kind="gdn", w=ct.cast(ct.pointer(w), ct.c_void_p),
                           ssm=pf(ssm), tail=pf(tail), buf=pf(buf)))
    else:
        w = FullW()
        w.attn_norm = pf(fa(p + "attn_norm.weight"))
        w.q_buf, w.q_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_q.weight"))
        w.k_buf, w.k_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_k.weight"))
        w.v_buf, w.v_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_v.weight"))
        w.q_norm = pf(fa(p + "attn_q_norm.weight"))
        w.k_norm = pf(fa(p + "attn_k_norm.weight"))
        w.o_buf, w.o_code = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_output.weight"))
        w.post_norm = pf(fa(p + "post_attention_norm.weight"))
        w.rope_inv = pf(rope_inv)
        w.nh = NH
        Kr = np.zeros(MAXT * NKV * HD_A, np.float32)
        Vc = np.zeros(MAXT * NKV * HD_A, np.float32)
        buf = np.zeros(40000, np.float32)
        KEEP += [Kr, Vc, buf, w]
        STATES += [Kr, Vc]
        LAYERS.append(dict(kind="full", w=ct.cast(ct.pointer(w), ct.c_void_p),
                           Kr=pf(Kr), Vc=pf(Vc), buf=pf(buf)))
    if N_EXP > 0:
        # MoE FFN（softmax 路由 + norm_w、无 w_scale —— 与 m6_granite_moe 同构）
        # + 带标量 sigmoid 门的共享专家（granite 没有的语义）
        ep = np.zeros(3 * N_EXP, np.uint64)
        ec = np.zeros(3 * N_EXP, np.int32)
        for idx, nm in ((0, "ffn_gate_exps.weight"), (1, "ffn_up_exps.weight"),
                        (2, "ffn_down_exps.weight")):
            buf = np.frombuffer(bytes(T[p + nm].data), np.uint8)
            KEEP.append(buf)
            per = buf.nbytes // N_EXP
            for e in range(N_EXP):
                ep[3 * e + idx] = buf.ctypes.data + e * per
                ec[3 * e + idx] = CODE[T[p + nm].tensor_type.name]
        gi = np.frombuffer(bytes(T[p + "ffn_gate_inp.weight"].data), np.float32).reshape(N_EXP, H).copy()
        shp = np.zeros(4, np.uint64)
        shc = np.zeros(4, np.int32)
        for idx, nm in enumerate(("ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                  "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight")):
            e = qb(p + nm)
            shp[idx], shc[idx] = e[0].ctypes.data, e[1]
        KEEP += [ep, ec, gi, shp, shc]
        LAYERS[-1]["moe"] = dict(
            ep=ep.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            ec=ec.ctypes.data_as(ct.POINTER(ct.c_int)),
            gi=pf(gi), shp=shp.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            shc=shc.ctypes.data_as(ct.POINTER(ct.c_int)),
            scratch=np.zeros(3 * N_USED * EXP_FFN + N_USED * H, np.float32),
            sg=np.zeros(1, np.float32), mo=np.zeros(H, np.float32), sh=np.zeros(2 * SHEXP_FFN, np.float32),
            shg=np.zeros(H, np.float32))
        KEEP += [LAYERS[-1]["moe"]["scratch"], LAYERS[-1]["moe"]["sg"],
                 LAYERS[-1]["moe"]["mo"], LAYERS[-1]["moe"]["sh"], LAYERS[-1]["moe"]["shg"]]
    else:
        # dense FFN（三层都是 f16）
        dp = DenseP()
        for fld, nm in (("g", "ffn_gate.weight"), ("u", "ffn_up.weight"), ("d", "ffn_down.weight")):
            e = qb(p + nm)
            setattr(dp, fld, e[0].ctypes.data)
            setattr(dp, "c" + fld, e[1])
        dp.n_ff, dp.h = FF, H
        KEEP.append(dp)
        LAYERS[-1]["dp"] = ct.cast(ct.pointer(dp), ct.c_void_p)

X = np.zeros(H, np.float32)
X2N = np.zeros(H, np.float32)
X2O = np.zeros(H, np.float32)
FFO = np.zeros(H, np.float32)
KRN = np.zeros(NKV * HD_A, np.float32)
VRN = np.zeros(NKV * HD_A, np.float32)
SCORES = np.zeros(NH * MAXT, np.float32)
LOGITS = np.zeros(VOCAB, np.float32)
HSC = np.zeros(H, np.float32)
ONORM = fa("output_norm.weight")
_hn = "output.weight" if "output.weight" in T else "token_embd.weight"
_HEAD, _HEAD_CODE = qb(_hn)   # ★ qwen35 2B 无 output（tied）；qwen35moe 有独立 output.weight
KEEP += [X, X2N, X2O, FFO, KRN, VRN, SCORES, LOGITS, HSC]


def _ffn(L, xn, out):
    """FFN 分派：dense → m6_dense_op；MoE → m6_granite_moe + 门控共享专家。
    共享专家门 = sigmoid(ffn_gate_inp_shexp · x)（单标量，×shexp 输出）。"""
    if "moe" not in L:
        M6E.m6_dense_op(pf(xn), L["dp"], pf(out))
        return
    M = L["moe"]
    M6E.m6_granite_moe(pf(xn), M["ep"], M["ec"], M["gi"], N_EXP, N_USED, EXP_FFN, H, H,
                       pf(M["scratch"]), pf(M["mo"]))
    # 共享专家：up/gate gemv → silu(gate)·up → down → ×sigmoid(标量门)
    M6E.m6_granite_shexp(pf(xn), M["shp"], M["shc"], pf(M["sg"]), pf(M["sh"]), pf(M["shg"]),
                         pf(out), SHEXP_FFN, H)
    # ★ FFN = moe_out + 门控共享专家。m6_granite_moe 写的是 M["mo"]，shexp 覆盖了 out
    #   —— 这一步漏掉时 FFN 只剩 shexp（norm 0.21 vs 0.71），逐层 cos 0.9 但分段"全对"
    #   （手工对账时手动做了相加，正好把引擎缺的这步掩盖了）。
    out[:] = M["mo"] + out


def forward(tid, pos):
    t = T["token_embd.weight"]
    X[:] = np.asarray(gguf.quants.dequantize(t.data[int(tid)], t.tensor_type), np.float32)
    for il, L in enumerate(LAYERS):
        if L["kind"] == "gdn":
            M6E.m6_gdn_attn(pf(X), L["w"], L["ssm"], L["tail"], L["buf"], pf(X2N), pf(X2O))
        else:
            M6E.m6_full_attn(pf(X), L["w"], L["Kr"], L["Vc"], pos, pos, L["buf"], pf(SCORES),
                             pf(X2N), pf(X2O), pf(KRN), pf(VRN))
            Kr = np.ctypeslib.as_array(L["Kr"], shape=(MAXT * NKV * HD_A,))
            Vc = np.ctypeslib.as_array(L["Vc"], shape=(MAXT * NKV * HD_A,))
            Kr[pos * NKV * HD_A:(pos + 1) * NKV * HD_A] = KRN
            Vc[pos * NKV * HD_A:(pos + 1) * NKV * HD_A] = VRN
        _ffn(L, X2N, FFO)
        X[:] = X2O + FFO
    M6E.m6_head_op(pf(X), pf(ONORM), H, EPS, _HEAD.ctypes.data, _HEAD_CODE, VOCAB, pf(LOGITS), pf(HSC))
    return LOGITS


def logits_of_x():
    return LOGITS


def reset():
    for b in STATES:
        b[...] = 0


# 对账探针（GPROBE=1：逐位置记录每层输出，与 dump 的 l_out-{il} 比）
PROBE_ON = os.environ.get("GPROBE", "0") == "1"
PROBE = {"by_pos": {}}


def _forward_probed(tid, pos):
    t = T["token_embd.weight"]
    X[:] = np.asarray(gguf.quants.dequantize(t.data[int(tid)], t.tensor_type), np.float32)
    rec = np.zeros((NL, H), np.float32) if PROBE_ON else None
    for il, L in enumerate(LAYERS):
        if L["kind"] == "gdn":
            M6E.m6_gdn_attn(pf(X), L["w"], L["ssm"], L["tail"], L["buf"], pf(X2N), pf(X2O))
        else:
            M6E.m6_full_attn(pf(X), L["w"], L["Kr"], L["Vc"], pos, pos, L["buf"], pf(SCORES),
                             pf(X2N), pf(X2O), pf(KRN), pf(VRN))
            Kr = np.ctypeslib.as_array(L["Kr"], shape=(MAXT * NKV * HD_A,))
            Vc = np.ctypeslib.as_array(L["Vc"], shape=(MAXT * NKV * HD_A,))
            Kr[pos * NKV * HD_A:(pos + 1) * NKV * HD_A] = KRN
            Vc[pos * NKV * HD_A:(pos + 1) * NKV * HD_A] = VRN
        _ffn(L, X2N, FFO)
        X[:] = X2O + FFO
        if PROBE_ON:
            rec[il] = X
    if PROBE_ON:
        PROBE["by_pos"][pos] = rec
    M6E.m6_head_op(pf(X), pf(ONORM), H, EPS, _HEAD.ctypes.data, _HEAD_CODE, VOCAB, pf(LOGITS), pf(HSC))
    return LOGITS


if PROBE_ON:
    forward = _forward_probed

if __name__ == "__main__":
    import time
    ids = [int(v) for v in os.environ.get("TOKS", "760,6511,314,9338,369").split(",")]
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
    dt = time.time() - t0
    print(f"[GEN] {N} 步 {dt:.2f}s → {N/max(1e-9, dt):.2f} tok/s")
    print(f"[GEN] 贪心 token: {gen}")

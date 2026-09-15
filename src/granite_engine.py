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

KEEP = []
M6 = ct.CDLL(os.path.join(BASE, "m6_engine.so"))
M6.m6_init_dl.argtypes = [ct.c_char_p] * 5
M6.m6_init_dl.restype = ct.c_int
_rc = M6.m6_init_dl(*[(os.path.join(BASE, "m5") + "/" + n).encode()
                      for n in ("m5_kern6.so", "m5_kern9.so", "m5_kern8.so", "m5_kern7.so", "m5_kernF.so")])
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

def ssm_layer0_forward(x_in, weights):
    """单步 S4D 层（t=0、状态零）。weights: {'in':…, 'conv':…, …} 由装载器填。

    与 llama.cpp 内核对过的语义：
      conv: tap 最快，状态帧在前、当前帧最后 ⇒ t=0 时只有 w[:, -1] 乘当前帧
      scan: dt = softplus(dt_pre + dt_bias)（无 clamp）；A={1,n_head} ⇒ 标量 dA；
            state[h,k,s] = x_dt[h,k]·B[s]（t=0）；y[h,k] = Σ_s state·C[s]
      门:   y = silu(z) · y（swiglu_split，z 是 in_proj 前 DI 段）
      D:    y[h,k] += x[h,k]·ssm_d[h]（按头）
    """
    w_in = weights["in"]                                # [6448, 1536]
    w_conv = weights["conv"]                            # [3328, 4]
    b_conv = weights["conv_b"]
    DI = D_INNER
    XBC = DI + 2 * N_GROUP * D_STATE
    zall = x_in @ w_in.T
    zseg = zall[:DI]; xBC = zall[DI:DI + XBC]; dt_pre = zall[DI + XBC:]
    conv_out = w_conv[:, -1] * xBC + b_conv             # t=0：状态零
    conv_out = conv_out / (1.0 + np.exp(-conv_out))     # silu
    x = conv_out[:DI]; B = conv_out[DI:DI + D_STATE]; Cseg = conv_out[DI + D_STATE:DI + 2 * D_STATE]
    dt = dt_pre + weights["dt_b"]
    A = weights["a"]; dsp = weights["d"]
    sdt = np.log1p(np.exp(dt))
    xh = x.reshape(NHEAD_OR_DT, HDIM_S4)
    hst = (xh * sdt[:, None])[:, :, None] * B[None, None, :]
    y_hk = np.tensordot(hst, Cseg, axes=([2], [0])) + xh * dsp[:, None]
    y = y_hk.T.reshape(-1)                              # {head_dim, n_head}，k 最快
    return (zseg / (1.0 + np.exp(-zseg))) * y

NHEAD_OR_DT = DT_RANK
HDIM_S4 = D_INNER // DT_RANK

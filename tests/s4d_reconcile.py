#!/usr/bin/env python3
"""S4D（granite-hybrid 的 SSM 段）链头对账 —— 用 llama.cpp 夹具在 pos=0（状态为零）dump 的
命名张量，把**每一层**的链头独立验证一遍（不需要背状态，因为 t=0 时 dA·s 项为零）。

夹具产物（hybrid/zaya_gdump，ZDUMP_POS=0）：
  attn_norm-{il}      层输入 = RMSNorm(inpL)（这正是 ssm_in 的输入）
  mamba2_y_add_d-{il} scan 读出 + D 跳连之后，形状 ne=(64,48) ⇒ 内存序 [head][dim]（dim 最快）
  mamba_out-{il}      ssm_out 投影之后（层最终输出）

对账做三件事（都由数据定，不靠读源码断言）：
  1. conv 的"当前帧 tap"到底是哪一列：k0 ∈ {0,1,2,3} 全试，看哪个对上
  2. y 的展开顺序：[head][dim]（dim 最快）还是它的转置
  3. 链尾（swiglu + grouped RMSNorm + ssm_out）是否一致
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf  # noqa: E402
import gguf_fast  # noqa: E402

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf")
D = os.environ.get("S4DDIR", "/tmp/grec0")

R = gguf_fast.FastGGUF(MODEL)
T = {t.name: t for t in R.tensors}


def kv(name, default=None):
    f = R.fields.get(f"granitehybrid.{name}")
    return f.value if f is not None else default


D_STATE = int(kv("ssm.state_size"))
D_CONV = int(kv("ssm.conv_kernel"))
DT_RANK = int(kv("ssm.time_step_rank"))
N_GROUP = int(kv("ssm.group_count"))
D_INNER = int(kv("ssm.inner_size"))
NL = int(kv("block_count"))
EPS = float(kv("attention.layer_norm_rms_epsilon", 1e-5))
ATTN_L = [il for il in range(NL) if f"blk.{il}.attn_q.weight" in T]
SSM_L = [il for il in range(NL) if f"blk.{il}.ssm_in.weight" in T]
XBC = D_INNER + 2 * N_GROUP * D_STATE
HDIM = D_INNER // DT_RANK
print(f"[CFG] 层={NL} SSM={len(SSM_L)} d_inner={D_INNER} dt_rank={DT_RANK} head_dim={HDIM} "
      f"d_state={D_STATE} n_group={N_GROUP} d_conv={D_CONV} xBC={XBC}")


def deq(name):
    """反量化 → 按 ggml 内存序（ne0 最快）reshape 成 [ne_last, ..., ne0]。"""
    t = T[name]
    arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32)
    return arr.reshape(tuple(int(v) for v in t.shape)[::-1])


def dump(name):
    """dump 文件名是 `<节点名>.<序号>.bin`（序号是图的节点号），按前缀 glob。"""
    import glob
    hits = [p for p in glob.glob(os.path.join(D, name + ".*.bin"))]
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: 命中 {len(hits)} 个文件 {hits[:3]}")
    with open(hits[0], "rb") as f:
        return np.frombuffer(f.read(), np.float32)


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)   # 数值稳定版


# ── 权重（只取一次，36 层共用；每层的权重在这里按需反量化并缓存）──
def layer_w(il):
    return dict(
        w_in=deq(f"blk.{il}.ssm_in.weight"),          # [6448, 1536]
        w_conv=deq(f"blk.{il}.ssm_conv1d.weight"),    # [3328, 4]
        b_conv=deq(f"blk.{il}.ssm_conv1d.bias"),      # [3328]
        a=deq(f"blk.{il}.ssm_a").ravel(),             # [48]
        d=deq(f"blk.{il}.ssm_d").ravel(),             # [48]
        dt_b=deq(f"blk.{il}.ssm_dt.bias").ravel(),    # [48]
        norm=deq(f"blk.{il}.ssm_norm.weight").ravel(),# [3072]
        w_out=deq(f"blk.{il}.ssm_out.weight"),        # [1536, 3072]
    )


def forward_t0(x_in, W, k0=3):
    """t=0、状态零。返回 (y_add_d[h,dim], mamba_out)。"""
    zall = x_in @ W["w_in"].T
    z = zall[:D_INNER]
    xBC = zall[D_INNER:D_INNER + XBC]
    dt_pre = zall[D_INNER + XBC:]
    conv_out = silu(W["w_conv"][:, k0] * xBC + W["b_conv"])
    xc = conv_out[:D_INNER]
    B = conv_out[D_INNER:D_INNER + D_STATE]
    Cc = conv_out[D_INNER + D_STATE:D_INNER + 2 * D_STATE]
    sdt = softplus(dt_pre + W["dt_b"])                     # t=0：dA·s=0，A 不参与
    xh = xc.reshape(DT_RANK, HDIM)
    st = sdt[:, None, None] * xh[:, :, None] * B[None, None, :]   # [head,dim,state]
    y_add = np.tensordot(st, Cc, axes=([2], [0])) + xh * W["d"][:, None]   # [head,dim]
    # 链尾：swiglu(z, y) → grouped RMSNorm → ssm_out
    for order in ("h_major", "transposed"):
        yflat = y_add.reshape(-1) if order == "h_major" else y_add.T.reshape(-1)
        g = silu(z) * yflat
        rms = np.sqrt(np.mean(g * g) + EPS)
        ng = (g / rms) * W["norm"]
        out = W["w_out"] @ ng
        yield order, y_add, yflat, out


print("\n=== 链头：conv tap 扫描（看哪一列是「当前帧」，同时定 y 的展开顺序）===")
ref0 = dump("mamba2_y_add_d-0").reshape(HDIM, DT_RANK)       # ne=(64,48) ⇒ [dim, head]
ref0_hk = ref0.T                                              # [head, dim]
x0 = dump("attn_norm-0")
W0 = layer_w(0)
for k0 in range(D_CONV):
    for order, y_add, yflat, out in forward_t0(x0, W0, k0):
        c_hk = cos(y_add, ref0_hk)
        c_T = cos(y_add.T, ref0_hk)
        print(f"  k0={k0} order={order:10s}  cos(y_add,ref)={c_hk:.6f}   "
              f"cos(y_add.T,ref)={c_T:.6f}")

print("\n=== 36 个 SSM 层：链头 + 链尾（k0 由上一节定，这里先把 4 个都跑一遍）===")
best = {}
for il in SSM_L:
    W = layer_w(il)
    x = dump(f"attn_norm-{il}")
    ref = dump(f"mamba2_y_add_d-{il}").reshape(HDIM, DT_RANK).T      # [head, dim]
    ref_out = dump(f"mamba_out-{il}")
    row = []
    for k0 in range(D_CONV):
        for order, y_add, yflat, out in forward_t0(x, W, k0):
            row.append((k0, order, cos(y_add, ref), cos(out, ref_out)))
    for k0, order, c1, c2 in row:
        best.setdefault((k0, order), []).append((c1, c2, il))

print(f"{'k0':>3} {'order':>10} {'链头 cos 中位':>13} {'链头 min':>10} {'链尾 cos 中位':>13} {'链尾 min':>10}")
for (k0, order), v in sorted(best.items()):
    c1 = np.array([a for a, b, il in v])
    c2 = np.array([b for a, b, il in v])
    print(f"{k0:>3} {order:>10} {np.median(c1):>13.6f} {c1.min():>10.6f} "
          f"{np.median(c2):>13.6f} {c2.min():>10.6f}")

# 选最优组合，打印最差的几层，看是不是"全对"还是"普遍不行"
kk = max(best, key=lambda k: np.median([a for a, b, il in best[k]]))
print(f"\n=== 最优组合 k0={kk[0]} order={kk[1]} 的逐层明细（按链头 cos 升序，只列最差 6 层）===")
v = sorted(best[kk])
for c1, c2, il in v[:6]:
    print(f"  blk.{il:<3} 链头 cos={c1:.6f} 链尾 cos={c2:.6f}")
print(f"  ... 最好一层: blk.{v[-1][2]} 链头 cos={v[-1][0]:.6f}")

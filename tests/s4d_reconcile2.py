#!/usr/bin/env python3
"""S4D 链头对账（正确版）—— 逐位置递推，验 conv 历史帧顺序 + A/dt 衰减 + 状态更新次序。

夹具：对 pos=0..P 各跑一次 `ZDUMP_POS=p zaya_gdump ...`（同一 token 序列，逐 token decode），
于是每一层的输入（attn_norm-{il}）与输出（mamba2_y_add_d-{il} / mamba_out-{il}）在每个位置都有。

★ 索引约定（第一版脚本就是在这里错的）：dump 的内存序是 ne0 最快 = [dim(64) 最快, head(48)]，
  所以 flat.reshape(48,64) 才是 [head, dim]；写成 flat.reshape(64,48).T 等于自己又转置了一次。
"""
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf  # noqa: E402
import gguf_fast  # noqa: E402

MODEL = "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf"
GD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zaya_gdump")
LDP = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"
TOKS = "100,101,102,103,104,105,106,107"
P = int(os.environ.get("S4DP", "3"))          # 对账到第几个位置

R = gguf_fast.FastGGUF(MODEL)
T = {t.name: t for t in R.tensors}
kv = lambda n, d=None: (R.fields[f"granitehybrid.{n}"].value if f"granitehybrid.{n}" in R.fields else d)
D_STATE, D_CONV = int(kv("ssm.state_size")), int(kv("ssm.conv_kernel"))
DT_RANK, N_GROUP = int(kv("ssm.time_step_rank")), int(kv("ssm.group_count"))
D_INNER, NL = int(kv("ssm.inner_size")), int(kv("block_count"))
EPS = float(kv("attention.layer_norm_rms_epsilon", 1e-5))
XBC = D_INNER + 2 * N_GROUP * D_STATE
HDIM = D_INNER // DT_RANK
SSM_L = [il for il in range(NL) if f"blk.{il}.ssm_in.weight" in T]

# ── 夹具：每个位置跑一次 ──
dirs = {}
for p in range(P + 1):
    d = f"/tmp/grec{p}"
    dirs[p] = d
    if os.path.isdir(d) and glob.glob(d + "/mamba2_y_add_d-*"):
        continue
    os.makedirs(d, exist_ok=True)
    print(f"[dump] pos={p} -> {d}", flush=True)
    subprocess.run([GD, MODEL, TOKS, d], env={**os.environ, "LD_LIBRARY_PATH": LDP, "ZDUMP_POS": str(p)},
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def dump(p, name):
    hits = glob.glob(os.path.join(dirs[p], name + ".*.bin"))
    if len(hits) != 1:
        raise FileNotFoundError(f"pos={p} {name}: {len(hits)} 个命中")
    return np.frombuffer(open(hits[0], "rb").read(), np.float32)


def deq(n):
    t = T[n]
    return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(
        tuple(int(v) for v in t.shape)[::-1])


cos = lambda a, b: float(np.dot(np.ravel(a) / np.linalg.norm(a), np.ravel(b) / np.linalg.norm(b)))
silu = lambda x: x / (1.0 + np.exp(-x))
softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def run_layer(il, P):
    """对第 il 层跑 pos=0..P 的递推，返回每位置的 (y_add[h,dim], mamba_out, z, y_scan, sdt)。"""
    W = dict(w_in=deq(f"blk.{il}.ssm_in.weight"), w_conv=deq(f"blk.{il}.ssm_conv1d.weight"),
             b_conv=deq(f"blk.{il}.ssm_conv1d.bias"), a=deq(f"blk.{il}.ssm_a").ravel(),
             d=deq(f"blk.{il}.ssm_d").ravel(), dt_b=deq(f"blk.{il}.ssm_dt.bias").ravel(),
             norm=deq(f"blk.{il}.ssm_norm.weight").ravel(), w_out=deq(f"blk.{il}.ssm_out.weight"))
    hist = np.zeros((D_CONV - 1, XBC), np.float32)      # conv 状态：最近 d_conv-1 帧，初始全零
    hst = np.zeros((DT_RANK, HDIM, D_STATE), np.float32)  # ssm 状态
    out = []
    for p in range(P + 1):
        x_in = dump(p, f"attn_norm-{il}")
        zall = x_in @ W["w_in"].T
        z = zall[:D_INNER]
        xBC = zall[D_INNER:D_INNER + XBC]
        sdt = softplus(zall[D_INNER + XBC:] + W["dt_b"])       # dt = softplus(pre + bias)
        s = np.vstack([hist, xBC[None, :]])                    # [d_conv, XBC]，当前帧在最后
        conv_out = silu((W["w_conv"].T * s).sum(axis=0) + W["b_conv"])   # w_conv[ch,tap] × s[tap,ch]
        hist = s[1:]                                          # 丢掉最老的一帧
        xc, B, Cc = (conv_out[:D_INNER], conv_out[D_INNER:D_INNER + D_STATE],
                     conv_out[D_INNER + D_STATE:D_INNER + 2 * D_STATE])
        xh = xc.reshape(DT_RANK, HDIM)
        dA = np.exp(sdt * W["a"])                             # A={1,48} ⇒ 每头一个标量衰减
        hst = hst * dA[:, None, None] + (xh * sdt[:, None])[:, :, None] * B[None, None, :]
        y_add = np.tensordot(hst, Cc, axes=([2], [0])) + xh * W["d"][:, None]
        g = silu(z) * y_add.reshape(-1)                       # ★ h 主序展开（不是转置！）
        rms = np.sqrt(np.mean(g * g) + EPS)
        out.append((y_add, W["w_out"] @ ((g / rms) * W["norm"]), z, np.tensordot(hst, Cc, axes=([2], [0]))))
    return out


print(f"\n{'pos':>3} {'层数':>4} {'链头 cos 中位':>13} {'链头 min':>11} {'链头 max|Δ|':>12} "
      f"{'链尾 cos 中位':>13} {'链尾 min':>11}")
for p in range(P + 1):
    c1, c2, dmax = [], [], []
    for il in SSM_L:
        mine = run_layer(il, p)
        y_add, m_out = mine[p][0], mine[p][1]
        ref = dump(p, f"mamba2_y_add_d-{il}").reshape(DT_RANK, HDIM)   # ★ [head, dim]
        c1.append(cos(y_add, ref))
        dmax.append(float(np.abs(y_add - ref).max()))
        c2.append(cos(m_out, dump(p, f"mamba_out-{il}")))
    print(f"{p:>3} {len(c1):>4} {np.median(c1):>13.6f} {min(c1):>11.6f} {max(dmax):>12.2e} "
          f"{np.median(c2):>13.6f} {min(c2):>11.6f}")

print("\n=== 对照：若把 y 展开写成转置（旧实现的写法），链尾会掉到多少 ===")
il = SSM_L[0]
mine = run_layer(il, P)
z, y_add = mine[P][2], mine[P][0]
ref_out = dump(P, f"mamba_out-{il}")
W = dict(norm=deq(f"blk.{il}.ssm_norm.weight").ravel(), w_out=deq(f"blk.{il}.ssm_out.weight"))
for tag, yflat in (("h_major（正确）", y_add.reshape(-1)), ("transposed（旧代码）", y_add.T.reshape(-1))):
    g = silu(z) * yflat
    o = W["w_out"] @ ((g / np.sqrt(np.mean(g * g) + EPS)) * W["norm"])
    print(f"  {tag:22s} 链尾 cos = {cos(o, ref_out):.6f}")

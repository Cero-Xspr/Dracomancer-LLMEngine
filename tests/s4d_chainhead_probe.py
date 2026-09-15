#!/usr/bin/env python3
"""链头矛盾的判决性诊断：链尾 0.9997（说明我的 z 与 y 都对）与链头 cos=0.0005（几乎正交）不能同时成立。
于是问三件事：
  A. 用**参考 dump 的** y 走链尾，能不能对上参考 mamba_out？（对 ⇒ 参考 y 自洽，我的 y 才是错的）
  B. 链尾对 y 到底敏不敏感？（不敏感 ⇒ 上一节的"链尾 0.9997"是空的，不能当证据）
  C. 参考 y 到底是什么：和 z / conv 输出 / x·D / 各层的 ref 分别是什么关系？
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf  # noqa: E402
import gguf_fast  # noqa: E402

MODEL = "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf"
D = os.environ.get("S4DDIR", "/tmp/grec0")
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
print(f"[CFG] SSM 层 {SSM_L}")


def deq(n):
    t = T[n]
    return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(
        tuple(int(v) for v in t.shape)[::-1])


def dump(name):
    hits = glob.glob(os.path.join(D, name + ".*.bin"))
    assert len(hits) == 1, (name, hits)
    return np.frombuffer(open(hits[0], "rb").read(), np.float32)


cos = lambda a, b: float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
silu = lambda x: x / (1.0 + np.exp(-x))


def tail(z, yflat, W):
    g = silu(z) * yflat
    rms = np.sqrt(np.mean(g * g) + EPS)
    return W["w_out"] @ ((g / rms) * W["norm"])


il = 0
W = dict(w_out=deq(f"blk.{il}.ssm_out.weight"), norm=deq(f"blk.{il}.ssm_norm.weight").ravel())
x_in = dump(f"attn_norm-{il}")
ref_y_flat = dump(f"mamba2_y_add_d-{il}")          # 内存序 [dim,head]（dim 最快）⇒ h 主序展开
ref_out = dump(f"mamba_out-{il}")
zall = x_in @ deq(f"blk.{il}.ssm_in.weight").T
z = zall[:D_INNER]

print("\n[A] 用参考 y 走链尾 vs 参考 mamba_out")
print(f"    cos = {cos(tail(z, ref_y_flat, W), ref_out):.6f}")

print("\n[B] 链尾对 y 的敏感度（把 y 换成同范数的随机/全 1/置换）")
rng = np.random.default_rng(0)
base = cos(tail(z, ref_y_flat, W), ref_out)
print(f"    参考 y            : {base:.6f}")
print(f"    随机 y（同范数）  : {cos(tail(z, rng.normal(size=ref_y_flat.size).astype(np.float32) * np.linalg.norm(ref_y_flat) / np.sqrt(ref_y_flat.size), W), ref_out):.6f}")
print(f"    全 1 y            : {cos(tail(z, np.ones_like(ref_y_flat), W), ref_out):.6f}")
print(f"    把 y 置零         : {cos(tail(z, np.zeros_like(ref_y_flat), W), ref_out):.6f}")

print("\n[C] 参考 y 与哪些量相关（第 0 层）")
ref_y_hk = ref_y_flat.reshape(DT_RANK, HDIM)      # [head, dim]
conv_w = deq(f"blk.{il}.ssm_conv1d.weight")
b_conv = deq(f"blk.{il}.ssm_conv1d.bias")
xBC = zall[D_INNER:D_INNER + XBC]
dt_pre = zall[D_INNER + XBC:]
conv_out = silu(conv_w[:, 3] * xBC + b_conv)
xc, B, Cc = conv_out[:D_INNER], conv_out[D_INNER:D_INNER + D_STATE], \
    conv_out[D_INNER + D_STATE:D_INNER + 2 * D_STATE]
sdt = np.log1p(np.exp(dt_pre + deq(f"blk.{il}.ssm_dt.bias").ravel()))
xh = xc.reshape(DT_RANK, HDIM)
st = sdt[:, None, None] * xh[:, :, None] * B[None, None, :]
scan = np.tensordot(st, Cc, axes=([2], [0]))
skip = xh * deq(f"blk.{il}.ssm_d").ravel()[:, None]
print(f"    cos(ref_y, 我的 y_add)      = {cos(ref_y_hk.ravel(), (scan + skip).ravel()):.6f}")
print(f"    cos(ref_y, 我的 scan 部分)  = {cos(ref_y_hk.ravel(), scan.ravel()):.6f}")
print(f"    cos(ref_y, 我的 x·D 部分)   = {cos(ref_y_hk.ravel(), skip.ravel()):.6f}")
print(f"    cos(ref_y, xc 原样)         = {cos(ref_y_hk.ravel(), xh.ravel()):.6f}")
print(f"    cos(ref_y, z 前 3072)       = {cos(ref_y_hk.ravel(), z):.6f}")
print(f"    cos(ref_y, conv_out 前 3072)= {cos(ref_y_hk.ravel(), xc):.6f}")
print(f"    ‖ref_y‖={np.linalg.norm(ref_y_hk):.4f}  ‖我的 y‖={np.linalg.norm(scan+skip):.4f}  "
      f"‖scan‖={np.linalg.norm(scan):.4f}  ‖skip‖={np.linalg.norm(skip):.4f}")

print("\n[D] 参考 y 是不是别的层的？(ref_y[0] 与全部 36 层 ref 的 cos)")
scores = [(cos(ref_y_flat, dump(f"mamba2_y_add_d-{j}")), j) for j in SSM_L]
scores.sort(reverse=True)
print("    top5:", [(round(c, 4), j) for c, j in scores[:5]])

print("\n[E] 参考 y 是不是「我自己算的其他层」？(ref_y[0] 与 全部36层用各自输入算出的 y 的 cos)")
best = []
for j in SSM_L:
    Wj = deq(f"blk.{j}.ssm_in.weight")
    xj = dump(f"attn_norm-{j}")
    zj = xj @ Wj.T
    cw = deq(f"blk.{j}.ssm_conv1d.weight")
    bb = deq(f"blk.{j}.ssm_conv1d.bias")
    co = silu(cw[:, 3] * zj[D_INNER:D_INNER + XBC] + bb)
    xj2 = co[:D_INNER].reshape(DT_RANK, HDIM)
    Bj = co[D_INNER:D_INNER + D_STATE]
    Cj = co[D_INNER + D_STATE:D_INNER + 2 * D_STATE]
    sd = np.log1p(np.exp(zj[D_INNER + XBC:] + deq(f"blk.{j}.ssm_dt.bias").ravel()))
    stj = sd[:, None, None] * xj2[:, :, None] * Bj[None, None, :]
    yj = (np.tensordot(stj, Cj, axes=([2], [0])) + xj2 * deq(f"blk.{j}.ssm_d").ravel()[:, None])
    best.append((cos(ref_y_flat, yj.ravel()), j))
best.sort(reverse=True)
print("    top5:", [(round(c, 4), j) for c, j in best[:5]])

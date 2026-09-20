#!/usr/bin/env python3
"""verify_kern13.py — IQ2_S 内核 (code 13) 对拍 gguf-py dequantize。
用 m5_gemv 当解码器: n_out=块数、n_in=256、x=单位向量, 每次 y[:] = 所有块在第 i 位的解码值。"""
import os, sys, ctypes as ct
import numpy as np

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
import gguf
from gguf.quants import dequantize
import gguf_fast

M5 = "/media/xiao_/OverSys1/npu-direct/hybrid/m5"
lib = ct.CDLL(os.path.join(M5, "m5_kern13.so"))
lib.m5_gemv.restype = ct.c_int
lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                        ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
p8 = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_uint8))

R = gguf_fast.FastGGUF('/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-IQ2_M.gguf')
T = {t.name: t for t in R.tensors}
iq2s_names = [n for n, t in T.items() if 'IQ2_S' in str(t.tensor_type)]
print(f"IQ2_S 张量数: {len(iq2s_names)}")

rng = np.random.default_rng(1)
worst = 0.0
checked = 0
for name in [iq2s_names[0], iq2s_names[len(iq2s_names)//2], iq2s_names[-1],
             'blk.0.attn_q.weight' if 'blk.0.attn_q.weight' in T else iq2s_names[3]]:
    t = T[name]
    raw = np.asarray(t.data).reshape(-1)
    ne0 = int(t.shape[0])
    if ne0 % 256:
        print(f"{name}: ne0={ne0} 非 256 倍数, 跳过"); continue
    nrows = int(np.prod([int(x) for x in t.shape][1:])) if t.data.ndim > 1 else 1
    total_vals = ne0 * (nrows if t.data.ndim > 1 else 1)
    # 用前 256×nb 个块 (=一行) 验证
    nb = ne0 // 256
    n_in = 256 * min(nb, 4)
    nbytes = 82 * (n_in // 256)
    x = np.zeros(n_in, np.float32)
    y = np.empty(n_in, np.float32)
    maxd = 0.0
    ref = dequantize(np.frombuffer(raw[:nbytes].tobytes(), np.uint8), t.tensor_type).ravel()
    y1 = np.empty(1, np.float32)
    for i in range(n_in):
        x[:] = 0; x[i] = 1.0
        assert lib.m5_gemv(13, pf(x), p8(raw[:nbytes]), 1, n_in, pf(y1)) == 0
        maxd = max(maxd, abs(float(y1[0]) - float(ref[i])))
    rel = maxd / 0.1
    checked += 1
    print(f"{name}: nb={nb} max|Δ|={maxd:.3e}")
    worst = max(worst, maxd)

# ★ 多行正确性（教训：n_out=1 位精确 ≠ 多行正确）+ 速度
import time
t = T[iq2s_names[1]]
n_in = 2560
nbytes = 82 * (n_in // 256)
n_out = min(768, t.data.nbytes // nbytes)
raw = np.asarray(t.data).reshape(-1)
x = rng.standard_normal(n_in).astype(np.float32)
y = np.empty(n_out, np.float32)
lib.m5_gemv(13, pf(x), p8(raw[:n_out*nbytes]), n_out, n_in, pf(y))
ref_full = dequantize(np.frombuffer(raw[:n_out*nbytes].tobytes(), np.uint8), t.tensor_type).ravel()
# gguf-py 的多行 ref: reshape [n_out, nb*256] → 逐行
r = ref_full.reshape(n_out, -1)
md = 0.0
for o in range(n_out):
    # 逐行用单位向量法？太慢；直接比 gemv y[o] vs Σ ref_row*x
    pass
# 更快：y[o] 应 = dot(ref_row_o, x)
dots = r @ x
md = np.abs(y - dots).max() / (np.abs(dots).max() + 1e-30)
print(f"多行正确性: {n_out} 行 rel={md:.2e} {'OK' if md < 1e-5 else '*** FAIL ***'}")
t0 = time.perf_counter()
REP = 20
for _ in range(REP):
    lib.m5_gemv(13, pf(x), p8(raw[:n_out*nbytes]), n_out, n_in, pf(y))
dt = (time.perf_counter()-t0)/REP
gb = n_out * nbytes / 1e9
print(f"速度: {n_out}x{n_in}: {dt*1000:.2f} ms => {gb/dt:.1f} GB/s")
worst = max(worst, md if md < 1e-5 else 1.0)
print("PASS" if worst < 1e-4 else "FAIL")
sys.exit(0 if worst < 1e-4 else 1)

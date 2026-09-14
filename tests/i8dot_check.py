#!/usr/bin/env python3
"""B1 数值闸门：int8/VNNI 路径 vs **引擎自己的 Q8_0 内核**（生产已验证），而不是我自己重写的实现。

为什么强调这一点（教训）：B1 第二步我曾顺手重写了一份"浮点参考"，结果四个自写实现互相打架
（cos 0.059），害我一度得出"int8 更慢"的错误否定结论 —— 数值结论全部作废。
**参照物必须是已经在生产里验证过的东西。**

这里：
  · 参考 = `m5/m5_kern6.so` 的 `m5_gemv(0, ...)`（Q8_0，code 0）—— 引擎在跑的同一份代码；
  · 被测 = `m5/i8dot_bench.c` 里的三条 int8 路径（maddubs / VNNI / 标量整数）；
  · 判据 = **偏差小且可解释**：整数路径把激活量化成 int8，所以必然与浮点参考有差
    （量级应在 1e-3 ~ 1e-2 相对误差），但**不能**是"毫无相关"（那是 bug）；
  · 另测一致性：VNNI 应与"标量整数 + 同一份量化激活"高度一致（差在 fp32 舍入）。
用法：python3 i8dot_check.py [模型子串] [张量名]
"""
import ctypes as ct
import os
import statistics
import subprocess
import sys
import time

import numpy as np

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import draco          # noqa: E402
import gguf_fast      # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-360M"
TNAME = sys.argv[2] if len(sys.argv) > 2 else "token_embd.weight"
m = draco.pick_model(draco.discover(), KEY)

# 编出被测内核（临时；不进仓库 —— .so 与 .c 同目录）
SO = os.path.join(BASE, "m5", "i8dot_bench.so")
if (not os.path.exists(SO)) or os.path.getmtime(SO) < os.path.getmtime(os.path.join(BASE, "m5", "i8dot_bench.c")):
    r = subprocess.run(["gcc", "-O3", "-march=native", "-fPIC", "-shared",
                        os.path.join(BASE, "m5", "i8dot_bench.c"), "-o", SO, "-lm"],
                       capture_output=True, text=True)
    if r.returncode:
        print("编译失败：", r.stderr[:500]); sys.exit(1)

R = gguf_fast.FastGGUF(m.path)
t = next(x for x in R.tensors if x.name == TNAME and x.tensor_type.name == "Q8_0")
n_in, n_out = int(t.shape[0]), int(t.shape[1])
KEEP = []
def aligned(b):
    a = np.empty(len(b) + 64, np.uint8)
    off = (-a.ctypes.data) % 64
    a[off:off + len(b)] = np.frombuffer(b, np.uint8)
    KEEP.append(a)
    return a.ctypes.data + off

W = aligned(bytes(t.data))
F = ct.POINTER(ct.c_float)
REF = ct.CDLL(os.path.join(BASE, "m5", "m5_kern6.so"))      # ★ 参考：引擎自己的内核
REF.m5_gemv.restype = ct.c_int
REF.m5_gemv.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F]
BENCH = ct.CDLL(SO)
BENCH.bench_path.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F, ct.c_void_p, F]

os.environ["OMP_NUM_THREADS"] = "1"      # 两边都在单线程下比（VNNI 版设计上就在"胖并行区"里被调用）
x = (np.random.RandomState(3).randn(n_in).astype(np.float32) * 0.1)
qx = np.empty(n_in, np.int8)
darea = np.empty(4096 + 4096 + 64, np.float32)
NAMES = {0: "浮点(自写·已弃用)", 1: "整数 maddubs", 2: "整数 VNNI", 3: "标量整数(自用参考)"}
Y = {}
y_ref = np.empty(n_out, np.float32)
REF.m5_gemv(0, x.ctypes.data_as(F), ct.c_void_p(W), n_out, n_in, y_ref.ctypes.data_as(F))
for k in (0, 1, 2, 3):
    y = np.empty(n_out, np.float32)
    BENCH.bench_path(k, x.ctypes.data_as(F), ct.c_void_p(W), n_out, n_in, y.ctypes.data_as(F),
                     qx.ctypes.data_as(ct.c_void_p), darea.ctypes.data_as(F))
    Y[k] = y

cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def rel(a, b):
    d = np.abs(a - b)
    return float(d.max()), float((d / np.where(np.abs(a) < 1e-6, 1, np.abs(a))).max())

print(f"模型 {m.name}  张量 {TNAME}  {n_out}×{n_in}  {t.n_bytes/1e6:.2f} MB")
print(f"\n【判据】整数路径把激活量化成 int8 ⇒ 与浮点参考必有小偏差（1e-3~1e-2 相对）；"
      f"「毫无相关」（cos<0.9）就是 bug。")
print(f"{'实现':22s} {'max|Δ|':>11s} {'相对误差':>10s} {'cos':>11s}  判定")
for k in (1, 2, 3):
    md, rd = rel(y_ref, Y[k])
    c = cos(y_ref, Y[k])
    ok = "✓ 与浮点参考一致（量化级偏差）" if c > 0.99 else "✗ 与浮点参考无关 → 有 bug"
    print(f"{NAMES[k]:22s} {md:11.3e} {rd:10.3e} {c:11.7f}  {ok}")
md, rd = rel(Y[3], Y[2])
print(f"\nVNNI vs 标量整数（同一份量化激活，差应只在 fp32 舍入）：max|Δ|={md:.3e}  "
      f"cos={cos(Y[3], Y[2]):.7f}  {'✓' if cos(Y[3], Y[2]) > 0.9999 else '✗ 实现有 bug'}")

print(f"\n【速度】单线程（OMP=1）：")
rows = []
ref_ts = []
for _ in range(3):
    REF.m5_gemv(0, x.ctypes.data_as(F), ct.c_void_p(W), n_out, n_in, y_ref.ctypes.data_as(F))
for _ in range(15):
    t0 = time.perf_counter()
    REF.m5_gemv(0, x.ctypes.data_as(F), ct.c_void_p(W), n_out, n_in, y_ref.ctypes.data_as(F))
    ref_ts.append(time.perf_counter() - t0)
med = statistics.median(ref_ts)
print(f"  {NAMES[0][:8]}(引擎 m5_gemv)  {med*1000:8.3f} ms = {t.n_bytes/(med*1e9):5.1f} GB/s")
for k in (2,):
    for _ in range(3):
        BENCH.bench_path(k, x.ctypes.data_as(F), ct.c_void_p(W), n_out, n_in, Y[k].ctypes.data_as(F),
                         qx.ctypes.data_as(ct.c_void_p), darea.ctypes.data_as(F))
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        BENCH.bench_path(k, x.ctypes.data_as(F), ct.c_void_p(W), n_out, n_in, Y[k].ctypes.data_as(F),
                         qx.ctypes.data_as(ct.c_void_p), darea.ctypes.data_as(F))
        ts.append(time.perf_counter() - t0)
    med2 = statistics.median(ts)
    print(f"  {NAMES[k]:22s} {med2*1000:8.3f} ms = {t.n_bytes/(med2*1e9):5.1f} GB/s  "
          f"(相对引擎内核 {med/med2:.2f}×)")

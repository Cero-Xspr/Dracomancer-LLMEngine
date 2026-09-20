#!/usr/bin/env python3
"""bench_hetero_bw2.py — F0.5：严谨的异构聚合带宽测量（同窗采样）。

窗口定义：llama-bench 单次长跑（-n 2048，约 60s）的起止时间为重叠窗 W。
CPU 侧：进程内循环调 m5_bw_scan（零 spawn 间隙），按调用区间与 W 的重叠时间
        折算"窗口内 CPU 字节流量 / 窗口时长"。
iGPU 侧：llama-bench 报告的 t/s × 模型字节数（tg 阶段每 token 读全量权重）。
输出：两侧窗口内速率、聚合值、与 89.6 GB/s 名义总线（DDR5-5600 双通道）的比值。
"""
import os, sys, time, subprocess
import ctypes
import numpy as np

VK = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-vk"
MODEL = "/media/Data-1/gguf/Qwen3.5-2B-Q8_0.gguf"
MODEL_BYTES = 1.92 * 2**30          # llama-bench 报的 1.92 GiB
NGEN = int(os.environ.get("NGEN", "2048"))
WINDOW_CPU = float(os.environ.get("WINDOW_CPU", "120"))

# ── CPU 侧：m5_bw_scan 进程内连续循环 ──
bw = ctypes.CDLL("/media/xiao_/OverSys1/npu-direct/hybrid/m5/m5_bw.so")
bw.m5_bw_scan.restype = ctypes.c_long
bw.m5_bw_scan.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
buf = np.full(4 * 1024 * 1024 * 1024, 0x5A, np.uint8)   # 4 GiB，写实页（np.zeros 是懒分配零页，读它=缓存空转）
ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
NBYTES = buf.size

os.environ.setdefault("OMP_NUM_THREADS", "20")

def cpu_loop_until(deadline, log):
    """循环扫描直到 deadline；每轮记录 (t0, t1, bytes)。返回区间列表。"""
    ivals = []
    while True:
        t0 = time.perf_counter()
        if t0 >= deadline:
            break
        bw.m5_bw_scan(ptr, NBYTES)
        t1 = time.perf_counter()
        ivals.append((t0, t1, NBYTES))
        log.append((t0, t1))
        if t1 >= deadline:
            break
    return ivals

print(f"[F0.5] 窗口 = llama-bench -n {NGEN} 全程（预计 ~{NGEN/33:.0f}s）+ CPU 连续读")

# ── 启动 iGPU 长跑 ──
bench = subprocess.Popen(
    [f"{VK}/bin/llama-bench", "-m", MODEL, "-p", "0", "-n", str(NGEN), "-ngl", "99"],
    cwd=VK, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

# 等 llama-bench 进入 tg 阶段（输出 tg 行即结束；开始时间用首次探测到进程稳定后）
time.sleep(6)                      # 模型加载+预热（2GB 已缓存，加载快）
bench_start = time.perf_counter()
ivals = cpu_loop_until(bench_start + WINDOW_CPU, [])
bench_end = time.perf_counter()
out = bench.communicate(timeout=300)[0]
bench_wall = None
for line in out.splitlines():
    if "tg128" in line or f"tg{NGEN}" in line:
        parts = line.split("|")
        rate = float(parts[-2].strip().split()[0])
        bench_wall = NGEN / rate
print(f"[F0.5] llama-bench 报告: {rate} t/s  (生成 {NGEN} tok, 全程均速)")

# CPU 窗口内速率：与 [bench_start, bench_start+bench_wall] 重叠的时间×字节
w0, w1 = bench_start, bench_start + (bench_wall or (bench_end - bench_start))
cpu_bytes = 0.0
for t0, t1, nbytes in ivals:
    o0, o1 = max(t0, w0), min(t1, w1)
    if o1 > o0:
        frac = (o1 - o0) / (t1 - t0)
        cpu_bytes += nbytes * frac
window = w1 - w0
cpu_rate = cpu_bytes / window / 1e9

igpu_rate = rate * MODEL_BYTES / 1e9
agg = cpu_rate + igpu_rate
NOMINAL = 89.6
print(f"\n窗口 {window:.1f}s（llama-bench 全程 {bench_wall:.1f}s）")
print(f"CPU  窗口内读带宽 : {cpu_rate:6.1f} GB/s   （单独基线 52.8）")
print(f"iGPU 窗口内流式   : {igpu_rate:6.1f} GB/s   （单独基线 32.66 t/s ≈ 68.8）")
print(f"聚合               : {agg:6.1f} GB/s   = 名义总线 89.6 的 {agg/NOMINAL*100:.0f}%")
print(f"CPU 单独聚合比     : {agg/52.8:.2f}×")
over = agg > NOMINAL
if over:
    print("⚠ 聚合超名义总线 ⇒ iGPU 均值≠窗口均值或存在读突发超额，需进一步分解")
else:
    print("✓ 聚合在名义总线内")

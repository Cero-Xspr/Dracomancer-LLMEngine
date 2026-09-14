#!/usr/bin/env python3
"""内核改动的严格 A/B（交替跑，取中位）—— 单次测量在这台机上有 5~10% 噪声，
不交替就分不清"优化"和"这一轮机器比较闲"。

用法：python3 bench_ab.py <模型子串> <线程列表> <每轮token> <轮数> <A.so> <B.so>
"""
import json
import os
import shutil
import statistics
import subprocess
import sys


def swap_so(src, dst):
    """★ 原子替换 .so：先把新内容写进同目录的临时文件再 os.replace。

    为什么不能直接 `shutil.copy`/`cp` 覆盖：目标一旦被**运行中的进程** dlopen/mmap 过
    （用户自己可能正开着 `draco.py chat`），cp 会先 truncate 目标 —— 那个瞬间正在运行的
    进程若刚好缺页，会拿到 SIGBUS 直接崩。os.replace 是原子换名，旧 inode 由内核
    保持到最后一个映射者退出，运行中的进程完全不受影响。
    """
    tmp = dst + ".new"
    shutil.copy(src, tmp)
    os.replace(tmp, dst)

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
SO = os.path.join(BASE, "m6_engine.so")
KEY = sys.argv[1]
THREADS = [int(x) for x in sys.argv[2].split(",")]
NGEN = int(sys.argv[3])
ROUNDS = int(sys.argv[4])
A_SO, B_SO = sys.argv[5], sys.argv[6]

sys.path.insert(0, BASE)
import draco  # noqa: E402
m = draco.pick_model(draco.discover(), KEY)

PROBE = r'''
import os, sys, time, json, statistics, ctypes as ct
import numpy as np
sys.path.insert(0, os.environ["BS_BASE"]); sys.path.insert(0, os.environ["BS_GGUF"])
os.environ["SKIP_BENCH"]="1"
import smol_engine as E
prof = E.M6E
prof.m6_prof_get.restype = ct.c_double; prof.m6_prof_cnt.restype = ct.c_long
N = int(os.environ["BS_N"])
def run(n):
    ts = []
    for i in range(n):
        t0 = time.perf_counter(); E.forward(100 + (i % 50), i); ts.append(time.perf_counter() - t0)
    return ts
run(10)
prof.m6_prof_reset()
ts = run(N)
print("BSJSON" + json.dumps({"ms": statistics.median(ts)*1000,
                             "attn": prof.m6_prof_get(1)*1000/N,
                             "ffn": prof.m6_prof_get(2)*1000/N}))
'''
env = dict(os.environ)
env.update(BS_BASE=BASE, BS_GGUF="/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py",
           MODEL=m.path, BS_N=str(NGEN), OMP_WAIT_POLICY="passive")

res = {("A", t): [] for t in THREADS}
res.update({("B", t): [] for t in THREADS})
for r in range(ROUNDS):
    for tag, so in (("A", A_SO), ("B", B_SO)):        # ★ 同一轮里交替：把机器负载漂移摊平
        swap_so(so, SO)
        for T in THREADS:
            env["OMP_NUM_THREADS"] = str(T)
            p = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("BSJSON")), None)
            if line:
                res[(tag, T)].append(json.loads(line[6:]))
swap_so(B_SO, SO)
try:
    sys.path.insert(0, BASE)
    from hwprobe import power_state
    print(power_state())
except Exception:
    pass
print(f"\n{ROUNDS} 轮交替，每轮 {NGEN} token（取各轮中位的再中位）\n")
print(f"{'线程':>4s} {'A ms':>8s} {'B ms':>8s} {'提速':>7s} | {'A attn':>7s} {'B attn':>7s} | {'A ffn':>7s} {'B ffn':>7s}")
for T in THREADS:
    a, b = res[("A", T)], res[("B", T)]
    if not a or not b:
        print(f"{T:4d}  数据不足"); continue
    ma = statistics.median(d["ms"] for d in a); mb = statistics.median(d["ms"] for d in b)
    aa = statistics.median(d["attn"] for d in a); ab = statistics.median(d["attn"] for d in b)
    fa = statistics.median(d["ffn"] for d in a); fb = statistics.median(d["ffn"] for d in b)
    print(f"{T:4d} {ma:8.2f} {mb:8.2f} {ma/mb:6.3f}× | {aa:7.2f} {ab:7.2f} | {fa:7.2f} {fb:7.2f}")

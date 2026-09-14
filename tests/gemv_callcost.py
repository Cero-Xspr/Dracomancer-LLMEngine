#!/usr/bin/env python3
"""B3 诊断：每 token 到底有多少次 gemv 调用、每次的固定开销是多少、总共值多少 ms。

为什么要先量：B1 那三轮的教训是"没确认瓶颈在哪就动手"。B3 的假设是"每算子 4~6µs × 200+ 次/token"，
但这是早期在少数形状上的印象分 —— 先把账算清楚：
  · **调用次数**：靠引擎自己的记录钩子（`M6_AUDIT=1` + field=7 读次数）在真推理里数，不靠推算；
  · **每次调用的实际耗时**：用**引擎自己的内核**（不是自写实现）在同一形状上单次计时；
  · **流式下限**：该形状的字节数 ÷ 该格式的**渐近带宽**（用大矩阵实测得到）；
  · **固定开销** = 实测 − 流式下限；再乘每 token 次数 ⇒ 这就是 B3 的预算上限。
用法：python3 gemv_callcost.py <模型子串> [线程数]
"""
import ctypes as ct
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import draco       # noqa: E402
import gguf_fast   # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-360M"
T = sys.argv[2] if len(sys.argv) > 2 else "8"
m = draco.pick_model(draco.discover(), KEY)
g = (getattr(m, "gguf_name", m.name) or "").lower()
ENG = "smol_engine" if "smollm2" in g else ("ling_engine" if "ling" in g else "zaya_gguf")

# ── ① 在真推理里数调用次数（引擎自己报，不推算）────────────────────────────
COLLECT = r'''
import os, sys, json, ctypes as ct
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
os.environ["M6_AUDIT"] = "1"; os.environ["SKIP_BENCH"] = "1"; os.environ["MODEL"] = os.environ["M"]
old = sys.argv; sys.argv = ["enz"]
E = __import__(os.environ["E"]); sys.argv = old
fwd = (lambda i, p: E.step(100 + (i % 50), p)) if os.environ["E"] == "zaya_gguf" else E.forward
NT = int(os.environ.get("NT", "4"))
for i in range(NT):
    fwd(i, i)
n = E.M6E.m6_audit_n()
recs = [[E.M6E.m6_audit_field(i, f) for f in range(9)] for i in range(n)]
print("BSJSON" + json.dumps({"nt": NT, "recs": recs}))
'''
r = subprocess.run([sys.executable, "-c", COLLECT], capture_output=True, text=True,
                   env={**os.environ, "M": m.path, "E": ENG, "OMP_NUM_THREADS": T, "NT": "4"})
line = next((l for l in r.stdout.splitlines() if l.startswith("BSJSON")), None)
if not line:
    print("采集失败：", r.stdout[-300:], r.stderr[-500:]); sys.exit(1)
d = json.loads(line[6:])
NT, recs = d["nt"], d["recs"]
try:
    sys.path.insert(0, BASE)
    from hwprobe import power_state
    print(power_state())
except Exception:
    pass
print(f"模型 {m.name}  引擎 {ENG}  线程 {T}  采样 {NT} 个 token")
print(f"引擎报告 {len(recs)} 个不同的 (格式, n_out, n_in) 调用组合\n")

CODENAME = {0: "Q8_0", 1: "IQ4_NL", 2: "IQ3_S", 3: "Q5_K", 4: "Q6_K", 5: "Q4_K",
            6: "IQ4_XS", 7: "F16", 8: "Q5_0", 9: "F32"}
LIBS = {0: "m5_kern6.so", 1: "m5_kern6.so", 2: "m5_kern6.so", 3: "m5_kern9.so",
        4: "m5_kern8.so", 5: "m5_kern7.so", 6: "m5_kern7.so", 7: "m5_kernF.so", 8: "m5_kern11.so"}
M6 = ct.CDLL(os.path.join(BASE, "m6_engine.so"))
M6.m6_rowbytes.restype = ct.c_size_t
M6.m6_rowbytes.argtypes = [ct.c_int, ct.c_int]
F = ct.POINTER(ct.c_float)
_cd = {}
def lib(code):
    f = LIBS.get(code)
    if f is None:
        return None
    if f not in _cd:
        h = ct.CDLL(os.path.join(BASE, "m5", f))
        h.m5_gemv.restype = ct.c_int
        h.m5_gemv.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F]
        _cd[f] = h
    return _cd[f]

R = gguf_fast.FastGGUF(m.path)
POOL = {}
for t in R.tensors:
    POOL.setdefault(t.tensor_type.name, bytes(t.data))

KEEP = []
def aligned(nb):
    a = np.empty(nb + 64, np.uint8)
    off = (-a.ctypes.data) % 64
    KEEP.append(a)
    return a.ctypes.data + off
def real_bytes(code, nb):
    p = POOL.get(CODENAME.get(code, ""), b"")
    if not p:
        return None
    return p[:nb] if len(p) >= nb else (p * (nb // len(p) + 1))[:nb]

# ── ② 渐近带宽（该格式的大矩阵，单次调用的下限参考）──────────────────────
print(f"{'格式':6s} {'形状':>13s} {'次/token':>8s} {'KB':>8s} {'实测µs':>8s} {'GB/s':>7s} "
      f"{'同格式渐近':>9s} {'固定开销µs':>9s} {'总µs/token':>10s}")
rows = []
for code, n_out, n_in, o0, o1, wmod, xmod, cnt, cnt_seg in sorted(recs, key=lambda r: -r[7]):
    L = lib(code)
    if L is None or n_in % 32 or n_in <= 0 or n_out <= 0:
        print(f"{CODENAME.get(code, '?'):6s} {n_out}×{n_in}  —— 无内核或形状不支持，跳过")
        continue
    rb = M6.m6_rowbytes(code, n_in)
    nb = n_out * rb
    wb = real_bytes(code, nb)
    if wb is None:
        continue
    W = aligned(nb)
    np.frombuffer((ct.c_uint8 * nb).from_address(W), dtype=np.uint8)[:] = np.frombuffer(wb, np.uint8)
    x = aligned(n_in * 4)
    np.frombuffer((ct.c_float * n_in).from_address(x), dtype=np.float32)[:] = \
        np.random.RandomState(1).randn(n_in).astype(np.float32) * 0.1
    y = np.empty(n_out, np.float32)
    for _ in range(5):
        L.m5_gemv(code, ct.cast(x, F), ct.c_void_p(W), n_out, n_in, y.ctypes.data_as(F))
    ts = []
    for _ in range(25):
        t0 = time.perf_counter()
        L.m5_gemv(code, ct.cast(x, F), ct.c_void_p(W), n_out, n_in, y.ctypes.data_as(F))
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    gbps = nb / med / 1e9
    rows.append(dict(code=code, n_out=n_out, n_in=n_in, cnt=cnt / NT, seg=cnt_seg / NT,
                     kb=nb / 1024, us=med * 1e6, gbps=gbps))
    print(f"{CODENAME.get(code, '?'):6s} {n_out}×{n_in:>6} 整调用 {cnt/NT:6.1f}/token "
          f"分段 {cnt_seg/NT:7.1f}/token  {nb/1024:7.1f}KB {med*1e6:7.2f}µs {gbps:6.1f}GB/s")

# 渐近带宽：按格式取"最大的那个形状"的 GB/s 当参考
best = {}
for r in rows:
    if r["code"] not in best or r["gbps"] > best[r["code"]]["gbps"]:
        best[r["code"]] = r
print(f"\n{'格式':6s} {'渐近 GB/s（取该格式最大形状）':>28s}   结论")
tot_over = 0.0
for code, r in sorted(best.items()):
    print(f"{CODENAME.get(code, '?'):6s} {r['gbps']:28.1f}   （{r['n_out']}×{r['n_in']}, "
          f"{r['kb']/1024:.1f}MB）")
print(f"\n【固定开销账】按每个形状的渐近带宽反推下限，超出部分×次数：")
print(f"{'格式':6s} {'形状':>13s} {'次/token':>8s} {'流式下限µs':>10s} {'实测µs':>8s} "
      f"{'开销µs/次':>9s} {'合计µs/token':>11s}")
for r in sorted(rows, key=lambda r: -(r["us"] - r["kb"] * 1024 / (best[r["code"]]["gbps"] * 1e9) * 1e6)):
    ref = best[r["code"]]["gbps"]
    floor_us = (r["kb"] * 1024) / (ref * 1e9) * 1e6
    over = r["us"] - floor_us
    tot = over * r["cnt"]
    tot_over += tot
    if over > 0.3:      # 只列有明显超出的
        shape = f"{r['n_out']}×{r['n_in']}"
        print(f"{CODENAME.get(r['code'], '?'):6s} {shape:>13s} {r['cnt']:8.1f} "
              f"{floor_us:10.2f} {r['us']:8.2f} {over:9.2f} {tot:11.1f}")
print(f"\n⇒ 全部形状的「超出流式下限」合计 ≈ **{tot_over/1000:.2f} ms/token**"
      f"（这是 B3 能争取的上限；对照：整 token 现在约 8~9 ms）")

#!/usr/bin/env python3
"""A1 正式版：**用引擎真实调用参数**审计 gemv 两个入口（gemv_any vs gemv_range_any）是否逐位等价。

与 gemv_entry_audit.py 的区别（那一版的缺陷）：
  它按 **GGUF 张量形状**推 n_out/n_in/指针 —— 一旦"引擎实际传的参数与形状映射不同"就会漏测/误测
  （2026-09-15 实测：v_b 的真实组合 Q6_K 128×512 就被漏掉，按形状推还读出过 NaN）。
本版改成：
  ① 让**引擎自己报出**每次 gemv 调用（C 侧 hook，M6_AUDIT=1；只记新出现的 (code,n_out,n_in) 组合）；
  ② 拿这些真实参数逐个重放：同一份随机权重/输入，分别调 `m5_gemv` 与 `m5_gemv_range(o0,o1)`；
  ③ 判据 = **逐位相同**（这两个入口理应做同样的乘加）。

用法：python3 gemv_audit_replay.py <模型子串>
"""
import ctypes as ct
import json
import os
import subprocess
import sys

import numpy as np

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import draco  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-360M"
m = draco.pick_model(draco.discover(), KEY)
g = (getattr(m, "gguf_name", m.name) or "").lower()
ENG = "smol_engine" if "smollm2" in g else ("ling_engine" if "ling" in g else "zaya_gguf")
print(f"模型：{m.name}   引擎模块：{ENG}")

# ── ① 让引擎在真实推理中报出所有 gemv 调用 ──────────────────────────────
COLLECT = r'''
import os, sys, json, ctypes as ct
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
os.environ["M6_AUDIT"] = "1"
os.environ["SKIP_BENCH"] = "1"
os.environ["MODEL"] = os.environ["M"]
old = sys.argv; sys.argv = ["enz"]
E = __import__(os.environ["E"]); sys.argv = old
fwd = (lambda i, p: E.step(100 + (i % 50), p)) if os.environ["E"] == "zaya_gguf" else E.forward
for i in range(int(os.environ.get("NT", "6"))):
    fwd(i, i)
n = E.M6E.m6_audit_n()
recs = [[E.M6E.m6_audit_field(i, f) for f in range(7)] for i in range(n)]
print("BSJSON" + json.dumps(recs))
'''
r = subprocess.run([sys.executable, "-c", COLLECT], capture_output=True, text=True,
                   env={**os.environ, "M": m.path, "E": ENG, "OMP_NUM_THREADS": "8", "NT": "6"})
line = next((l for l in r.stdout.splitlines() if l.startswith("BSJSON")), None)
if not line:
    print("采集失败：", r.stdout[-400:], r.stderr[-600:])
    sys.exit(1)
recs = json.loads(line[6:])
print(f"引擎在真实推理中共报告 {len(recs)} 个不同的 (格式, n_out, n_in) 调用组合\n")

# ── ② 逐个组合重放两个入口 ──────────────────────────────────────────────
LIBS = {  # code → 内核库（与 m6_layer.c 的分派、以及各引擎的 CODE 表一致）
    0: "m5_kern6.so", 1: "m5_kern6.so", 2: "m5_kern6.so",
    3: "m5_kern9.so", 4: "m5_kern8.so", 5: "m5_kern7.so", 6: "m5_kern7.so",
    7: "m5_kernF.so", 8: "m5_kern11.so",
}
NAME = {0: "Q8_0", 1: "IQ4_NL", 2: "IQ3_S", 3: "Q5_K", 4: "Q6_K", 5: "Q4_K", 6: "IQ4_XS",
        7: "F16", 8: "Q5_0"}
M6 = ct.CDLL(os.path.join(BASE, "m6_engine.so"))
M6.m6_rowbytes.restype = ct.c_size_t
M6.m6_rowbytes.argtypes = [ct.c_int, ct.c_int]
F = ct.POINTER(ct.c_float)
_cdll = {}
def lib_for(code):
    f = LIBS.get(code)
    if f is None:
        return None
    if f not in _cdll:
        h = ct.CDLL(os.path.join(BASE, "m5", f))
        h.m5_gemv.restype = ct.c_int
        h.m5_gemv.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F]
        h.m5_gemv_range.restype = ct.c_int
        h.m5_gemv_range.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F, ct.c_int, ct.c_int]
        _cdll[f] = h
    return _cdll[f]

# ★ 权重必须用**真实量化张量的字节**：随便填随机字节会让 f16 尺度位变成 NaN/次正规 ⇒ 全 NaN
#   （我第一版就是这么错的，6 个组合全报 nan）。
import gguf_fast
_R = gguf_fast.FastGGUF(m.path)
POOL = {}
for t in _R.tensors:
    POOL.setdefault(t.tensor_type.name, bytes(t.data))
CODENAME = {0: "Q8_0", 1: "IQ4_NL", 2: "IQ3_S", 3: "Q5_K", 4: "Q6_K", 5: "Q4_K",
            6: "IQ4_XS", 7: "F16", 8: "Q5_0"}
def real_bytes(code, nbytes):
    pool = POOL.get(CODENAME.get(code, ""), b"")
    if not pool:
        return None
    if len(pool) >= nbytes:
        return pool[:nbytes]
    return (pool * (nbytes // len(pool) + 1))[:nbytes]

KEEP = []
def aligned(nbytes):
    a = np.empty(nbytes + 64, np.uint8)
    off = (-a.ctypes.data) % 64
    KEEP.append(a)
    return a.ctypes.data + off

bad = []
print(f"{'格式':7s} {'n_out':>6s} {'n_in':>6s} {'o0':>5s} {'o1':>6s} {'W%64':>5s} {'逐位相同':>8s} {'max|Δ|':>10s}")
for code, n_out, n_in, o0, o1, wmod, xmod in recs:
    lib = lib_for(code)
    if lib is None:
        print(f"{NAME.get(code, '?'):7s} {n_out:6d} {n_in:6d} —— 没有对应内核，跳过")
        continue
    rb = M6.m6_rowbytes(code, n_in)
    nb = n_out * rb
    wb = real_bytes(code, nb)
    if wb is None:
        print(f"{NAME.get(code, '?'):7s} {n_out:6d} {n_in:6d} —— 模型里没有该格式的真实张量，跳过")
        continue
    W = aligned(nb)
    np.frombuffer((ct.c_uint8 * nb).from_address(W), dtype=np.uint8)[:] = np.frombuffer(wb, np.uint8)
    x = aligned(n_in * 4)
    np.frombuffer((ct.c_float * n_in).from_address(x), dtype=np.float32)[:] = \
        np.random.RandomState(7).randn(n_in).astype(np.float32) * 0.1
    y1 = np.full(n_out, np.nan, np.float32)
    y2 = np.full(n_out, np.nan, np.float32)
    lib.m5_gemv(code, ct.cast(x, F), ct.c_void_p(W), n_out, n_in, y1.ctypes.data_as(F))
    # 重放"分段调用"（这是引擎在胖并行区里的真实形态）：按 o0..o1 调用
    for o in range(o0, o1, 32):
        lib.m5_gemv_range(code, ct.cast(x, F), ct.c_void_p(W), n_out, n_in,
                          y2.ctypes.data_as(F), o, min(o + 32, o1))
    # ★ 只比 [o0, o1) 这一段：range 调用只负责这些行，其余行本来就是空的
    seg1, seg2 = y1[o0:o1], y2[o0:o1]
    same = np.array_equal(seg1, seg2)
    d = float(np.abs(seg1 - seg2).max())
    if not same:
        bad.append((NAME.get(code), n_out, n_in, o0, o1, d))
    print(f"{NAME.get(code, '?'):7s} {n_out:6d} {n_in:6d} {o0:5d} {o1:6d} {wmod:5d} "
          f"{'✓' if same else '✗':>8s} {d:10.3e}")

print()
if bad:
    print(f"❌ {len(bad)} 个组合不等价：")
    for b in bad:
        print(f"   {b[0]} n_out={b[1]} n_in={b[2]} o0..o1={b[3]}..{b[4]} max|Δ|={b[5]:.3e}")
else:
    print(f"✅ 全部 {len(recs)} 个**真实调用组合**逐位等价 —— MoE 批量路径/胖并行区里的分段调用"
          f"与整调用入口一致")

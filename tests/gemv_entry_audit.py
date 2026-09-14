#!/usr/bin/env python3
"""A1：对账 `gemv_any`（整调用，内含 OMP）与 `gemv_range_any`（行区间，无 OMP）是否**数值等价**。

为什么必须做：2026-09-15 发现两者在 Ling 的 k_b/v_b（Q8_0，512×128 / 128×512）上**不等价**
（MLA 输出 cos 0.9938），而 **MoE 批量路径 m6_moe_batch4 正在用 range 入口**。
性质是"静默算错"（不崩、不 NaN），最难发现 —— 与 ZAYA 的 `-ub 1` 同类。

判据：**逐位相同**才算等价（这两个入口理应做同样的乘加；任何差异都意味着某条路走了不同实现）。
用法：python3 gemv_entry_audit.py <模型子串>
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
ENG = "smol_engine" if "smollm2" in getattr(m, "gguf_name", m.name).lower() else \
      ("ling_engine" if "ling" in getattr(m, "gguf_name", m.name).lower() else "zaya_gguf")
print(f"模型：{m.name}   引擎模块：{ENG}\n")

PROBE = r'''
import os, sys, json, ctypes as ct
import numpy as np
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
os.environ["SKIP_BENCH"] = "1"; os.environ["MODEL"] = os.environ["M"]
old = sys.argv; sys.argv = ["enz"]
E = __import__(os.environ["E"]); sys.argv = old
M6 = ct.CDLL(os.environ["BASE"] + "/m6_engine.so")
# 直接问 m6_layer.c 里的两个分派入口：它们是 static inline，拿不到符号 ⇒ 走 m5 库的两个导出
# （m6_layer 的 gemv_any/gemv_range_any 就是按 code 选库后调用这两个导出）
# 两个引擎族的接口不同：smol/zaya 用 E.T + E.CODE + E._M5LIB；Ling 把表放在 ling_proto(LP)
if hasattr(E, "_M5LIB"):
    TENS = E.T
    def lookup(t):
        code = E.CODE[t.tensor_type.name]
        return E._M5LIB.get(code), code
else:
    import ling_proto as LP
    TENS = LP.T
    def lookup(t):
        lib, code = LP.CODE[t.tensor_type.name]
        return lib, code
KEEP = []
def A(raw):
    a = np.frombuffer(bytes(raw), np.uint8) if not isinstance(raw, np.ndarray) else \
        np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    big = np.empty(a.nbytes + 64, np.uint8); off = (-big.ctypes.data) % 64
    big[off:off + a.nbytes] = a; KEEP.append(big); return big.ctypes.data + off
F = ct.POINTER(ct.c_float)
rows = []
seen = set()
for t in TENS.values():
    nm = t.name
    if len(t.shape) < 2:      # ★ 3 维的也要测（Ling 的 k_b/v_b 是 (128,512,16)，出事的就是它）
        continue
    n_in, n_out = int(t.shape[0]), int(t.shape[1])
    if n_in % 32 or n_out < 32:
        continue
    try:
        lib, code = lookup(t)
    except KeyError:
        continue          # 该格式没走量化内核（F32 等）
    if lib is None or code is None:
        continue
    key = (code, n_out, n_in)
    if key in seen:            # 每个 (格式,形状) 只测一次
        continue
    seen.add(key)
    lib.m5_gemv.restype = ct.c_int
    lib.m5_gemv.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F]
    lib.m5_gemv_range.restype = ct.c_int
    lib.m5_gemv_range.argtypes = [ct.c_int, F, ct.c_void_p, ct.c_int, ct.c_int, F, ct.c_int, ct.c_int]
    x = (np.random.RandomState(7).randn(n_in).astype(np.float32) * 0.1)
    ap = A(t.data)
    y1 = np.full(n_out, np.nan, np.float32)
    y2 = np.full(n_out, np.nan, np.float32)
    rc1 = lib.m5_gemv(code, x.ctypes.data_as(F), ct.c_void_p(ap), n_out, n_in, y1.ctypes.data_as(F))
    # 与 m6_layer 的 range 分派等价：按 32 行分段调用
    for o in range(0, n_out, 32):
        lib.m5_gemv_range(code, x.ctypes.data_as(F), ct.c_void_p(ap), n_out, n_in,
                          y2.ctypes.data_as(F), o, min(o + 32, n_out))
    same = bool(np.array_equal(y1, y2))
    d = float(np.abs(y1 - y2).max())
    cos = float(np.dot(y1, y2) / (np.linalg.norm(y1) * np.linalg.norm(y2) + 1e-30))
    rows.append({"type": t.tensor_type.name, "code": code, "n_out": n_out, "n_in": n_in,
                 "tensor": nm, "same": same, "maxd": d, "cos": cos, "rc1": rc1})
print("BSJSON" + json.dumps(rows))
'''
r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                   env={**os.environ, "M": m.path, "E": ENG, "BASE": BASE, "OMP_NUM_THREADS": "8"})
line = next((l for l in r.stdout.splitlines() if l.startswith("BSJSON")), None)
if not line:
    print("失败：", r.stdout[-500:], r.stderr[-800:])
    sys.exit(1)
rows = json.loads(line[6:])
bad = [x for x in rows if not x["same"]]
print(f"{'格式':6s} {'形状':>14s} {'逐位相同':>8s} {'max|Δ|':>10s} {'cos':>10s}")
for x in sorted(rows, key=lambda d: (d["type"], -d["n_out"] * d["n_in"])):
    print(f"{x['type']:6s} {str(x['n_out'])+'x'+str(x['n_in']):>14s} "
          f"{'✓' if x['same'] else '✗':>8s} {x['maxd']:10.3e} {x['cos']:10.8f}")
print(f"\n共 {len(rows)} 个 (格式,形状) 组合：{'全部等价 ✓' if not bad else f'❌ {len(bad)} 个不等价'}")
for x in bad:
    print(f"  ✗ {x['type']} {x['n_out']}x{x['n_in']}  max|Δ|={x['maxd']:.3e} cos={x['cos']:.8f}")

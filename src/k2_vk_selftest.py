#!/usr/bin/env python3
"""k2_vk_selftest.py — libvkrun + iq2s_gemv3 vs CPU kern13 对拍（真实 GGUF 切片）。

用法：python3 k2_vk_selftest.py
"""
import os, sys, ctypes as ct
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from k2_vk import VKCtx, VGMat, X_FLOATS, Y_FLOATS, NB_H

MODEL_DIR = "/media/xiao_/OverSys1/gguf/k2-horizon"
SLICE = os.path.join(BASE, "vk", "vexp0.iq2s.bin")

NOUT, NB = 1024, 10
ROWB = NB * 82

# CPU 参照
lib13 = ct.CDLL(os.path.join(BASE, "m5", "m5_kern13.so"))
lib13.m5_gemv.restype = ct.c_int
lib13.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                          ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]

W = np.fromfile(SLICE, np.uint8)[:NOUT * ROWB]
assert W.nbytes == NOUT * ROWB, (W.nbytes, NOUT * ROWB)
rng = np.random.default_rng(7)
h = (rng.standard_normal(2560) * 0.5).astype(np.float32)

# 简易上下文：一个缓冲（比 W 大，留给 ④ 的偏移副本）
import k2_vk
vk = VKCtx.__new__(VKCtx)
vk.lib = vk._VKCtx__lib()
info = k2_vk.VGInfo()
hp = ct.c_void_p()
BUF = 2 * (1024 * NOUT * ROWB + (1 << 20))
rc = vk.lib.vg_init(ct.byref(hp), BUF, k2_vk.SPV.encode(),
                    X_FLOATS * 4, Y_FLOATS * 4, 64 << 20, ct.byref(info))
assert rc == 0, f"vg_init {rc}"
vk.h = hp
vk.n_w = info.n_w
print(f"[st] W 缓冲 {info.n_w} 块")
g16 = np.fromfile(k2_vk.G16, np.uint16).astype(np.uint32)
assert vk.lib.vg_upload_grid(vk.h, g16.ctypes.data_as(ct.c_void_p)) == 0
vk.xv = np.frombuffer((ct.c_float * X_FLOATS).from_address(vk.lib.vg_xmap(vk.h)), np.float32)
vk.yv = np.frombuffer((ct.c_float * Y_FLOATS).from_address(vk.lib.vg_ymap(vk.h)), np.float32)
assert vk.lib.vg_upload(vk.h, 0, 0, W.ctypes.data_as(ct.c_void_p), W.nbytes) == 0

def cpu_ref(x, w, n_out, n_in):
    y = np.empty(n_out, np.float32)
    assert lib13.m5_gemv(13, x.ctypes.data_as(ct.POINTER(ct.c_float)),
                         w.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                         n_out, n_in, y.ctypes.data_as(ct.POINTER(ct.c_float))) == 0
    return y

ok = True

# ① 单矩阵（v2 等价路径：x_off=0, y_off=0）
yc = cpu_ref(h, W, NOUT, 2560)
vk._run([(k2_vk.G(0, 0), 0, 0, NOUT, NB)], h, 2560)
d = np.abs(vk.yv[:NOUT] - yc).max()
print(f"① 单矩阵 max|Δ|={d:.3e} {'PASS' if d < 1e-3 else 'FAIL'}")
ok &= d < 1e-3

# ② x_off/y_off 路径：x 写进 X 缓冲偏移 4096，y 放偏移 2048
vk.xv[4096:4096 + 2560] = h
vk._run([(k2_vk.G(0, 0), 4096, 2048, NOUT, NB)], None, X_FLOATS)
d = np.abs(vk.yv[2048:2048 + NOUT] - yc).max()
print(f"② x_off/y_off max|Δ|={d:.3e} {'PASS' if d < 1e-3 else 'FAIL'}")
ok &= d < 1e-3

# ③ 多矩阵一次 submit：同一 W 两个 y 窗 + nb=3 的 768 输入矩阵（行 246B 非整字对齐）
w3 = W[:768 * 3 * 82].copy()   # 当作 768 行×nb=3 的矩阵用（数值无意义，只验寻址）
yc3 = cpu_ref(h[:768], w3, 768, 768)
vk._run([(k2_vk.G(0, 0), 0, 0, NOUT, NB),
         (k2_vk.G(0, 0), 0, 8192, 768, 3)], h, 2560)
d1 = np.abs(vk.yv[:NOUT] - yc).max()
d2 = np.abs(vk.yv[8192:8192 + 768] - yc3).max()
print(f"③ 多矩阵+nb3 max|Δ|={max(d1, d2):.3e} {'PASS' if max(d1, d2) < 1e-3 else 'FAIL'}")
ok &= max(d1, d2) < 1e-3

# ④ w_off 非 0：把 W 原样再传到 4 对齐偏移 off2
off2 = (NOUT * ROWB + 15) & ~15
assert vk.lib.vg_upload(vk.h, 0, off2, W.ctypes.data_as(ct.c_void_p), W.nbytes) == 0
vk._run([(k2_vk.G(0, off2), 0, 0, NOUT, NB)], h, 2560)
d = np.abs(vk.yv[:NOUT] - yc).max()
print(f"④ w_off 偏移 max|Δ|={d:.3e} {'PASS' if d < 1e-3 else 'FAIL'}")
ok &= d < 1e-3

# ⑤ 计时：真实相位形状（行数 × dispatch 数），复用同一块 W（数值不看，只看时间）
import time as _t

def bench_phase(mats, label, reps=30):
    ts = []
    for _ in range(reps):
        t = _t.perf_counter()
        vk._run(mats, h, 2560)
        ts.append(_t.perf_counter() - t)
    ts = ts[3:]   # 预热
    rows = sum(m[3] for m in mats)
    mb = rows * NB * 82 / 1e6 * (len(mats) and 1)
    print(f"⑤ {label:28s} {len(mats):2d}disp {rows:5d}行 {st_md(ts):7.3f} ms  "
          f"({rows*NB*82/1e9/st_md(ts):6.1f} GB/s)")

def st_md(ts):
    import statistics
    return statistics.median(ts)

# 合成大矩阵内容：把 W 平铺到 4096 行（q 大小）
Wbig = np.tile(W, 4)   # 4096 行
vk.lib.vg_upload(vk.h, 0, 0, Wbig.ctypes.data_as(ct.c_void_p), Wbig.nbytes)
yc4 = cpu_ref(h, Wbig, NOUT * 4, 2560) if False else None
bench_phase([(k2_vk.G(0, 0), 0, 0, 4096, NB)], "单矩阵4096行(≈q)")
bench_phase([(k2_vk.G(0, 0), 0, k * 768, 768, NB) for k in range(16)], "16×768行(≈gate_up)")
bench_phase([(k2_vk.G(0, 0), 0, k * 2560, 2560, 3) for k in range(4)], "4×2560行nb3(≈down)", reps=20)

print("== 自测", "PASS ==" if ok else "FAIL ==")
sys.exit(0 if ok else 1)

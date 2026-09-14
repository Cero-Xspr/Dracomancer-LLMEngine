#!/usr/bin/env python3
"""B4-M1：Q8_0 的 OpenCL gemv —— 能不能打过 CPU 内核？

过关线（ROADMAP B4-M1）：**明显超过 CPU**，否则 B4 停在这里不往下投。
CPU 靶子（今天 `iova` 实测，用引擎自己的 m5_kern6 内核）：
  · Q8_0 320×960（0.3MB）：单次调用 ~27~28 GB/s
  · Q8_0 49152×960（50MB，token_embd/head）：单线程 40.4 GB/s
参考上限：iGPU 纯读带宽 68.5 GB/s（16MB）/ 81.5 GB/s（64MB）。
用法：PYTHONPATH=<gguf-py> python3 gemv_q8_0.py [形状]
"""
import ctypes
import os
import statistics
import sys
import time

import numpy as np

CL_GPU, CL_R, CL_W, CL_RW, CL_UHP = 4, 4, 2, 1, 8
CL_PROGRAM_BUILD_LOG = 0x1183
lib = ctypes.CDLL("libOpenCL.so.1")
for f, a, r in [
    ("clGetPlatformIDs", [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)], ctypes.c_int),
    ("clGetDeviceIDs", [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)], ctypes.c_int),
    ("clCreateContext", [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_void_p),
    ("clCreateCommandQueue", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_int)], ctypes.c_void_p),
    ("clCreateBuffer", [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_void_p),
    ("clCreateProgramWithSource", [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)], ctypes.c_void_p),
    ("clBuildProgram", [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    ("clGetProgramBuildInfo", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
    ("clCreateKernel", [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_void_p),
    ("clSetKernelArg", [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p], ctypes.c_int),
    ("clEnqueueNDRangeKernel", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    ("clEnqueueReadBuffer", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    ("clFinish", [ctypes.c_void_p], ctypes.c_int),
]:
    fn = getattr(lib, f)
    fn.argtypes = a
    fn.restype = r

def ck(e, what):
    if e:
        raise RuntimeError(f"{what} err={e}")

pf = (ctypes.c_void_p * 2)(); npf = ctypes.c_uint(2)
ck(lib.clGetPlatformIDs(2, pf, ctypes.byref(npf)), "plat")
dev = (ctypes.c_void_p * 4)(); nd = ctypes.c_uint()
ck(lib.clGetDeviceIDs(pf[0], CL_GPU, 4, dev, ctypes.byref(nd)), "dev")
_e = ctypes.c_int()
devp = ctypes.c_void_p(dev[0])
ctx = lib.clCreateContext(None, 1, ctypes.byref(devp), None, None, ctypes.byref(_e)); ck(_e.value, "ctx")
q = lib.clCreateCommandQueue(ctx, devp, 0, ctypes.byref(_e)); ck(_e.value, "queue")

KSRC = ("""
// Q8_0 gemv：每个 work-item 一行；x 先进 local（同一 work-group 复用），权重从显存流。
// f16 尺度手写解码（不依赖 cl_khr_fp16 扩展）。
inline float h2f(ushort h) {
    uint s = (uint)(h & 0x8000u) << 16;
    uint e = (uint)(h & 0x7C00u);
    uint m = (uint)(h & 0x03FFu);
    return as_float(s | ((e + 0x1C000u) << 13) | (m << 13));
}
__kernel void gemv_q8_0(__global const float* x, __global const uchar* W,
                        __global float* y, const int n_out, const int n_in) {
    __local float xs[4096];
    const int tid = get_local_id(0);
    const int nb = n_in / 32;
    for (int i = tid; i < n_in; i += get_local_size(0)) xs[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int o = get_global_id(0);
    if (o >= n_out) return;
    const __global uchar* row = W + (size_t)o * nb * 34;
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const __global uchar* blk = row + b * 34;
        const float d = h2f(*(const __global ushort*)blk);
        const __global char* qs = (const __global char*)(blk + 2);
        const __local float* xb = xs + b * 32;
        float4 a = (float4)(0.f);
        for (int i = 0; i < 32; i += 4)
            a += convert_float4(vload4(i >> 2, qs)) * vload4(i >> 2, xb);
        acc += d * (a.x + a.y + a.z + a.w);
    }
    y[o] = acc;
}
""").encode()
srcp = (ctypes.c_char_p * 1)(KSRC); srclen = (ctypes.c_size_t * 1)(len(KSRC))
prog = lib.clCreateProgramWithSource(ctx, 1, srcp, srclen, ctypes.byref(_e)); ck(_e.value, "prog")
rc = lib.clBuildProgram(prog, 1, ctypes.byref(devp), b"-cl-fast-relaxed-math", None, None)
if rc:
    logb = ctypes.create_string_buffer(1 << 20); got = ctypes.c_size_t()
    lib.clGetProgramBuildInfo(prog, dev[0], CL_PROGRAM_BUILD_LOG, 1 << 20, logb, ctypes.byref(got))
    print(logb.value.decode()[:2000]); sys.exit(1)
print("OpenCL build OK")

def mkbuf(sz, ptr=None, flags=CL_R):
    if ptr is not None:
        flags |= CL_UHP
    e = ctypes.c_int()
    b = lib.clCreateBuffer(ctx, flags, sz, ptr, ctypes.byref(e)); ck(e.value, "buf")
    return b

def launch(kern, gws, lws, args, niter):
    for i, (sz, val) in enumerate(args):
        ck(lib.clSetKernelArg(kern, i, sz, ctypes.byref(val)), "arg")
    g = (ctypes.c_size_t * 1)(gws); l = (ctypes.c_size_t * 1)(lws)
    t0 = time.perf_counter()
    for _ in range(niter):
        ck(lib.clEnqueueNDRangeKernel(q, kern, 1, None, g, l, 0, None, None), "ndr")
    ck(lib.clFinish(q), "fin")
    return (time.perf_counter() - t0) / niter

# ── 取真实 Q8_0 张量（GGUF 布局）──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf_fast
MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/smol/SmolLM2-360M-Instruct-Q4_K_M.gguf")
R = gguf_fast.FastGGUF(MODEL)
want = sys.argv[1] if len(sys.argv) > 1 else "320x960"
cands = [t for t in R.tensors if t.tensor_type.name == "Q8_0"]
t = next((x for x in cands if f"{int(x.shape[1])}x{int(x.shape[0])}" == want.replace("×", "x")), None)
if t is None:
    t = next((x for x in cands if int(x.shape[0]) == 960 and int(x.shape[1]) == 320), cands[0])
n_in, n_out = int(t.shape[0]), int(t.shape[1])
raw = bytes(t.data)
print(f"张量 {t.name}  n_out={n_out} n_in={n_in}  {len(raw)/1e6:.3f} MB")

KEEP = []
def aligned(b):
    a = np.empty(len(b) + 64, np.uint8)
    o = (-a.ctypes.data) % 64
    a[o:o + len(b)] = np.frombuffer(b, np.uint8)
    KEEP.append(a)
    return a.ctypes.data + o

x = (np.random.RandomState(3).randn(n_in).astype(np.float32) * 0.1)
y_cl = np.zeros(n_out, np.float32)
y_cpu = np.zeros(n_out, np.float32)
W_u8 = np.frombuffer(raw, np.uint8)
bx = mkbuf(x.nbytes, x.ctypes.data_as(ctypes.c_void_p))
bw = mkbuf(W_u8.nbytes, W_u8.ctypes.data_as(ctypes.c_void_p))
by = mkbuf(y_cl.nbytes)

# ── CPU 靶子（引擎自己的内核）──────────────────────────────────────────
CPU = ctypes.CDLL("/media/xiao_/OverSys1/npu-direct/hybrid/m5/m5_kern6.so")
CPU.m5_gemv.restype = ctypes.c_int
F = ctypes.POINTER(ctypes.c_float)
CPU.m5_gemv.argtypes = [ctypes.c_int, F, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, F]
Wp = aligned(raw)
CPU.m5_gemv(0, x.ctypes.data_as(F), ctypes.c_void_p(Wp), n_out, n_in, y_cpu.ctypes.data_as(F))
cts = []
for _ in range(20):
    t0 = time.perf_counter()
    CPU.m5_gemv(0, x.ctypes.data_as(F), ctypes.c_void_p(Wp), n_out, n_in, y_cpu.ctypes.data_as(F))
    cts.append(time.perf_counter() - t0)
cpu_s = statistics.median(cts)
print(f"CPU （引擎 m5_kern6，单线程，OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','?')}）: "
      f"{cpu_s*1e3:7.3f} ms  {len(raw)/cpu_s/1e9:6.1f} GB/s")

# ── iGPU ─────────────────────────────────────────────────────────────
k = lib.clCreateKernel(prog, b"gemv_q8_0", ctypes.byref(_e)); ck(_e.value, "kern")
print(f"\n{'lws':>6s} {'ms':>9s} {'GB/s':>8s}  {'vs CPU':>8s}  正确性")
best = None
for lws in (64, 128, 256):
    # gws 必须是 lws 的整数倍（且 ≥ n_out）
    gws = ((max(n_out, 1) + lws - 1) // lws) * lws
    bxh, bwh, byh = ctypes.c_void_p(bx), ctypes.c_void_p(bw), ctypes.c_void_p(by)
    args = [(8, bxh), (8, bwh), (8, byh), (4, ctypes.c_int(n_out)), (4, ctypes.c_int(n_in))]
    launch(k, gws, lws, args, 2)
    dt = statistics.median([launch(k, gws, lws, args, 1) for _ in range(9)])
    ck(lib.clEnqueueReadBuffer(q, by, 1, 0, y_cl.nbytes, y_cl.ctypes.data_as(ctypes.c_void_p), 0, None, None), "read")
    ck(lib.clFinish(q), "fin")
    cos = float(np.dot(y_cl, y_cpu) / (np.linalg.norm(y_cl) * np.linalg.norm(y_cpu) + 1e-30))
    gb = len(raw) / dt / 1e9
    print(f"{lws:6d} {dt*1e3:9.3f} {gb:8.1f}  {gb/(len(raw)/cpu_s/1e9):7.2f}×  "
          f"cos={cos:.7f} max|Δ|={np.abs(y_cl-y_cpu).max():.2e}")
    if cos > 0.999 and (best is None or gb > best[0]):
        best = (gb, lws, cos)
print()
if best and best[0] > len(raw) / cpu_s / 1e9 * 1.05:
    print(f"⇒ ✅ iGPU 打过 CPU：{best[0]:.1f} vs {len(raw)/cpu_s/1e9:.1f} GB/s（lws={best[1]}）"
          f"⇒ 按 M1 过关线，继续 M2")
else:
    print(f"⇒ ❌ iGPU 未明显超过 CPU（最好 {best[0]:.1f} vs {len(raw)/cpu_s/1e9:.1f} GB/s）"
          f"⇒ 按 ROADMAP B4 的纪律：**停在这里**，不往 M2/M3 投")

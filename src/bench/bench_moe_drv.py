#!/usr/bin/env python3
"""bench_moe 驱动：同进程交错 A/B 各变体（抵时钟漂移），输出逐位校验。"""
import ctypes as ct, os, sys, time, statistics
import numpy as np

HERE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf

GGUF = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/Qwen3.6-35B-A3B-REAP-48-v2.gguf")
ROUNDS = int(os.environ.get("ROUNDS", "15"))
N_USED = int(os.environ.get("N_USED", "8"))
CHUNK = int(os.environ.get("CHUNK", "64"))
OMP_N = os.environ.get("OMP_NUM_THREADS", "8")

def load(name):
    lib = ct.CDLL(os.path.join(HERE, "m5", name + ".so"))
    lib.m5_gemv_range.restype = ct.c_int
    lib.m5_gemv_range.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                                  ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.c_int, ct.c_int]
    return lib

k6, k12 = load("m5_kern6"), load("m5_kern12")
bm = ct.CDLL(os.path.join(HERE, "m5", "bench_moe.so"))
bm.bm_set_kern.argtypes = [ct.c_void_p, ct.c_void_p]
bm.bm_moe.argtypes = [ct.POINTER(ct.c_float)] + [ct.POINTER(ct.c_uint8)]*0 + [ct.c_void_p]*3 + \
    [ct.POINTER(ct.c_float), ct.c_int, ct.c_int, ct.c_int, ct.c_int,
     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int, ct.c_int]
bm.bm_time.restype = ct.c_double
bm.bm_set_kern(k6.m5_gemv_range, k12.m5_gemv_range)   # ★ 之前漏了这行：KRNG=NULL 段错误

r = gguf.GGUFReader(GGUF)
tmap = {t.name: t for t in r.tensors}
def pf(a): return a.ctypes.data_as(ct.POINTER(ct.c_float))
def p8(a): return a.ctypes.data_as(ct.POINTER(ct.c_uint8))
def pv(a): return ct.cast(a.ctypes.data, ct.c_void_p)

# 取一个 MoE 层的专家张量
L = os.environ.get("LAYER", "0")
tg_ = tmap[f"blk.{L}.ffn_gate_exps.weight"]
tu_ = tmap[f"blk.{L}.ffn_up_exps.weight"]
td_ = tmap[f"blk.{L}.ffn_down_exps.weight"]
g = np.frombuffer(tg_.data, np.uint8)
u = np.frombuffer(tu_.data, np.uint8)
d = np.frombuffer(td_.data, np.uint8)
NEXP = int(tg_.shape[0])
print(f"gate shape {[int(v) for v in tg_.shape]} down {[int(v) for v in td_.shape]}  OMP={OMP_N} n_used={N_USED} chunk={CHUNK}")
row_g = g.nbytes // NEXP
H, FF = 2048, 512      # n_in, inter
rng = np.random.default_rng(7)
x = rng.standard_normal(H).astype(np.float32)
wtop = np.full(N_USED, 1.0/N_USED, np.float32)
scratch = np.zeros(3*N_USED*FF + N_USED*H, np.float32)
out = np.zeros(H, np.float32)
pe = [ct.cast(ct.c_void_p(a.ctypes.data + e*row_g), ct.POINTER(ct.c_uint8))
      for a in (g, u, d) for e in range(N_USED)]
GP = (ct.POINTER(ct.c_uint8)*N_USED)(*[pe[0*N_USED+e] for e in range(N_USED)])
UP = (ct.POINTER(ct.c_uint8)*N_USED)(*[pe[1*N_USED+e] for e in range(N_USED)])
DP = (ct.POINTER(ct.c_uint8)*N_USED)(*[pe[2*N_USED+e] for e in range(N_USED)])

VARIANTS = [(0, 0), (1, CHUNK), (1, 32), (1, 128), (3, 0), (4, CHUNK)]
# 预热 + 正确性：变体0 vs 变体1/3/4 输出必须逐位一致
for v, c in VARIANTS:
    print(f"  >> run v{v} c{c}", flush=True)
    bm.bm_moe(pf(x), GP, UP, DP, pf(wtop), N_USED, FF, H, H, pf(scratch), pf(out), v, c)
    if v == 0:
        ref = out.copy()
    else:
        same = np.array_equal(ref, out)
        print(f"  校验 variant {v} chunk {c}: {'逐位一致 ✓' if same else '不一致 ✗ (' + str(np.abs(ref-out).max()) + ')'}")

print(f"\n每层耗时（{ROUNDS} 轮中位，同进程交错）:")
res = {vc: [] for vc in VARIANTS}
for _ in range(ROUNDS):
    for v, c in VARIANTS:
        t0 = bm.bm_time()
        for rep in range(10):
            bm.bm_moe(pf(x), GP, UP, DP, pf(wtop), N_USED, FF, H, H, pf(scratch), pf(out), v, c)
        res[(v, c)].append((bm.bm_time() - t0) / 10 * 1000)
base = statistics.median(res[(0, 0)])
bytes_per = N_USED * 3 * H * FF * (110/256)
for (v, c), ts in res.items():
    m = statistics.median(ts)
    lbl = {(0,0): "现状(整矩阵/专家)", (3,0): "单线程基线", (4,CHUNK): f"分块{kern if False else ''}+kern12"}.get((v, c), f"分块{c}")
    print(f"  v{v} {lbl:22s} {m:7.3f} ms  ({base/m:5.2f}× vs 现状)  有效带宽 {bytes_per/m*1e-6:5.1f} GB/s")

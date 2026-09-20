#!/usr/bin/env python3
"""test_batch8k.py — m5_batch8_k 的位精确验证：批量 N 专家 vs 逐专家 m5_gemv。"""
import os, sys, ctypes as ct
import numpy as np

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
M5 = "/media/xiao_/OverSys1/npu-direct/hybrid/m5"
from gguf.quants import dequantize
import gguf
Q4K = gguf.GGMLQuantizationType.Q4_K
Q6K = gguf.GGMLQuantizationType.Q6_K

batch = ct.CDLL(os.path.join(M5, "m5_batch8_k.so"))
k7 = ct.CDLL(os.path.join(M5, "m5_kern7.so"))
k8 = ct.CDLL(os.path.join(M5, "m5_kern8.so"))
for lib in (k7, k8):
    lib.m5_gemv.restype = ct.c_int
    lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                            ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
batch.m5_gemvn_q4k.restype = ct.c_int
batch.m5_gemvn_q4k.argtypes = [ct.POINTER(ct.c_float), ct.c_long, ct.POINTER(ct.c_uint8),
                               ct.POINTER(ct.c_int), ct.c_long, ct.c_int, ct.c_int, ct.c_int,
                               ct.POINTER(ct.c_float)]
batch.m5_gemvn_q6k.restype = ct.c_int
batch.m5_gemvn_q6k.argtypes = batch.m5_gemvn_q4k.argtypes

pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
p8 = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_uint8))
pi = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_int))

rng = np.random.default_rng(0)

def make_q(t, n_out, n_in):
    v = (rng.standard_normal((n_out, n_in)) * 0.02).astype(np.float32)
    raw = np.frombuffer(dequantize(np.frombuffer(
        np.concatenate([bytes(0) for _ in range(0)]) if False else
        (lambda q: q)(np.zeros(1, np.uint8)), t), t), np.float32) if False else None
    # 直接用 gguf-py 的 quantize? gguf-py 无 Q4_K 量化器 → 借 m5 解码器反向构造: 用现成模型张量!
    return v

# ★ 用真模型张量验证（gguf-py 没有量化器）
import gguf_fast
R = gguf_fast.FastGGUF('/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf')
T = {t.name: t for t in R.tensors}

def run_case(name, tname, n_exp, per_expert_rows, n_in, ttype, code, kern_lib):
    t = T[tname]
    total_rows = int(t.shape[1]) if len(t.shape) > 1 else 1
    per = int(t.data.nbytes // n_exp) if False else None
    # 该张量本身是堆叠的: 直接用前 n_exp 个专家
    stride = t.data.nbytes // n_exp
    W = np.asarray(t.data).reshape(-1)
    x = rng.standard_normal(n_in).astype(np.float32)
    idx = np.arange(n_exp, dtype=np.int32)
    # 批量
    yb = np.zeros((n_exp, per_expert_rows), np.float32)
    fn = batch.m5_gemvn_q4k if code == 5 else batch.m5_gemvn_q6k
    rc = fn(pf(np.ascontiguousarray(x)), 0, p8(W), pi(idx), stride, per_expert_rows, n_in, n_exp, pf(yb))
    assert rc == 0, f"rc={rc}"
    # 逐专家参照
    ok = True
    for e in range(n_exp):
        yr = np.zeros(per_expert_rows, np.float32)
        assert kern_lib.m5_gemv(code, pf(x), p8(W[e*stride:(e+1)*stride]), per_expert_rows, n_in, pf(yr)) == 0
        d = np.abs(yb[e] - yr).max()
        if d != 0.0:
            ok = False
            print(f"  {name} 专家{e}: max|Δ|={d:.3e}  ***")
    print(f"{name}: {'位精确 OK ✓' if ok else 'FAIL'}  (n_exp={n_exp}, {per_expert_rows}x{n_in})")
    return ok

ok1 = run_case("Q4K gate/up 形状", "blk.3.ffn_gate_exps.weight", 4, 768, 2560, Q4K, 5, k7)
ok2 = run_case("Q6K down 形状",   "blk.3.ffn_down_exps.weight", 4, 2560, 768, Q6K, 4, k8)
ok3 = run_case("Q4K v_experts",   "blk.3.attn_v_exps.weight", 4, 1024, 2560, Q4K, 5, k7)
# 乱序索引
t = T["blk.3.ffn_gate_exps.weight"]
stride = t.data.nbytes // 100
W = np.asarray(t.data).reshape(-1)
x = rng.standard_normal(2560).astype(np.float32)
idx = np.array([77, 3, 99, 12], np.int32)
yb = np.zeros((4, 768), np.float32)
batch.m5_gemvn_q4k(pf(x), 0, p8(W), pi(idx), stride, 768, 2560, 4, pf(yb))
ok4 = True
for j, e in enumerate(idx):
    yr = np.zeros(768, np.float32)
    k7.m5_gemv(5, pf(x), p8(W[e*stride:(e+1)*stride]), 768, 2560, pf(yr))
    d = np.abs(yb[j] - yr).max()
    if d != 0.0: ok4 = False; print(f"  乱序专家{e}: max|Δ|={d:.3e}")
print(f"Q4K 乱序索引: {'位精确 OK ✓' if ok4 else 'FAIL'}")
sys.exit(0 if (ok1 and ok2 and ok3 and ok4) else 1)

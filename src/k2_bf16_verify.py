#!/usr/bin/env python3
"""k2_bf16_verify.py — 官方 IFM BF16 GGUF vs NANI Q4_K_M vs safetensors 三方大张量验证。
背景：IFM 自家 GGUF 的 norm 是隔位插零坏布局；本脚本验证其大张量是否也坏。"""
import os, sys, pickle, json
os.environ.setdefault("GGUF_PY_DIR", "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
import numpy as np
import gguf_fast
from gguf.quants import dequantize

IFM = '/media/Data-1/gguf/k2-horizon/K2-Horizon-36B-BF16.gguf'
NANI = '/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf'

# IFM 头（已解析过）——直接从文件重读（文件现已完整）
import struct
data = open(IFM, 'rb').read(32 * 1024 * 1024)
n_tensors = None
# 物理表起点 11439996（此前定标过）
off = 11439996
tensors = []
for i in range(798):
    n, = struct.unpack('<Q', data[off:off+8]); off += 8
    name = data[off:off+n].decode(); off += n
    nd, = struct.unpack('<I', data[off:off+4]); off += 4
    dims = struct.unpack(f'<{nd}Q', data[off:off+8*nd]); off += 8*nd
    tt, = struct.unpack('<I', data[off:off+4]); off += 4
    o, = struct.unpack('<Q', data[off:off+8]); off += 8
    tensors.append((name, dims, tt, o))
data_start = (off + 31) // 32 * 32
tmap = {n: (d, t, o) for n, d, t, o in tensors}
print(f"IFM 头: {len(tensors)} 张量, data_start={data_start}")

f_ifm = open(IFM, 'rb')
def ifm_deq(name):
    d, t, o = tmap[name]
    ne = int(np.prod(d))
    f_ifm.seek(data_start + o)
    raw = f_ifm.read(ne * 2 if t == 30 else ne * 4)
    if t == 30:
        u = np.frombuffer(raw, np.uint16).astype(np.uint32)
        a = (u << np.uint32(16)).view(np.float32)
    else:
        a = np.frombuffer(raw, np.float32)
    return a.reshape(d[::-1] if len(d) > 1 else d)

R = gguf_fast.FastGGUF(NANI)
T = {t.name: t for t in R.tensors}
def nani(name):
    t = T[name]
    return np.asarray(dequantize(t.data, t.tensor_type), np.float32)

def relerr(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))

print(f"\n{'张量':36} {'norm含义':10} {'IFM零占比':>9} {'NANI零占比':>10} {'corr':>8} 判定")
results = {}
for name in ['blk.0.attn_norm.weight', 'blk.0.ffn_norm.weight', 'blk.3.attn_norm.weight']:
    a = ifm_deq(name).ravel(); b = nani(name).ravel()
    ca = np.corrcoef(a, b)[0, 1]
    za, zb = np.count_nonzero(a == 0) / len(a), np.count_nonzero(b == 0) / len(b)
    print(f"{name:36} {'norm':10} {za:9.2f} {zb:10.2f} {ca:8.4f} {'IFM坏(隔位插零)' if za > 0.4 else 'OK'}")
    results[name] = {'ifm_zero': za, 'nani_zero': zb, 'corr': float(ca)}

# 大张量：ffn_down（blk.0 稠密）与 experts（blk.3）
for name in ['blk.0.ffn_down.weight', 'blk.3.ffn_gate_exps.weight', 'blk.3.attn_q.weight',
             'blk.3.ffn_down_exps.weight']:
    a = ifm_deq(name); b = nani(name)
    if a.shape != b.shape:
        print(f"{name}: 形状不合 官{a.shape} 本{b.shape}"); continue
    c = np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1]
    e = relerr(a, b)
    verdict = 'IFM==NANI(量化噪声内)' if c > 0.99 else ('*** IFM 与 NANI 不同(检查谁坏)' if abs(c) < 0.9 else '弱相关')
    print(f"{name:36} corr={c:8.4f} rel={e:.4f} {verdict}")
    results[name] = {'corr': float(c), 'rel': e}

json.dump(results, open('/media/xiao_/OverSys1/npu-direct/hybrid/tests/k2_bf16_verify.json', 'w'), indent=1)
print("\n已存 tests/k2_bf16_verify.json")

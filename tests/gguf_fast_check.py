#!/usr/bin/env python3
"""gguf_fast vs gguf.GGUFReader 全量对账（字段值 + 张量元数据 + 数据字节 + 耗时）。

判据（全部必须为 0）：
  · 字段：名字集合相同；标量值 ==；字符串 ==；字符串数组逐元素 ==；数值数组 np.array_equal
  · 张量：名字集合相同；tensor_type.name / shape / n_processes / n_bytes / data_offset 相同
  · 字节：每张量 data.tobytes() 与 gguf-py 的**逐字节**相同（sha256 前缀比对，避免 5GB 内存峰值）
用法：python3 gguf_fast_check.py <gguf> [<gguf> ...]
"""
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import numpy as np
import gguf
import gguf_fast


def h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def check(path):
    print(f"\n=== {os.path.basename(path)}  ({os.path.getsize(path)/2**30:.2f}GB)")
    t0 = time.time(); A = gguf.GGUFReader(path); ta = time.time() - t0
    t0 = time.time(); B = gguf_fast.FastGGUF(path); tb = time.time() - t0
    t0 = time.time(); B2 = gguf_fast.FastGGUF(path); tb2 = time.time() - t0
    print(f"  gguf-py   {ta:7.2f}s        gguf_fast {tb:6.2f}s   （缓存命中 {tb2*1000:6.1f}ms）")
    bad = 0

    # ---- 字段 ----
    # GGUF.version/tensor_count/kv_count 是 gguf-py 自己塞的**伪字段**（不在文件的 KV 段里），
    # gguf_fast 不造 ⇒ 比对前从 gguf-py 侧剔除，否则每次都是假差异。
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    fa = {f.name for f in A.fields.values()} - skip
    fb = set(B.fields.keys())
    if fa != fb:
        print(f"  ✗ 字段名集合不同  只在 gguf-py: {sorted(fa-fb)[:5]}  只在 fast: {sorted(fb-fa)[:5]}")
        bad += 1
    ndiff = 0
    for name in sorted(fa & fb):
        va, vb = A.fields[name].contents(), B.fields[name].contents()
        ok = None
        if isinstance(va, list) and isinstance(vb, list):
            ok = len(va) == len(vb) and all(x == y for x, y in zip(va, vb))
        elif isinstance(va, (int, float, str)) or isinstance(vb, (int, float, str)):
            ok = va == vb
        else:                                   # numpy 数组
            try:
                ok = np.array_equal(np.asarray(va), np.asarray(vb))
            except Exception as e:
                ok = f"比较失败 {e}"
        if ok is not True:
            print(f"  ✗ 字段 {name}: {str(ok)[:60]}")
            ndiff += 1
    if ndiff:
        bad += 1
    else:
        print(f"  ✓ 字段 {len(fa & fb)} 个全部一致")

    # ---- 张量元数据 ----
    TA, TB = {t.name: t for t in A.tensors}, {t.name: t for t in B.tensors}
    if set(TA) != set(TB):
        print(f"  ✗ 张量名不同: {sorted(set(TA) ^ set(TB))[:5]}")
        bad += 1
    meta_bad = [n for n in TA if n in TB and not (
        TA[n].tensor_type.name == TB[n].tensor_type.name
        and tuple(int(x) for x in TA[n].shape) == tuple(int(x) for x in TB[n].shape)
        and TA[n].n_elements == TB[n].n_elements
        and TA[n].n_bytes == TB[n].n_bytes
        and TA[n].data_offset == TB[n].data_offset)]
    if meta_bad:
        print(f"  ✗ 元数据不一致 {len(meta_bad)} 个: {meta_bad[:5]}")
        bad += 1
    else:
        print(f"  ✓ 张量元数据 {len(TB)} 个全部一致（type/shape/元素数/字节数/偏移）")

    # ---- 数据字节（逐张量 sha256）----
    t0 = time.time()
    byte_bad = []
    for n in TB:
        if TB[n].data.nbytes != TA[n].data.nbytes:
            byte_bad.append(n); continue
        if h(TB[n].data.tobytes()) != h(TA[n].data.tobytes()):
            byte_bad.append(n)
    if byte_bad:
        print(f"  ✗ 数据字节不一致 {len(byte_bad)} 个: {byte_bad[:5]}")
        bad += 1
    else:
        tot = sum(TB[n].data.nbytes for n in TB) / 2**30
        print(f"  ✓ 数据字节 {len(TB)} 张量 / {tot:.2f}GB 全部逐字节一致（校验耗时 {time.time()-t0:.1f}s）")
    return bad


if __name__ == "__main__":
    paths = sys.argv[1:] or ["/media/xiao_/OverSys1/gguf/zaya1/ZAYA1-8B-Q4_K_M.gguf"]
    nb = 0
    for p in paths:
        nb += check(p)
    print(f"\n{'✅ 全部一致' if nb == 0 else f'❌ {nb} 处不一致'}")
    sys.exit(1 if nb else 0)

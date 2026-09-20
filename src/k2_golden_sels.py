#!/usr/bin/env python3
"""k2_golden_sels.py — 采集 CPU 引擎在金标准 prompt 末位每层的路由选择。
产物给 K2_FORCE_SEL 用：闸门强制 GPU 与 CPU 走同一组专家（消除 fp 序噪声
引发的 argsort 翻转级联），只比数学。
用法：MODEL=<gguf> OUT=... python3 k2_golden_sels.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

assert os.environ.get("MODEL")
import k2_engine as KE   # K2_VK 必须未设（CPU 路径）

gold_p = os.environ.get("GOLDEN", os.path.join(KE.BASE, "tests/golden/k2horizon_golden_iq2m.json"))
ids = json.load(open(gold_p))["ids"]

KE.reset()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p)

sels = []
for li in range(KE.NL):
    L = KE.LAYERS[li]
    if L.sparse:
        sv, sm = KE._LAST_SEL.get(li, [None, None])
        sels.append([[int(e) for e in (sv if sv is not None else [])],
                     [int(e) for e in (sm if sm is not None else [])]])
    else:
        sels.append([[], []])

out = {"model": os.path.basename(KE.MODEL), "pos": len(ids) - 1, "last_sels": sels}
path = os.environ.get("OUT", os.path.join(KE.BASE, "tests/golden/k2horizon_sels_iq2m.json"))
json.dump(out, open(path, "w"))
print("已写", path, " 稀疏层样例 li=5:", sels[5])

#!/usr/bin/env python3
"""chunked prefill 等价性闸门：chunk 路径 vs 逐 token 路径。
判据：末位 logits cos ≥ 0.999 + 贪心续接 12 token 逐 token 一致。
用法：MODEL=... ENG=qwen35|qwen35moe python3 prefill_chunk_gate.py"""
import os, sys, time
assert os.environ.get("MODEL") and os.environ.get("ENG")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import draco_engine_server as S
S.ARGS = argparse.Namespace(model=os.environ["MODEL"], port=1,
                            engine=os.environ["ENG"], threads=8, ctx=0)
S.load_engine()
import qwen35_engine as QE
from qwen35_prefill import ChunkPrefiller

CP = ChunkPrefiller(QE)
T = int(os.environ.get("NTOK", "96"))
text = ("机器学习是人工智能的一个分支，它让计算机从数据中学习模式。"
        "深度学习使用多层神经网络，在图像和语音上进步巨大。"
        "批量处理可以提高吞吐量，因为权重只需要读取一次。") * ((T // 48) + 1)
ids = S.AP["encode"](text)[:T]

S.AP["reset"]()
for p, t in enumerate(ids):
    QE.forward(int(t), p)
lg0 = QE.LOGITS.copy()
g0 = []
for i in range(12):
    nid = int(QE.LOGITS.argmax())
    g0.append(nid)
    QE.forward(nid, len(ids) + i)

S.AP["reset"]()
lg1 = CP.prefill_chunk(ids, 0).copy()
cos = float(lg0 @ lg1 / (np.linalg.norm(lg0) * np.linalg.norm(lg1) + 1e-30))
g1 = []
for i in range(12):
    nid = int(QE.LOGITS.argmax())
    g1.append(nid)
    QE.forward(nid, len(ids) + i)

ok = cos >= 0.999 and g0 == g1
print(f"[{os.environ['ENG']}] T={T}  cos={cos:.6f}  贪心续接 {'一致' if g0 == g1 else '不一致 ' + str((g0[:4], g1[:4]))}"
      f"  → {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""granite C 驱动器 vs llama.cpp：逐位置比 logits（同一 token 序列，逐 token decode）。

参考来自夹具 dump（zaya_gdump，ZDUMP_POS=p）的 logits.bin —— 那个位置最后一层的 logits。
判据（沿用 ZAYA 那轮的经验）：**cos 会因量化误差累积而偏低，看 top-1/argmax 是否一致**，
外加 top-5 重合度；cos 只当参考。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TOKS = [int(v) for v in os.environ.get("TOKS", "100,101,102,103,104,105,106,107").split(",")]
P = int(os.environ.get("S4DP", "3"))

import granite_engine as G  # noqa: E402  (装载模型 + 建描述符)

for p in range(P + 1):
    for pos in range(p + 1):
        G.forward(TOKS[pos], pos)
    mine = G.logits_of_x().copy()
    ref = np.frombuffer(open(f"/tmp/grec{p}/logits.bin", "rb").read(), np.float32)
    c = float(np.dot(mine / np.linalg.norm(mine), ref / np.linalg.norm(ref)))
    tm, tr = np.argsort(-mine)[:5], np.argsort(-ref)[:5]
    a_m, a_r = int(tm[0]), int(tr[0])
    print(f"pos={p}  cos={c:.5f}  argmax mine={a_m} ref={a_r} {'✓' if a_m == a_r else '✗'}  "
          f"top5 重合={len(set(tm.tolist()) & set(tr.tolist()))}/5", flush=True)
    if p == P:
        print(f"  mine top5={tm.tolist()} logit={mine[tm[0]]:.4f}")
        print(f"  ref  top5={tr.tolist()} logit={ref[tr[0]]:.4f}")
        print(f"  logits: max|Δ|={np.abs(mine-ref).max():.4f}  "
              f"mine std={mine.std():.4f} ref std={ref.std():.4f}")

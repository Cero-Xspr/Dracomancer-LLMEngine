#!/usr/bin/env python3
"""granite C 引擎 vs llama.cpp 的**逐层**对账：定位第一处偏离，并判断是"数值累积"还是"结构性跳变"。

判据：
  · 我的每层输出 vs dump 的 l_out-{il}：cos 应当逐层缓慢下滑（量化误差），
    若某层**突然**掉（比如 0.99 → 0.5），那一层就有结构性错误。
  · 再比 attn_norm-{il}（层输入）：若某层的**输入**已经错了，说明错误来自更早的层。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["GPROBE"] = "1"
TOKS = [int(v) for v in os.environ.get("TOKS", "100,101,102,103,104,105,106,107").split(",")]
P = int(os.environ.get("S4DP", "3"))

import granite_engine as G  # noqa: E402


def dump(il, name):
    hits = glob.glob(f"/tmp/grec{P}/{name}-{il}.*.bin")
    assert len(hits) == 1, (name, il, hits)
    return np.frombuffer(open(hits[0], "rb").read(), np.float32)


def cos(a, b):
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


for pos in range(P + 1):
    G.forward(TOKS[pos], pos)
mine = G.PROBE["layer_out"].copy()

print(f"\n=== pos={P}：每层最终输出 l_out-{{il}} vs dump ===")
print(f"{'il':>3} {'类型':>5} {'cos(l_out)':>11} {'max|Δ|':>9} {'‖mine‖':>8} {'‖ref‖':>8}")
prev = None
for il in range(G.NL):
    r = dump(il, "l_out")
    c = cos(mine[il], r)
    d = float(np.abs(mine[il] - r).max())
    tag = "attn" if il in G.ATTN_SET else "ssm"
    flag = ""
    if prev is not None and prev - c > 0.05:
        flag = "  ← ★ 骤降"
    print(f"{il:>3} {tag:>5} {c:>11.5f} {d:>9.4f} {np.linalg.norm(mine[il]):>8.3f} "
          f"{np.linalg.norm(r):>8.3f}{flag}")
    prev = c

print(f"\n=== 分支输出对比（前 8 层 + 所有 attn 层）===")
for il in list(range(8)) + [x for x in G.ATTN_SET if x >= 8]:
    name = "attn_out" if il in G.ATTN_SET else "mamba_out"
    try:
        r = dump(il, name)
    except AssertionError:
        print(f"{il:>3} {name}: 无 dump")
        continue
    print(f"{il:>3} {name:>10}  ‖ref‖={np.linalg.norm(r):>9.3f}")
print("\n（分支输出我的引擎不单独留探针；用 l_out 的骤降位置判断即可）")

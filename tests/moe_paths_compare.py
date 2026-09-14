#!/usr/bin/env python3
"""A1 的第二半：同一 MoE 层输入，比较**四个实现**是否语义一致。

为什么需要它：MoE 有几条并存的实现路径，谁在用哪条都不一样 ——
  · `m6_bailing_moe` —— 融合路由（C 侧自己算 sigmoid+偏置+分组 top2+top-NUSED）
  · `m6_moe_batch4` / `m6_moe_batch3` —— 批处理（路由权重由 Python 算好传进去）
  · 纯 Python 逐专家循环 —— **语义参考**（可读的那份定义）
如果它们之间有"陈旧数据/越界/顺序错"这类问题，症状会和 A1 前半段那个悬案一样：
不崩、不报错，只是数值微微偏 —— 所以要用参考实现钉住。

判据：**cos=1.0 且 max|Δ| 在 1~2 ULP 量级**（累加顺序/FMA 差异）算一致；
      任何更大的偏差都说明有实质分歧（不是"精度问题"）。

用法：python3 moe_paths_compare.py [模型GGUF] [层号]
"""
import os
import sys

import numpy as np

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
os.environ.setdefault("MODEL", sys.argv[1] if len(sys.argv) > 1
                      else "/media/xiao_/OverSys1/gguf/ling/Ling-3.0-tiny-Q4_K_M.gguf")
import ling_proto as LP  # noqa: E402

M6 = LP._M6
if M6 is None:
    print("引擎没加载（ling_proto 的 _M6 为 None）—— 这台机器上应该加载了，检查内核 .so")
    sys.exit(1)

want = int(sys.argv[2]) if len(sys.argv) > 2 else None
L = next((l for l in LP.LAYERS if "exp_ptrs" in l and (want is None or int(l["il"]) == want)), None)
if L is None:
    print("找不到带专家指针的 MoE 层")
    sys.exit(1)
print(f"层 {L['il']}  专家 {LP.NEXP} 选 {LP.NUSED}  组 {LP.N_GROUP}/{LP.N_GROUP_USED}  "
      f"norm_w={LP.NORM_W}  w_scale={LP.W_SCALE}  H={LP.H}")

cur = (np.random.RandomState(5).randn(LP.H).astype(np.float32) * 0.3)


def run(blmoe, batch4):
    LP._BLMOE, LP._BATCH4 = blmoe, batch4
    out = np.asarray(LP.moe_step_ling(cur, L), np.float32).copy()
    return out


LP._M6, keep = None, LP._M6                              # ① 纯 Python 参考
ref = np.asarray(LP.moe_step_ling(cur, L), np.float32).copy()
LP._M6 = keep
paths = (("m6_bailing_moe(融合路由)", run(True, True)),
         ("m6_moe_batch4", run(False, True)),
         ("m6_moe_batch3", run(False, False)))
LP._BLMOE, LP._BATCH4 = True, True                       # 还原默认

print(f"\n参考实现 |out|max = {np.abs(ref).max():.4f}   （差异判为"一致"的门槛：cos=1.0 且 1~2 ULP）")
print(f"{'实现':24s} {'max|Δ|':>11s} {'cos':>14s}  结论")
bad = 0
for tag, r in paths:
    d = float(np.abs(ref - r).max())
    cos = float(np.dot(ref, r) / (np.linalg.norm(ref) * np.linalg.norm(r) + 1e-30))
    # ULP 量级：ULP ≈ 2^-23 * 量级 ⇒ 允许 4 ULP
    ulp = 2 ** -23 * max(float(np.abs(ref).max()), 1e-6)
    ok = cos > 0.999999 and d <= 4 * ulp
    bad += 0 if ok else 1
    print(f"{tag:24s} {d:11.3e} {cos:14.8f}  {'✓ 一致（累加顺序级）' if ok else '✗ 实质分歧'}")
print(f"\n{'✅ 全部一致' if bad == 0 else f'❌ {bad} 条有实质分歧'}"
      f"（差异只应来自累加顺序/FMA，不应来自陈旧数据或越界）")

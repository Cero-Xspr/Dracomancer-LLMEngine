#!/usr/bin/env python3
"""k2_prefill_gate.py — 批量 prefill vs 引擎逐 token 路径的一致性闸门 + 速度对比。"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_engine as KE
import k2_prefill as KP
import falcon_tok

MODEL = os.environ["MODEL"]
tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")

# 与金标准同 prompt，另加一个中文短句测多 chunk 位置
CASES = [
    ("en-5tok", "The capital of France is"),
    ("zh-8tok", "中国的首都是北京，那么日本的首都是"),
]

pf = KP.Prefiller(KE)
ok_all = True
for name, prompt in CASES:
    ids = tk.encode(prompt).ids
    # 引擎逐 token（参照）
    KE.reset()
    t0 = time.perf_counter()
    for p, tid in enumerate(ids):
        KE.forward(int(tid), p)
    t_seq = time.perf_counter() - t0
    ref = np.asarray(KE.LOGITS).copy()
    # 批量 prefill（reset KV 再跑）
    KE.reset()
    t0 = time.perf_counter()
    lg_batch, _ = pf.prefill(ids, pos0=0)
    t_batch = time.perf_counter() - t0
    top_ref = np.argsort(-ref)[:8]
    top_bat = np.argsort(-lg_batch)[:8]
    rel = np.abs(ref - lg_batch).max() / (np.abs(ref).max() + 1e-30)
    ok = int(top_bat[0]) == int(top_ref[0]) and rel < 5e-3
    ok_all = ok_all and ok
    print(f"[{name}] T={len(ids)} 逐token {t_seq*1000:.0f}ms  批量 {t_batch*1000:.0f}ms "
          f"({t_seq/t_batch:.1f}x)  rel={rel:.2e}  top1 {int(top_bat[0])} vs {int(top_ref[0])} "
          f"{'PASS' if ok else '*** FAIL'}")
    print(f"   ref top8: {[int(i) for i in top_ref]}")
    print(f"   bat top8: {[int(i) for i in top_bat]}")

print("\n总体:", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)

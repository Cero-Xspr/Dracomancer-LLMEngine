#!/usr/bin/env python3
"""K2-Horizon 闸门：引擎贪心 vs 金标准（numpy oracle，经 torch 机制对拍校准）。"""
import os, sys, json, time
assert os.environ.get("MODEL"), "需要 MODEL 环境变量"
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_engine as KE
import falcon_tok

tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
gold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "tests/golden/k2horizon_golden.json")))
ids = gold["ids"]
N = int(os.environ.get("NGEN", str(len(gold["greedy8"]))))

KE.reset()
t0 = time.perf_counter()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p)
pref = time.perf_counter() - t0
lg0 = np.asarray(KE.LOGITS).copy()
gen = []
t0 = time.perf_counter()
for i in range(N):
    nid = int(np.asarray(KE.LOGITS).argmax())
    gen.append(nid)
    KE.forward(nid, len(ids) + i)
dt = (time.perf_counter() - t0) / max(1, N)

# prompt 末位 logits 的 top8 与金标准比对（宽松：top1 必须一致）
gold_top1 = gold["last_top8"][0][0]
top1 = int(np.argsort(-lg0)[0])

ok_greedy = gen[:N] == gold["greedy8"][:N]
ok_top1 = top1 == gold_top1
print(f"[k2] prefill {len(ids)}tok {pref:.1f}s  decode {dt*1000:.0f} ms/tok")
print(f"  引擎贪心: {gen}")
print(f"  金标准  : {gold['greedy8'][:N]}")
print(f"  引擎 top1={top1}  金标准 top1={gold_top1}")
print(f"  {'PASS' if (ok_greedy and ok_top1) else 'FAIL'}")
sys.exit(0 if (ok_greedy and ok_top1) else 1)

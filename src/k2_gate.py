#!/usr/bin/env python3
"""K2-Horizon 闸门 v3：引擎 vs 批量 numpy oracle。
比对：① prompt 末位 top1/top8 ② 全部 48 层**末 token** 的 |x|.sum 与 |x|².sum 校验和。
"""
import os, sys, json, time
assert os.environ.get("MODEL"), "需要 MODEL 环境变量"
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_engine as KE
import falcon_tok

tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
gold = json.load(open(os.environ.get("GOLDEN", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "tests/golden/k2horizon_golden.json"))))
ids = gold["ids"]

sums = {}


def hook(li, x):
    xa = np.asarray(x)
    sums[li] = (float(np.abs(xa[0] if xa.ndim == 2 else xa).sum()),
                float(np.square(xa[0] if xa.ndim == 2 else xa).sum()))


KE.reset()
t0 = time.perf_counter()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p, layer_hook=hook if p == len(ids) - 1 else None)
pref = time.perf_counter() - t0
lg0 = np.asarray(KE.LOGITS).copy()

top8 = np.argsort(-lg0)[:8]
gold_top8 = [t[0] for t in gold["last_top8"]]
top1_ok = int(top8[0]) == gold_top8[0]
top5_ok = set(int(i) for i in top8[:5]) >= set(gold_top8[:3])

# 逐层校验和
ok_layer, n_bad = True, 0
worst = 0.0
for li, s1, s2 in gold["layer_sums"]:
    e1, e2 = sums.get(li, (float("nan"), float("nan")))
    r1 = abs(e1 - s1) / max(abs(s1), 1e-9)
    r2 = abs(e2 - s2) / max(abs(s2), 1e-9)
    worst = max(worst, r1)
    if r1 > 2e-4 or r2 > 2e-4:
        ok_layer = False
        n_bad += 1
        if n_bad <= 5:
            print(f"  层{li}: |x|.sum rel={r1:.2e}  |x|².sum rel={r2:.2e}")

print(f"[k2] prefill {len(ids)}tok {pref:.1f}s")
print(f"  引擎 top8: {[int(i) for i in top8]}")
print(f"  金标准top8: {gold_top8}")
print(f"  top1 {'一致' if top1_ok else '不一致'}; 逐层校验和: {'48层全一致' if ok_layer else f'{n_bad}层不符'} (最大 rel={worst:.2e})")
ok = top1_ok and ok_layer
print(f"  {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)

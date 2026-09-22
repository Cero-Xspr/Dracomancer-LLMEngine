#!/usr/bin/env python3
"""k2_speed_long.py — 长生成分段计速（静机长上下文平坦度验收）。
采样固定 temp=0.7/top_p=0.95/seed=7（与档案一致）。分段报 t/s。

用法：MODEL=<gguf> MAXT=2048 N=1500 python3 k2_speed_long.py
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

N = int(os.environ.get("N", "1500"))
PROMPT = os.environ.get(
    "PROMPT",
    "Write a long, detailed, well-structured essay (at least 1500 words) about "
    "the history and future of renewable energy technology.")

import k2_engine as KE
import falcon_tok

tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
import re as _re
from jinja2.sandbox import ImmutableSandboxedEnvironment
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tests/k2ref/chat_template.jinja")).read()
src = _re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "",
              _re.sub(r"\{%-?\s*generation\s*-?%\}", "", src))
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
tm = env.from_string(src)
ids = tk.encode(tm.render(messages=[{"role": "user", "content": PROMPT}],
                          bos_token="<|ifm|begin_of_text|>",
                          eos_token="<|ifm|endoftext|>",
                          add_generation_prompt=True)).ids
print(f"prompt {len(ids)} tok  MAXT={KE.MAXT} N={N}", flush=True)

KE.reset()
t0 = time.perf_counter()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p)
pf = time.perf_counter() - t0
print(f"prefill {len(ids)} tok: {pf:.2f}s ({len(ids)/pf:.1f} t/s)", flush=True)

rng = random.Random(7)
gen, times = [], []
for i in range(N):
    lg = np.asarray(KE.LOGITS, np.float64) / 0.7
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    order = np.argsort(-p)
    cut = int(np.searchsorted(np.cumsum(p[order]), 0.95) + 1)
    keep = order[:cut]
    pk = p[keep] / p[keep].sum()
    nid = int(rng.choices(keep.tolist(), weights=pk.tolist())[0])
    if nid in (1, 250019):
        print(f"[EOS @ {len(gen)}]", flush=True)
        break
    t0 = time.perf_counter()
    gen.append(nid)
    KE.forward(nid, len(ids) + i)
    times.append(time.perf_counter() - t0)

n = len(gen)
if n:
    tot = sum(times)
    print(f"decode {n} tok: {tot:.2f}s ({n/tot:.2f} t/s, {tot/n*1000:.0f} ms/tok)", flush=True)
    # 分段平坦度
    segs = [(0, 50), (400, 500), (900, 1000), (1400, 1500)]
    for a, b in segs:
        if a >= n:
            break
        b = min(b, n)
        st = sum(times[a:b])
        print(f"  段 {a:4d}-{b:4d}: {b-a:3d} tok {st*1000/(b-a):6.1f} ms/tok "
              f"{(b-a)/st:5.2f} t/s", flush=True)
    if n > 1400:
        first = sum(times[:50]) / max(1, min(50, n))
        last = sum(times[-50:]) / 50
        print(f"平坦度: 首50 {first*1000:.0f}ms → 尾50 {last*1000:.0f}ms  "
              f"(掉幅 {last/first:.2f}×) {'PASS' if last/first < 1.4 else 'CHECK'}", flush=True)
print("文本头:", tk.decode(gen[:60])[:80], flush=True)

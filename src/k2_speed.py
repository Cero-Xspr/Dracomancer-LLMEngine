#!/usr/bin/env python3
"""k2_speed.py — 最小 decode 测速驱动（greedy，单 prompt，带 K2_PROF 汇总）。

用法：MODEL=<gguf> K2_FUSE=1 [K2_VK=1] [N=16] python3 k2_speed.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

N = int(os.environ.get("N", "16"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")

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
print(f"prompt {len(ids)} tok", flush=True)

KE.reset()
t0 = time.perf_counter()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p)
pf = time.perf_counter() - t0
print(f"prefill {len(ids)} tok: {pf:.2f}s ({len(ids)/pf:.1f} t/s)", flush=True)

gen = []
KE._PROF.clear()
t0 = time.perf_counter()
for i in range(N):
    nid = int(np.asarray(KE.LOGITS).argmax())
    if nid in (1, 250019):
        break
    gen.append(nid)
    KE.forward(nid, len(ids) + i)
dt = time.perf_counter() - t0
print(f"decode {len(gen)} tok: {dt:.2f}s -> {len(gen)/dt:.2f} t/s ({dt/max(1,len(gen))*1000:.0f} ms/tok)", flush=True)
print("文本:", tk.decode(gen)[:200], flush=True)
if os.environ.get("K2_PROF") == "1":
    tot = sum(KE._PROF.values())
    print(f"-- prof (合计 {tot*1000:.0f} ms/{len(gen)}tok) --", flush=True)
    for k, v in sorted(KE._PROF.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v*1000/len(gen):7.1f} ms/tok", flush=True)

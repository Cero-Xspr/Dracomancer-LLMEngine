#!/usr/bin/env python3
"""同一提示词、同一采样、无历史 —— 两个后端的**单轮**对拍。

为什么必须单轮：draco chat 每轮把整段历史发回去，而小模型首轮回答一旦分叉，
第二轮就不是"同输入比输出"了（我拿 135M 比过一次，得到的是噪声结论）。
用法：python3 backend_ab.py <模型子串> [max_tokens]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

KEY = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-360M"
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 32
BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
PROMPT = "What is 17*23? Answer with just the number."

sys.path.insert(0, BASE)
import draco  # noqa: E402

ms = draco.discover()
m = draco.pick_model(ms, KEY)
print(f"模型：{m.name}   {m.path}")

results = {}
for backend in ("dengine", "igpu", "cpu"):
    if backend not in draco.BACKENDS:
        continue
    if backend in ("cpu", "igpu") and not m.supported:
        continue
    try:
        srv = draco.Server(m, backend, 2048, 8, verbose=False)
    except SystemExit as e:
        print(f"[{backend}] 起不来：{e}")
        continue
    try:
        t0 = time.time()
        srv.wait_ready()
        load = time.time() - t0
        body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                           "max_tokens": MT, "temperature": 0, "seed": 0,
                           "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(srv.url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            js = json.loads(r.read())
        dt = time.time() - t0
        msg = js["choices"][0]["message"]
        tim = js.get("timings") or {}
        results[backend] = msg.get("content") or ""
        print(f"[{backend:8s}] 装载 {load:5.2f}s  首答 {dt:5.2f}s  "
              f"decode {tim.get('predicted_per_second')} t/s   {results[backend]!r}")
    except Exception as e:
        print(f"[{backend}] 请求失败：{type(e).__name__} {e}")
    finally:
        srv.stop()

print()
vals = {k: v.strip() for k, v in results.items()}
uniq = set(vals.values())
print(f"各后端答案：{vals}")
print("→ 全部一致 ✅" if len(uniq) == 1 else f"→ {len(uniq)} 种不同答案 ⚠️")

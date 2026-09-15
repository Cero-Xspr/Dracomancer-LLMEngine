#!/usr/bin/env python3
"""并发压测 e2e（手工跑，不进自动套件——要起服务装模型）：
并发 N 个**不同会话**的请求，验证 ①全部完成 ②答案正确 ③互不串状态（整代锁语义）。

用法：先起服务（任一 dengine 引擎），再：
    python3 tests/concurrency_e2e.py <port> [并发数]
"""
import json
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 34661
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
U = f"http://127.0.0.1:{PORT}/v1/chat/completions"

QA = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
      ("Spain", "Madrid"), ("Egypt", "Cairo")][:N]


def chat(msgs, mx=16):
    body = json.dumps({"messages": msgs, "max_tokens": mx, "temperature": 0}).encode()
    r = urllib.request.Request(U, data=body, headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    return d["choices"][0]["message"]["content"] or "", d["timings"]


results = [None] * N


def worker(i):
    country, expect = QA[i]
    try:
        t0 = time.time()
        a1, tm1 = chat([{"role": "user", "content": f"What is the capital of {country}? One word."}])
        a2, tm2 = chat([{"role": "user", "content": f"Reply with exactly one word: {expect}?"}])
        results[i] = (country, expect, a1.strip(), a2.strip(),
                      (expect.lower() in a1.lower()), tm1.get("prompt_cached", 0), time.time() - t0)
    except Exception as e:
        results[i] = (country, expect, f"ERROR {type(e).__name__}: {e}", "", False, 0, 0)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"并发 {N} 会话，墙钟 {time.time()-t0:.1f}s")
bad = 0
for r in results:
    country, expect, a1, a2, ok1, cached, dt = r
    ok = ok1 and expect.lower() in a2.lower()
    bad += 0 if ok else 1
    print(f"  {'✅' if ok else '❌'} {country:8s} 首答={a1[:24]!r:28s} 二答={a2[:24]!r:28s} "
          f"cached={cached} 本线 {dt:.1f}s")
if bad:
    print(f"❌ {bad}/{N} 失败（若为答案串线/错乱 ⇒ 整代锁或状态槽有 bug；若为 ERROR ⇒ 看服务端日志）")
    sys.exit(1)
print("✅ 并发串行语义正确：全部完成、答案无串线")

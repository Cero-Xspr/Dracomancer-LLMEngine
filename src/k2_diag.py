#!/usr/bin/env python3
"""k2_diag.py — K2 输出质量判别：区分 量化坏 / 模板错 / 模型本身不行。

用法: MODEL=<gguf> python3 k2_diag.py
四个探针（同一进程、顺序跑）:
  A. 官方模板 + 贪心 120 tok          —— 贪心是否循环/乱码
  B. 官方模板 + temp1.0/top_p0.95 256 —— 官方推荐采样下是否连贯（判别模型本身）
  C. reasoning_effort=fast + 采样     —— 思考更短是否改善
  D. 裸文本续写（无模板）+ 采样       —— 基座知识是否在
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import re
import k2_engine as KE
import falcon_tok
from jinja2.sandbox import ImmutableSandboxedEnvironment

def render(user, effort=None):
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tests/k2ref/chat_template.jinja")).read()
    src = re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "",
                 re.sub(r"\{%-?\s*generation\s*-?%\}", "", src))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    kw = {"reasoning_effort": effort} if effort else {}
    return env.from_string(src).render(
        messages=[{"role": "user", "content": user}],
        bos_token="<|ifm|begin_of_text|>", eos_token="<|ifm|endoftext|>",
        add_generation_prompt=True, **kw)

def top_p_sample(logits, temp=1.0, top_p=0.95):
    lg = np.asarray(logits, np.float64) / max(temp, 1e-6)
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    cut = int(np.searchsorted(csum, top_p) + 1)
    keep = order[:cut]
    pk = p[keep] / p[keep].sum()
    return int(np.random.choice(keep, p=pk))

def run(name, ids, n_gen, greedy=False, temp=1.0, top_p=0.95, skip_tags=("</ifm|think>",)):
    KE.reset()
    t0 = time.perf_counter()
    for p, tid in enumerate(ids):
        KE.forward(int(tid), p)
    pf = time.perf_counter() - t0
    gen, out_ids = [], []
    t0 = time.perf_counter()
    eos = 1
    think_closed = False
    answer_start = 0
    for i in range(n_gen):
        lg = np.asarray(KE.LOGITS)
        nid = int(lg.argmax()) if greedy else top_p_sample(lg, temp, top_p)
        gen.append(nid); out_ids.append(nid)
        if nid == eos:
            break
        # 记录思考块闭合点，分开显示思考与回答
        if not think_closed and nid == 250030:  # </ifm|think>
            think_closed = True
            answer_start = len(out_ids)
        KE.forward(nid, len(ids) + i)
    dt = (time.perf_counter() - t0) / max(1, len(gen))
    text = tk.decode(out_ids)
    print(f"\n=== {name} === prefill {len(ids)}tok/{pf:.1f}s  decode {dt*1000:.0f}ms/tok  n={len(gen)}")
    if think_closed:
        print(f"  [思考] {tk.decode(out_ids[:answer_start])[:600]!r}")
        print(f"  [回答] {tk.decode(out_ids[answer_start:])[:600]!r}")
    else:
        print(f"  [全文] {text[:600]!r}")
    return out_ids

tk, R = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")

p = render("Hello, who are you?")
ids = tk.encode(p).ids
print("模板渲染 token 数:", len(ids))
run("A 模板+贪心120", ids, 120, greedy=True)
np.random.seed(7)
run("B 模板+temp1.0/top_p0.95 256", ids, 256)

p2 = render("What is the capital of France?", effort="low")
ids2 = tk.encode(p2).ids
run("C fast思考+采样 200", ids2, 200)

raw = "The capital of France is"
ids3 = tk.encode(raw).ids
run("D 裸文本续写+采样 60", ids3, 60, temp=0.8, top_p=0.9)

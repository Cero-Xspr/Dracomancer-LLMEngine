#!/usr/bin/env python3
"""K2 模型能力探针：多领域 prompt × 采样模式 × 量化档对比。

用法：
  MODEL=<gguf> python3 k2_bench.py [--max-tokens 200] [--save results.json]
                                   [--sample]      # temp=1.0/top_p=0.95（官方推荐）
                                   [--effort low|medium|high]
探针覆盖：事实 QA / 推理 / 代码 / 中文 / 摘要 / 创意。
推理模型：思考在 <ifm|think>…</ifm|think>，答案在其后——分开统计与展示。
"""
import os, sys, json, time, argparse, re
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROBES = [
    ("事实-QA",   "What is the capital of France? Answer in one sentence."),
    ("事实-QA2",  "Who wrote Romeo and Juliet?"),
    ("推理",      "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?"),
    ("推理2",     "Sarah has 3 apples. She buys 5 more and gives 2 to her friend. How many apples does she have?"),
    ("代码-py",   "Write a Python function that reverses a string."),
    ("中文-QA",   "中国的首都是哪里？请用一句话回答。"),
    ("中文-翻译", "Translate to English: 今天天气很好，我们一起去公园散步吧。"),
    ("摘要",      "Summarize in one sentence: The Industrial Revolution began in Britain in the late 18th century. It transformed manufacturing from hand production to machine production, leading to urbanization and significant social changes."),
    ("创意",      "Write a haiku about autumn leaves."),
]

_THINK_OPEN = {250029, 250050, 250052}
_THINK_CLOSE = {250030, 250051, 250053}


def load_template():
    """官方 chat_template.jinja（tests/k2ref），剥掉 transformers 专用 generation 标签。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/k2ref/chat_template.jinja")
    src = open(p).read()
    src = re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "",
                 re.sub(r"\{%-?\s*generation\s*-?%\}", "", src))
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    return env.from_string(src)


def top_p_sample(logits, temp=1.0, top_p=0.95):
    lg = np.asarray(logits, np.float64) / max(temp, 1e-6)
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    order = np.argsort(-p)
    cut = int(np.searchsorted(np.cumsum(p[order]), top_p) + 1)
    keep = order[:cut]
    pk = p[keep] / p[keep].sum()
    return int(np.random.choice(keep, p=pk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--save", default=None)
    ap.add_argument("--sample", action="store_true", help="temp=1.0 top_p=0.95（官方推荐）")
    ap.add_argument("--effort", default=None, choices=["high", "medium", "low"])
    args = ap.parse_args()

    import k2_engine as KE
    import falcon_tok
    tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
    tm = load_template()
    kw = {"reasoning_effort": args.effort} if args.effort else {}

    def render(user):
        return tm.render(messages=[{"role": "user", "content": user}],
                         bos_token="<|ifm|begin_of_text|>",
                         eos_token="<|ifm|endoftext|>",
                         add_generation_prompt=True, **kw)

    results = []
    sum_pf = sum_gen = sum_tok = 0.0
    n_tok = 0
    for pi, (label, prompt) in enumerate(PROBES):
        ids = tk.encode(render(prompt)).ids
        KE.reset()
        t0 = time.perf_counter()
        for p, tid in enumerate(ids):
            KE.forward(int(tid), p)
        ttft = time.perf_counter() - t0
        sum_pf += ttft

        if args.sample:
            np.random.seed(1234 + pi)
        gen, think_end, answer_start = [], None, -1
        t0 = time.perf_counter()
        for i in range(args.max_tokens):
            lg = np.asarray(KE.LOGITS)
            nid = int(lg.argmax()) if not args.sample else top_p_sample(lg)
            if nid in (1, 250019):   # endoftext / im_end
                break
            gen.append(nid)
            if nid in _THINK_OPEN:
                think_end = len(gen)
            elif think_end is not None and answer_start < 0 and nid in _THINK_CLOSE:
                answer_start = len(gen)
            KE.forward(nid, len(ids) + i)
        dt = time.perf_counter() - t0
        sum_gen += dt
        n_tok += len(gen)

        think = tk.decode(gen[:think_end]) if think_end else ""
        body_start = answer_start if answer_start > 0 else (think_end or 0)
        answer = tk.decode(gen[body_start:])
        tps = len(gen) / dt if dt > 0 else 0.0
        results.append({"label": label, "prompt": prompt, "think_chars": len(think),
                        "answer": answer.strip()[:400], "n_gen": len(gen),
                        "ttft_s": round(ttft, 2), "tps": round(tps, 1)})
        print(f"[{label}] {len(gen)}tok {tps:.1f}t/s TTFT {ttft:.1f}s 思考{len(think)}字", flush=True)
        print(f"  答: {answer.strip()[:220]!r}", flush=True)

    print(f"\n== 汇总: TTFT均值 {sum_pf/len(PROBES):.2f}s  decode均值 "
          f"{sum_gen/max(1,n_tok)*1000:.0f}ms/tok  采样="
          f"{'temp1.0/top_p0.95' if args.sample else 'greedy'} ==")
    if args.save:
        json.dump({"model": os.path.basename(KE.MODEL), "sample": args.sample,
                   "effort": args.effort, "results": results},
                  open(args.save, "w"), ensure_ascii=False, indent=1)
        print("已存", args.save)


if __name__ == "__main__":
    main()

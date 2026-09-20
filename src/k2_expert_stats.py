#!/usr/bin/env python3
"""k2_expert_stats.py — K2 剪枝可行性：统计 MoE(100×8) 与 MoVA(64×4) 的专家激活频率。

对一组多样化 prompt 逐 token 记录每层被选中的专家（MoE 与 MoVA 各一套），
输出：激活分布、top-k 覆盖率（若剪枝保留 top-k 专家，多少 token 的选择会被破坏）、
以及「频率重排后 top-k 静态路由」的近似质量上限。

用法: MODEL=<gguf> python3 k2_expert_stats.py [--prompts 多样化集]
"""
import os, sys, json, time, argparse, re
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "中国的首都是哪里？",
    "If a train travels 120 km in 1.5 hours, what is its average speed?",
    "Explain quantum entanglement in simple terms.",
    "Write a haiku about autumn leaves.",
    "Summarize: The Industrial Revolution began in Britain in the late 18th century.",
    "Translate to English: 今天天气很好。",
    "Who wrote Romeo and Juliet?",
    "def fibonacci(n): # complete this",
    "What causes rain?",
    "Tell me a short story about a cat.",
]


def load_template():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/k2ref/chat_template.jinja")
    src = open(p).read()
    src = re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "",
                 re.sub(r"\{%-?\s*generation\s*-?%\}", "", src))
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    return env.from_string(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    import k2_engine as KE
    import falcon_tok
    tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
    tm = load_template()

    NL = KE.NL
    moe_counts = np.zeros((NL, KE.NEXP), np.int64)
    mova_counts = np.zeros((NL, KE.MEXP), np.int64)
    n_tok = 0

    orig_attn = KE.attn_ffn_common
    orig_moe = KE.moe_ffn

    def attn_rec(x, L, pos):
        h = KE.grouped_rms1(x, L.norm_a)
        if L.sparse:
            vl = KE.gemv1(L.vgc, L.vgb, KE.MEXP, KE.H, h)
            sc = KE.sigmoid1(vl) if KE.GATING_FUNC == 2 else KE.softmax(vl)
            sel = np.argsort(-(sc + L.vgate_b), kind="stable")[:KE.MUSED]
            mova_counts[L.li, sel] += 1
        return orig_attn(x, L, pos)

    def moe_rec(h2, L):
        logits = L.gate_inp @ h2
        sc = KE.sigmoid1(logits) if KE.GATING_FUNC == 2 else KE.softmax(logits)
        sel = np.argsort(-(sc + L.probs_b), kind="stable")[:KE.NUSED]
        moe_counts[L.li, sel] += 1
        return orig_moe(h2, L)

    KE.attn_ffn_common = attn_rec
    KE.moe_ffn = moe_rec

    t0 = time.time()
    for prompt in PROMPTS:
        ids = tk.encode(tm.render(
            messages=[{"role": "user", "content": prompt}],
            bos_token="<|ifm|begin_of_text|>", eos_token="<|ifm|endoftext|>",
            add_generation_prompt=True)).ids
        KE.reset()
        for p, tid in enumerate(ids):
            KE.forward(int(tid), p)
        for i in range(args.max_tokens):
            nid = int(np.asarray(KE.LOGITS).argmax())
            if nid in (1, 250019):
                break
            n_tok += 1
            KE.forward(nid, len(ids) + i)
    dt = time.time() - t0
    print(f"[k2] {len(PROMPTS)} prompts, {n_tok} 生成 token, {dt:.0f}s", flush=True)

    res = {"n_tokens": n_tok, "moe_top_coverage": [], "mova_top_coverage": [],
           "moe_top8": {}, "mova_top4": {}}
    print(f"\n层 | MoE: top-k专家(按频率) 覆盖率@8/16/32 | MoVA: 覆盖率@4/8/16")
    cov8_all, cov16_all, cov32_all = [], [], []
    mc4_all, mc8_all, mc16_all = [], [], []
    for li in range(3, NL):
        m = moe_counts[li]; total = m.sum()
        order = np.argsort(-m)
        c8 = m[order[:8]].sum() / total; c16 = m[order[:16]].sum() / total; c32 = m[order[:32]].sum() / total
        cov8_all.append(c8); cov16_all.append(c16); cov32_all.append(c32)
        res["moe_top8"][li] = [int(i) for i in order[:8]]
        v = mova_counts[li]; vt = v.sum()
        vo = np.argsort(-v)
        c4 = v[vo[:4]].sum() / vt; c8 = v[vo[:8]].sum() / vt; c16 = v[vo[:16]].sum() / vt
        mc4_all.append(c4); mc8_all.append(c8); mc16_all.append(c16)
        res["mova_top4"][li] = [int(i) for i in vo[:4]]
        if li % 8 == 3:
            print(f"{li:3d} | {c8:.2f}/{c16:.2f}/{c32:.2f} | MoVA {c4:.2f}/{c8:.2f}/{c16:.2f}")
    print(f"\n均值: MoE 覆盖率@8={np.mean(cov8_all):.3f} @16={np.mean(cov16_all):.3f} @32={np.mean(cov32_all):.3f}")
    print(f"      MoVA 覆盖率@4={np.mean(mc4_all):.3f} @8={np.mean(mc8_all):.3f} @16={np.mean(mc16_all):.3f}")
    print("""
剪枝判读（对 12 个日常 prompt）:
  · MoE @8 覆盖率 = 若只保留每层最高频的 8 个专家（-92.5% 专家内存），日常流量中被命中原 top-8 选择的比例。
    注意：这是「频率重排」的上限口径，剪枝后的真实质量需要重对账。
  · MoVA @4 同理（v 专家内存 -87.5%）。""")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/k2_expert_stats.json")
    json.dump(res, open(out, "w"))
    print("已存", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""k2_neuron_stats.py — 神经元级剪枝可行性：测 MoE 中间激活 (silu(g)*u) 的集中度。

对多样化 prompt 收集每 (层,专家,神经元) 的 |激活| 统计，
若 top-k 神经元承载绝大部分质量 ⇒ 按 kept 行剪 gate/up、按 kept 列剪 down，
每 token 字节流量按比例下降（这是激活参数剪枝的正确打开方式）。
"""
import os, sys, time, argparse
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
    "Solve: 2x + 6 = 20, find x.",
    "List three uses of copper in electronics.",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query joining orders and customers.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    import k2_engine as KE
    import falcon_tok
    # 模板
    import re
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/k2ref/chat_template.jinja")
    src = open(p).read()
    src = re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "", re.sub(r"\{%-?\s*generation\s*-?%\}", "", src))
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    tm = env.from_string(src)

    tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
    NL, NEXP, NEUR = KE.NL, KE.NEXP, KE.MOE_INTER
    sum1 = np.zeros((NL, NEXP, NEUR), np.float64)   # Σ|gu|
    cnt = np.zeros((NL, NEXP), np.int64)

    import k2_engine  # noqa
    # 简化：直接复刻引擎 moe（带 rw），统计在旁边记
    def moe_full(h2, L):
        logits = L.gate_inp @ h2
        sc = KE.sigmoid1(logits) if KE.GATING_FUNC == 2 else KE.softmax(logits)
        sel = np.argsort(-(sc + L.probs_b), kind="stable")[:KE.NUSED]
        rw = sc[sel]; rw = rw / rw.sum() * KE.R_SCALING
        out = np.zeros(KE.H, np.float32)
        for k, e in enumerate(sel):
            g = KE.gemv1(L.exc, L.exb[e * L.ex_per:], KE.MOE_INTER, KE.H, h2)
            u = KE.gemv1(L.uxc, L.uxb[e * L.ux_per:], KE.MOE_INTER, KE.H, h2)
            gu = KE.silu(g) * u
            sum1[L.li, e] += np.abs(gu)
            cnt[L.li, e] += 1
            d = KE.gemv1(L.dxc, L.dxb[e * L.dx_per:], KE.H, KE.MOE_INTER, gu)
            out += d * rw[k]
        g = KE.gemv1(L.sgc, L.sgb, KE.MOE_INTER, KE.H, h2)
        u = KE.gemv1(L.suc, L.sub, KE.MOE_INTER, KE.H, h2)
        d = KE.gemv1(L.sdc, L.sdb, KE.H, KE.MOE_INTER, KE.silu(g) * u)
        return out + d
    KE.moe_ffn = moe_full

    t0 = time.time()
    n_tok = 0
    for prompt in PROMPTS:
        ids = tk.encode(tm.render(messages=[{"role": "user", "content": prompt}],
                                  bos_token="<|ifm|begin_of_text|>",
                                  eos_token="<|ifm|endoftext|>",
                                  add_generation_prompt=True)).ids
        KE.reset()
        for p, tid in enumerate(ids):
            KE.forward(int(tid), p)
        for i in range(args.max_tokens):
            nid = int(np.asarray(KE.LOGITS).argmax())
            if nid in (1, 250019): break
            n_tok += 1
            KE.forward(nid, len(ids) + i)
    dt = time.time() - t0
    print(f"[k2] {len(PROMPTS)} prompts {n_tok} tok {dt:.0f}s")

    # 分析：每 (层,专家) 激活质量的 top-k 神经元占比曲线
    fracs = [0.125, 0.25, 0.5]
    agg = {f: [] for f in fracs}
    sparse50 = 0; total_pairs = 0
    for li in range(3, NL):
        for e in range(NEXP):
            if cnt[li, e] == 0: continue
            imp = sum1[li, e] / cnt[li, e]
            s = np.sort(imp)[::-1]
            tot = s.sum()
            if tot <= 0: continue
            total_pairs += 1
            for f in fracs:
                agg[f].append(s[:int(NEUR*f)].sum() / tot)
            if (s[:NEUR//2].sum() / tot) > 0.9: sparse50 += 1
    print(f"\n激活质量集中度（{total_pairs} 个 层×专家 对）:")
    for f in fracs:
        print(f"  top {int(f*100):3d}% 神经元承载质量: 均值 {np.mean(agg[f]):.3f}  中位 {np.median(agg[f]):.3f}")
    print(f"  top50% 质量 >90% 的 (层,专家) 占比: {sparse50}/{total_pairs} = {sparse50/max(1,total_pairs):.2f}")
    print("""
判读：
  · 若 top50% ≥ ~0.93：砍 50% 神经元（gate/up 砍行、down 砍列）≈ 误差可控，
    每 token 专家字节 -50% ⇒ decode 直接接近 2×。
  · 若 top50% ~0.8：剪枝后误差显著，需要校准式剪枝（最小化输出差）才可能。
  · 注意：剪枝收益 = 字节流减少，但质量损失需重跑闸门与诊断确认。""")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/k2_neuron_stats.npz")
    np.savez_compressed(out, sum1=sum1, cnt=cnt)
    print("已存", out)


if __name__ == "__main__":
    main()

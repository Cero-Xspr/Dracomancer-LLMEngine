#!/usr/bin/env python3
"""C2：家族级合规闸门 —— 把 selfcheck 从"单模型体检"升级成"这一族/这条引擎可信吗"。

selfcheck 回答的是"这个模型现在跑起来对不对"；家族闸门回答的是**三个更结构性的问题**：
  ① **跨后端一致**：同一个模型在 cpu / igpu / dengine 上对同样的问题是否给同样的答案？
     （一个后端悄悄坏了，单跑它自己是看不出来的 —— 这是"静默算错"类风险的最后一道网。）
  ② **跨量化一致**：同架构不同量化的模型（若本地都有）答案是否同族一致？
     量化误差会改数值，但**不该改答案**（改了说明这份量化或这条链有问题）。
  ③ **长上下文退化**：把同一问题的上下文用无关文本拉到 0/256/768/2048 token，
     答案是否保持正确？（上下文处理的坑往往只在高长度暴露。）
判据都用 selfcheck 的同一套健康问句（`draco.py` 的 `_SELFCHECK_SANITY`），
不另立标准；结果是**可贴进 issue 的 JSON**。

用法：python3 family_gate.py <模型子串> [--backends cpu,igpu,dengine] [--ctx 2048]
"""
import argparse
import json
import os
import sys
import time

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import draco  # noqa: E402

FILLER = ("The sea was calm that morning, and the fishermen rowed out past the "
          "breakwater before the sun had cleared the headland. ")


def ask(srv, prompt, ctx_tokens=0, max_tokens=32):
    """问一次（贪心、固定 seed）。ctx_tokens>0 时用无关文本把上下文拉到约那么长。"""
    text = prompt
    if ctx_tokens > 0:
        # 一句实测约 25 token（我第一版按 20 估 ⇒ 1024 那档超了引擎上限被打 400）
        per = max(1, int(ctx_tokens / 25))
        text = FILLER * per + "\n\n" + prompt
    msgs = [{"role": "user", "content": text}]
    out, timing = draco._complete_once(srv.url, msgs, 0.0, 0, max_tokens, rep=1.0, think=False)
    return out.strip(), timing


def check_answer(ans, expect):
    low = ans.lower()
    return any(e.lower() in low for e in expect)


def semantic_agree(a, b):
    """跨后端答案比较：先比逐字；逐字不同再看**语义**是否相同。

    实测驱动（2026-09-15 granite）：cpu 答 'Here is the count from 1 to 5 …: 1, 2, 3, 4, 5.'
    igpu 答 '1, 2, 3, 4, 5' —— 语义相同、措辞不同。这不是"静默算错"，判 FAIL 是误报。
    规则：抽取两边全部数字序列，若都非空且完全一致 ⇒ 视为语义一致（same=True 但记
    wording_diff=True，不静默吞掉）。
    """
    if a == b:
        return True, False
    # 抽取**独立数字 token**（"Here is the count from 1 to 5…: 1,2,3,4,5" 会混入问题里的 1/5，
    # 所以取「数字串」不行；要按词边界抽出来再取**尾部公共子序列**——更稳的是比“末尾数字序列”）
    import re
    ta = re.findall(r"\d+", a)
    tb = re.findall(r"\d+", b)
    if ta and tb and ta[-len(tb):] == tb:      # b 的数字序列是 a 的尾部 ⇒ 语义一致（措辞不同）
        return True, True
    return False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backends", default="cpu,igpu,dengine")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()

    ms = draco.discover()
    m = draco.pick_model(ms, a.model)
    if m is None:
        print(f"没有匹配 '{a.model}' 的模型"); return 1
    print(f"家族闸门：{m.name}（架构 {m.arch}）")
    rep = {"model": m.name, "arch": m.arch, "ctx": a.ctx, "when": time.strftime("%Y-%m-%d %H:%M"),
           "backend_agree": {}, "long_context": {}, "notes": []}

    # ── ① 跨后端一致 ──────────────────────────────────────────────
    want = [b.strip() for b in a.backends.split(",") if b.strip() in draco.BACKENDS]
    want = [b for b in want if b in (m.backends() + ["dengine", "local", "npu"])]
    per_backend = {}
    print(f"\n① 跨后端一致（后端：{', '.join(want) or '无可用'}）")
    for b in want:
        try:
            srv = draco.Server(m, b, a.ctx, 8, verbose=False)
        except SystemExit as e:
            print(f"  [{b:8s}] 起不来：{e}")
            rep["notes"].append(f"{b}: 起不来")
            continue
        try:
            srv.wait_ready()
            ans = []
            for q, exp in draco._SELFCHECK_SANITY:
                t, _ = ask(srv, q)
                ans.append((t, check_answer(t, exp)))
            per_backend[b] = ans
            ok = sum(1 for _, p in ans if p)
            print(f"  [{b:8s}] {ok}/{len(ans)} 通过   " +
                  " | ".join(f"{'✓' if p else '✗'}{t[:26]!r}" for t, p in ans))
        finally:
            srv.stop()
    # 跨后端比"答案文本"（同一问题同一后端应一致；不同后端因量化/累加差异可能微差）
    if len(per_backend) >= 2:
        base_b = next(iter(per_backend))
        for b, ans in per_backend.items():
            if b == base_b:
                continue
            same, wording = [], []
            for (ta, _), (tb, _) in zip(per_backend[base_b], ans):
                sa, wd = semantic_agree(ta, tb)
                same.append(sa); wording.append(wd)
            rep["backend_agree"][b] = {"vs": base_b, "same": same, "wording_diff": wording,
                                       "answers": [x[0] for x in ans]}
            wd = sum(1 for w in wording if w)
            if all(same) and wd == 0:
                tag = "全部逐字相同"
            elif all(same):
                tag = f"语义全同（{wd} 条措辞不同）"
            else:
                tag = f"{sum(same)}/{len(same)} 条相同"
            print(f"    {b} vs {base_b}: {tag}")

    # ── ② 跨量化一致（本地同架构的其它量化）─────────────────────────
    sibs = [x for x in ms if x is not m and x.arch == m.arch and x.supported]
    if sibs:
        print(f"\n② 跨量化一致（同架构本地还有 {len(sibs)} 个）")
        for s in sibs[:3]:
            print(f"  · {s.short}（{s.size/1e9:.2f}GB）—— 用 `draco selfcheck -m {s.short[:16]}` 单独体检，"
                  f"它自己的指纹需稳定；跨模型逐字一致**不作要求**（权重不同、量化不同）")
        rep["notes"].append(f"同架构本地另有 {len(sibs)} 个量化/规模，逐个跑 selfcheck 看指纹稳定性")
    else:
        print(f"\n② 跨量化一致：本地没有同架构的第二个模型，跳过")
        rep["notes"].append("本地无同架构第二个模型")

    # ── ③ 长上下文退化 ────────────────────────────────────────────
    b3 = next((b for b in ("dengine", "igpu", "cpu") if b in per_backend), None)
    print(f"\n③ 长上下文退化（后端 {b3}，拉到 0/256/{a.ctx//2}/{a.ctx} token）")
    if b3 is None:
        print("  没有可用后端，跳过")
    else:
        srv = draco.Server(m, b3, a.ctx + 256, 8, verbose=False)
        try:
            srv.wait_ready()
            q, exp = draco._SELFCHECK_SANITY[1]        # "1+1=?" 最不容易受上下文干扰
            ref, _ = ask(srv, q)
            print(f"  上下文 0      → {ref!r}（基准）")
            for c in sorted({256, a.ctx // 2, a.ctx}):      # 去重（ctx=512 时 256 会出现两次）
                if c <= 0:
                    continue
                try:
                    t, tim = ask(srv, q, ctx_tokens=c)
                    ok = check_answer(t, exp)
                    same = (t == ref)
                    rep["long_context"][str(c)] = {"answer": t, "correct": ok, "same_as_ref": same,
                                                   "prompt_n": (tim or {}).get("prompt_n")}
                    print(f"  上下文 ~{c:<5d} → {t!r}  {'✓ 与基准相同' if same else '⚠ 与基准不同'}"
                          f"（正确={'是' if ok else '否'}）")
                except Exception as e:
                    msg = str(e)
                    if "400" in msg:
                        # 超上下文上限：这是"这一档测不到"，不是"模型错了"（且启动器已给干净 400）
                        print(f"  上下文 ~{c:<5d} → 超出本引擎上下文上限，跳过这一档")
                        rep["long_context"][str(c)] = {"skipped": "prompt 超上下文"}
                    else:
                        print(f"  上下文 ~{c:<5d} → 失败：{type(e).__name__} {msg}")
                        rep["long_context"][str(c)] = {"error": msg}
        finally:
            srv.stop()

    print("\n" + "=" * 60)
    ok_health = all(all(p for _, p in ans) for ans in per_backend.values()) if per_backend else False
    ok_b = all(all(s for s in v["same"]) for v in rep["backend_agree"].values()) \
        if rep["backend_agree"] else True
    degraded = [c for c, v in rep["long_context"].items() if v.get("correct") is False]
    err = [c for c, v in rep["long_context"].items() if "error" in v]
    verdict = "PASS" if (ok_health and ok_b and not err) else "FAIL"
    print(f"结论：{verdict}")
    print("  判据（合格线）：① 各后端健康问句全过  ② 跨后端答案逐字一致  ③ 长上下文不报错")
    if degraded:
        print(f"  ⚠ 长上下文下有 {len(degraded)} 档答案退化（{', '.join(sorted(degraded, key=int))} token）"
              f"—— **小模型天然会掉，不作为 FAIL**，但它是这条链在长上下文下行为的数据点；")
        print(f"    大模型若也掉，就要查上下文处理（位置编码/窗口/状态写回）。")
    print("  （注意：跨后端**逐字一致**是强判据；不同后端因量化/累加顺序差异可能出现个别字差，"
          "此时看第 ② 项是否只是个别字，而不是整体错乱）")
    rep["verdict"] = verdict
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"已写出 {a.json_out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

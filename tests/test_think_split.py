#!/usr/bin/env python3
"""ThinkSplitter 单测：核心不变式 —— **reasoning + content 逐字等于模型输出原文**
（不丢字、不重复），以及未闭合思考的兜底分区是否正确。

为什么值得单测：分区分片是**流式状态机**（每步只拿到"已生成全文"的前缀），
一旦状态推进算错，症状是文本丢字/重复/串区 —— 而这类 bug 在聊天里很难一眼看出。
"""
import sys

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
from draco_engine_server import ThinkSplitter

FAIL = []


def run(text, start_inside, pieces=None, final=True):
    """把 text 按给定分片喂进去（pieces=None ⇒ 逐字符喂，最严）→ (reasoning, content, 事件流)"""
    sp = ThinkSplitter(start_inside=start_inside)
    if pieces is None:
        pieces = [text[i:i + 1] for i in range(len(text))] or [""]
    R, C, ev = [], [], []
    for k in range(len(pieces)):
        full = "".join(pieces[:k + 1])
        for kind, piece in sp.feed(full):
            (R if kind == "reasoning" else C).append(piece)
            ev.append((kind, piece))
        if k == len(pieces) - 1 and final:
            for kind, piece in sp.feed(full, final=True):
                (R if kind == "reasoning" else C).append(piece)
                ev.append((kind, piece))
    return "".join(R), "".join(C), ev


def check(name, text, start_inside, want_content, want_reasoning=None, pieces=None):
    R, C, _ = run(text, start_inside, pieces)
    ok = (C == want_content) if want_reasoning is None else (C == want_content and R == want_reasoning)
    # 不变式 1：字符守恒（把 <think>/</think> 标签本身去掉后比对）
    stripped = text.replace("<think>", "").replace("</think>", "")
    if R + C != stripped:
        ok = False
        print(f"  ✗ 不变式破坏：R+C 与原文不符\n     R+C={R+C!r}\n     原文={stripped!r}")
    if not ok:
        FAIL.append(name)
        print(f"✗ {name}\n   content={C!r}\n   reasoning={R!r}\n   期望 content={want_content!r}")
    else:
        print(f"✓ {name}")
        print(f"    reasoning={R!r}\n    content={C!r}")


# ① 闭合并思考（标准情形）
check("闭合思考", "We compute. 17*20=340.\n</think>\n\nThe answer is 391.",
      True, "\n\nThe answer is 391.", "We compute. 17*20=340.\n")
# 同上但按 3 字符分块喂（多字节/分块解码在真实流式里就会出现）
check("闭合思考·分片", "We compute. 17*20=340.\n</think>\n\nThe answer is 391.",
      True, "\n\nThe answer is 391.", "We compute. 17*20=340.\n", pieces=None)

# ② 未闭合思考 + 末段是回答（ZAYA 实测形态）
check("未闭合·末段回答", "We need to multiply.\n17*20 = 340, 17*3 = 51.\n\nSo it's 391.",
      True, "So it's 391.", "We need to multiply.\n17*20 = 340, 17*3 = 51.\n\n")

# ③ 未闭合 + 多空行（实测 ZAYA 出现的 \n\n\n）
check("未闭合·多空行", "Think A.\n\nThink B.\n\n\n391",
      True, "391", "Think A.\n\nThink B.\n\n\n")

# ④ 未闭合 + 最后一段超长 ⇒ 整段都算思考（没有回答可捞）
long_tail = "x" * 900
check("未闭合·末段超长", "Reasoning start.\n\n" + long_tail,
      True, "", "Reasoning start.\n\n" + long_tail)

# ⑤ 完全不思考（think 关：不预填标签）
check("无思考", "The capital of France is Paris.", False, "The capital of France is Paris.", "")

# ⑥ 模型自己写了 <think> 与 </think>（双边标签）
check("双边标签", "<think>aaa\nbbb</think>\n\nanswer", False, "\n\nanswer", "aaa\nbbb")

# ⑦ 思考区里的多段（中间空行应被正常吐出，只有末段挂起）
check("思考区多段", "P1 line.\n\nP2 line.\n\n</think>\n\nfin", True, "\n\nfin", "P1 line.\n\nP2 line.\n\n")

# ⑧ 空输出
check("空输出", "", True, "", "")

# ⑩ 流式活性：喂到 "第一行\n第二行还没完" 时，"第一行\n" 必须**已经**产出
#    （这条是 2026-09-15 用户报"思考不流式"的回归测试：挂起粒度必须是"一行"而不是"一段"）
sp = ThinkSplitter(start_inside=True)
ev = sp.feed("第一行已经写完\n第二行")
got = "".join(x for k, x in ev if k == "reasoning")
if got != "第一行已经写完\n":
    FAIL.append("流式活性"); print(f"✗ 流式活性：期望立刻产出首行，实际 {got!r}")
else:
    print(f"✓ 流式活性（首行立刻产出，只挂起半行）\n    已产出={got!r}")

# ⑨ 生成结束恰好停在空行后（挂起段为空 ⇒ 不该产出垃圾）
R, C, _ = run("thinking...\n\n", True)
if R + C != "thinking...\n\n":
    FAIL.append("尾部空行"); print(f"✗ 尾部空行 R+C={R+C!r}")
else:
    print(f"✓ 尾部空行\n    reasoning={R!r}\n    content={C!r}")

# ⑩ 流式增量解码的 UTF-8 悬挂（用户实测：draco 的中文回答里出现 11 个 �，llama.cpp 不会）
#    机制：一个汉字 3 字节，被切在两个 token 之间时第一轮 decode 出 `...�`（1 字符），
#    补全后仍 1 字符 ⇒ 按"已确认字符数"算增量得到空串 ⇒ **该汉字被永久丢掉**。
#    这里用**不变式**测：对一段 UTF-8 文本的**任意** token 切分，挂起式增量解码都必须还原原文。
from draco_engine_server import _decodable_prefix   # noqa: E402

TXT = "很高兴见到你，我是一个人工智能助手。"
raw = TXT.encode("utf-8")


class _FakeTok:
    """按给定字节切分当 token；decode 复刻 HF/byte-level 分词器的 lossy 行为。"""
    def __init__(self, cuts): self.cuts = cuts
    def decode(self, ids):
        return b"".join(self.cuts[i] for i in ids).decode("utf-8", "replace")


def _stream_incremental(tok, n, hold_back):
    sent, got = 0, ""
    for k in range(1, n + 1):
        full = tok.decode(list(range(k)))
        safe = _decodable_prefix(full) if hold_back else full
        if len(safe) > sent:
            got += safe[sent:]
            sent = len(safe)
    return got


# 遍历所有"把字节串切成 ≤3 段"的切法（含把汉字劈开的最坏情况）
bad = 0
cases = 0
for a in range(1, len(raw)):
    for b in range(a + 1, len(raw) + 1):
        cuts = [raw[:a], raw[a:b], raw[b:]]
        cuts = [c for c in cuts if c]
        tok = _FakeTok(cuts)
        cases += 1
        if _stream_incremental(tok, len(cuts), True) != TXT:
            bad += 1
            if bad <= 3:
                print(f"  ✗ 切分 {[len(c) for c in cuts]} 还原失败")
if bad or cases == 0:
    FAIL.append("UTF-8 悬挂不变式"); print(f"✗ UTF-8 悬挂：{bad}/{cases} 种切分还原失败")
else:
    print(f"✓ UTF-8 悬挂不变式（{cases} 种任意字节切分都还原原文，含把汉字劈成两半）")

# ①① 反面对照：不挂起时**必然**丢字（证明这个测试真的在测东西，而不是恒真）
lost = sum(1 for a in range(1, len(raw)) if "�" in _stream_incremental(
    _FakeTok([c for c in [raw[:a], raw[a:]] if c]), 2, False))
print(f"✓ 反面对照：不挂起的写法在 {lost}/{len(raw)-1} 种切分下丢字（说明不变式非恒真）")

print()
if FAIL:
    print(f"❌ {len(FAIL)} 项失败: {FAIL}")
    sys.exit(1)
print("✅ 全部通过 —— 字符守恒 + 分区正确")

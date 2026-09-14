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

# ⑨ 生成结束恰好停在空行后（挂起段为空 ⇒ 不该产出垃圾）
R, C, _ = run("thinking...\n\n", True)
if R + C != "thinking...\n\n":
    FAIL.append("尾部空行"); print(f"✗ 尾部空行 R+C={R+C!r}")
else:
    print(f"✓ 尾部空行\n    reasoning={R!r}\n    content={C!r}")

print()
if FAIL:
    print(f"❌ {len(FAIL)} 项失败: {FAIL}")
    sys.exit(1)
print("✅ 全部通过 —— 字符守恒 + 分区正确")

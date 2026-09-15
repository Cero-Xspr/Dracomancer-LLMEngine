#!/usr/bin/env python3
"""m6_engine 重构的回归测试：同一个 prompt、同一个 .so 版本 → 逐 token 的 logits 矢量对比。

为什么需要它：本轮重构把 `m6_bailing_moe` 的"三相位"抽成共用函数 `moe_experts`、
把 `m6_llama_attn_op` 抽成 `llama_attn_core`（加 scale 参数）。这是**共享代码路径**，
"没报错"完全不能说明数值没变（上次改共享签名就是靠自检才发现三个模型全空）。

用法（每个引擎各跑两次：旧 .so / 新 .so）：
    python3 m6_regress.py <engine 模块名> <so 路径> <输出 .npy>
然后比对两个 .npy 是否**逐位相同**。
"""
import os
import sys

import numpy as np

ENGINE, SO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NSTEP = int(os.environ.get("NSTEPS", "12"))
PROMPT = os.environ.get("PROMPT", "中国的首都是")
os.environ["M6_SO"] = SO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

E = __import__(ENGINE)
if ENGINE == "ling_engine":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/media/xiao_/OverSys1/npu-direct/hybrid/tok-ling")
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.BASEDIR + "tok-smol")

ids = tok.encode(PROMPT)
rows = []
for pos, tid in enumerate(ids):
    E.forward(int(tid), pos)
rows.append(np.asarray(E.logits_of_x(), np.float32).copy())
gen = [int(np.argmax(rows[-1]))]
for s in range(NSTEP):
    E.forward(int(gen[-1]), len(ids) + s)
    lg = np.asarray(E.logits_of_x(), np.float32).copy()
    rows.append(lg)
    gen.append(int(np.argmax(lg)))
np.save(OUT, np.stack(rows))
print(f"[{ENGINE} {os.path.basename(SO)}] 步数={len(rows)} argmax 序列={gen}")
print(f"  文本={tok.decode(gen)!r}")
print(f"  logits[0][:4]={rows[0][:4].tolist()}")

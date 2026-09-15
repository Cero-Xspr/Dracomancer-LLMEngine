#!/usr/bin/env python3
"""draco_engine_server —— 把**自研引擎**（m6 内核的 Python 驱动）包成 OpenAI 兼容的 HTTP 服务。

为什么存在：draco.py 的四个后端里没有自研引擎 —— cpu/igpu/local 走 llama-server、npu 走 FLM，
而我们自己的内核（m5_kern*/m6_engine）只有独立脚本。这个服务器补上那座桥：
draco 从此可以用 `-b dengine` 直接驱动**我们自己的引擎**跑 chat / serve / selfcheck / tune。

设计（都是刻意的）：
  · **只依赖**：标准库 + numpy + tokenizers(读 tokenizer.json) + 引擎模块本体。不引 FastAPI、不引 transformers。
  · **每请求全量重 prefill**：draco chat 每轮发完整历史，而 m6 的状态（KV）在 C 结构体里、
    没有 reset API —— 但状态本体（kc/vc/tlen）是 **Python 侧持有的 numpy 数组**，清零即可。
    135M 全量重 prefill 在本机 <1s，比引入会话状态管理简单一个量级。
  · **chat 模板手工渲染 ChatML**（SmolLM2 家族）。不引 jinja —— 我们的场景模板固定，
    手写渲染比引依赖更稳（模板字段变了会显式报错，而不是静默渲染错）。
  · SSE 流式 + 最终 chunk 带 llama-server 风格的 `timings`（draco 的速度显示直接吃这套）。

用法：
  python3 draco_engine_server.py --model <gguf> --port N [--engine smol]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.environ.get("DRACO_ENGINE_DIR", "/media/xiao_/OverSys1/npu-direct/hybrid")
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")

import numpy as np  # noqa: E402
import gguf_fast  # noqa: E402  ★ 元数据解析：gguf.GGUFReader 在 ZAYA(5.19GB/262k 词表) 上要 13.6s
                 #   且是纯 CPU（磁盘读 0 字节）—— 全花在"每字符串元素一次 numpy 封装"。
                 #   gguf_fast 0.29s；逐字节对账见 gguf_fast_check.py。适配器里两个 reader 都换掉。

ARGS = None
AP = None           # 引擎适配器（forward/logits/reset/encode/decode/eos/im_end）
MAXT = 1024
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def _tok_json(sub):
    """★ 直接用 `tokenizers` 读 tokenizer.json，**不 import transformers**。

    实测（2026-09-15）：`import transformers` 2.83s，而 `tokenizers` 0.01s + from_file 37~255ms。
    省下的 2.8s 是 smol/ling 装载时间的一半以上。正确性已逐 id 对账：对同一份 tokenizer.json，
    `tokenizers.Tokenizer.encode(s, add_special_tokens=True/False).ids` 与
    `AutoTokenizer.encode(...)` **完全一致**（含 eos/im_end/decode），所以这不是"近似"替换。
    """
    import tokenizers
    return tokenizers.Tokenizer.from_file(os.path.join(BASE, sub, "tokenizer.json"))


def _gguf_ids(path, *names):
    """从 GGUF 读 special token id（免 transformers 的 eos_token 查询）。"""
    r = gguf_fast.FastGGUF(path)
    out = []
    for n in names:
        f = r.fields.get(n)
        out.append(int(f.value) if f is not None else None)
    return out


def _template_renderer(model_path, think_default=True):
    """用模型**自带的** `tokenizer.chat_template` 渲染对话（jinja2）→ render(msgs, think)。

    ★★ 为什么 ZAYA 不能用我手写的 ChatML（2026-09-15，用户报"开启思考后回答不变白"）：
       ZAYA 的模板在生成前缀之前**固定**输出 `'\\n\\n<|im_start|>system\\n<|im_end|>\\n'`
       —— 也就是**一个空的 system 轮**。我手写的渲染器把它整个漏掉（只发 user + assistant 前缀），
       提示词于是偏离训练格式：模型照样答对，但**不写 `</think>`**，
       于是整段回答都留在 reasoning_content 里 ⇒ 表现就是"思考完的答案不变白"。
       （对账：smol 的模板与我手写的输出**逐字节一致**，所以 smol 没这个问题；
         ling 的格式压根不是 ChatML，本来就走 jinja。）
       ⇒ 结论：**有模板就用模板**，手写渲染只留作"GGUF 里没有 chat_template"的兜底。
    返回 None 表示该 GGUF 无模板（调用方退回手写渲染）。
    """
    r = gguf_fast.FastGGUF(model_path)
    f = r.fields.get("tokenizer.chat_template")
    if f is None:
        return None
    import jinja2
    tpl = jinja2.Environment().from_string(f.contents())
    # 模板里可能引用 <role>/<|role_end|> 这类字面量 token —— 从 token_id 反查文本喂进去
    vars_ = {}
    tf = r.fields.get("tokenizer.ggml.tokens")
    if tf is not None:
        toks = tf.contents()
        for name, ff in r.fields.items():
            if name.startswith("tokenizer.ggml.") and name.endswith("_token_id"):
                try:
                    vars_[name.split(".")[-1][:-3]] = toks[int(ff.value)]
                except Exception:
                    pass

    def render(msgs, think):
        return tpl.render(messages=list(msgs), add_generation_prompt=True,
                          enable_thinking=(think_default if think is None else bool(think)),
                          **vars_)
    return render


def _load_smol():
    """SmolLM2（llama 家族）→ smol_engine.py。状态在 KEEP 的 kc/vc/tlen，memset 即复位。"""
    global ENG, TOK, EOS_ID, IM_END, MAXT
    import os
    os.environ["MODEL"] = ARGS.model
    import smol_engine as E
    ENG = E
    TOK = _tok_json("tok-smol")
    EOS_ID = _gguf_ids(ARGS.model, "tokenizer.ggml.eos_token_id")[0]
    IM_END = TOK.token_to_id("<|im_end|>")
    if IM_END is None:
        IM_END = EOS_ID
    MAXT = int(getattr(E, "MAXT", 1024))

    def reset():
        import ctypes as ct
        for item in E.KEEP:
            if isinstance(item, E.M6LlamaAttnP):
                n = E.MAXT * item.n_kv * item.head_dim
                ct.memset(item.kcache, 0, n * 4)
                ct.memset(item.vcache, 0, n * 4)
                item.tlen[0] = 0
    return dict(forward=E.forward, logits=E.logits_of_x, reset=reset,
                encode=lambda t: TOK.encode(t, add_special_tokens=True).ids,
                decode=lambda ids: TOK.decode(ids), eos=EOS_ID, im_end=IM_END, max_t=MAXT,
                system_default=DEFAULT_SYSTEM, bos=None, think_block=False, think_default=False,
                render=lambda msgs, think: render_chatml_generic(msgs, think, None, DEFAULT_SYSTEM, False))


def _load_zaya():
    """ZAYA1-8B（CCA+MoE 自研架构）→ zaya_gguf.py（40 层 = 80 blk，RSS ~5GB）。

    ★ zaya_gguf 的三层坑：
      1. import 时会跑基准 ⇒ 必须先设 WARM=1/MEAS=1（否则装载 240s）；
      2. 它从 sys.argv[1] 读层数 ⇒ 先替换 argv；
      3. head/tokenizer 全在 `if GEN:` 块里 import 拿不到 ⇒ 这里自包含实现
        （rms+共享词嵌入 head 用 m5_gemv Q6_K；tokenizer = 全词表 Viterbi + 字节回退，
         从 GGUF 读 tokens/scores/types；这些逻辑与 zaya_gguf 的 GEN 块逐行同源）。
    ★ 状态复位：conv_state/kbuf/vbuf/vdel/tlen/RH 都是 Python 侧 numpy 数组（memset 即可），
      且 op 在 pos==0 按参考语义复位 conv_state。
    """
    global ENG, MAXT
    import os
    os.environ["MODEL"] = ARGS.model
    # ★ SKIP_BENCH：zaya_gguf 导入时会跑一段基准（40 层上 ~3s，含 16 次 decode），
    #   独立脚本/诊断要这些数字，桥接进程只要"装载 + forward" ⇒ 跳过。
    #   （把 WARM/MEAS 调到 1 也省不掉：那段是固定的 16 次 decode + 剖面 + 逐 token 计时。）
    os.environ["SKIP_BENCH"] = "1"
    os.environ.setdefault("WARM", "1")
    os.environ.setdefault("MEAS", "1")
    old_argv = sys.argv
    sys.argv = ["zaya_gguf"]
    try:
        import zaya_gguf as Z
    finally:
        sys.argv = old_argv
    ENG = Z
    import ctypes as ct
    import gguf as _g

    def reset():
        for pcp in Z.cca_ps:
            for f, n in (("conv_state", 2 * 1280), ("kbuf", 1024 * 256),
                         ("vbuf", 1024 * 256), ("vdel", 128)):
                buf = getattr(pcp, f)
                if buf:
                    ct.memset(ct.cast(buf, ct.c_void_p), 0, n * 4)
            if pcp.tlen:
                ct.cast(pcp.tlen, ct.POINTER(ct.c_int))[0] = 0
        Z.RH[:] = 0.0                      # EDA 递推状态

    # ---- head（共享词嵌入）：与 zaya_gguf GEN 块同源 ----
    onw = np.frombuffer(bytes(Z.T["output_norm.weight"].data), np.float32).copy()
    _emb_al = Z.tptr(Z._EMB_T)             # ★ 64 字节对齐；已对齐 ⇒ 零拷贝（省 420MB 常驻）
    _m5g = ct.CDLL(os.path.join(BASE, "m5", "m5_kern8.so"))
    _m5g.m5_gemv.restype = ct.c_int
    _m5g.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.c_void_p,
                             ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]

    def logits_of(x):
        v = np.ascontiguousarray(x, np.float32)
        s2 = float((v * v).mean())
        xn = v / np.sqrt(s2 + 1e-5) * onw
        xn = np.ascontiguousarray(xn, np.float32)
        lo = np.empty(vsz, np.float32)
        _m5g.m5_gemv(4, xn.ctypes.data_as(ct.POINTER(ct.c_float)), _emb_al, vsz, H_Z,
                     lo.ctypes.data_as(ct.POINTER(ct.c_float)))
        return lo

    # ---- tokenizer：全词表 Viterbi + 字节回退（GEN 块同源）----
    _r = gguf_fast.FastGGUF(ARGS.model)     # ★ 13.6s → 0.29s（同一个文件，gguf-py 每字符串一次 numpy 封装）
    _f = {t.name: t for t in _r.fields.values()}
    toks = [bytes(t) if isinstance(t, bytes) else t
            for t in _f["tokenizer.ggml.tokens"].contents()]
    toks = [t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t) for t in toks]
    tt = list(_f["tokenizer.ggml.token_type"].contents())
    sc = list(_f["tokenizer.ggml.scores"].contents())
    vsz = len(toks)
    pid = {}
    for i, t in enumerate(toks):
        if tt[i] in (1, 4, 5, 6) and t not in pid:
            pid[t] = (i, float(sc[i]))
    MAXP = max(len(k) for k in pid)
    BYTE = {}
    for b in range(256):
        k = f"<0x{b:02X}>"
        if k in pid:
            BYTE[b] = pid[k][0]

    def enc(text):
        spm = text.replace(" ", "\u2581")
        n = len(spm)
        NEG = -1e18
        best = [NEG] * (n + 1); prev = [-1] * (n + 1); pie = [-1] * (n + 1)
        best[0] = 0.0
        for i in range(n):
            if best[i] == NEG:
                continue
            for L in range(1, min(MAXP, n - i) + 1):
                e = pid.get(spm[i:i + L])
                if e is None:
                    continue
                if best[i] + e[1] > best[i + L]:
                    best[i + L] = best[i] + e[1]; prev[i + L] = i; pie[i + L] = e[0]
        if best[n] == NEG:
            return [pid[c][0] for c in spm if c in pid]
        out, j = [], n
        while j > 0:
            out.append(pie[j]); j = prev[j]
        return out[::-1]

    def decode(ids):
        return "".join(toks[i].replace("\u2581", " ") for i in ids)

    eos = int(_f["tokenizer.ggml.eos_token_id"].contents())
    tbl = {t: i for i, t in enumerate(toks)}
    im_end = tbl.get("<|im_end|>", eos)
    MAXT = int(getattr(Z, "ZMAXT", 1024))     # ★ 取引擎真实上下文（跟随 --ctx），别写死
    H_Z = Z.H

    # ★ zaya 的 step 返回隐状态、logits_of 需要它 ⇒ 闭包里存"最后隐状态"，
    #   对齐 smol 的 logits() 无参接口。
    last = {"x": None}

    def forward(tid, pos):
        last["x"] = Z.step(tid, pos)
        return last["x"]

    def logits():
        return logits_of(last["x"])

    # ★★ 渲染改用模型自己的模板（见 _template_renderer 的注释：手写渲染漏了模板开头的
    #    空 system 轮 ⇒ 模型不写 </think> ⇒ 回答不变白）。模板缺失才退回手写 ChatML。
    _rt = _template_renderer(ARGS.model, think_default=True)
    render = _rt or (lambda msgs, think:
                     render_chatml_generic(msgs, think, None, None, True))
    return dict(forward=forward, logits=logits, reset=reset,
                encode=enc, decode=decode, eos=eos, im_end=im_end, max_t=MAXT,
                system_default=None, bos=2, vsz=vsz,
                think_block=True, think_default=True,   # ZAYA 模板默认开思考，用空 think 块关
                render=render)


def _load_ling():
    """Ling 3.0 Tiny（bailingmoe3：MLA + KDA 线性注意力 + 分组 MoE）→ ling_engine.py。

    ★ 与 smol/zaya 的关键差别：**它的对话格式不是 ChatML**
      （实测模板渲染出 `<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>…<|role_end|><role>ASSISTANT</role>`）
      ⇒ 必须用 jinja2 跑模型自己的模板，手工渲染必错（这也是之前把这步推迟的原因）。
    ★ 状态复位：MLA 层的 kcache/vcache/tlen + KDA 层的 conv_state/S（都是 Python 侧缓冲的
      **副本**，memset 即可）。KDA 的 delta-net 状态 S 是 [NH,128,128]（1MB/层）。
    """
    global ENG, MAXT
    import os
    os.environ["MODEL"] = ARGS.model
    os.environ.setdefault("MAXT", "1024")
    import ling_engine as E
    import ling_proto as LP
    import jinja2
    import gguf as _g
    ENG = E
    tk = _tok_json("tok-ling")          # ★ 同 smol：免 transformers 的 2.8s 导入

    # ---- 模板（从 GGUF 读，jinja2 渲染）----
    render = _template_renderer(ARGS.model, think_default=True)
    if render is None:
        raise SystemExit("Ling 的 GGUF 里没有 chat_template —— 它的格式不是 ChatML，无法手写兜底")

    def reset():
        import ctypes as ct
        for item in E.KEEP:
            if isinstance(item, E.M6MlaP):
                ks = LP.KV_LORA + LP.ROT
                ct.memset(ct.cast(item.kcache, ct.c_void_p), 0, item.max_t * ks * 4)
                ct.memset(ct.cast(item.vcache, ct.c_void_p), 0, item.max_t * LP.KV_LORA * 4)
                item.tlen[0] = 0
            elif isinstance(item, E.M6KdaP):
                ct.memset(ct.cast(item.conv_state, ct.c_void_p), 0,
                          3 * LP.D_INNER * (LP.CONV_K - 1) * 4)
                ct.memset(ct.cast(item.S, ct.c_void_p), 0,
                          LP.NH * LP.KDA_HEAD * LP.KDA_HEAD * 4)

    _eos = _gguf_ids(ARGS.model, "tokenizer.ggml.eos_token_id")[0]
    MAXT = E.MAXT
    # ★★ think_block=True（2026-09-15 修正）：Bailing 模板的生成前缀同样是
    #    换行 + `<think>`（thinking on）或换行 + `<think></think>`（thinking off），
    #    也就是**开思考的 <think> 也是提示词预填的** —— 我先前设成 False，
    #    于是思考段被判成 content（不加 [思考] 前缀、不暗色），正是用户说的"Ling 的问题"。
    #    模板的历史格式也是 `<role>ASSISTANT</role>` + 换行 + `<think>{reasoning}</think>{content}`，
    #    与 ZAYA 同族约定。
    # think_default：模板里 enable_thinking 未给时 thinking_option='on' ⇒ 默认开，与模型一致
    #    （要直接给答案用 `/think off`，那会渲染成空 think 块）。
    return dict(forward=E.forward, logits=E.logits_of_x, reset=reset,
                encode=lambda t: tk.encode(t, add_special_tokens=False).ids,
                decode=lambda ids: tk.decode(ids), eos=_eos,
                im_end=_eos, max_t=MAXT,
                system_default=None, bos=None, think_block=True, think_default=True,
                render=render)


def _load_granite():
    """granite-h-tiny（granitehybrid：S4D/Mamba-1 × 36 + GQA 注意力 × 4 + MoE 64选6 + 共享专家）
    → granite_engine.py（C 侧 m6_granite_s4d_op / m6_granite_attn_op / m6_granite_moe）。

    ★ 三条与其它架构都不同的语义（都从 llama.cpp 逐行核对 + 实测对账，别照抄别的适配器）：
      · **注意力层是 NoPE**：GGUF `rope.scaling.finetuned=0` ⇒ llama.cpp 的 granite-hybrid 把
        rope_pattern 全填 false ⇒ 图里根本不加 rope。加 rope 反而错。
      · **kq_scale = 1/head_dim**（=attention.scale=0.0078125），不是 llama 系的 1/sqrt(head_dim)。
      · MoE 是 **softmax 路由**（Bailing 那套是 sigmoid）+ **没有 expert_weights_scale**。
    ★ 分词器不是 transformers 目录：直接用 GGUF 的 vocab/merges 搭（granite_tok.py），
      已与 llama-tokenize 逐 id 对账（9/9 串一致，含 <|start_of_role|> 这类特殊 token）。
    ★ 状态复位：S4D 的 conv 态 hist / ssm 态 hst、注意力 KV 缓存与 tlen，全是 Python 侧数组。
    """
    global ENG, MAXT
    import os
    os.environ["MODEL"] = ARGS.model
    os.environ.setdefault("MAXT", str(ARGS.ctx or 1024))
    import granite_engine as E
    import granite_tok as GT
    ENG = E
    tk, _R = GT.build(ARGS.model)

    render = _template_renderer(ARGS.model)
    if render is None:
        raise SystemExit("granite 的 GGUF 里没有 chat_template —— 无法渲染对话格式")

    def reset():
        E.reset()

    _eos = _gguf_ids(ARGS.model, "tokenizer.ggml.eos_token_id")[0]
    # granite-h-tiny **不是推理模型**：模板里没有 think/reasoning ⇒ 思考分区关闭
    #   （think_block/think_default=False；`/think on` 对它没有意义）
    return dict(forward=E.forward, logits=E.logits_of_x, reset=reset,
                encode=lambda t: tk.encode(t, add_special_tokens=False).ids,
                decode=lambda ids: tk.decode(ids, skip_special_tokens=False),
                eos=_eos, im_end=_eos, max_t=E.MAXT,
                system_default=None, bos=None,
                think_block=False, think_default=False,
                render=render)


def _load_falcon():
    """Falcon-H1（falcon-h1：每层 注意力∥Mamba-2 并行 + dense FFN，0.5B dense）→ falcon_engine.py。

    ★ falcon 专属语义（与 granite 同走 mamba2 图，但注意这四条）：
      · **NEOX rope** + freq_base≈1e11（llama.cpp 把 FALCON_H1 归在半区旋转那一段）
      · **没有 ssm_norm**（loader 里 TENSOR_NOT_REQUIRED ⇒ C 侧跳过分组归一化）
      · ffn_norm 的张量名**没有 .weight 后缀**
      · **没有任何 scale**（无 residual/embedding/logit scale），KV/激活全 fp32
      · tokenizer pre=falcon-h1 ⇒ llama3 同款正则 + **add_bos=17**（falcon_tok.py，
        与 llama-tokenize 逐 id 对账 9/9 串一致）
    ★ general.name 是无意义的 "Original" ⇒ draco 按架构 falcon-h1 匹配适配器。
    """
    global ENG, MAXT
    import os
    os.environ["MODEL"] = ARGS.model
    os.environ.setdefault("MAXT", str(ARGS.ctx or 1024))
    import falcon_engine as E
    import falcon_tok as FT
    ENG = E
    tk, _R = FT.build(ARGS.model)

    render = _template_renderer(ARGS.model)
    if render is None:
        raise SystemExit("falcon 的 GGUF 里没有 chat_template")

    def reset():
        E.reset()

    _eos = _gguf_ids(ARGS.model, "tokenizer.ggml.eos_token_id")[0]
    # 模板用 <|im_end|> 收尾：从词表里按文本找它的 id（找不到就退回 eos）
    toks = list(_R.fields["tokenizer.ggml.tokens"].value)
    im_end = toks.index("<|im_end|>") if "<|im_end|>" in toks else _eos
    # 非推理模型：模板无 think/reasoning ⇒ 思考分区关闭
    return dict(forward=E.forward, logits=E.logits_of_x, reset=reset,
                encode=lambda t: tk.encode(t).ids,
                decode=lambda ids: tk.decode(ids, skip_special_tokens=False),
                eos=_eos, im_end=im_end, max_t=E.MAXT,
                system_default=None, bos=None,
                think_block=False, think_default=False,
                render=render)


def load_engine():
    """按 --engine 分派到适配器。每个适配器返回统一的算子接口 dict。"""
    global AP, MAXT
    if ARGS.engine == "smol":
        AP = _load_smol()
    elif ARGS.engine == "zaya":
        AP = _load_zaya()
    elif ARGS.engine == "ling":
        AP = _load_ling()
    elif ARGS.engine == "granite":
        AP = _load_granite()
    elif ARGS.engine == "falcon":
        AP = _load_falcon()
    else:
        raise SystemExit(f"未知引擎 {ARGS.engine}（当前支持：smol / zaya / ling / granite / falcon）")
    MAXT = AP["max_t"]
    print(f"[SRV] 引擎就绪：{ARGS.model}  ctx<={MAXT}", flush=True)


def reset_state():
    """清零跨 token 状态（KV/卷积态/EDA）。每请求都调（全量重 prefill 语义）。

    ★ 各引擎的状态本体都是 Python 侧持有的 numpy 数组，C 结构体里只存指针 ⇒
    不改 C 引擎就有 reset；具体清什么由适配器的 reset() 决定。
    """
    AP["reset"]()


def render_chatml_generic(messages, enable_thinking, system_default, default_system, think_block):
    """ChatML（SmolLM2 与 ZAYA 同族）。★ 手写渲染而非 jinja：模板字段变了显式报错。

    ★★ 生成前缀必须**逐字照模型自己的模板**（2026-09-14 用户报「无法关思考、思考撑满 max_tokens」）：
        {%- if enable_thinking %}   <|im_start|>assistant\n<think>\n
        {%- else %}                 <|im_start|>assistant\n<think>\n</think>\n\n
    关思考靠的是**预填一个空的 think 块**（模型看到空块就直接作答）。我第一版只发
    `<|im_start|>assistant\n`，既没读 enable_thinking 也没预填 ⇒ 永远雷霆大思考。
    """
    parts = []
    msgs = list(messages)
    sysdef = system_default or default_system
    if sysdef and (not msgs or msgs[0].get("role") != "system"):
        msgs.insert(0, {"role": "system", "content": sysdef})
    for m in msgs:
        parts.append(f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>\n")
    think = True if enable_thinking is None else bool(enable_thinking)
    if think_block:
        parts.append("<|im_start|>assistant\n<think>\n" if think
                     else "<|im_start|>assistant\n<think>\n</think>\n\n")
    else:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def sample(logits, temp, seed_rng, repeat_penalty, recent):
    """温度采样 + 简单重复惩罚。temp=0 ⇒ 贪心。"""
    z = logits.astype(np.float64).copy()
    if repeat_penalty and repeat_penalty > 1.0 and recent:
        for t in list(recent)[-64:]:
            z[t] /= repeat_penalty if z[t] > 0 else z[t] * repeat_penalty
    if temp and temp > 0:
        p = np.exp((z - z.max()) / temp)
        p /= p.sum()
        # ★ random.Random.choice 没有 p= 参数（那是 numpy 的）—— 用 choices(weights=)
        return int(seed_rng.choices(range(len(p)), weights=p.tolist())[0])
    return int(np.argmax(z))


def generate(messages, max_tokens, temp, seed, repeat_penalty, enable_thinking=None, hold=None):
    """**同步**做完准备工作（渲染/分词/上下文预算），返回真正的生成器。

    ★ 为什么拆两层（2026-09-15 实测）：整个函数若是生成器，异常要等第一次 next() 才抛 ——
      那时 SSE 响应头已经发出去了，客户端只能看到"连接被意外关闭"（我实测撞到过：
      提示词超过上下文时，前端就是这个表现）。现在超长会在**发头之前**变成干净的 400 +
      明确文案（含实际 token 数与上下文）。
    """
    think_now = (AP.get("think_default", True) if enable_thinking is None
                 else bool(enable_thinking))
    # ★ 渲染交给适配器：有 chat_template 的模型一律用模型自己的 jinja 模板（见
    #   _template_renderer 的注释：手写 ChatML 漏过 ZAYA 的空 system 轮）。
    text = AP["render"](messages, enable_thinking)
    ids = AP["encode"](text)
    if AP.get("bos"):
        ids = [AP["bos"]] + ids
    budget = MAXT - len(ids) - 8
    if budget <= 0:
        raise ValueError(f"提示词太长：{len(ids)} token > 本引擎上下文 {MAXT}。"
                         f"（draco 侧可用 -c 调大，桥接会用同一个值）")
    n_gen = min(max_tokens if max_tokens and max_tokens > 0 else 256, budget)
    return _gen(ids, n_gen, temp, seed, repeat_penalty, think_now, hold)


def _decodable_prefix(text):
    """去掉尾部**可能不完整**的 UTF-8 字符（分词器解码时它已经被写成 U+FFFD 了）。

    ★ 为什么流式增量解码必须挂起它：一个汉字是 3 字节，如果被切在两个 token 之间，
      第一轮 `decode(ids)` 出来是 `...�`（1 个字符），第二轮补齐后是 `...很`（**还是 1 个字符**）
      ⇒ 按"已确认字符数"算增量会得到空串，**那个汉字被永久丢掉，只剩一个 �**。
      实测（granite 分词器）：`很高兴见到你` 切 8 个 token，现在写法产出 `�高�见到你`，
      挂起尾部后产出 `很高兴见到你` ✓。
    粒度是"尾部连续的一段 �"：只可能是"结尾处尚未补全的字节序列"（合法的 � 字符只会出现在
    正文中间），挂起最多延迟一轮；生成结束时 `feed(..., final=True)` 用完整文本冲刷，不丢东西。
    """
    n = 0
    while n < len(text) and text[-1 - n] == "\ufffd":
        n += 1
    return text[:len(text) - n] if n else text


def _gen(ids, n_gen, temp, seed, repeat_penalty, think_now, hold=None):
    """生成器：yield (kind, 文本增量, 计时dict)。"""
    rng = random.Random(seed if seed and seed > 0 else None)

    reset_state()
    t0 = time.time()
    for pos, tid in enumerate(ids):
        AP["forward"](int(tid), pos)
    prefill_s = time.time() - t0

    # ★ 真·逐 token 流式：每步解码"已生成前缀"再产出**新增的那段**。
    #   我之前是"整段生成完再切块 yield" —— 用户一眼看出没流式（而 llama-server 是流式的）。
    out, gen_ids = [], []
    t1 = time.time()
    t_first = None
    # ★ 可选分段计时（DRACO_ENG_PROF=1）：实测过"引擎裸跑 159.5 t/s、走桥接只有 123.4 t/s"，
    #   差的 1.8ms/token 必须落到具体一段上才谈得上优化（别猜）。
    _prof = os.environ.get("DRACO_ENG_PROF") == "1"
    _p = {"logits": 0.0, "sample": 0.0, "decode": 0.0, "split": 0.0, "fwd": 0.0}
    # ★ 分区分片是有状态的（见 ThinkSplitter：未闭合思考要挂起最后一段做兜底）
    sp = ThinkSplitter(start_inside=bool(AP.get("think_block")) and think_now, hold=hold)
    for i in range(n_gen):
        _q = time.perf_counter()
        lg = AP["logits"]()
        nid = sample(lg, temp, rng, repeat_penalty, gen_ids)
        if _prof:
            _q2 = time.perf_counter(); _p["logits"] += _q2 - _q
        if nid == AP["eos"] or nid == AP["im_end"]:
            break
        out.append(nid)
        gen_ids.append(nid)
        _q = time.perf_counter()
        full = AP["decode"](out)
        if _prof:
            _q2 = time.perf_counter(); _p["decode"] += _q2 - _q; _q = _q2
        full_safe = _decodable_prefix(full)      # ★ 见 _decodable_prefix：不挂起就会丢汉字
        if len(full_safe) > sp.sent:
            if t_first is None:
                t_first = time.time()
            for kind, piece in sp.feed(full_safe):
                if piece:
                    yield kind, piece, {}
        if _prof:
            _q2 = time.perf_counter(); _p["split"] += _q2 - _q; _q = _q2
        AP["forward"](nid, len(ids) + i)
        if _prof:
            _p["fwd"] += time.perf_counter() - _q
    full = AP["decode"](out)
    for kind, piece in sp.feed(full, final=True):    # ★ 收尾：未闭合的思考末段按 content 发出
        if piece:
            yield kind, piece, {}
    dec_s = time.time() - t1
    ttft_ms = round((t_first - t1) * 1000, 1) if t_first else None
    if _prof and out:
        n = len(out)
        _p["其余(HTTP/JSON/发生器)"] = dec_s - sum(_p.values())
        print("[PROF] 每 token 分摊 ms: "
              + "  ".join(f"{k}={v / n * 1000:.3f}" for k, v in _p.items())
              + f"  |  合计 {dec_s / n * 1000:.3f}", file=sys.stderr, flush=True)
    timings = {
        "prompt_n": len(ids), "prompt_per_second": round(len(ids) / prefill_s, 1) if prefill_s else 0,
        "predicted_n": len(out), "predicted_per_second": round(len(out) / dec_s, 1) if dec_s else 0,
        "ttft_ms": ttft_ms,
    }
    yield "content", "", timings


class ThinkSplitter:
    """把流式生成的文本切成 (kind, 文本)，kind ∈ {"reasoning","content"}。

    规则主体与 llama-server 一致：`<think>...</think>` 内是 reasoning，其余是 content。

    ★ start_inside：**开思考的 `<think>` 是提示词预填的**（ZAYA 模板的生成前缀就带 `<think>\n`），
      所以模型输出里只有 `</think>`、没有开标签。只统计输出里的标签会判成"不在思考中"，
      于是思考文本跑到 content 里（我第一版就是这样，用户看到的现象是"思考没被标出来"）。

    ★★★ 未闭合思考的兜底（2026-09-15，用户报"开思考后回答不变白，仅 zaya"）：
      诊断结论 —— **ZAYA1-8B（本 GGUF）就是不写 `</think>`**，不是我们引擎的 bug：
      用同一提示词跑 llama.cpp（`--jinja`，同一模型），它的回答同样整段落在 reasoning_content、
      content 为空（实测记录见 /tmp/loadcpp 那次对拍）。模型自己的模板也承认这件事：
        · 规范的 assistant 输出 = `{reasoning}\\n</think>\\n\\n{content}`（模板就是按这个把历史拼回去的）
        · 模板注释原文："Allow downstream logic to take care of broken thought" ——
          **坏掉的思考明确由下游处理**，而"下游"就是我们。
      ⇒ 于是这里按同一约定兜底：思考中**只挂起"当前这一行"**（最后一个换行之后还没结束的那段）。
         出现换行、或这一行超过 LINE_MAX 字符 ⇒ 确认为思考并**立即吐出**；
         生成结束时仍未闭合 ⇒ 挂起的那一行按 **content** 发出（回答于是显示为白字）。
      ★★ 2026-09-15 用户报"思考内容不会流式传输进来"——根因是**我第一版挂起粒度过粗**：
         当时挂起的是"最后一个**空行段**"，而思考常常整段没有空行 ⇒ 用户要等到生成结束
         （或憋够 800 字符）才看到思考一次性蹦出来。实测：整段 382 字的思考只送来 **6 个分片**，
         首个分片在 t=6.05s 且一次 270 字符。改成"按行挂起"后显示延迟降到一行，
         而兜底仍能捞到回答 —— 因为要捞的那段本来就在最后一行（ZAYA 的 "391" 就是）。

    不变式（单测保证）：所有产出的 reasoning+content 拼接 **逐字等于**模型输出原文，
    不丢字、不重复。
    """
    LINE_MAX = 240             # 挂起的那一行超过这么多字符就吐出去，别一直憋着
    _TAGS = ("<think>", "</think>")

    # ★★ 挂起粒度（环境变量 DRACO_THINK_HOLD，2026-09-15 用户问"能分得更细吗/完全逐 token 吗"）：
    #   "line"（默认）—— 按行挂起：思考延迟一行，收尾能把末行当回答（回答变白字）。
    #   "token"  —— **完全直出**（不挂起，思考逐 token 实时可见），代价是收尾**不做兜底**
    #               ⇒ 不写 </think> 的模型（ZAYA）回答会一直是暗色。
    #   "dup"    —— 完全直出 + 收尾把末行**再发一次**作为 content：思考实时、回答也变白，
    #               代价是那一小段会**显示两次**（先暗后白）。
    #   为什么必须二选一：把某段判成"回答"要**事后**才知道（模型可能写出 </think>），
    #   所以"零延迟"与"零重复地正确分区"在流式协议下不可兼得 —— 让用户挑，而不是我替他挑。
    # 默认值取环境变量；**每请求可用请求体字段 `draco_think_hold` 覆盖**（draco 的 `/hold` 命令走这条）。
    HOLD_DEFAULT = (os.environ.get("DRACO_THINK_HOLD") or "line").strip().lower()

    def __init__(self, start_inside=False, hold=None):
        # ★ 类型兜底：客户端可能把数字/None 之类发进来（实测踩过：Float 导致 .strip() 抛异常，
        #   生成器中途崩 ⇒ 客户端收到"空回复"）。**桥接不该因为一个可选字段的坏值崩掉**。
        hold = str(hold).strip().lower() if hold is not None else ""
        hold = hold or self.HOLD_DEFAULT
        if hold not in ("line", "token", "dup"):
            hold = "line"
        self.inside = start_inside
        self.live = hold in ("token", "dup")          # 完全直出（不挂起）
        self.dup = hold == "dup"                      # 收尾补发末行（会有一次重复）
        self.hold_mode = hold
        self.sent = 0            # 已确认并产出的字符数
        self.capped = False      # 当前挂起的这一行是否已被 LINE_MAX 截断过（截断过的不能再当回答）

    def _emit(self, full, upto):
        """把 [sent, upto) 按当前所在区产出。"""
        if upto <= self.sent:
            return []
        piece = full[self.sent:upto]
        self.sent = upto
        return [("reasoning" if self.inside else "content", piece)]

    def _hold(self, full):
        """末尾**还不能吐**的字符数。

        ★ 流式标签的经典坑：不能吐出可能是标签前缀的尾巴。若把 `<` 当正文吐掉，
          后面 `</think>` 补齐时 `find` 就从 sent 之后找不到了 ⇒ 永远不切换到正文区
          （单测 "双边标签" 抓到的就是这个）。所以挂起"任意标签的最长真前缀"。
        """
        n = min(7, len(full) - self.sent)
        for j in range(n, 0, -1):
            suf = full[len(full) - j:]
            if any(t.startswith(suf) for t in self._TAGS):
                return j
        return 0

    def feed(self, full, final=False):
        out = []
        while True:
            nxt_o = full.find("<think>", self.sent)
            nxt_c = full.find("</think>", self.sent)
            if nxt_o != -1 and (nxt_c == -1 or nxt_o < nxt_c):
                out += self._emit(full, nxt_o)          # 标签前的文字按"当前区"确认
                self.inside = True
                self.sent = nxt_o + len("<think>")
                continue
            if nxt_c != -1:
                out += self._emit(full, nxt_c)
                self.inside = False
                self.sent = nxt_c + len("</think>")
                continue
            break
        # 后面没有标签了 —— 尾部处理
        if not self.inside:
            out += self._emit(full, len(full) if final else len(full) - self._hold(full))
            return out
        if self.live:                       # 完全直出：思考逐 token 实时送出（见 HOLD 的说明）
            out += self._emit(full, len(full) if final else len(full) - self._hold(full))
            if final and self.inside and self.dup:
                # 收尾补发：把**最后一行**当回答再发一次（这一次是 content，所以是白字）
                tail = full[self.sent:]
                i = tail.rfind("\n")
                if i != -1:
                    out.append(("content", tail[i + 1:]))
                    self.sent = len(full)
            return out
        seg = full[self.sent:]
        k = seg.rfind("\n")
        if k != -1:
            # 按行确认：最后一个换行（含）之前都算思考，换行之后的那半行先挂起
            out += self._emit(full, self.sent + k + 1)
            self.capped = False          # 新的一行开始了，重新计数
        elif len(seg) > self.LINE_MAX:
            # 单行太长（模型一口气写一大段没有换行）⇒ 不能一直憋着，先吐出去
            out += self._emit(full, len(full) if final else len(full) - self._hold(full))
            self.capped = True           # ★ 这一行被截断过 ⇒ 收尾时不能再把残段当"回答"
        if final and self.inside:
            # ★ 收尾仍未闭合：挂起的那一行（非空且不长）就是回答，按 content 发出；
            #   否则保持 reasoning（空/太长都可能 —— 单行太长的话上面已经吐过、这里挂起为空）
            if not self.capped and 0 < len(full) - self.sent <= self.LINE_MAX:
                self.inside = False
            out += self._emit(full, len(full))
        return out


def _split_think_stream(full, sent, start_inside=False):
    """兼容旧签名的无状态版本（单测/复用场景）。"""
    sp = ThinkSplitter(start_inside=start_inside)
    sp.sent = sent
    return sp.feed(full, final=True)


def text_out_chunks(text, size=8):
    """把完整文本切成小块（流式观感）。"""
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):      # 静默（draco 非 verbose 时把 stdout 吞了，无所谓）
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            import os
            return self._json(200, {"data": [{"id": os.path.basename(ARGS.model),
                                              "object": "model", "owned_by": "dracomancer"}]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": str(e)})
        msgs = req.get("messages") or []
        mt = req.get("max_tokens") or 0
        temp = float(req.get("temperature") or 0)
        seed = int(req.get("seed") or -1)
        rp = float(req.get("repeat_penalty") or 1.0)
        ctk = req.get("chat_template_kwargs") or {}
        think = ctk.get("enable_thinking")          # None = 按模型默认
        hold = req.get("draco_think_hold")          # 思考流式粒度（line/token/dup）；None=按环境变量
        rid = f"draco-eng-{int(time.time()*1000)}"
        # ★ 准备阶段在发响应头之前完成 ⇒ 超长/渲染失败都能回一个像样的 400
        try:
            gen = generate(msgs, mt, temp, seed, rp, think, hold)
        except ValueError as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        except Exception as e:
            return self._json(500, {"error": {"message": f"准备失败：{e}", "type": "server_error"}})
        try:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                timing = {}
                for kind, delta, tim in gen:
                    if tim:
                        timing = tim
                    # kind=reasoning → reasoning_content（draco 会暗色显示为 [思考]）
                    dkey = "reasoning_content" if kind == "reasoning" else "content"
                    chunk = {"id": rid, "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {dkey: delta or None},
                                          "finish_reason": None}]}
                    if tim:      # 最后一片：附 usage/timings（draco 两种格式都认）
                        chunk["usage"] = {"completion_tokens": timing["predicted_n"],
                                          "prompt_tokens": timing["prompt_n"],
                                          "decoding_speed_tps": timing["predicted_per_second"],
                                          "prefill_speed_tps": timing["prompt_per_second"]}
                        chunk["choices"][0]["delta"] = {"content": None}
                        chunk["choices"][0]["finish_reason"] = "stop"
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                parts, reasoning, timing = [], [], {}
                for kind, delta, tim in gen:
                    if tim:
                        timing = tim
                    (reasoning if kind == "reasoning" else parts).append(delta)
                return self._json(200, {
                    "id": rid, "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant",
                                                         "content": "".join(parts) or None,
                                                         "reasoning_content": "".join(reasoning) or None},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": timing.get("predicted_n", 0),
                              "prompt_tokens": timing.get("prompt_n", 0),
                              "decoding_speed_tps": timing.get("predicted_per_second"),
                              "prefill_speed_tps": timing.get("prompt_per_second")},
                    "timings": timing,
                })
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            # ★ 响应头已经发出（SSE 200）就不可能再发 500 —— 只能记录并断开。
            #   我第一版在这里 _json(500, ...)，结果把 500 塞进了响应体，客户端更懵。
            import traceback
            print(f"[SRV] 请求处理异常：{e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--engine", default="smol",
                    help="smol / zaya / ling / granite（自研引擎各自的适配器）")
    ap.add_argument("--threads", type=int, default=4,
                    help="OMP 线程数（小模型 4 最快；默认全核反而慢 2.5×，实测）")
    ap.add_argument("--ctx", type=int, default=0,
                    help="上下文长度（0=引擎默认 1024）。★ 必须与 draco 的 -c 一致，"
                         "否则会出现「能发出去但引擎拒收」的落差")
    ARGS = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
    if ARGS.ctx and ARGS.ctx > 0:
        os.environ["MAXT"] = str(ARGS.ctx)      # 引擎在 import 时读这个值分配 KV/score 缓冲
    load_engine()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"[SRV] listening on http://127.0.0.1:{ARGS.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

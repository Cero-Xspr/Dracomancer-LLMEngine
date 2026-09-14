#!/usr/bin/env python3
"""draco_engine_server —— 把**自研引擎**（m6 内核的 Python 驱动）包成 OpenAI 兼容的 HTTP 服务。

为什么存在：draco.py 的四个后端里没有自研引擎 —— cpu/igpu/local 走 llama-server、npu 走 FLM，
而我们自己的内核（m5_kern*/m6_engine）只有独立脚本。这个服务器补上那座桥：
draco 从此可以用 `-b dengine` 直接驱动**我们自己的引擎**跑 chat / serve / selfcheck / tune。

设计（都是刻意的）：
  · **只依赖**：标准库 + numpy + transformers(仅 tokenizer) + 引擎模块本体。不引 FastAPI。
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

ARGS = None
AP = None           # 引擎适配器（forward/logits/reset/encode/decode/eos/im_end）
MAXT = 1024
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def _load_smol():
    """SmolLM2（llama 家族）→ smol_engine.py。状态在 KEEP 的 kc/vc/tlen，memset 即复位。"""
    global ENG, TOK, EOS_ID, IM_END, MAXT
    import os
    os.environ["MODEL"] = ARGS.model
    import smol_engine as E
    ENG = E
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(os.path.join(BASE, "tok-smol"))
    EOS_ID = TOK.eos_token_id
    IM_END = TOK.convert_tokens_to_ids("<|im_end|>")
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
                encode=lambda t: TOK.encode(t, add_special_tokens=True),
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
    _emb_al = Z.A(bytes(Z._EMB_T.data))    # 64 字节对齐（gemv 用对齐加载）
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
    _r = _g.GGUFReader(ARGS.model)
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
    MAXT = 1024
    H_Z = Z.H

    # ★ zaya 的 step 返回隐状态、logits_of 需要它 ⇒ 闭包里存"最后隐状态"，
    #   对齐 smol 的 logits() 无参接口。
    last = {"x": None}

    def forward(tid, pos):
        last["x"] = Z.step(tid, pos)
        return last["x"]

    def logits():
        return logits_of(last["x"])

    return dict(forward=forward, logits=logits, reset=reset,
                encode=enc, decode=decode, eos=eos, im_end=im_end, max_t=MAXT,
                system_default=None, bos=2, vsz=vsz,
                think_block=True, think_default=True,   # ZAYA 模板默认开思考，用空 think 块关
                render=lambda msgs, think: render_chatml_generic(msgs, think, None, None, True))


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
    from transformers import AutoTokenizer
    ENG = E
    tk = AutoTokenizer.from_pretrained(os.path.join(BASE, "tok-ling"))

    # ---- 模板与特殊 token（从 GGUF 读，jinja2 渲染）----
    _r = _g.GGUFReader(ARGS.model)
    _f = {t.name: t for t in _r.fields.values()}
    _toks = [t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t)
             for t in _f["tokenizer.ggml.tokens"].contents()]
    _vars = {}
    for _n, _t in _f.items():
        if _n.startswith("tokenizer.ggml.") and _n.endswith("_token_id"):
            try:
                _vars[_n.split(".")[-1][:-3]] = _toks[int(_t.contents())]
            except Exception:
                pass
    _tpl = jinja2.Environment().from_string(_f["tokenizer.chat_template"].contents())

    def render(msgs, think):
        return _tpl.render(messages=msgs, add_generation_prompt=True,
                           enable_thinking=(True if think is None else bool(think)), **_vars)

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

    MAXT = E.MAXT
    return dict(forward=E.forward, logits=E.logits_of_x, reset=reset,
                encode=lambda t: tk.encode(t, add_special_tokens=False),
                decode=lambda ids: tk.decode(ids), eos=tk.eos_token_id,
                im_end=tk.eos_token_id, max_t=MAXT,
                system_default=None, bos=None, think_block=False, think_default=False,
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
    else:
        raise SystemExit(f"未知引擎 {ARGS.engine}（当前支持：smol / zaya / ling）")
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


def generate(messages, max_tokens, temp, seed, repeat_penalty, enable_thinking=None):
    """生成器：yield (文本增量, 计时dict)。最后一次 yield 后返回 timing。"""
    rng = random.Random(seed if seed and seed > 0 else None)
    think_now = (AP.get("think_default", True) if enable_thinking is None
                 else bool(enable_thinking))
    # ★ 渲染交给适配器：SmolLM2/ZAYA 用已验证的手写 ChatML；Ling 的格式完全不同
    #   （<role>HUMAN</role>...<|role_end|>）⇒ 必须用模型自己的 jinja 模板渲染。
    text = AP["render"](messages, enable_thinking)
    ids = AP["encode"](text)
    if AP.get("bos"):
        ids = [AP["bos"]] + ids
    budget = MAXT - len(ids) - 8
    if budget <= 0:
        raise ValueError(f"prompt 太长（{len(ids)} token > ctx {MAXT}）")
    n_gen = min(max_tokens if max_tokens and max_tokens > 0 else 256, budget)

    reset_state()
    t0 = time.time()
    for pos, tid in enumerate(ids):
        AP["forward"](int(tid), pos)
    prefill_s = time.time() - t0

    # ★ 真·逐 token 流式：每步解码"已生成前缀"再产出**新增的那段**。
    #   我之前是"整段生成完再切块 yield" —— 用户一眼看出没流式（而 llama-server 是流式的）。
    out, gen_ids = [], []
    t1 = time.time()
    sent = 0                      # 已产出的字符数（前缀解码可能出现多字节片段，按字符增量发最稳）
    t_first = None
    for i in range(n_gen):
        lg = AP["logits"]()
        nid = sample(lg, temp, rng, repeat_penalty, gen_ids)
        if nid == AP["eos"] or nid == AP["im_end"]:
            break
        out.append(nid)
        gen_ids.append(nid)
        full = AP["decode"](out)
        if len(full) > sent:
            if t_first is None:
                t_first = time.time()
            for kind, piece in _split_think_stream(
                    full, sent,
                    start_inside=bool(AP.get("think_block")) and think_now):
                if piece:
                    yield kind, piece, {}
            sent = len(full)
        AP["forward"](nid, len(ids) + i)
    dec_s = time.time() - t1
    ttft_ms = round((t_first - t1) * 1000, 1) if t_first else None
    timings = {
        "prompt_n": len(ids), "prompt_per_second": round(len(ids) / prefill_s, 1) if prefill_s else 0,
        "predicted_n": len(out), "predicted_per_second": round(len(out) / dec_s, 1) if dec_s else 0,
        "ttft_ms": ttft_ms,
    }
    yield "content", "", timings


def _split_think_stream(full, sent, start_inside=False):
    """把**已生成全文**里 sent 之后的部分，按当前是否在 <think> 内切成 (kind, 文本)。

    kind="reasoning" 落在 <think>...</think> 内（draco 会暗色加 [思考] 前缀显示），
    其余为 "content" —— 与 llama-server 对推理模型的行为一致。

    ★ start_inside：**开思考的 <think> 是提示词预填的**（ZAYA 模板的生成前缀就带 `<think>\n`），
      所以模型输出里只有 `</think>`、没有开标签。只统计输出里的标签会判成"不在思考中"，
      于是思考文本跑到 content 里（我第一版就是这样，用户看到的现象是"思考没被标出来"）。
    """
    out = []
    i = sent
    inside = start_inside
    while i < len(full):
        nxt_open = full.find("<think>", i)
        nxt_close = full.find("</think>", i)
        if nxt_open != -1 and (nxt_close == -1 or nxt_open < nxt_close):
            out.append(("content", full[i:nxt_open]))
            inside = True
            i = nxt_open + len("<think>")
            continue
        if nxt_close != -1:
            out.append(("reasoning" if inside else "content", full[i:nxt_close]))
            inside = False
            i = nxt_close + len("</think>")
            continue
        out.append(("reasoning" if inside else "content", full[i:]))
        i = len(full)
    return out


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
        rid = f"draco-eng-{int(time.time()*1000)}"
        try:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                timing = {}
                for kind, delta, tim in generate(msgs, mt, temp, seed, rp, think):
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
                for kind, delta, tim in generate(msgs, mt, temp, seed, rp, think):
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
    ap.add_argument("--engine", default="smol")
    ap.add_argument("--threads", type=int, default=4,
                    help="OMP 线程数（小模型 4 最快；默认全核反而慢 2.5×，实测）")
    ARGS = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
    load_engine()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"[SRV] listening on http://127.0.0.1:{ARGS.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

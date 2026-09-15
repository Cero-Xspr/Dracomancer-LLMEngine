#!/usr/bin/env python3
"""Falcon-H1 的分词器：GGUF vocab/merges + llama.cpp 的 llama3 正则（falcon-h1 pre 类型
在 llama-vocab.cpp 里与 llama3 共用 LLAMA_VOCAB_PRE_TYPE_LLAMA3，且 **add_bos=true**）。

与 granite_tok 的差别只有两点：正则同款但 **要加 BOS**（id 17）；对账时 llama-tokenize
**不带 --no-bos**（默认加 bos），两边才可比。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_fast  # noqa: E402

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/falcon-h1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf")
LLAMA_TOK = os.environ.get("LLAMA_TOK", "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/"
                                          "build-dbg/bin/llama-tokenize")
LDP = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"

# llama-vocab.cpp 的 LLAMA_VOCAB_PRE_TYPE_LLAMA3 正则（falcon-h1 与 llama3 共用）
LLAMA3_PAT = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
              r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def build(model=MODEL, add_bos=True):
    """add_bos=True：encode 前置 BOS（falcon/llama 的裸文本口径，与 llama-tokenize 默认一致）。
    add_bos=False：原样 BPE——给「模板渲染文本已含 bos_token」的适配器用（llama 3.2 模板
    以 {{- bos_token }} 开头，前置会变成双 BOS）。"""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex
    R = gguf_fast.FastGGUF(model)
    toks = list(R.fields["tokenizer.ggml.tokens"].value)
    merges = [tuple(m.split(" ")) for m in R.fields["tokenizer.ggml.merges"].value]
    ttypes = list(R.fields["tokenizer.ggml.token_type"].value)
    vocab = {t: i for i, t in enumerate(toks)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3_PAT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    specials = [t for i, t in enumerate(toks) if int(ttypes[i]) != 1]
    tk.add_special_tokens(specials)
    bf = R.fields.get("tokenizer.ggml.bos_token_id")
    bos = int(bf.value) if bf is not None else None    # ★ qwen35 等没有 bos 键 ⇒ None（不前置）
    eos = int(R.fields["tokenizer.ggml.eos_token_id"].value)     # 11

    class _Enc:
        def __init__(self, ab):
            self._ab = ab
            self.bos = bos

        def encode(self, text, add_special_tokens=False):
            e = tk.encode(text, add_special_tokens=False)
            ids = ([bos] + e.ids) if (self._ab and bos is not None) else e.ids
            return type("R", (), {"ids": ids})()

        def decode(self, ids, skip_special_tokens=False):
            return tk.decode([i for i in ids if bos is None or i != bos],
                             skip_special_tokens=skip_special_tokens)

        def id_to_token(self, i):
            return toks[i]

    bs = f"{bos}({toks[bos]!r})" if bos is not None else "无"
    print(f"[tok] vocab={len(toks)} merges={len(merges)} special={len(specials)} "
          f"bos={bs} eos={eos}({toks[eos]!r})", flush=True)
    return _Enc(add_bos), R


def llama_ids(s, model=MODEL, no_bos=False):
    cmd = [LLAMA_TOK, "-m", model, "-p", s, "--ids"] + (["--no-bos"] if no_bos else [])
    out = subprocess.run(cmd, env={**os.environ, "LD_LIBRARY_PATH": LDP},
                         capture_output=True, text=True, check=True).stdout
    return [int(v) for v in out.replace("[", " ").replace("]", " ").replace(",", " ").split()]


if __name__ == "__main__":
    tk, R = build()
    if "--ids" in sys.argv:
        print(tk.encode(sys.argv[sys.argv.index("--ids") + 1]).ids)
        sys.exit(0)
    if "--check" in sys.argv:
        cases = ["The capital of France is", "中国的首都是北京", "Hello   world!\n\nSecond line.",
                 "It's a 1234 number, isn't it?", "def foo(x):  # comment\n    return x**2",
                 "<|im_start|>user\nhi<|im_end|>", "  leading spaces", "tab\there",
                 "emoji 🎉 and ünïcödé"]
        bad = 0
        assert tk.encode("").ids == [tk.bos], "空串 = [BOS]（add_bos=true）"
        print(f"✓ '' 空串 → [{tk.bos}]（BOS）")
        for s in cases:
            mine = tk.encode(s).ids
            ref = llama_ids(s)
            ok = mine == ref
            bad += 0 if ok else 1
            print(f"{'✓' if ok else '✗'} {s[:40]!r:44s} mine={mine[:12]} ref={ref[:12]}")
            if not ok:
                print(f"    mine全={mine}\n    ref全 ={ref}")
        print(f"\n{'全部一致' if bad == 0 else f'{bad}/{len(cases)} 串不一致'}")

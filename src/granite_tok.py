#!/usr/bin/env python3
"""granite 的 Python 分词器：直接用 GGUF 里的 vocab/merges 搭（不依赖 transformers 模型目录）。

为什么能这么搭（都是从 llama.cpp 源码抄来的、不是猜的）：
  · `tokenizer.ggml.model = gpt2` ⇒ ByteLevel BPE，tokens 里存的就是 GPT-2 的 unicode 映射串（Ġ…）
  · `tokenizer.ggml.pre = dbrx` ⇒ llama-vocab.cpp 里 DBRX/SMAUG 与 **llama3 共用同一条正则**
    （"(?:'[sS]|'[tT]|…)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"）
  · token_type != 1(NORMAL) 的都算 special（granite 的 <|start_of_role|> 等是 3=CONTROL）

★ 搭完必须对账（`--check`）：与 build-dbg/bin/llama-tokenize 的 --ids 逐串比。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_fast  # noqa: E402

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf")
LLAMA_TOK = os.environ.get("LLAMA_TOK", "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/"
                                          "build-dbg/bin/llama-tokenize")
LDP = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"

# llama.cpp llama-vocab.cpp 的 DBRX 正则（与 llama3 同一串）
LLAMA3_PAT = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
              r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def build(model=MODEL):
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
    print(f"[tok] vocab={len(toks)} merges={len(merges)} special={len(specials)} "
          f"bos={R.fields['tokenizer.ggml.bos_token_id'].value} "
          f"eos={R.fields['tokenizer.ggml.eos_token_id'].value}", flush=True)
    return tk, R


def llama_ids(s, model=MODEL):
    out = subprocess.run([LLAMA_TOK, "-m", model, "-p", s, "--ids", "--no-bos"],
                         env={**os.environ, "LD_LIBRARY_PATH": LDP},
                         capture_output=True, text=True, check=True).stdout
    return [int(v) for v in out.replace("[", " ").replace("]", " ").replace(",", " ").split()]


if __name__ == "__main__":
    tk, R = build()
    if "--check" in sys.argv:
        cases = ["The capital of France is", "中国的首都是北京", "Hello   world!\n\nSecond line.",
                 "It's a 1234 number, isn't it?", "def foo(x):  # comment\n    return x**2",
                 "<|start_of_role|>user<|end_of_role|>hi<|end_of_text|>", "  leading spaces",
                 "tab\there", "emoji 🎉 and ünïcödé"]
        bad = 0
        # 空串单独测：llama-tokenize 不接受 -p ""（退出码 1），拿不到参考 ⇒ 只本地断言
        assert tk.encode("", add_special_tokens=False).ids == [], "空串应当切成 0 个 token"
        print("✓ ''（空串）本地断言：0 个 token（llama-tokenize 拒绝 -p ''，无法给参考）")
        for s in cases:
            mine = tk.encode(s, add_special_tokens=False).ids
            ref = llama_ids(s)
            ok = mine == ref
            bad += 0 if ok else 1
            print(f"{'✓' if ok else '✗'} {s[:40]!r:44s} mine={mine[:12]} ref={ref[:12]}")
            if not ok:
                print(f"    mine全={mine}\n    ref全 ={ref}")
            back = tk.decode(mine, skip_special_tokens=False)
            if back != s:
                print(f"    ⚠ 解码不还原: {back!r} != {s!r}")
        # 特殊 token：模板里的角色标记必须映射成单个 id
        ids = tk.encode("<|start_of_role|>user<|end_of_role|>", add_special_tokens=False).ids
        print(f"特殊 token 检查: {ids}  (应含 100264/100265 之类的单词 id)")
        print(f"\n{'全部一致' if bad == 0 else f'{bad}/{len(cases)} 串不一致'}")

    if "--ids" in sys.argv:
        print(tk.encode(sys.argv[sys.argv.index("--ids") + 1], add_special_tokens=False).ids)

#!/usr/bin/env python3
"""k2_golden_fast.py — 金标准 v3：批量 numpy oracle 一次前向（与 torch 机制对拍过的实现）。

存：prompt 末位 top8 + 全部 48 层**末 token** 的 |x|.sum / |x|².sum 校验和。
引擎闸门（k2_gate.py v3）用同样的钩子逐层对拍。
内存：LazyW 逐层逐出（含专家元组键）⇒ 单进程 ~6GB。运行 ~2 分钟。
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
import gguf
import gguf.quants as Q
import k2_numpy as KN

GGUF = os.environ.get("MODEL", "/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests/golden/k2horizon_golden.json")

r = gguf.GGUFReader(GGUF)
T_ = {t.name: t for t in r.tensors}
meta = {}
for m in r.fields.values():
    try:
        meta[m.name] = bytes(m.parts[m.data[0]]).decode() if m.types[0] == gguf.GGUFValueType.STRING else m.contents[0]
    except Exception:
        pass


def mval(suffix, default):
    for k, v in meta.items():
        if k.endswith(suffix):
            return v
    return default


n_head = int(mval("attention.head_count", 32))
n_kv = int(mval("attention.head_count_kv", 8))
hd = int(mval("attention.key_length", 128))
H = int(mval("embedding_length", 2560))
NL = int(mval("block_count", 48))
EPS = float(mval("attention.layer_norm_rms_epsilon", 1e-6))
NEXP = int(mval("expert_count", 100))
NUSED = int(mval("expert_used_count", 8))
MEXP = int(mval("mova.expert_count", 64))
MUSED = int(mval("mova.expert_used_count", 4))
THETA = float(mval("rope.theta", 1e7))
SCALING = float(mval("expert_weights_scale", 2.5))
GATING = "sigmoid" if int(mval("expert_gating_func", 2)) == 2 else "softmax"
print(f"[k2] L={NL} H={H} MoE={NEXP}x{NUSED} MoVA={MEXP}x{MUSED} gating={GATING} scale={SCALING}", flush=True)


class LazyW:
    KEEP = ("token_embd.weight", "output.weight", "output_norm.weight")

    def __init__(self, T_):
        self.T = T_
        self.cache = {}
        self.cur_blk = None

    def expert(self, name, e, n_exp):
        key = (name, e)
        if key in self.cache:
            return self.cache[key]
        t = self.T[name]
        per = t.data.nbytes // n_exp
        raw = np.frombuffer(t.data, np.uint8)[e * per:(e + 1) * per]
        v = np.asarray(Q.dequantize(raw, t.tensor_type), np.float32)
        shp = tuple(int(x) for x in t.shape)[:-1][::-1]
        v = v.reshape(shp)
        self.cache[key] = v
        return v

    def __getitem__(self, name):
        if name in self.cache:
            return self.cache[name]
        t = self.T[name]
        v = np.asarray(Q.dequantize(t.data, t.tensor_type), np.float32)
        if name not in self.KEEP:
            parts = name.split(".")
            blk = int(parts[1]) if parts[0] == "blk" else None
            if blk != self.cur_blk:
                for k in [k for k in self.cache
                          if (isinstance(k, str) and k.startswith("blk.")) or isinstance(k, tuple)]:
                    del self.cache[k]
                self.cur_blk = blk
        self.cache[name] = v
        return v

    def __contains__(self, name):
        return name in self.T


lazy = LazyW(T_)


class WMap:
    @staticmethod
    def hf2gguf(name):
        if name == "model.embed_tokens.weight": return "token_embd.weight"
        if name == "model.norm.weight": return "output_norm.weight"
        if name == "lm_head.weight": return "output.weight"
        if name.startswith("model.layers."):
            rest = name[len("model.layers."):]
            blk, sub = rest.split(".", 1)
            sub = (sub.replace("self_attn.q_proj", "attn_q")
                      .replace("self_attn.k_proj", "attn_k")
                      .replace("self_attn.v_proj", "attn_v")
                      .replace("self_attn.o_proj", "attn_output")
                      .replace("self_attn.gate_proj", "attn_gate")
                      .replace("self_attn.v_router", "attn_v_gate")
                      .replace("input_layernorm", "attn_norm")
                      .replace("post_attention_layernorm", "ffn_norm")
                      .replace("mlp.gate_proj", "ffn_gate")
                      .replace("mlp.up_proj", "ffn_up")
                      .replace("mlp.down_proj", "ffn_down")
                      .replace("mlp.gate.bias", "exp_probs_b.bias")
                      .replace("mlp.gate", "ffn_gate_inp"))
            if sub.startswith("mlp.experts."):
                return None
            if sub.startswith("mlp."):
                sub = sub.replace("mlp.shared_experts.gate_proj", "ffn_gate_shexp") \
                         .replace("mlp.shared_experts.up_proj", "ffn_up_shexp") \
                         .replace("mlp.shared_experts.down_proj", "ffn_down_shexp") \
                         .replace("mlp.", "ffn_")
            return f"blk.{blk}.{sub}"
        return name

    def __init__(self, lazy):
        self.lazy = lazy

    def __getitem__(self, name):
        if ".experts." in name:
            parts = name.split(".")
            li, ei = int(parts[2]), int(parts[5])
            kind = parts[6].replace("_proj", "")
            base = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}[kind] + ".weight"
            return self.lazy.expert(f"blk.{li}.{base}", ei, NEXP)
        if ".v_experts." in name:
            li = int(name.split(".")[2])
            ei = int(name.split("v_experts.")[1].split(".")[0])
            return self.lazy.expert(f"blk.{li}.attn_v_exps.weight", ei, MEXP)
        gg = self.hf2gguf(name)
        return self.lazy[gg]

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name):
        if ".experts." in name or ".v_experts." in name:
            return True
        return self.hf2gguf(name) in self.lazy


wm = WMap(lazy)

import falcon_tok
tk, _ = falcon_tok.build(GGUF, add_bos=False, pre="llama3")
prompt = os.environ.get("PROMPT", "The capital of France is")
ids = tk.encode(prompt).ids
print(f"[k2] prompt ids({len(ids)}): {ids}", flush=True)

cfg = {
    "num_attention_heads": n_head, "num_key_value_heads": n_kv, "head_dim": hd,
    "hidden_size": H, "num_hidden_layers": NL,
    "num_experts": NEXP, "num_experts_per_tok": NUSED, "num_shared_experts": 1,
    "mova_num_experts": MEXP, "mova_num_experts_per_tok": MUSED,
    "mlp_only_layers": [0, 1, 2], "layernorm_num_groups": 2, "rms_norm_eps": EPS,
    "rope_theta": THETA, "router_scaling_factor": SCALING, "query_key_norm": False,
    "router_score_func": GATING,
}

# ── 一次批量前向 + 逐层校验和（末 token）──
layer_sums = []


def hook(il, xl):
    last = np.asarray(xl)[-1]
    layer_sums.append((int(il), float(np.abs(last).sum()), float(np.square(last).sum())))


t0 = time.time()
lg = KN.model_forward(np.array(ids), wm, cfg, layer_hook=hook)
print(f"[k2] 前向 {len(ids)} tok {time.time()-t0:.0f}s", flush=True)
lg = np.asarray(lg)[-1]
top8 = np.argsort(-lg)[:8]
print("末位 top8:", [(int(i), round(float(lg[i]), 3)) for i in top8], flush=True)

out = {
    "model": os.path.basename(GGUF),
    "prompt": prompt, "ids": ids,
    "last_top8": [[int(i), float(lg[i])] for i in top8],
    "greedy8": [int(top8[0])],
    "layer_sums": layer_sums,
}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print("已写", OUT, flush=True)

#!/usr/bin/env python3
"""k2_golden.py — 真模型金标准：numpy oracle（惰性反查表）跑 K2-Horizon-MoVA-36B。

输出 tests/golden/k2horizon_golden.json：prompt 末位 top8 + 贪心续接 8 token。
权重名翻译：HF 名（k2_numpy 用）→ GGUF 名（blk.N 惯例），堆叠专家张量按行切片。
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
import gguf
import gguf.quants as Q
import k2_numpy as KN

GGUF = "/media/xiao_/OverSys1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf"

t00 = time.time()
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
print(f"[k2] L={NL} H={H} heads={n_head}/{n_kv} hd={hd} MoE={NEXP}x{NUSED} MoVA={MEXP}x{MUSED}", flush=True)

experts_shape = tuple(int(v) for v in T_["blk.3.ffn_gate_exps.weight"].shape)
vexps_shape = tuple(int(v) for v in T_["blk.3.attn_v_exps.weight"].shape)
print(f"[k2] ffn_exps {experts_shape}  v_exps {vexps_shape}", flush=True)


class LazyW:
    """GGUF 名 → fp32（惰性反查）。保留最近一层的张量；embed/lm_head/output_norm 常驻。"""

    KEEP = ("token_embd.weight", "output.weight", "output_norm.weight")

    def __init__(self, T_):
        self.T = T_
        self.cache = {}
        self.cur_blk = None

    def __getitem__(self, name):
        if name in self.cache:
            return self.cache[name]
        t = self.T[name]
        v = np.asarray(Q.dequantize(t.data, t.tensor_type), np.float32)
        if name not in self.KEEP:
            parts = name.split(".")
            blk = int(parts[1]) if parts[0] == "blk" else None
            if blk != self.cur_blk:
                for k in [k for k in self.cache if k.startswith("blk.")]:
                    del self.cache[k]
                self.cur_blk = blk
        self.cache[name] = v
        return v

    def __contains__(self, name):
        return name in self.T


lazy = LazyW(T_)


class WMap:
    """HF 名 → GGUF 名 + 堆叠专家切片。"""

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
                      .replace("mlp.gate_proj", "ffn_gate")     # ★ 稠密 FFN（必须先于 mlp.gate 规则）
                      .replace("mlp.up_proj", "ffn_up")
                      .replace("mlp.down_proj", "ffn_down")
                      .replace("mlp.gate.bias", "exp_probs_b.bias")  # ★★ MoE 路由 bias（moe_gate_bias=true 时官方语义要加，缺它=静默路由错误）
                      .replace("mlp.gate", "ffn_gate_inp"))     # MoE 路由器 weight
            if sub.startswith("mlp.experts."):
                return None   # 堆叠张量，走专用接口
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
        # 堆叠专家：mlp.experts.E.{gate,up,down}_proj.weight → ffn_*_exps.weight[E]
        if ".experts." in name:
            # ffn_*_exps 反查表形状 [H, inter, NEXP]，专家 e = full[:, :, e].T → [inter, H]
            parts = name.split(".")
            li, ei = int(parts[2]), int(parts[5])   # layers.N.mlp.experts.E.kind_proj
            kind = parts[6].replace("_proj", "")
            base = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}[kind] + ".weight"
            full = self.lazy[f"blk.{li}.{base}"]
            return full[ei]   # dequantize 输出 [NEXP, out, in]（gguf-py 反转），按专家轴切
        if ".v_experts." in name:
            li = int(name.split(".")[2])
            ei = int(name.split("v_experts.")[1].split(".")[0])
            full = self.lazy[f"blk.{li}.attn_v_exps.weight"]
            return full[ei]
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

# ── tokenizer ──
import falcon_tok
tk, _ = falcon_tok.build(GGUF, add_bos=False, pre="llama3")   # pre=k2-horizon 的正则与 llama3 相同（tokenizer.json 实证）
prompt = "The capital of France is"
ids = tk.encode(prompt).ids
print(f"[k2] prompt ids({len(ids)}): {ids}", flush=True)

cfg = {
    "num_attention_heads": n_head, "num_key_value_heads": n_kv, "head_dim": hd,
    "hidden_size": H, "num_hidden_layers": NL,
    "num_experts": NEXP, "num_experts_per_tok": NUSED, "num_shared_experts": 1,
    "mova_num_experts": MEXP, "mova_num_experts_per_tok": MUSED,
    "mlp_only_layers": [0, 1, 2], "layernorm_num_groups": 2, "rms_norm_eps": EPS,
    "rope_theta": THETA, "router_scaling_factor": 1.0, "query_key_norm": False,
}

# ── prompt 前向 ──
t0 = time.time()
logits = KN.model_forward(np.array(ids), wm, cfg)
print(f"[k2] prompt {len(ids)} tok 前向 {time.time()-t0:.1f}s", flush=True)
last = logits[-1]
top8 = np.argsort(-last)[:8]
print("末位 top8:", [(int(i), round(float(last[i]), 3)) for i in top8])

# ── 贪心 8 token（全量重跑，惰性缓存按层逐出）──
gen = [int(np.argmax(last))]
cur = list(ids)
for i in range(7):
    t0 = time.time()
    lg = KN.model_forward(np.array(cur), wm, cfg)
    nxt = int(lg[-1].argmax())
    gen.append(nxt)
    cur.append(nxt)
    print(f"  gen {i+1}/7: {time.time()-t0:.1f}s tok={nxt}", flush=True)

out = {
    "model": "K2-Horizon-MoVA-36B-A4B-Q4_K_M",
    "prompt": prompt, "ids": ids,
    "last_top8": [(int(i), float(last[i])) for i in top8],
    "greedy8": gen,
    "oracle": "k2_numpy（机制对齐官方 modeling_k2_horizon.py，torch 对拍 max|Δ|=1.4e-7）",
}
gd = "/media/xiao_/OverSys1/npu-direct/hybrid/tests/golden/k2horizon_golden.json"
os.makedirs(os.path.dirname(gd), exist_ok=True)
json.dump(out, open(gd, "w"), indent=1)
print("金标准已写入", gd)

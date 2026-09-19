#!/usr/bin/env python3
"""机制对拍：官方 torch 实现的小随机 K2Horizon vs 我的 numpy 实现。
对拍范围：分组 RMSNorm / softplus 门 / MoVA 路由+silu / MoE+共享专家 / rope。"""
import os, sys, importlib.util
import numpy as np
import torch

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/hybrid")
import k2_numpy as KN

# ── 加载官方建模代码（包结构，相对导入可用）──
sys.path.insert(0, "/tmp")
mod = importlib.import_module("k2ref.modeling_k2_horizon")

TinyCfg = __import__('k2ref.configuration_k2_horizon', fromlist=['K2HorizonConfig']).K2HorizonConfig
cfg = TinyCfg(
    vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
    moe_intermediate_size=24, decoder_sparse_step=1, mlp_only_layers=[0],
    mova_num_experts=8, mova_num_experts_per_tok=2,
    attention_gate_func="softplus", moe_gate_bias=True,
    norm_topk_prob=True, query_key_norm=True,          # ★ 顺带覆盖 q/k 分组 norm 路径
    rope_theta=10000.0, rope_head_dim=16, rms_norm_eps=1e-6,
    layernorm_num_groups=2, tie_word_embeddings=False,
    max_position_embeddings=512,
)
torch.manual_seed(0)
model = mod.K2HorizonForCausalLM(cfg)
model.eval()

ids = torch.tensor([[3, 17, 5, 99, 42, 7, 1, 60, 23, 8]])
with torch.no_grad():
    logits_t = model(ids).logits[0].float()

# ── 权重导出 ──
w = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
cfg_np = {
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "hidden_size": 64, "num_hidden_layers": 4, "num_experts": 8,
    "num_experts_per_tok": 2, "num_shared_experts": 1, "hidden_size_ffn": 96,
    "mova_num_experts": 8, "mova_num_experts_per_tok": 2,
    "mlp_only_layers": [0], "layernorm_num_groups": 2, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "head_dim": 16, "router_scaling_factor": 1.0, "query_key_norm": True,
}

logits_n = KN.model_forward(ids[0].numpy(), w, cfg_np)

d = np.abs(logits_t.numpy() - logits_n)
rel = d.max() / (np.abs(logits_t.numpy()).max() + 1e-30)
print(f"logits 形状 torch {tuple(logits_t.shape)} vs numpy {logits_n.shape}")
print(f"max|Δ| = {d.max():.3e}   相对 = {rel:.3e}")
print("PASS" if rel < 1e-4 else "FAIL —— 需要定位机制分歧")

#!/usr/bin/env python3
"""k2_numpy.py — K2-Horizon (k2_horizon 架构) 的 numpy 参照实现。

语义逐式对齐 IFM 官方 modeling_k2_horizon.py（torch）：
  · 分组 RMSNorm（T5 式：n_groups 组内方差）
  · MoVA 注意力：V 投影 64 专家路由（softmax 分数、bias 仅参与 top-k 选择、权重归一）、
    专家输出过 silu 再加权；输出门 gate_proj → softplus(beta=ln2)
  · MoE FFN：softmax 路由 + bias 选择 + norm_topk + 1 共享专家
  · 前 mlp_only_layers 层稠密；rope = HF rotate_half 约定（theta 1e7）
权重来源：与 HF state_dict 同键名的 numpy dict。fp32 计算。
"""
import numpy as np


def grouped_rms(x, w, n_groups, eps):
    """K2HorizonRMSNorm：组内方差 T5 式 RMSNorm。x [..., H] fp32。"""
    shape = x.shape
    g = x.reshape(*shape[:-1], n_groups, -1)
    var = (g * g).mean(-1, keepdims=True)
    g = g * (1.0 / np.sqrt(var + eps))
    return (g.reshape(shape)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus_ln2(x):
    """F.softplus(x, beta=ln2) 的稳定实现。"""
    b = np.log(2.0)
    return np.logaddexp(0.0, b * x) / b


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(q, k, cos, sin):
    # q/k: [nh, T, hd]；cos/sin: [T, hd] → 头维广播
    cos = cos[None, :, :]
    sin = sin[None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, groups):
    # HF repeat_kv：沿头维 repeat_interleave（相邻 q 头共享同一 kv 头）
    return np.repeat(x, groups, axis=0)


def softmax_last(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def calc_router(logits, bias, top_k, scaling):
    """calc_router_weights：bias 只参与选择；权重=原始分数 gather 后归一 ×scaling。"""
    scores = softmax_last(logits)
    sel_scores = scores + bias if bias is not None else scores
    idx = np.argsort(-sel_scores, axis=-1, kind="stable")[..., :top_k]
    w = np.take_along_axis(scores, idx, axis=-1)
    if top_k > 1:
        w = w / w.sum(-1, keepdims=True)
    if scaling is not None:
        w = w * scaling
    return w, idx


def mlp_forward(x, w, prefix):
    return (silu(x @ w[prefix + "gate_proj.weight"].T) * (x @ w[prefix + "up_proj.weight"].T)) \
        @ w[prefix + "down_proj.weight"].T


def _causal_mask(T):
    m = np.full((T, T), -np.inf, np.float32)
    return np.triu(m, 1)[None, :, :]


def _attention_core(q, k, v, nh, nkv, hd, T):
    k = repeat_kv(k, nh // nkv)
    v = repeat_kv(v, nh // nkv)
    att = softmax_last((q @ k.transpose(0, 2, 1) / np.sqrt(hd)) + _causal_mask(T))
    return (att @ v).transpose(1, 0, 2).reshape(T, nh * hd)


def mova_attention(x, w, prefix, cfg, cos, sin):
    """K2HorizonMoVAAttention（无缓存单遍）。x [T, H]。"""
    T, H = x.shape
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    logits = x @ w[prefix + "v_router.weight"].T
    bias = w.get(prefix + "v_router.bias")
    topk = cfg["mova_num_experts_per_tok"]
    rw, sel = calc_router(logits, bias, topk, cfg.get("router_scaling_factor"))
    V = np.zeros((T, nkv * hd), np.float32)
    flat_e = sel.ravel()
    flat_t = np.repeat(np.arange(T), topk)
    flat_w = rw.ravel()
    for e in np.unique(flat_e):
        m = flat_e == e
        ts = flat_t[m]
        V[ts] += silu(x[ts] @ w[f"{prefix}v_experts.{e}.weight"].T) * flat_w[m][:, None]
    Vh = V.reshape(T, nkv, hd).transpose(1, 0, 2)
    qf = x @ w[prefix + "q_proj.weight"].T
    kf = x @ w[prefix + "k_proj.weight"].T
    if cfg.get("query_key_norm"):
        qf = grouped_rms(qf, w[prefix + "q_norm.weight"], nh, cfg["rms_norm_eps"])
        kf = grouped_rms(kf, w[prefix + "k_norm.weight"], nkv, cfg["rms_norm_eps"])
    q = qf.reshape(T, nh, hd).transpose(1, 0, 2)
    k = kf.reshape(T, nkv, hd).transpose(1, 0, 2)
    q, k = apply_rope(q, k, cos, sin)
    out = _attention_core(q, k, Vh, nh, nkv, hd, T)
    out = out * softplus_ln2(x @ w[prefix + "gate_proj.weight"].T)
    return out @ w[prefix + "o_proj.weight"].T


def dense_attention(x, w, prefix, cfg, cos, sin):
    """K2HorizonAttention（稠密层：标准 v_proj，同样有 softplus 输出门）。"""
    T, H = x.shape
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    qf = x @ w[prefix + "q_proj.weight"].T
    kf = x @ w[prefix + "k_proj.weight"].T
    if cfg.get("query_key_norm"):
        qf = grouped_rms(qf, w[prefix + "q_norm.weight"], nh, cfg["rms_norm_eps"])
        kf = grouped_rms(kf, w[prefix + "k_norm.weight"], nkv, cfg["rms_norm_eps"])
    q = qf.reshape(T, nh, hd).transpose(1, 0, 2)
    k = kf.reshape(T, nkv, hd).transpose(1, 0, 2)
    v = (x @ w[prefix + "v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
    q, k = apply_rope(q, k, cos, sin)
    out = _attention_core(q, k, v, nh, nkv, hd, T)
    out = out * softplus_ln2(x @ w[prefix + "gate_proj.weight"].T)
    return out @ w[prefix + "o_proj.weight"].T


def sparse_moe(x, w, prefix, cfg):
    """K2HorizonSparseMoeBlock：路由 + top-k 专家 + 共享专家。"""
    T = x.shape[0]
    logits = x @ w[prefix + "gate.weight"].T
    bias = w.get(prefix + "gate.bias")
    topk = cfg["num_experts_per_tok"]
    rw, sel = calc_router(logits, bias, topk, cfg.get("router_scaling_factor"))
    out = np.zeros((T, cfg["hidden_size"]), np.float32)
    flat_e = sel.ravel()
    flat_t = np.repeat(np.arange(T), topk)
    flat_w = rw.ravel()
    for e in np.unique(flat_e):
        m = flat_e == e
        ts = flat_t[m]
        out[ts] += mlp_forward(x[ts], w, f"{prefix}experts.{e}.") * flat_w[m][:, None]
    if cfg.get("num_shared_experts", 0) > 0:
        out = out + mlp_forward(x, w, prefix + "shared_experts.")
    return out


def model_forward(ids, w, cfg, layer_hook=None):
    """完整 forward。ids [T] int；w：HF state_dict 同键名 fp32 numpy dict。返回 logits [T, vocab]。
    layer_hook(li, x): 每层完成后的回调（调试/对账用）。"""
    T = len(ids)
    x = w["model.embed_tokens.weight"][ids]
    hd = cfg["head_dim"]
    inv = 1.0 / (cfg["rope_theta"] ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    pos = np.arange(T, dtype=np.float32)
    ang = pos[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    # ★ HF 约定：freqs 块复制 cat([f,f])（rotate_half 的前后两半各配同一组频率），不是交错
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    ngroups = cfg["layernorm_num_groups"]
    for il in range(cfg["num_hidden_layers"]):
        pre = f"model.layers.{il}."
        sparse = il not in cfg.get("mlp_only_layers", [])
        h = grouped_rms(x, w[pre + "input_layernorm.weight"], ngroups, cfg["rms_norm_eps"])
        if sparse and cfg.get("mova_num_experts", 0) > 0:
            attn = mova_attention(h, w, pre + "self_attn.", cfg, cos, sin)
        else:
            attn = dense_attention(h, w, pre + "self_attn.", cfg, cos, sin)
        x = x + attn
        h2 = grouped_rms(x, w[pre + "post_attention_layernorm.weight"], ngroups, cfg["rms_norm_eps"])
        if sparse:
            x = x + sparse_moe(h2, w, pre + "mlp.", cfg)
        else:
            x = x + mlp_forward(h2, w, pre + "mlp.")
        if layer_hook is not None:
            layer_hook(il, x.copy())
    if layer_hook is not None:
        layer_hook(-1, x.copy())
    x = grouped_rms(x, w["model.norm.weight"], ngroups, cfg["rms_norm_eps"])
    return x @ w["lm_head.weight"].T

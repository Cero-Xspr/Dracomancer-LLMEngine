#!/usr/bin/env python3
"""C3 收尾③（KDA 族）：声明式求值器的 KDA 段 vs `ling_proto`（已对账参考），Ling 层 0 真权重。

参考链：ling_proto.kda_step ↔ llama.cpp（logits cos 0.998814、top-5 一致）。
本脚本验证：按 SEMANTICS_KDA/LAYER_STEPS_KDA 的语义**独立重写**的求值路径，与参考一致。

判据：cos=1.0（同机同精度下应到 1e-7 级）。
历史教训（写死在这）：第一版三个 conv 分支全错用了 `conv_q` ⇒ cos 0.056 ——
**逐段二分**（conv / delta / 读出）才定位到。对账时先分段，别全段一把梭。
用法：python3 kda_reconcile.py [层号]
"""
import os
import sys

import numpy as np

BASE = "/media/xiao_/OverSys1/npu-direct/hybrid"
sys.path.insert(0, BASE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
os.environ.setdefault("MODEL", "/media/xiao_/OverSys1/gguf/ling/Ling-3.0-tiny-Q4_K_M.gguf")
os.environ.setdefault("MAXT", "64")
import ling_proto as LP  # noqa: E402
import gguf  # noqa: E402

il = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if il in LP.ATTN_L:
    print(f"层 {il} 是 MLA 层（ATTN_L={sorted(LP.ATTN_L)}），本脚本只对 KDA 层"); sys.exit(1)
L = LP.LAYERS[il]
H, NH, HD, CK, DI = LP.H, LP.NH, LP.KDA_HEAD, LP.CONV_K, LP.D_INNER


def dq_name(name):
    """按 GGUF 存储序反排取反量化权重（与 ling_proto 的 farr/ftarr 同源，不自己另写一份）。"""
    t = LP.T[name]
    return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(
        tuple(int(v) for v in t.shape)[::-1])


rng = np.random.RandomState(5)
cur = (rng.randn(H).astype(np.float32) * 0.5)
p = f"blk.{il}."

# ── 参考：ling_proto.kda_step（真权重、量化 gemv 内核）──
LP.KDA_S[il][0][:] = 0; LP.KDA_S[il][1][:] = 0; LP.KDA_S[il][2][:] = 0; LP.KDA_H[il][:] = 0
ref = np.asarray(LP.kda_step(cur.copy(), L, il), np.float32)

# ── 被测：按声明式语义独立实现（fp32 权重 + numpy）──
wq, wk, wv, wo = (dq_name(p + "attn_q.weight"), dq_name(p + "attn_k.weight"),
                  dq_name(p + "attn_v.weight"), dq_name(p + "attn_output.weight"))


def conv1d(proj, state, cw):
    cin = np.concatenate([state, proj[:, None]], 1)
    o = (cw * cin).sum(1)
    np.copyto(state, cin[:, -(CK - 1):])
    return (o / (1.0 + np.exp(-o))).reshape(NH, HD)      # silu


def l2n(x):
    return x / np.maximum(np.sqrt((x * x).sum(-1, keepdims=True)), LP.EPS)


LP.KDA_S[il][0][:] = 0; LP.KDA_S[il][1][:] = 0; LP.KDA_S[il][2][:] = 0; LP.KDA_H[il][:] = 0
q = l2n(conv1d(cur @ wq.T, LP.KDA_S[il][0], L['conv_q']))
k = l2n(conv1d(cur @ wk.T, LP.KDA_S[il][1], L['conv_k']))
v = conv1d(cur @ wv.T, LP.KDA_S[il][2], L['conv_v'])
gate = (cur @ L['f_a'][1] + L['dt_b']).reshape(NH, HD)
g = 1.0 / (1.0 + np.exp(-gate * L['ssm_a'][:, None])) * LP.GATE_LB
beta = 1.0 / (1.0 + np.exp(-cur @ L['beta'][1]))
S = LP.KDA_H[il]
S *= np.exp(g)[:, :, None]
kv = np.einsum("hij,hi->hj", S, k)
S += k[:, :, None] * ((v - kv) * beta[:, None])[:, None, :]
o = np.einsum("hij,hi->hj", S, q) * (HD ** -0.5)
o = o / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-6) * L['o_norm']
og = 1.0 / (1.0 + np.exp(-cur @ L['g_a'][1])).reshape(NH, HD)
ours = (o * og).reshape(DI) @ wo.T

cos = float(np.dot(ref, ours) / (np.linalg.norm(ref) * np.linalg.norm(ours) + 1e-30))
print(f"KDA 层 {il}：声明式语义路径 vs ling_proto（已对账参考）")
print(f"  cos={cos:.8f}  max|Δ|={np.abs(ref - ours).max():.3e}")
ok = cos > 0.999999
print(f"  ⇒ {'✅ 一致（声明式语义与已对账参考相同）' if ok else '❌ 分歧 —— 先逐段二分（conv/delta/读出）'}")
sys.exit(0 if ok else 1)

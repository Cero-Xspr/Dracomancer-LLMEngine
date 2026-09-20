#!/usr/bin/env python3
"""k2_vk_ab.py — 层3 逐相位 GPU vs CPU 同输入对拍（定位 K2_VK 层3 校验和发散源）。"""
import os, sys
os.environ.setdefault("K2_VK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_engine as KE

assert KE.VK is not None, "需要 K2_VK=1"
VK = KE.VK

# 复用金标准 prompt
import json
gold = json.load(open(os.environ.get("GOLDEN", os.path.join(KE.BASE, "tests/golden/k2horizon_golden_iq2m.json"))))
ids = gold["ids"]

x_capture = {}
def hook(li, x):
    x_capture[li] = x

KE.reset()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p, layer_hook=hook)

x3 = x_capture[2]                      # 层3 输入 = 层2 输出
L = KE.LAYERS[3]
h = KE.grouped_rms1(x3, L.norm_a)
h2 = h                                  # 数值对拍用任意一致输入即可

def rel(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()
    s = max(np.abs(np.asarray(b, np.float64)).max(), 1e-9)
    return d, d / s

# ① refault 正确性：CPU 视图 vs 文件重读
import gguf_fast
raw = open(KE.MODEL, "rb")
t = KE.T["blk.3.attn_v_gate.weight"]
raw.seek(t.data_offset)
fb = np.frombuffer(raw.read(t.n_bytes), np.uint8)
d, r = rel(np.frombuffer(t.data, np.uint8), fb)
print(f"① refault: v_gate 视图 vs 文件 max|Δ|={d} rel={r:.2e} {'OK' if d == 0 else '*** 数据损坏 ***'}")

# ② fused4
qg, g4g, kg, vg_g = VK.fused4(h, L.qg, L.gg, L.kg, L.vgg)
qc, g4c, kc, vg_c = KE.gemv_fused4(h, L.qb, L.gb, L.kb, L.vgb, KE.NH*KE.HD, KE.NH*KE.HD, KE.NKV*KE.HD, KE.MEXP)
for nm, a, b in [("q", qg, qc), ("gate", g4g, g4c), ("k", kg, kc), ("v_router", vg_g, vg_c)]:
    d, r = rel(a, b)
    print(f"② fused4[{nm:8s}] max|Δ|={d:.3e} rel={r:.2e}")
sc_g = KE.sigmoid1(vg_g); sc_c = KE.sigmoid1(vg_c)
sel_g = np.argsort(-(sc_g + L.vgate_b), kind="stable")[:KE.MUSED]
sel_c = np.argsort(-(sc_c + L.vgate_b), kind="stable")[:KE.MUSED]
print(f"   v_router sel: GPU={list(sel_g)} CPU={list(sel_c)} {'SAME' if (sel_g==sel_c).all() else '*** 不同 ***'}")
sc_m = KE.sigmoid1(KE.L.gate_inp @ h2) if False else None

# ③ mova（固定用 CPU 的 sel）
sel = sel_c
wts = sc_c[sel]; wts = wts / wts.sum() * KE.R_SCALING
V4g = VK.mova4(h, L.veg, sel, L.ve_per, KE.VOUT)
V4c = np.stack([KE.gemv1(L.vec, L.veb[e*L.ve_per:], KE.VOUT, KE.H, h) for e in sel])
d, r = rel(V4g, V4c)
print(f"③ mova V4 max|Δ|={d:.3e} rel={r:.2e}")
vg_ = (KE.silu(V4g) * wts[:, None]).sum(0); vc_ = (KE.silu(V4c) * wts[:, None]).sum(0)
d, r = rel(vg_, vc_)
print(f"   v(合成) max|Δ|={d:.3e} rel={r:.2e}")

# ④ moe gate_up
sel_m = np.arange(8)                   # 固定 8 个专家（绕过路由分歧）
G8g, U8g = VK.gate_up(h2, L.exg, L.uxg, sel_m, L.ex_per, L.ux_per, KE.MOE_INTER)
G8c, U8c = KE.gemv_fused_gu(h2, L.exb, L.uxb, L.ex_per, L.ux_per, sel_m, KE.MOE_INTER)
d, r = rel(G8g, G8c); print(f"④ moe G8 max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(U8g, U8c); print(f"   moe U8 max|Δ|={d:.3e} rel={r:.2e}")

# ⑤ moe down
gu = KE.silu(G8c) * U8c
if L.dxg is not None:
    D8g = VK.down(gu, L.dxg, sel_m, L.dx_per, KE.H, KE.MOE_INTER)
    D8c = np.stack([KE.gemv1(L.dxc, L.dxb[e*L.dx_per:], KE.H, KE.MOE_INTER, gu[k]) for k, e in enumerate(sel_m)])
    d, r = rel(D8g, D8c)
    print(f"⑤ moe D8 max|Δ|={d:.3e} rel={r:.2e}")
else:
    print("⑤ 层3 ffn_down_exps 非 IQ2_S（走 CPU），跳过")

# ⑥ 路由敏感性：真实 h2 的 MoE 路由分数间隙
logits = L.gate_inp @ h2
sc = KE.sigmoid1(logits)
order = np.sort(sc + L.probs_b)[::-1]
gaps = np.diff(np.sort(sc + L.probs_b)[::-1][:NUSED+2])
print(f"⑥ MoE 路由分数 top10 间隙: {np.array2string(gaps[:5], precision=3)} (最小 {gaps.min():.3e})")
scv = KE.sigmoid1(vg_c) + L.vgate_b
gv = np.sort(scv)[::-1]
print(f"   MoVA 路由分数 top6: {np.array2string(gv[:6], precision=6)} 间隙 {np.array2string(-np.diff(gv[:6]), precision=3)}")

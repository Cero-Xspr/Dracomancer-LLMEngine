#!/usr/bin/env python3
"""k2_dbg5.py — 捕获 gate 流中层5 的 GPU 相位输出 vs CPU 重算（同输入）。"""
import os, sys
os.environ.setdefault("K2_VK", "1")
_gold = "/media/Data-1/npu-direct/hybrid/tests/golden/k2horizon_golden_iq2m.json"
os.environ["K2_FORCE_SEL"] = "/media/Data-1/npu-direct/hybrid/tests/golden/k2horizon_sels_iq2m.json"
os.environ["K2_FORCE_POS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import json
import k2_engine as KE

assert KE.VK is not None and KE.FORCE is not None, "FORCE 未生效"
print("FORCE_POS =", KE.FORCE_POS)

VK = KE.VK
REC = {}
_ORIG = {n: getattr(VK, n) for n in ("fused4", "mova4", "gate_up9", "down9")}

L5 = KE.LAYERS[5]
MARK = {"fused4": (1, L5.qg[0]), "mova4": (1, L5.veg), "gate_up9": (1, L5.exg), "down9": (2, L5.dxg)}
def wrap(name):
    fn = _ORIG[name]
    idx, target = MARK[name]
    def w(*a, **k):
        r = fn(*a, **k)
        g = a[idx]
        if isinstance(g, tuple):
            g = g[0]
        if g is target:
            REC[name] = (a, r)
        return r
    return w

for n, f in _ORIG.items():
    setattr(VK, n, wrap(n))

MOE_REC = {}
_moe_orig = KE.moe_ffn
def _moe_w(h2, L, pos=-1):
    r = _moe_orig(h2, L, pos)
    if L.li == 5:
        MOE_REC["in"] = h2
        MOE_REC["ret"] = r
    return r
KE.moe_ffn = _moe_w

gold = json.load(open(_gold))
ids = gold["ids"]
cap = {}
KE.reset()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p, layer_hook=lambda li, x: cap.__setitem__(li, x))

print("捕获:", {k: len(v) for k, v in REC.items()})
# 取末位（最后一次调用）
fused = REC["fused4"]
mova = REC["mova4"]
gu9 = REC["gate_up9"]
dn9 = REC["down9"]

L = KE.LAYERS[5]
x5 = cap[4]
x5g = cap[5]

def rel(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()
    return d, d / max(np.abs(np.asarray(b, np.float64)).max(), 1e-9)

# ── CPU 全重算（同 x5、同强制 sel）──
h = KE.grouped_rms1(x5, L.norm_a)
qc, g4, kc, vc = KE.gemv_fused4(h, L.qb, L.gb, L.kb, L.vgb, KE.NH*KE.HD, KE.NH*KE.HD, KE.NKV*KE.HD, KE.MEXP)
q = qc.reshape(KE.NH, KE.HD)
k = KE.rope1(kc.reshape(KE.NKV, KE.HD), 4)
L.K[4] = k
scv = KE.sigmoid1(vc)
sel_v = np.array(KE.FORCE[5][0])
wts = scv[sel_v]; wts = wts / wts.sum() * KE.R_SCALING
V4 = np.stack([KE.gemv1(L.vec, L.veb[e*L.ve_per:], KE.VOUT, KE.H, h) for e in sel_v])
v = (KE.silu(V4) * wts[:, None]).sum(0)
L.V[4] = v.reshape(KE.NKV, KE.HD)
qr = KE.rope1(q, 4)
Kc = np.repeat(L.K[:5].transpose(1, 0, 2), KE.NH // KE.NKV, axis=0)
Vc = np.repeat(L.V[:5].transpose(1, 0, 2), KE.NH // KE.NKV, axis=0)
att = KE.softmax_last((qr[:, None, :] @ Kc.transpose(0, 2, 1)) * np.float32(KE.HD ** -0.5))
outa = (att @ Vc)[:, 0, :].reshape(KE.NH * KE.HD)
gate = KE.softplus_ln2(g4)
o = KE.gemv1(L.oc, L.ob, KE.H, KE.NH * KE.HD, outa * gate)
x_mid = x5 + o
h2 = KE.grouped_rms1(x_mid, L.norm_f)
logits = L.gate_inp @ h2
sc = KE.sigmoid1(logits)
sel_m = np.array(KE.FORCE[5][1])
rw = sc[sel_m]; rw = rw / rw.sum() * KE.R_SCALING
G8, U8 = KE.gemv_fused_gu(h2, L.exb, L.uxb, L.ex_per, L.ux_per, sel_m, KE.MOE_INTER)
gu8 = KE.silu(G8) * U8
out = np.zeros(KE.H, np.float32)
for kk, e in enumerate(sel_m):
    out += KE.gemv1(L.dxc, L.dxb[e*L.dx_per:], KE.H, KE.MOE_INTER, gu8[kk]) * rw[kk]
sg = KE.gemv1(L.sgc, L.sgb, KE.MOE_INTER, KE.H, h2)
su = KE.gemv1(L.suc, L.sub, KE.MOE_INTER, KE.H, h2)
sd = KE.gemv1(L.sdc, L.sdb, KE.H, KE.MOE_INTER, KE.silu(sg) * su)
x5_cpu = x_mid + out + sd

# ── GPU 侧中间量（从 REC 拿输入/输出）──
h_g = fused[0][0]
d, r = rel(h_g, h); print(f"h (rms)      max|Δ|={d:.3e} rel={r:.2e}")
qg, g4g, kg, vgg_ = fused[1]
d, r = rel(qg, qc); print(f"q            max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(kg, kc); print(f"k            max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(vgg_, vc); print(f"v_router     max|Δ|={d:.3e} rel={r:.2e}")
V4g = mova[1]
d, r = rel(V4g, V4); print(f"mova V4      max|Δ|={d:.3e} rel={r:.2e}")
(h2_g, ) = gu9[0][:1]
d, r = rel(h2_g, h2); print(f"h2 (rms ffn) max|Δ|={d:.3e} rel={r:.2e}")
G8g, U8g, sgg_, sug_ = gu9[1]
d, r = rel(G8g, G8); print(f"G8           max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(sgg_, sg); print(f"sg(shexp)    max|Δ|={d:.3e} rel={r:.2e}")
gu_g = dn9[0][0]
d, r = rel(gu_g, gu8); print(f"gu8 (silu*U) max|Δ|={d:.3e} rel={r:.2e}")
D8g, d9g = dn9[1]
D8c = np.stack([KE.gemv1(L.dxc, L.dxb[e*L.dx_per:], KE.H, KE.MOE_INTER, gu8[kk]) for kk, e in enumerate(sel_m)])
d, r = rel(D8g, D8c); print(f"D8           max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(d9g, sd); print(f"sd(shexp d)  max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(x5g, x5_cpu); print(f"x5 (层输出)  max|Δ|={d:.3e} rel={r:.2e}")
moe_e = MOE_REC["ret"]
moe_m = out + sd
d, r = rel(moe_e, moe_m); print(f"moe 返回值   max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(MOE_REC["in"], h2); print(f"moe 输入 h2  max|Δ|={d:.3e} rel={r:.2e}")
# 引擎 moe = (D8e*rw_e).sum + d9e ⇒ 反解 rw_e：对每维不行的，但可对 D8 行 0 单独试
# 直接对比 x_mid：x_mid_e = x5g - moe_e ; x_mid_m = x5 + o
xmid_e = x5g - moe_e
xmid_m = x5 + o
d, r = rel(xmid_e, xmid_m); print(f"x_mid 反解   max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(o, o); print(f"(sanity o=o) max|Δ|={d:.3e}")

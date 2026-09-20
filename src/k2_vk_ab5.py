#!/usr/bin/env python3
"""k2_vk_ab5.py — 层5 gate_up9/down9 逐相位 GPU vs CPU 对拍。"""
import os, sys
os.environ.setdefault("K2_VK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_engine as KE
import json

assert KE.VK is not None
VK = KE.VK
gold = json.load(open(os.path.join(KE.BASE, "tests/golden/k2horizon_golden_iq2m.json")))
ids = gold["ids"]
cap = {}
KE.reset()
for p, tid in enumerate(ids):
    KE.forward(int(tid), p, layer_hook=lambda li, x: cap.__setitem__(li, x))

L = KE.LAYERS[5]
x5 = cap[4]
h = KE.groupedms1(x5, L.norm_a) if False else KE.grouped_rms1(x5, L.norm_a)
h2 = h
import json as _j
_f = _j.load(open('/media/Data-1/npu-direct/hybrid/tests/golden/k2horizon_sels_iq2m.json'))
sel_m = np.array(_f["last_sels"][5][1])
sel_v = np.array(_f["last_sels"][5][0])
print("强制 sel_moe:", list(sel_m), " sel_v:", list(sel_v))
sel = sel_m

def rel(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()
    return d, d / max(np.abs(np.asarray(b, np.float64)).max(), 1e-9)

# gate_up9
G8g, U8g, sgg_, sug_ = VK.gate_up9(h2, L.exg, L.uxg, sel, L.ex_per, L.ux_per, L.sgg, L.sug, KE.MOE_INTER)
G8c, U8c = KE.gemv_fused_gu(h2, L.exb, L.uxb, L.ex_per, L.ux_per, sel, KE.MOE_INTER)
sgc = KE.gemv1(L.sgc, L.sgb, KE.MOE_INTER, KE.H, h2)
suc = KE.gemv1(L.suc, L.sub, KE.MOE_INTER, KE.H, h2)
for nm, a, b in [("G8", G8g, G8c), ("U8", U8g, U8c), ("sg", sgg_, sgc), ("su", sug_, suc)]:
    d, r = rel(a, b)
    print(f"gate_up9[{nm}] max|Δ|={d:.3e} rel={r:.2e}")

# down9
gu8 = KE.silu(G8c) * U8c
su_in = KE.silu(sgc) * suc
D8g, d9g = VK.down9(gu8, su_in, L.dxg, sel, L.dx_per, L.sdg, KE.H, KE.MOE_INTER)
D8c = np.stack([KE.gemv1(L.dxc, L.dxb[e*L.dx_per:], KE.H, KE.MOE_INTER, gu8[k]) for k, e in enumerate(sel)])
d9c = KE.gemv1(L.sdc, L.sdb, KE.H, KE.MOE_INTER, su_in)
d, r = rel(D8g, D8c); print(f"down9[D8] max|Δ|={d:.3e} rel={r:.2e}")
d, r = rel(d9g, d9c); print(f"down9[d9] max|Δ|={d:.3e} rel={r:.2e}")
# 分专家看 D8 哪行错
for k in range(8):
    d, r = rel(D8g[k], D8c[k])
    print(f"  D8[{k}] rel={r:.2e}")

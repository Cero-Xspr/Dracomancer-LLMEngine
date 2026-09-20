#!/usr/bin/env python3
"""k2_map_audit.py — NANI GGUF 映射 vs 官方 safetensors 逐张量对拍。
用法: python3 k2_map_audit.py   (数据: /tmp/st_l3/*.bin + 本地 Q4_K_M)
判据: Q4_K 相对误差 ~1% 内 = 同权重（对上了）；~50-100% = 映射错。"""
import os, sys, json, glob
os.environ.setdefault("GGUF_PY_DIR", "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gguf_fast
from gguf.quants import dequantize

ST = "/tmp/st_l3"
_SHAPES = json.load(open("/tmp/st4_pick.json"))
def st_deq(fn):
    u = np.fromfile(fn, np.uint16).astype(np.uint32)
    a = (u << np.uint32(16)).view(np.float32)
    key = os.path.basename(fn)[:-4]
    shp = _SHAPES[key][1]
    return a.reshape(shp)

def gg_get(R, T, name):
    t = T[name]
    return np.asarray(dequantize(t.data, t.tensor_type), np.float32)

def relerr(a, b):
    if a.shape != b.shape:
        return None
    denom = np.abs(b).max() + 1e-30
    return float(np.abs(a - b).max() / denom)

R = gguf_fast.FastGGUF('/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf')
T = {t.name: t for t in R.tensors}

def candidates(a, shapes_from_st):
    """a: gguf dequant；返回 {描述: 数组} 候选对齐"""
    c = {"as-is": a, "T": a.T}
    return c

pairs = [
    # (HF 名, GGUF 名, 变换)
    ("model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight", "T"),
    ("model.layers.3.self_attn.k_proj.weight", "blk.3.attn_k.weight", "T"),
    ("model.layers.3.self_attn.o_proj.weight", "blk.3.attn_output.weight", "T"),
    ("model.layers.3.gate_proj.weight", "blk.3.attn_gate.weight", "T"),
    ("model.layers.3.mlp.gate.weight", "blk.3.ffn_gate_inp.weight", "T"),
    ("model.layers.3.mlp.gate.bias", "blk.3.exp_probs_b.bias", None),
    ("model.layers.3.self_attn.v_router.weight", "blk.3.attn_v_gate.weight", "T"),
    ("model.layers.3.self_attn.v_router.bias", "blk.3.attn_v_gate.bias", None),
    ("model.layers.3.mlp.shared_experts.gate_proj.weight", "blk.3.ffn_gate_shexp.weight", "T"),
    ("model.layers.3.mlp.shared_experts.up_proj.weight", "blk.3.ffn_up_shexp.weight", "T"),
    ("model.layers.3.mlp.shared_experts.down_proj.weight", "blk.3.ffn_down_shexp.weight", "T"),
    ("model.layers.3.mlp.experts.0.gate_proj.weight", "blk.3.ffn_gate_exps.weight", "e0"),
    ("model.layers.3.mlp.experts.0.up_proj.weight", "blk.3.ffn_up_exps.weight", "e0"),
    ("model.layers.3.mlp.experts.0.down_proj.weight", "blk.3.ffn_down_exps.weight", "e0"),
    ("model.layers.3.mlp.experts.1.gate_proj.weight", "blk.3.ffn_gate_exps.weight", "e1"),
    ("model.layers.3.mlp.experts.1.up_proj.weight", "blk.3.ffn_up_exps.weight", "e1"),
    ("model.layers.3.mlp.experts.1.down_proj.weight", "blk.3.ffn_down_exps.weight", "e1"),
]

print(f"{'HF 张量':52} {'GGUF':28} {'HF shape':14} {'GG shape':16} rel-err  判定")
for hf, gn, how in pairs:
    fn = os.path.join(ST, hf + ".bin")
    if not os.path.exists(fn):
        print(f"{hf:52} 文件缺失"); continue
    a_st = st_deq(fn)
    if how and "exp" in how and gn in T:
        g = gg_get(R, T, gn)
        idx = 0 if how == "e0" else 1
        # 三种可能的堆叠轴序都试
        cands = {"[:, :, e]": g[:, :, idx] if g.ndim == 3 else None,
                 "[e, :, :]": g[idx] if g.ndim == 3 else None,
                 "[e, :, :].T": g[idx].T if g.ndim == 3 else None,
                 "[:, :, e].T": g[:, :, idx].T if g.ndim == 3 else None}
        best = min(((k, v) for k, v in cands.items() if v is not None),
                   key=lambda kv: float('inf') if relerr(a_st, kv[1]) is None else relerr(a_st, kv[1]))
        e = relerr(a_st, best[1])
        print(f"{hf:52} {gn}[{best[0]}]{'':10} {str(a_st.shape):14} {str(best[1].shape):16} "
              f"{'---' if e is None else f'{e:.4f}'}  {'MATCH' if e is not None and e < 0.05 else '*** MISMATCH'}")
    else:
        g = gg_get(R, T, gn)
        if how == "T":
            e = relerr(a_st, g.T)
            e2 = relerr(a_st, g)
            ee, tag = (e, "T") if (e2 is None or (e is not None and e <= e2)) else (e2, "as-is")
        else:
            ee, tag = relerr(a_st, g), "as-is"
        print(f"{hf:52} {gn}{(28-len(gn))*' '} {str(a_st.shape):14} {str(g.shape):16} "
              f"{'---' if ee is None else f'{ee:.4f}'}  {'MATCH' if ee is not None and ee < 0.05 else '*** MISMATCH'}")

# v_experts: HF 64 个 [1024,2560] vs GGUF (2560,1024,64)
for i in (0, 1):
    hf = f"model.layers.3.self_attn.v_experts.{i}.weight"
    fn = os.path.join(ST, hf + ".bin")
    if not os.path.exists(fn):
        print(f"{hf:52} 文件缺失"); continue
    a_st = st_deq(fn)
    g = gg_get(R, T, "blk.3.attn_v_exps.weight")  # (2560,1024,64) dequant 后 ne 序
    cands = {}
    if g.ndim == 3:
        cands["[:,:,e]"] = g[:, :, i]
        cands["[e,:,:]"] = g[i] if i < g.shape[0] else None
        cands["[:,:,e].T"] = g[:, :, i].T
        cands["[e,:,:].T"] = g[i].T if i < g.shape[0] else None
    best, be = None, None
    for kk, vv in cands.items():
        if vv is None: continue
        e = relerr(a_st, vv)
        if e is not None and (be is None or e < be): best, be = kk, e
    print(f"{hf:52} v_exps[{best}]{'':14} {str(a_st.shape):14} {str(g.shape):16} "
          f"{'---' if be is None else f'{be:.4f}'}  {'MATCH' if be is not None and be < 0.05 else '*** MISMATCH'}")
#!/usr/bin/env python3
"""k2_prefill.py — K2-Horizon 的批量（chunked）prefill。

逐层把 T 个 token 的投影批成 GEMM（权重每层只读一次），MoE/MoVA 按专家分组
（专家权重也只读一次），注意力用 numpy 批量因果 softmax。KV 写回引擎缓存，
与 k2_engine 的逐 token 路径共用同一份权重与缓存布局。

用法：
  import k2_engine as KE; import k2_prefill as KP
  pf = KP.Prefiller(KE)
  logits_last, x_all = pf.prefill(ids, pos0=0)
"""
import os
import ctypes as ct
import numpy as np

import gguf.quants as Q


def _pf(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_float))


def _p8(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_uint8))


class Prefiller:
    def __init__(self, KE):
        self.KE = KE
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m5")
        self.G = ct.CDLL(os.path.join(base, "m5_gemm.so"))
        self.G.m5_gemm.restype = ct.c_int
        self.G.m5_gemm.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                                   ct.c_int, ct.c_int, ct.c_int,
                                   ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)]
        H = KE.H
        self.wbuf = np.empty(64 * (H + 16), np.float32)   # 每线程一份行反量化缓冲
        self._rowcache = {}   # name -> (nbytes, dequantized f32 [rows]) 小矩阵缓存？不缓存大矩阵

    # —— 批量投影：Y[T, n_out] = X[T, n_in] · Wᵀ ——
    def gemm(self, X, wname, L, which, T):
        KE = self.KE
        code, wb, n_out, n_in, per = self._winfo(L, which)
        W = wb
        if per is not None:   # 堆叠专家：取第 0 个专家？—— 由调用方传指针，不走这里
            raise RuntimeError("use gemm_expert")
        Y = np.empty((T, n_out), np.float32)
        assert self.G.m5_gemm(code, _pf(np.ascontiguousarray(X)), _p8(W), T, n_out, n_in,
                              _pf(Y), _pf(self.wbuf)) == 0
        return Y

    def _winfo(self, L, which):
        KE = self.KE
        m = {
            "q": (L.qc, L.qb, KE.NH * KE.HD, KE.H, None),
            "k": (L.kc, L.kb, KE.NKV * KE.HD, KE.H, None),
            "v": (L.vc, L.vb, KE.NKV * KE.HD, KE.H, None),
            "gate_a": (L.gc, L.gb, KE.NH * KE.HD, KE.H, None),
            "o": (L.oc, L.ob, KE.H, KE.NH * KE.HD, None),
            "g1": (L.g1c, L.g1b, 6144, KE.H, None) if hasattr(L, "g1c") else (None, None, 0, 0, None),
            "u1": (L.u1c, L.u1b, 6144, KE.H, None) if hasattr(L, "u1c") else (None, None, 0, 0, None),
            "d1": (L.d1c, L.d1b, KE.H, 6144, None) if hasattr(L, "d1c") else (None, None, 0, 0, None),
            "sg": (L.sgc, L.sgb, KE.MOE_INTER, KE.H, None),
            "su": (L.suc, L.sub, KE.MOE_INTER, KE.H, None),
            "sd": (L.sdc, L.sdb, KE.H, KE.MOE_INTER, None),
        }
        code, wb, n_out, n_in, per = m[which]
        return code, wb, n_out, n_in, per

    def gemm_w(self, X, code, W, n_out, n_in, T):
        Y = np.empty((T, n_out), np.float32)
        assert self.G.m5_gemm(code, _pf(np.ascontiguousarray(X)), _p8(W), T, n_out, n_in,
                              _pf(Y), _pf(self.wbuf)) == 0
        return Y

    # —— MoVA V：按专家分组 ——
    def mova_v(self, h, L, T):
        KE = self.KE
        vlogits = h @ self._router_w(L).T                                   # [T, 64]
        sc = KE.sigmoid1(vlogits) if KE.GATING_FUNC == 2 else KE.softmax(vlogits)
        sel_scores = sc + L.vgate_b[None, :]
        sel = np.argsort(-sel_scores, axis=1, kind="stable")[:, :KE.MUSED]  # [T, 4]
        rw = np.take_along_axis(sc, sel, 1)
        rw = rw / rw.sum(1, keepdims=True) * KE.R_SCALING
        V = np.zeros((T, KE.NKV * KE.HD), np.float32)   # ★ 必须 zeros：V[ts] += 是累加
        for e in np.unique(sel.ravel()):
            ts, kk = np.nonzero(sel == e)
            if len(ts) >= 3:
                ve = self.gemm_w(h[ts], L.vec, L.veb[e * L.ve_per:], KE.NKV * KE.HD, KE.H, len(ts))
            else:
                ve = np.stack([KE.gemv1(L.vec, L.veb[e * L.ve_per:], KE.NKV * KE.HD, KE.H, h[tix])
                               for tix in ts])
            V[ts] += KE.silu(ve) * rw[ts, kk][:, None]
        return V, sel, rw

    def _router_w(self, L):
        # v_router 权重 [64,2560] 小矩阵：反量化一次缓存在 L 上（稠密层没有，返回 None）
        if not hasattr(L, "_vrw"):
            if not getattr(L, "sparse", False):
                L._vrw = None
                return None
            tt = self.KE.T[f"blk.{L.li}.attn_v_gate.weight"]
            L._vrw = np.asarray(Q.dequantize(tt.data, tt.tensor_type), np.float32)
        return L._vrw

    # —— MoE FFN：按专家分组 ——
    def moe_chunk(self, h2, L, T):
        KE = self.KE
        logits = h2 @ L.gate_inp.T                                          # [T, 100]
        sc = KE.sigmoid1(logits) if KE.GATING_FUNC == 2 else KE.softmax(logits)
        sel_scores = sc + L.probs_b[None, :]
        sel = np.argsort(-sel_scores, axis=1, kind="stable")[:, :KE.NUSED]
        rw = np.take_along_axis(sc, sel, 1)
        rw = rw / rw.sum(1, keepdims=True) * KE.R_SCALING
        out = np.zeros((T, KE.H), np.float32)
        for e in np.unique(sel.ravel()):
            ts, kk = np.nonzero(sel == e)
            ne = len(ts)
            if ne >= 3:   # 批量 GEMM：去量化一次摊 T 行
                g = self.gemm_w(h2[ts], L.exc, L.exb[e * L.ex_per:], KE.MOE_INTER, KE.H, ne)
                u = self.gemm_w(h2[ts], L.uxc, L.uxb[e * L.ux_per:], KE.MOE_INTER, KE.H, ne)
                gu = KE.silu(g) * u
                d = self.gemm_w(gu, L.dxc, L.dxb[e * L.dx_per:], KE.H, KE.MOE_INTER, ne)
            else:         # 稀疏行：直接 gemv（免 dequant+GEMM 的固定开销）
                gs = np.empty((ne, KE.MOE_INTER), np.float32)
                us = np.empty((ne, KE.MOE_INTER), np.float32)
                ds = np.empty((ne, KE.H), np.float32)
                for j, tix in enumerate(ts):
                    gs[j] = KE.gemv1(L.exc, L.exb[e * L.ex_per:], KE.MOE_INTER, KE.H, h2[tix])
                    us[j] = KE.gemv1(L.uxc, L.uxb[e * L.ux_per:], KE.MOE_INTER, KE.H, h2[tix])
                    ds[j] = KE.gemv1(L.dxc, L.dxb[e * L.dx_per:], KE.H, KE.MOE_INTER, KE.silu(gs[j]) * us[j])
                d = ds
            out[ts] += d * rw[ts, kk][:, None]
        # 共享专家（批量）
        g = self.gemm_w(h2, L.sgc, L.sgb, KE.MOE_INTER, KE.H, T)
        u = self.gemm_w(h2, L.suc, L.sub, KE.MOE_INTER, KE.H, T)
        d = self.gemm_w(KE.silu(g) * u, L.sdc, L.sdb, KE.H, KE.MOE_INTER, T)
        return out + d

    # —— 批量 rope（HF rotate_half 约定，位置 pos0..pos0+T-1）——
    @staticmethod
    def rope_batch(v, pos0, hd, theta):
        """v [T, n_head, hd]（或 [T, hd]）；行=token，位置 pos0..。HF rotate_half 约定。"""
        T = v.shape[0]
        inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        ang = (pos0 + np.arange(T, dtype=np.float32))[:, None] * inv[None, :]
        cos = np.cos(ang).astype(np.float32)[:, None, :] if v.ndim == 3 else np.cos(ang).astype(np.float32)
        sin = np.sin(ang).astype(np.float32)[:, None, :] if v.ndim == 3 else np.sin(ang).astype(np.float32)
        cos = np.concatenate([cos, cos], -1)
        sin = np.concatenate([sin, sin], -1)
        half = hd // 2
        rot = np.concatenate([-v[..., half:], v[..., :half]], -1)
        return v * cos + rot * sin

    # —— 主入口 ——
    def prefill(self, ids, pos0=0, want_all_logits=False, layer_hook=None):
        KE = self.KE
        T = len(ids)
        H, NH, NKV, HD = KE.H, KE.NH, KE.NKV, KE.HD
        eps = KE.EPS
        # 嵌入批量反量化
        t = KE.T["token_embd.weight"]
        x = np.empty((T, H), np.float32)
        for i, tid in enumerate(ids):
            x[i] = Q.dequantize(t.data[int(tid)], t.tensor_type)
        grp = NH // NKV
        DBG = {}
        self._dbg = DBG
        for li, L in enumerate(KE.LAYERS):
            h = KE.grouped_rms1(x, L.norm_a) if x.ndim == 1 else self._rms_batch(x, L.norm_a, 2, eps)
            # Q/K 批量
            qf = self.gemm_w(h, L.qc, L.qb, NH * HD, H, T)
            kf = self.gemm_w(h, L.kc, L.kb, NKV * HD, H, T)
            q = self.rope_batch(qf.reshape(T, NH, HD), pos0, HD, KE.THETA)   # ★ q 必须过 rope
            k = self.rope_batch(kf.reshape(T, NKV, HD), pos0, HD, KE.THETA)
            # V
            if L.sparse:
                V, _, _ = self.mova_v(h, L, T)
            else:
                V = self.gemm_w(h, L.vc, L.vb, NKV * HD, H, T)
            if os.environ.get("K2DBG3") and li == 3:
                DBG.update(h3=h.copy(), q3=q.copy(), k3=k.copy(), V3=V.copy())
            # 写 KV 缓存
            L.K[pos0:pos0 + T] = k
            L.V[pos0:pos0 + T] = V.reshape(T, NKV, HD)
            # 批量注意力（因果）
            kv_n = pos0 + T
            Kc = np.repeat(L.K[:kv_n].transpose(1, 0, 2), grp, axis=0)   # [NH, kv_n, HD]
            Vc = np.repeat(L.V[:kv_n].transpose(1, 0, 2), grp, axis=0)
            qh = np.ascontiguousarray(q.transpose(1, 0, 2))              # [NH, T, HD]
            att = KE.softmax_last(
                (qh @ Kc.transpose(0, 2, 1)) * np.float32(HD ** -0.5)
                + self._causal_mask(T, pos0))                            # [NH, T, kv_n]
            out = (att @ Vc).transpose(1, 0, 2).reshape(T, NH * HD)
            gate = KE.softplus_ln2(self.gemm_w(h, L.gc, L.gb, NH * HD, H, T))
            attn = self.gemm_w(out * gate, L.oc, L.ob, H, NH * HD, T)
            x = x + attn
            h2 = self._rms_batch(x, L.norm_f, 2, eps)
            if os.environ.get("K2DBG3") and li == 3:
                DBG.update(att3=att.copy(), out3=out.copy(), gate3=gate.copy(),
                           attn3=attn.copy(), h23=h2.copy())
            if L.sparse:
                mo = self.moe_chunk(h2, L, T)
                if os.environ.get("K2DBG3") and li == 3:
                    DBG["moe3"] = mo.copy()
                x = x + mo
            else:
                g = self.gemm_w(h2, L.g1c, L.g1b, 6144, H, T)
                u = self.gemm_w(h2, L.u1c, L.u1b, 6144, H, T)
                x = x + self.gemm_w(KE.silu(g) * u, L.d1c, L.d1b, H, 6144, T)
            if layer_hook is not None:
                layer_hook(li, x.copy())   # 全行（调试口径）
        xf = self._rms_batch(x, KE.ONORM, 2, eps)
        logits = self.gemm_w(xf if want_all_logits else xf[-1:],
                             KE.tcode("output.weight"), KE.wview("output.weight"),
                             KE.VOCAB, H, 1 if not want_all_logits else T)
        return (logits, x) if want_all_logits else (logits[0], x)

    @staticmethod
    def _rms_batch(x, w, n_groups, eps):
        shp = x.shape
        g = x.reshape(*shp[:-1], n_groups, -1)
        var = (g * g).mean(-1, keepdims=True)
        g = g * (1.0 / np.sqrt(var + eps))
        return g.reshape(shp) * w

    @staticmethod
    def _causal_mask(T, pos0):
        """query i (绝对位置 pos0+i) 可见 key 0..pos0+i。返回 [T, kv_n] 的 0/-inf。"""
        kv_n = pos0 + T
        m = np.zeros((T, kv_n), np.float32)
        for i in range(T):
            m[i, pos0 + i + 1:] = -np.inf
        return m[None, :, :]   # 广播到 heads


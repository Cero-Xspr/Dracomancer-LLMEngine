#!/usr/bin/env python3
"""qwen35 家族的 chunked prefill（M1：两段式 GDN + 批注意力 + 批投影）。

把"逐 token forward 的 prefill"换成：
  每层投影（qkv/gate/q/k/v/o/ssm_out）对全部 T 个位置一次 GEMM（权重只反量化一遍），
  GDN 递归在 m6_gdn_scan 里逐 token 走（只碰激活与 S/tail 状态），
  MoE FFN 仍逐 token 调既有 C 算子（M2 再批）。
状态（S/tail/Kr/Vc）直接落在引擎原缓冲 ⇒ 扫完即可无缝续 decode。

等价性闸门：末位 logits cos ≥0.999 + 贪心续接与逐 token 路径逐 token 一致。
"""
import os, sys, ctypes as ct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf

EPS = 1e-6           # 与 m6_layer.c #define EPS 一致（attention.layer_norm_rms_epsilon）


def pf(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_float))


def rms_mean(X, w, h=None):
    """attn_norm/post_norm 版 RMS：inv = 1/sqrt(ms/H + eps)。X: [H] 或 [T,H]。"""
    Hh = X.shape[-1]
    ms = (X * X).sum(-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms / Hh + EPS)
    return X * inv * w


def softmax_causal(Qr, Kr_np, Vc_np, p0, T, nh, nkv, hd, gate, scale):
    """Qr[T,nh,hd]，KV 取 [0..p0+T) 行，因果遮罩。返回 [T, nh*hd]。"""
    Tt = p0 + T
    outs = np.empty((T, nh * hd), np.float32)
    for h in range(nh):
        kv = h // (nh // nkv)
        sc = (Qr[:, h, :] @ Kr_np[:Tt, kv*hd:(kv+1)*hd].T) * scale   # [T, Tt]
        sc += np.where(np.arange(Tt)[None, :] <= (p0 + np.arange(T))[:, None], 0.0, -1e30)
        sc -= sc.max(1, keepdims=True)
        P = np.exp(sc)
        P /= P.sum(1, keepdims=True)
        outs[:, h*hd:(h+1)*hd] = P @ Vc_np[:Tt, kv*hd:(kv+1)*hd]
    return outs * gate


def rope_(H3, pos0, rope_inv, half):
    """原地 rope：每头内维度对 (t, t+half)，t<half。H3: [T, heads, hd]，最后一维是头内维度。"""
    ang = (pos0 + np.arange(H3.shape[0]))              # [T]
    for t in range(half):
        c = np.cos(ang * rope_inv[t])[:, None]         # [T,1]
        s = np.sin(ang * rope_inv[t])[:, None]
        a = H3[..., t].copy()
        b = H3[..., t + half].copy()
        H3[..., t] = a * c - b * s
        H3[..., t + half] = a * s + b * c
    return H3


class ChunkPrefiller:
    def __init__(self, QE):
        self.QE = QE
        base = QE.BASE
        self.G = ct.CDLL(os.path.join(base, "m5", "m5_gemm.so"))
        self.G.m5_gemm.restype = ct.c_int
        self.G.m5_gemm.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                                   ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        self.SC = ct.CDLL(os.path.join(base, "m6_gdn_scan.so"))
        self.MT = ct.CDLL(os.path.join(base, "m6_moe_tok.so"))
        self.MT.m6_moe_tok.restype = None
        self.MT.m6_moe_tok.argtypes = [ct.c_void_p] * 8 + [ct.c_int] * 6 + [ct.c_void_p] * 3
        self._dq_fp = ct.cast(self.G.m5_dequant_row, ct.c_void_p)  # 注入反量化入口（v3 无嵌套 OMP）
        self.FG = ct.CDLL(os.path.join(base, "m5", "m5_gemm_fused.so"))   # IQ3_S 融合 GEMM
        self._fused_fp = ct.cast(self.FG.m5_gemm_iq3s, ct.c_void_p)
        self.SC.m6_gdn_scan.restype = None
        self.SC.m6_gdn_scan.argtypes = [ct.c_void_p]*8 + [ct.c_int]*3 + [ct.c_void_p]*2
        self.NTH = (os.cpu_count() or 8)
        # ★ M2 融合版已转正（2026-09-19）：m6_moe_tok 对全 IQ3_S 专家走 m5_gemm_fused
        #   （寄存器级反量化 × 8-token 分块，v16/dg 复用），端到端热态交替 2.15-2.20×（vs 逐 token），
        #   末位 cos=1.000000 + 贪心续接一致。DRACO_MOE_BATCH=0 回 M1 逐 token FFN。
        self.moe_batch = os.environ.get("DRACO_MOE_BATCH", "1") == "1"
        self.wbuf = np.zeros(self.NTH * 8208, np.float32)
        self.f32p = ct.POINTER(ct.c_float)
        self.u8p = ct.POINTER(ct.c_uint8)

    def _f32view(self, cptr, n):
        return np.ctypeslib.as_array(ct.cast(cptr, self.f32p), shape=(n,))

    def _gemm(self, code, X, wbuf_ptr, n_out, n_in):
        T = X.shape[0]
        Y = np.empty((T, n_out), np.float32)
        self.G.m5_gemm(code, X.ctypes.data, wbuf_ptr, T, n_out, n_in,
                       Y.ctypes.data, self.wbuf.ctypes.data)
        return Y

    def _gdn(self, w, L, X, pos0):
        QE = self.QE
        T, H = X.shape
        attn_norm = self._f32view(w.attn_norm, H)
        Xn = rms_mean(X, attn_norm)
        QKV = self._gemm(w.qkv_code, Xn, w.qkv_buf, w.n_qkv, H)
        Zm = self._gemm(w.gate_code, Xn, w.gate_buf, w.n_vh * 128, H)
        bw = self._f32view(w.beta_wt, H * w.n_vh).reshape(H, w.n_vh)
        aw = self._f32view(w.alpha_wt, H * w.n_vh).reshape(H, w.n_vh)
        ssm_a = self._f32view(w.ssm_a, w.n_vh)
        ssm_dt = self._f32view(w.ssm_dt, w.n_vh)
        B = 1.0 / (1.0 + np.exp(-(Xn @ bw)))                                   # sigmoid
        av = Xn @ aw + ssm_dt[None, :]
        sp = np.logaddexp(0.0, av)                                             # softplus
        Gm = np.exp(sp * ssm_a[None, :])
        conv = self._f32view(w.conv, w.n_qkv * 4).reshape(w.n_qkv, 4)
        ssm_norm = self._f32view(w.ssm_norm, 128)
        tail = np.ctypeslib.as_array(ct.cast(L["tail"], self.f32p), shape=(w.n_qkv * 3,))
        S = np.ctypeslib.as_array(ct.cast(L["ssm"], self.f32p), shape=(w.n_vh * 128 * 128,))
        ON = np.empty((T, w.n_vh * 128), np.float32)
        work = np.empty(w.n_qkv, np.float32)
        self.SC.m6_gdn_scan(pf(QKV), pf(Zm), pf(B), pf(Gm), pf(conv), pf(ssm_norm),
                            tail.ctypes.data, S.ctypes.data, T, w.n_qkv, w.n_vh,
                            pf(ON), pf(work))
        attn = self._gemm(w.ssm_out_code, ON, w.ssm_out_buf, H, w.d_in)
        return attn, w.post_norm

    def _full(self, w, L, X, pos0):
        QE = self.QE
        T, H = X.shape
        nh, nkv, hd = w.nh, QE.NKV, QE.HD_A
        attn_norm = self._f32view(w.attn_norm, H)
        Xn = rms_mean(X, attn_norm)
        Q = self._gemm(w.q_code, Xn, w.q_buf, nh * 2 * hd, H)      # [T, 16×512]
        K = self._gemm(w.k_code, Xn, w.k_buf, nkv * hd, H)
        V = self._gemm(w.v_code, Xn, w.v_buf, nkv * hd, H)
        qn = self._f32view(w.q_norm, hd)
        kn = self._f32view(w.k_norm, hd)
        # q: rms + rope；门控留在后半
        Qr = Q.reshape(T, nh, 2*hd)[:, :, :hd].copy()
        gate = 1.0 / (1.0 + np.exp(-Q.reshape(T, nh, 2*hd)[:, :, hd:]))
        # ★ 全注意力的 q/k rms 是「均值式」n2/HD+eps（与 GDN 的 SUM 式不同！）
        ms = (Qr * Qr).sum(-1)
        inv = 1.0 / np.sqrt(ms / hd + EPS)
        Qr *= inv[:, :, None] * qn[None, None, :]
        rope_(Qr, pos0, QE.rope_inv, 32)
        K2 = K.reshape(T, nkv, hd).copy()
        ms = (K2 * K2).sum(-1)
        inv = 1.0 / np.sqrt(ms / hd + EPS)
        K2 *= inv[:, :, None] * kn[None, None, :]
        rope_(K2, pos0, QE.rope_inv, 32)
        Kr_np = np.ctypeslib.as_array(ct.cast(L["Kr"], self.f32p), shape=(QE.MAXT, nkv*hd))
        Vc_np = np.ctypeslib.as_array(ct.cast(L["Vc"], self.f32p), shape=(QE.MAXT, nkv*hd))
        Kr_np[pos0:pos0+T] = K2.reshape(T, nkv*hd)
        Vc_np[pos0:pos0+T] = V
        attn = softmax_causal(Qr, Kr_np, Vc_np, pos0, T, nh, nkv, hd, gate.reshape(T, nh*hd),
                              1.0 / np.sqrt(float(hd)))
        o_attn = self._gemm(w.o_code, np.ascontiguousarray(attn), w.o_buf, H, nh*hd)
        return o_attn, w.post_norm

    def _ffn_chunk(self, L, X2n):
        """M2：MoE FFN 按位置批处理 —— 路由一次算完全部位置，专家按并集分组，
        每个被选中的专家**反量化一次**服务它的全部 token（m5_gemm），共享专家整批。
        语义与 m6_granite_moe/m6_granite_shexp 同式（softmax 路由 + stable topk +
        norm_w + clamp；门控共享专家 = sigmoid(sg·x) 标量门）。"""
        QE = self.QE
        M = L["moe"]
        T, H = X2n.shape
        FF = QE.EXP_FFN
        NEXP, NUSED = QE.N_EXP, QE.N_USED
        gi = self._f32view(M["gi"], NEXP * H).reshape(NEXP, H)
        # 路由
        lg = X2n @ gi.T
        lg -= lg.max(1, keepdims=True)
        P = np.exp(lg)
        P /= P.sum(1, keepdims=True)
        order = np.argsort(-P, axis=1, kind="stable")[:, :NUSED]
        wtop = np.take_along_axis(P, order, 1)
        wtop /= np.maximum(wtop.sum(1, keepdims=True), 6.103515625e-5)
        out = np.zeros((T, H), np.float32)
        # 专家并集分组：整体下沉到 C（gather → 每专家批 GEMM → silu → 加权 scatter）
        order_c = np.ascontiguousarray(order, np.int32)
        wtop_c = np.ascontiguousarray(wtop, np.float32)
        work = np.empty(T * NUSED * (H + 3 * FF + H), np.float32)
        iwork = np.empty(T * NUSED * 2 + NEXP + 1, np.int32)
        self.MT.m6_moe_tok(self._dq_fp, self._fused_fp, self.wbuf.ctypes.data,
                           pf(X2n), pf(order_c), pf(wtop_c), M["ep"], M["ec"],
                           T, NUSED, FF, H, H, NEXP, pf(out), pf(work), pf(iwork))
        # 共享专家（整批）
        shp = np.ctypeslib.as_array(M["shp"], shape=(4,))
        shc = np.ctypeslib.as_array(M["shc"], shape=(4,))
        sg_w = self._f32view(int(shp[3]), H)
        gsc = 1.0 / (1.0 + np.exp(-(X2n @ sg_w)))
        shg = self._gemm(int(shc[0]), X2n, int(shp[0]), QE.SHEXP_FFN, H)
        shu = self._gemm(int(shc[1]), X2n, int(shp[1]), QE.SHEXP_FFN, H)
        act = np.ascontiguousarray(shg / (1.0 + np.exp(-shg)) * shu)
        shd = self._gemm(int(shc[2]), act, int(shp[2]), H, QE.SHEXP_FFN)
        return out + shd * gsc[:, None]

    def prefill_chunk(self, ids, pos0):
        """等价于对 ids 逐 token forward(ids[i], pos0+i) 后的引擎状态；返回末位 logits。"""
        QE = self.QE
        T = len(ids)
        X = np.empty((T, QE.H), np.float32)
        t_emb = QE.T["token_embd.weight"]
        for i, tid in enumerate(ids):
            X[i] = np.asarray(gguf.quants.dequantize(t_emb.data[int(tid)], t_emb.tensor_type), np.float32)
        FFO = np.empty(QE.H, np.float32)
        for L in QE.LAYERS:
            w = ct.cast(L["w"], ct.POINTER(QE.GdnW if L["kind"] == "gdn" else QE.FullW)).contents
            if L["kind"] == "gdn":
                attn, post = self._gdn(w, L, X, pos0)
            else:
                attn, post = self._full(w, L, X, pos0)
            post_arr = self._f32view(post, QE.H)
            X2 = X + attn
            X2n = rms_mean(X2, post_arr)
            if "moe" in L and self.moe_batch:
                # M2：专家并集批处理
                X = X2 + self._ffn_chunk(L, X2n)
                continue
            # MoE/dense FFN：M1 逐 token（复用引擎算子与缓冲）
            for i in range(T):
                QE._ffn(L, X2n[i], FFO)
                X2[i] += FFO
            X = X2
        # 头：只算末位（m6_head_op 内部自带 final rms）
        QE.M6E.m6_head_op(pf(np.ascontiguousarray(X[-1])), pf(QE.ONORM), QE.H, QE.EPS,
                          QE._HEAD.ctypes.data, QE._HEAD_CODE, QE.VOCAB, pf(QE.LOGITS), pf(QE.HSC))
        return QE.LOGITS

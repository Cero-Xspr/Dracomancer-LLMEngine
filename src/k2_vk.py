#!/usr/bin/env python3
"""k2_vk.py — F1c：K2 引擎的 iGPU 侧（libvkrun.so 的 ctypes 封装 + 张量驻留登记）。

职责：
  · 建 Vulkan 上下文（W 缓冲 heap1 优先/GTT 溢出、X/Y/G16 GTT 常映射）
  · 登记 IQ2_S 张量 → (wset, w_off)，一次 arm() 完成全部上传（staging/GTT 直拷）
  · 六个相位调用：fused4 / mova4 / gate_up / down / shared_gu / shared_d
所有输出Y 经 GTT 常映射直接读（fence 后保证可见）。
"""
import os
import time
import ctypes as ct
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(BASE, "vk", "libvkrun.so")
SPV = os.environ.get("K2_VK_SPV", os.path.join(BASE, "vk", "iq2s_gemv3.spv"))
G16 = os.path.join(BASE, "vk", "g16.bin")

H = 2560
NB_H = H // 256          # n_in=2560 → 10 块
NB_I = 768 // 256        # n_in=768  → 3 块（MoE down）

# X 布局（float 下标）：[0,2560)=h/h2；[4096,4096+6144)=8×gu；[8192,8960)=shared silu·u
X_FLOATS = 12288        # 4096+8×768=10240 需求取整
Y_FLOATS = 24576         # 单次调用最大输出 8×2560


class VGMat(ct.Structure):
    _fields_ = [("w_off", ct.c_uint32), ("x_off", ct.c_uint32), ("y_off", ct.c_uint32),
                ("n_out", ct.c_uint32), ("nb", ct.c_uint32), ("wset", ct.c_uint32)]


class VGInfo(ct.Structure):
    _fields_ = [("n_w", ct.c_int),
                ("w_bytes", ct.c_ulong * 8), ("w_heap", ct.c_int * 8),
                ("w_map", ct.c_void_p * 8)]


class G:
    """(wset, w_off) —— 一个 GPU 侧矩阵组基址（张量级）。"""
    __slots__ = ("wset", "w_off")

    def __init__(self, wset, w_off):
        self.wset = wset
        self.w_off = w_off


class VKCtx:
    def __lib(self):
        lib = ct.CDLL(LIB)
        lib.vg_init.restype = ct.c_int
        lib.vg_init.argtypes = [ct.POINTER(ct.c_void_p), ct.c_ulong, ct.c_char_p,
                                ct.c_ulong, ct.c_ulong, ct.c_ulong, ct.POINTER(VGInfo)]
        lib.vg_upload_grid.restype = ct.c_int
        lib.vg_upload_grid.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.vg_upload.restype = ct.c_int
        lib.vg_upload.argtypes = [ct.c_void_p, ct.c_int, ct.c_ulong, ct.c_void_p, ct.c_ulong]
        lib.vg_run.restype = ct.c_int
        lib.vg_run.argtypes = [ct.c_void_p, ct.POINTER(VGMat), ct.c_int,
                               ct.c_void_p, ct.c_ulong]
        lib.vg_xmap.restype = ct.c_void_p
        lib.vg_xmap.argtypes = [ct.c_void_p]
        lib.vg_ymap.restype = ct.c_void_p
        lib.vg_ymap.argtypes = [ct.c_void_p]
        return lib

    def __init__(self, tensors, wanted):
        """tensors: {name: gguf_fast tensor}；wanted: 有序 name 列表（仅 IQ2_S 大张量）。
        vg_init 在 arm() 里做（需要总字节数）。"""
        self.T = tensors
        self.lib = self.__lib()
        self.h = None
        self.reg = {}
        self.order = [n for n in wanted if n in tensors]

    def arm(self, upload=True):
        total = sum(self.T[n].n_bytes for n in self.order)
        info = VGInfo()
        h = ct.c_void_p()
        rc = self.lib.vg_init(ct.byref(h), total, SPV.encode(),
                              X_FLOATS * 4, Y_FLOATS * 4, 128 << 20, ct.byref(info))
        if rc != 0:
            raise RuntimeError(f"vg_init rc={rc}")
        self.h = h
        self.n_w = info.n_w
        print(f"[vk] W 缓冲 {self.n_w} 块: " +
              " ".join(f"[{i}]{info.w_bytes[i]/2**30:.2f}GB({'DL' if info.w_heap[i] else 'GTT'})"
                       for i in range(self.n_w)), flush=True)
        # 登记：first-fit（整张量不跨缓冲，小张量可回填前缓冲的尾部）
        offs = [0] * self.n_w
        for n in self.order:
            nb = self.T[n].n_bytes
            for bi in range(self.n_w):
                if offs[bi] + nb <= int(info.w_bytes[bi]):
                    self.reg[n] = G(bi, offs[bi])
                    offs[bi] += (nb + 15) & ~15
                    break
            else:
                raise RuntimeError(f"{n} 放不下")
        # G16 码表（uint32[1024]）
        g16 = np.fromfile(G16, np.uint16).astype(np.uint32)
        assert g16.size == 1024
        rc = self.lib.vg_upload_grid(self.h, g16.ctypes.data_as(ct.c_void_p))
        assert rc == 0
        # X/Y numpy 视图
        self.xv = np.frombuffer((ct.c_float * X_FLOATS).from_address(
            self.lib.vg_xmap(self.h)), np.float32)
        self.yv = np.frombuffer((ct.c_float * Y_FLOATS).from_address(
            self.lib.vg_ymap(self.h)), np.float32)
        if upload:
            self.upload_all()
        return self

    def upload_all(self):
        """逐张量上传 + 立即 madvise（page cache 峰值 ≈ staging 而非整模型，防 OOM/swap）。
        ★ GTT 权重页被换出后 GPU 每次派发要从 swap 换入（实测秒级/tok）——内存纪律是性能前提。"""
        import mmap as _mmap
        t0 = time.perf_counter()
        tot = sum(self.T[n].n_bytes for n in self.order)
        done = 0
        next_mark = 1 << 30
        n_drop = 0
        for n in self.order:
            t = self.T[n]
            src = np.frombuffer(t.data, np.uint8)
            g = self.reg[n]
            rc = self.lib.vg_upload(self.h, g.wset, g.w_off,
                                    src.ctypes.data_as(ct.c_void_p), src.nbytes)
            if rc != 0:
                raise RuntimeError(f"upload {n} rc={rc}")
            done += src.nbytes
            mm = getattr(t, "_mm", None)
            if mm is not None:
                try:
                    off_pg = t.data_offset & ~0xFFF
                    end = (t.data_offset + t.n_bytes + 0xFFF) & ~0xFFF
                    mm.madvise(_mmap.MADV_DONTNEED, off_pg, end - off_pg)
                    n_drop += t.n_bytes
                except Exception:
                    pass
            if done >= next_mark or done == tot:
                print(f"[vk] 上传 {done/2**30:.1f}/{tot/2**30:.1f} GB "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
                next_mark += 1 << 30
        print(f"[vk] 上传完成 {time.perf_counter()-t0:.1f}s，释放 cache ~{n_drop/2**30:.1f} GB",
              flush=True)

    def drop_cache(self):
        """上传完的张量从 page cache 释放（madvise DONTNEED，回收 ~10GB RAM）。
        仅对 GPU 路径不再回读 CPU 视图的前提下安全（engine 调用点已按驻留分派）。"""
        import mmap as _mmap
        n_drop = 0
        for n in self.order:
            t = self.T[n]
            mm = getattr(t, "_mm", None)
            if mm is None:
                continue
            try:
                off_pg = t.data_offset & ~0xFFF
                end = (t.data_offset + t.n_bytes + 0xFFF) & ~0xFFF
                mm.madvise(_mmap.MADV_DONTNEED, off_pg, end - off_pg)
                n_drop += t.n_bytes
            except Exception:
                pass
        print(f"[vk] madvise DONTNEED ~{n_drop/2**30:.2f} GB", flush=True)

    # ── 低层 ──
    def _run(self, mats, x, x_floats):
        arr = (VGMat * len(mats))()
        for i, (g, xo, yo, nout, nb) in enumerate(mats):
            arr[i].w_off, arr[i].x_off, arr[i].y_off = g.w_off, xo, yo
            arr[i].n_out, arr[i].nb, arr[i].wset = nout, nb, g.wset
        xp = None if x is None else x.ctypes.data_as(ct.c_void_p)
        rc = self.lib.vg_run(self.h, arr, len(mats), xp, x_floats)
        if rc != 0:
            raise RuntimeError(f"vg_run rc={rc}")

    # ── 六个相位 ──
    def fused4(self, h, qb, gb, kb, vgb):
        """q/gate/k/v_router 共享 h，入参为 ((G, n_out)×4)。返回 (q, gate_in, k, vlogits)。"""
        n_q = qb[1]; n_g = gb[1]; n_k = kb[1]; n_v = vgb[1]
        mats = [(qb[0], 0, 0, n_q, NB_H), (gb[0], 0, n_q, n_g, NB_H),
                (kb[0], 0, n_q + n_g, n_k, NB_H), (vgb[0], 0, n_q + n_g + n_k, n_v, NB_H)]
        self._run(mats, h, H)
        y = self.yv[:n_q + n_g + n_k + n_v]
        return (y[:n_q].copy(), y[n_q:n_q + n_g].copy(), y[n_q + n_g:n_q + n_g + n_k].copy(),
                y[n_q + n_g + n_k:].copy())

    def mova4(self, h, vg, sel, per, n_out):
        """MoVA 4 专家共享 h。返回 V4 [ns, n_out]。"""
        mats = [(G(vg.wset, vg.w_off + int(e) * per), 0, k * n_out, n_out, NB_H)
                for k, e in enumerate(sel)]
        self._run(mats, h, H)
        return self.yv[:len(sel) * n_out].copy().reshape(len(sel), n_out)

    def gate_up(self, h2, eg, ug, sel, per_e, per_u, n_inter):
        """MoE 8 专家 gate+up 共享 h2。返回 (G, U) 各 [ns, n_inter]。"""
        ns = len(sel)
        mats = []
        for k, e in enumerate(sel):
            mats.append((G(eg.wset, eg.w_off + int(e) * per_e), 0, 2 * k * n_inter, n_inter, NB_H))
            mats.append((G(ug.wset, ug.w_off + int(e) * per_u), 0, 2 * k * n_inter + n_inter,
                         n_inter, NB_H))
        self._run(mats, h2, H)
        y = self.yv[:ns * 2 * n_inter].reshape(ns, 2, n_inter)
        return y[:, 0, :].copy(), y[:, 1, :].copy()

    def down(self, gu, dg, sel, per, n_out, n_inter):
        """MoE 8 专家 down：输入互不相同（gu[k] 写进 X 的独立窗口）。返回 D [ns, n_out]。"""
        ns = len(sel)
        self.xv[4096:4096 + ns * n_inter] = gu.reshape(-1)
        mats = [(G(dg.wset, dg.w_off + int(e) * per), 4096 + k * n_inter, k * n_out, n_out, NB_I)
                for k, e in enumerate(sel)]
        self._run(mats, None, X_FLOATS)
        return self.yv[:ns * n_out].copy().reshape(ns, n_out)

    def shared_gu(self, h2, sgg, sug, n_inter):
        mats = [(sgg, 0, 0, n_inter, NB_H), (sug, 0, n_inter, n_inter, NB_H)]
        self._run(mats, h2, H)
        y = self.yv[:2 * n_inter].copy()
        return y[:n_inter], y[n_inter:]

    def shared_d(self, x, sdg, n_out, n_inter):
        self.xv[8192:8192 + n_inter] = x
        mats = [(sdg, 8192, 0, n_out, NB_I)]
        self._run(mats, None, X_FLOATS)
        return self.yv[:n_out].copy()

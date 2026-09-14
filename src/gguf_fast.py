#!/usr/bin/env python3
"""gguf_fast —— 只读 GGUF 解析器，接口是 gguf-py 的**兼容子集**，但快 20~30 倍。

为什么需要它（2026-09-15 实测，ZAYA1-8B-Q4_K_M 5.19GB / 1283 张量 / 262k 词表）：
    gguf.GGUFReader(path)         13.9 s    （磁盘读 0 字节 —— 纯 CPU！）
    我们一次装模型要建 **两个** reader（引擎本体 + 适配器读 tokenizer）  ⇒ 28 s / 39 s
cProfile 指认的元凶是 gguf-py 的 `_get_str`：**每个字符串元素**都走一遍
    self.data[o:o+n].view(dtype)[:count].view(newbyteorder(...))
即 4 次 numpy 对象构造（`__array_finalize__` / `view` / `hasattr` / `may_share_memory`）。
词表三件套（tokens/types/scores）合计 78 万个元素 ⇒ 830 万次 numpy 调用 = 14 秒。
GGUF 的字符串是 `u64 长度 + 裸字节`，根本不需要 numpy —— 用 mmap + struct 直接切。

实现要点（都是刻意的）：
  · 元数据用 `struct.unpack_from` 打在 **mmap 对象**上（mmap 支持缓冲协议），
    完全绕开 numpy 的标量数组封装；字符串就是 `mm[a:b]`（C 层切片，返回 bytes）。
  · 张量 `.data` 是 `np.frombuffer(mm, ...)` 的**只读视图**（零拷贝、零磁盘读），
    与 gguf-py 一样对量化类型做 `quant_shape_to_byte_shape`，所以 `quants.dequantize(t.data, ...)`
    这类既有代码原样可用。
  · `FastGGUF(path)` 按 (路径, 大小, mtime) 缓存 —— 同一个进程里第二次拿同一个文件
    是纯内存操作。这正是"装一个模型要建两个 reader"的解药：第二个几乎免费。
  · 只支持小端主机 + 小端 GGUF（即全部现实情况）；大端主机显式报错，不静默算错。

不提供：写文件、tensor 增删、大端字节序、`.contents()` 的全部重载语义（只覆盖我们用到的那几种）。
"""
from __future__ import annotations

import mmap as _mmap
import os
import struct
import sys
from collections import OrderedDict

import numpy as np

# ★ gguf-py 的路径兜底：我们自己只借它的**类型表与 dequantize**（元数据解析已经不吃它了）。
#   以前每个脚本都要靠外部 `PYTHONPATH=.../llama.cpp-b10819/gguf-py` 才能 import gguf，
#   少一个环境变量就 "No module named 'gguf'"。这里补上，让调用方少一个隐性契约。
try:
    import gguf.constants  # noqa: F401
except ImportError:                                     # pragma: no cover
    for _p in (os.environ.get("GGUF_PY_DIR"),
               "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py"):
        if _p and os.path.isdir(os.path.join(_p, "gguf")):
            sys.path.insert(0, _p)
            break

from gguf.constants import (
    GGML_MAX_DIMS,
    GGML_QUANT_SIZES,
    GGUF_DEFAULT_ALIGNMENT,
    GGMLQuantizationType,
    GGUFValueType,
)
from gguf.quants import quant_shape_to_byte_shape

__all__ = ["FastGGUF", "FastTensor", "FastField"]

# GGUFValueType → (struct 字符, numpy dtype, 字节数)。BOOL 在 GGUF 里就是 1 字节整数。
_SCALAR = {
    GGUFValueType.UINT8:   ("B", np.uint8, 1),
    GGUFValueType.INT8:    ("b", np.int8, 1),
    GGUFValueType.UINT16:  ("H", np.uint16, 2),
    GGUFValueType.INT16:   ("h", np.int16, 2),
    GGUFValueType.UINT32:  ("I", np.uint32, 4),
    GGUFValueType.INT32:   ("i", np.int32, 4),
    GGUFValueType.FLOAT32: ("f", np.float32, 4),
    GGUFValueType.BOOL:    ("B", np.uint8, 1),
    GGUFValueType.UINT64:  ("Q", np.uint64, 8),
    GGUFValueType.INT64:   ("q", np.int64, 8),
    GGUFValueType.FLOAT64: ("d", np.float64, 8),
}
# 浮点/整型张量按原类型直读；量化类型按原始字节读（与 gguf-py 的 item_type 选择一致）
_ITERM = {
    GGMLQuantizationType.F16: np.float16,
    GGMLQuantizationType.F32: np.float32,
    GGMLQuantizationType.F64: np.float64,
    GGMLQuantizationType.I8:  np.int8,
    GGMLQuantizationType.I16: np.int16,
    GGMLQuantizationType.I32: np.int32,
    GGMLQuantizationType.I64: np.int64,
}

_CACHE: dict[str, "FastGGUF"] = {}


class FastField:
    """gguf-py ReaderField 的兼容子集。

    `contents()` 支持四种形态（够我们用）：
        STRING         → str
        标量           → python 标量
        ARRAY[STRING]  → list[str]
        ARRAY[数值]    → list（扁平化，与 gguf-py 一致）
    """

    __slots__ = ("offset", "name", "value", "types", "data", "parts")

    def __init__(self, offset, name, value, types):
        self.offset = offset
        self.name = name
        self.value = value
        self.types = types
        self.data = None
        self.parts = None

    def contents(self, index_or_slice=slice(None)):
        v = self.value
        if isinstance(v, list) and types_are_string(self.types):
            return v[index_or_slice] if isinstance(index_or_slice, int) else list(v[index_or_slice])
        if isinstance(v, np.ndarray):
            return v.tolist() if isinstance(index_or_slice, slice) else \
                v.tolist()[index_or_slice]
        return v

    def __repr__(self):
        return f"FastField({self.name!r})"


def types_are_string(types):
    return bool(types) and types[-1] == GGUFValueType.STRING


class FastTensor:
    """gguf-py ReaderTensor 的兼容子集：name/tensor_type/shape/n_bytes/data_offset/data。

    ★ 差别：`.data` 是**只读视图**（量化类型按 byte-shape reshape，浮点按原类型），
      不再复制一份到匿名内存；`.tobytes()` / `bytes(t.data)` 照旧可用。
    """

    __slots__ = ("name", "tensor_type", "shape", "n_elements", "n_bytes",
                 "data_offset", "data", "_mm")

    def __init__(self, name, tensor_type, dims, data_offset, mm, data_off_abs, n_elements, n_bytes):
        self.name = name
        self.tensor_type = tensor_type
        self.shape = dims                                   # np.uint32，ne[0] 在前（同 gguf-py）
        self.n_elements = n_elements
        self.n_bytes = n_bytes
        self.data_offset = data_off_abs
        self._mm = mm
        item = _ITERM.get(tensor_type)
        np_dims = tuple(int(d) for d in reversed(dims.tolist()))
        if item is not None:
            self.data = np.frombuffer(mm, dtype=item, count=n_elements, offset=data_off_abs)
            if self.data.size == n_elements and np_dims:
                self.data = self.data.reshape(np_dims)
        else:
            self.data = np.frombuffer(mm, dtype=np.uint8, count=n_bytes, offset=data_off_abs)
            if np_dims:
                self.data = self.data.reshape(quant_shape_to_byte_shape(np_dims, tensor_type))

    def raw_bytes(self):
        """该张量的裸字节视图（零拷贝）。"""
        return memoryview(self._mm)[self.data_offset:self.data_offset + self.n_bytes]

    def __repr__(self):
        return f"FastTensor({self.name!r}, {self.tensor_type.name}, {tuple(self.shape.tolist())})"


class FastGGUF:
    """只读、零拷贝、按文件缓存。字段名/方法与 GGUFReader 对齐（子集）。"""

    def __new__(cls, path, mode="r"):
        key = os.path.abspath(os.fspath(path))
        st = os.stat(key)
        sig = (st.st_size, st.st_mtime_ns)
        hit = _CACHE.get(key)
        if hit is not None and hit._sig == sig:
            return hit
        obj = super().__new__(cls)
        obj._key, obj._sig = key, sig
        _CACHE[key] = obj
        return obj

    def __init__(self, path, mode="r"):
        if getattr(self, "_ready", False):
            return                                          # 缓存命中：不重复解析
        if sys.byteorder != "little":
            raise RuntimeError("gguf_fast 只支持小端主机（本机不是）")
        self.path = self._key
        self.fields: OrderedDict[str, FastField] = OrderedDict()
        self.tensors: list[FastTensor] = []
        self._fd = os.open(self.path, os.O_RDONLY)
        self._mm = _mmap.mmap(self._fd, 0, access=_mmap.ACCESS_READ)
        self.data = np.frombuffer(self._mm, dtype=np.uint8)   # gguf-py 风格的全文件视图
        try:
            self._parse()
        except Exception:
            self._mm.close()
            os.close(self._fd)
            _CACHE.pop(self._key, None)
            raise
        self._ready = True

    # ---------- 基础读取（全部打在 mmap 上，绕开 numpy 标量封装）----------
    def _u32(self, o):
        return struct.unpack_from("<I", self._mm, o)[0]

    def _u64(self, o):
        return struct.unpack_from("<Q", self._mm, o)[0]

    def _str(self, o):
        """(新偏移, str)。★ 返回解码后字符串：我们只需要它，避免调用方再解一遍。"""
        n = self._u64(o)
        b = self._mm[o + 8:o + 8 + n]
        return o + 8 + n, b.decode("utf-8", "replace")

    def _value(self, o, t):
        """按类型读一个值 → (新偏移, 值, 类型链)。"""
        if t == GGUFValueType.STRING:
            o, s = self._str(o)
            return o, s, [t]
        if t == GGUFValueType.ARRAY:
            it = self._u32(o)
            cnt = self._u64(o + 4)
            o += 12
            if it == GGUFValueType.STRING:
                vals = []
                ap = vals.append
                mm = self._mm
                for _ in range(cnt):
                    n = struct.unpack_from("<Q", mm, o)[0]
                    ap(mm[o + 8:o + 8 + n].decode("utf-8", "replace"))
                    o += 8 + n
                # ★ 不做类型链展开（gguf-py 会为每个元素 append 一次）—— 调用方只看 types[-1]
                return o, vals, [t, GGUFValueType.STRING]
            fmt, dt, sz = _SCALAR[GGUFValueType(it)]
            arr = np.frombuffer(self._mm, dtype=dt, count=cnt, offset=o)
            return o + cnt * sz, arr, [t, GGUFValueType(it)]
        fmt, dt, sz = _SCALAR[t]
        return o + sz, struct.unpack_from("<" + fmt, self._mm, o)[0], [t]

    def _parse(self):
        if self._mm[:4] != b"GGUF":
            raise ValueError(f"不是 GGUF 文件：{self.path}")
        ver = self._u32(4)
        if ver not in (2, 3):
            raise ValueError(f"不支持的 GGUF 版本 {ver}（gguf_fast 支持 2/3）")
        self.version = ver
        o = 8
        o, tensor_count, _ = self._value(o, GGUFValueType.UINT64)
        o, kv_count, _ = self._value(o, GGUFValueType.UINT64)
        self.tensor_count, self.kv_count = int(tensor_count), int(kv_count)

        # —— KV 段 ——
        for _ in range(self.kv_count):
            start = o
            o, k, _ = self._value(o, GGUFValueType.STRING)
            o, rt, _ = self._value(o, GGUFValueType.UINT32)
            o, v, chain = self._value(o, GGUFValueType(int(rt)))
            self.fields[k] = FastField(start, k, v, chain)

        # —— 张量信息段 ——
        infos = []
        for _ in range(self.tensor_count):
            o, name, _ = self._value(o, GGUFValueType.STRING)
            o, nd, _ = self._value(o, GGUFValueType.UINT32)
            nd = int(nd)
            if nd > GGML_MAX_DIMS:
                raise ValueError(f"张量 {name} 维度数 {nd} 超过 GGML_MAX_DIMS")
            o, dims, _ = self._value_array(o, GGUFValueType.UINT64, nd)
            o, raw_dtype, _ = self._value(o, GGUFValueType.UINT32)
            o, t_off, _ = self._value(o, GGUFValueType.UINT64)
            infos.append((name, dims, int(raw_dtype), int(t_off)))

        # —— 张量数据段起始（按 general.alignment 对齐）——
        al = self.fields.get("general.alignment")
        self.alignment = int(al.value) if al is not None else GGUF_DEFAULT_ALIGNMENT
        if self.alignment == 0 or (self.alignment & (self.alignment - 1)):
            raise ValueError(f"非法对齐值 {self.alignment}")
        pad = o % self.alignment
        if pad:
            o += self.alignment - pad
        self.data_offset = o

        seen = set()
        for name, dims, raw_dtype, t_off in infos:
            if name in seen:
                raise ValueError(f"张量重名：{name}")
            seen.add(name)
            gt = GGMLQuantizationType(raw_dtype)
            n_elems = 1
            for d in dims.tolist():
                n_elems *= int(d)
            bs, ts = GGML_QUANT_SIZES[gt]
            n_bytes = n_elems * ts // bs
            abs_off = o + t_off
            if abs_off + n_bytes > self._mm.size():
                raise ValueError(f"张量 {name} 数据越界（{n_bytes} 字节 @ {abs_off}）")
            self.tensors.append(FastTensor(name, gt, dims, t_off, self._mm, abs_off,
                                           n_elements=n_elems, n_bytes=n_bytes))
        self._by_name = {t.name: t for t in self.tensors}

    def _value_array(self, o, t, n):
        """连续读 n 个标量 → np 数组（一次 frombuffer）。"""
        if t == GGUFValueType.STRING or t == GGUFValueType.ARRAY:
            raise ValueError("_value_array 只处理连续标量")
        fmt, dt, sz = _SCALAR[t]
        arr = np.frombuffer(self._mm, dtype=dt, count=n, offset=o)
        return o + n * sz, arr, [t]

    # ---------- 对外 ----------
    def get_field(self, key):
        return self.fields.get(key)

    def get_tensor(self, idx):
        return self.tensors[idx]

    def tensor(self, name):
        return self._by_name.get(name)

    def tensor_bytes(self, name):
        """张量裸字节（bytes 复制一份；对齐敏感的内核拷贝请用 tensor_bytes_into）。"""
        return bytes(self.tensor(name).raw_bytes())

    def close(self):
        _CACHE.pop(self._key, None)
        self._mm.close()
        os.close(self._fd)


def load(path):
    """便捷入口，同名张量直接给 dict。"""
    r = FastGGUF(path)
    return r, {t.name: t for t in r.tensors}

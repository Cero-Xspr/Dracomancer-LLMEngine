#!/usr/bin/env python3
"""adapt_schema —— 模型适配档案（models.d/*.json）的 **唯一** schema 实现。

为什么单独一个文件：这份校验被两边用 —— draco 启动时读注册表、以及提交校验器
（release/validate_submission.py）。两处各写一份必然漂移，所以只留这一份。
本文件**只依赖标准库**，可以原样拷进公开仓库独立运行。

档案 = 纯数据（白名单字段，永不执行代码）。白名单之外一律拒绝 —— 这是安全边界：
社区提交的东西只会被当成**参数**读，不会被当成**程序**跑。

schema 的人话说明见 AGENT_ADAPTER.md §3。
"""
from __future__ import annotations

import json
import os
import struct
import sys

PROFILE_SCHEMA_VERSION = 1

# 已知后端名（backend_hint 只能取其中之一）。与 draco.BACKENDS 的键保持一致；
# 这里写死是为了让本模块不依赖 draco（拿一份就能独立校验）。
KNOWN_BACKENDS = ("cpu", "igpu", "local", "npu")

_TOP_KEYS = ("schema_version", "match", "requires_local_build", "launch",
             "sampling", "backend_hint", "notes", "source")
_LAUNCH_KEYS = ("extra_args", "ngl")
_SAMPLING_KEYS = ("temp", "repeat_penalty", "max_tokens_default", "think_default")

# 内置兜底档案：models.d/ 整个丢失时也不至于让已知的坑复现
# （zaya 的批量 prefill 会被 Q8_K 量化台阶放大，见 STAGE1_NPU.md 里程碑 18）。
BUILTIN_PROFILES = [
    {
        "schema_version": 1,
        "match": {"arch": "zaya"},
        "requires_local_build": True,
        "launch": {"extra_args": ["-ub", "1"]},
        "backend_hint": "local",
        "notes": "内置兜底档案（models.d/zaya.json 丢失时生效）",
        "_file": "(内置)",
    },
]


def validate_profile(p, origin="<submission>"):
    """校验档案结构。不合法就 raise ValueError（调用方带上文件名报错）。"""
    if not isinstance(p, dict):
        raise ValueError("顶层必须是对象")
    if p.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须是 {PROFILE_SCHEMA_VERSION}")
    m = p.get("match")
    if not isinstance(m, dict) or not m.get("arch"):
        raise ValueError("match.arch 必填（GGUF 的 general.architecture）")
    for k in m:
        if k not in ("arch", "name_contains", "max_size_gb"):
            raise ValueError(f"match.'{k}' 不在白名单（允许：arch/name_contains/max_size_gb）")
    for k in p:
        if k not in _TOP_KEYS:
            raise ValueError(f"未知字段 '{k}'（白名单外一律拒绝）")
    if "requires_local_build" in p and not isinstance(p["requires_local_build"], bool):
        raise ValueError("requires_local_build 必须是布尔")
    if "launch" in p:
        l = p["launch"]
        if not isinstance(l, dict):
            raise ValueError("launch 必须是对象")
        for k in l:
            if k not in _LAUNCH_KEYS:
                raise ValueError(f"launch.'{k}' 不在白名单（允许：extra_args/ngl）")
        if "extra_args" in l:
            ea = l["extra_args"]
            if not isinstance(ea, list) or not all(isinstance(x, str) for x in ea):
                raise ValueError("launch.extra_args 必须是字符串数组")
    if "sampling" in p:
        s = p["sampling"]
        if not isinstance(s, dict):
            raise ValueError("sampling 必须是对象")
        for k in s:
            if k not in _SAMPLING_KEYS:
                raise ValueError(f"sampling.'{k}' 不在白名单（允许：{', '.join(_SAMPLING_KEYS)}）")
        for k in ("temp", "repeat_penalty", "max_tokens_default"):
            if k in s and not isinstance(s[k], (int, float)):
                raise ValueError(f"sampling.{k} 必须是数字")
        if "think_default" in s and not isinstance(s["think_default"], bool):
            raise ValueError("sampling.think_default 必须是布尔")
    if "backend_hint" in p and p["backend_hint"] not in KNOWN_BACKENDS:
        raise ValueError(f"backend_hint '{p['backend_hint']}' 不是已知后端"
                         f"（{'/'.join(KNOWN_BACKENDS)}）")
    if "notes" in p and not isinstance(p["notes"], str):
        raise ValueError("notes 必须是字符串")
    if "source" in p and not isinstance(p["source"], dict):
        raise ValueError("source 必须是对象")


def load_profiles(dirpath):
    """读 <dirpath>/*.json → (profiles, builtin_used)。

    坏文件**出声**跳过（绝不静默 —— 静默退回是本仓库踩过的坑）；
    多份档案命中同一模型由 profile_for 出声取第一。"""
    profs = []
    if not os.path.isdir(dirpath):
        print(f"[adapt] 警告：注册表目录 {dirpath} 不存在，"
              f"使用 {len(BUILTIN_PROFILES)} 条内置兜底档案", file=sys.stderr)
        return list(BUILTIN_PROFILES), True
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(dirpath, fn)
        try:
            with open(path, encoding="utf-8") as f:
                p = json.load(f)
            validate_profile(p, path)
            p["_file"] = fn
            profs.append(p)
        except Exception as e:
            print(f"[adapt] 错误：跳过坏档案 {fn}：{e}", file=sys.stderr)
    if not profs:
        print(f"[adapt] 警告：{dirpath} 里没有可用档案，"
              f"使用 {len(BUILTIN_PROFILES)} 条内置兜底档案", file=sys.stderr)
        return list(BUILTIN_PROFILES), True
    return profs, False


def profile_for(arch, name, size_bytes, profiles):
    """按 match 规则找第一条命中的档案；多条命中出声（第一个 wins）。"""
    hits = []
    for p in profiles:
        m = p.get("match") or {}
        if m.get("arch") != arch:
            continue
        sub = m.get("name_contains")
        if sub and sub.lower() not in (name or "").lower():
            continue
        mx = m.get("max_size_gb")
        if mx is not None and size_bytes > mx * 1e9:
            continue
        hits.append(p)
    if not hits:
        return None
    if len(hits) > 1:
        names = ", ".join(h.get("_file", "?") for h in hits)
        print(f"[adapt] 警告：{name} 命中 {len(hits)} 份档案（{names}），"
              f"用第一份（{hits[0].get('_file', '内置')}）", file=sys.stderr)
    return hits[0]


# ─────────────────────────── 极简 GGUF 头读取 ───────────────────────────
# 只为拿 general.architecture / general.name，避免依赖 gguf 包与 PYTHONPATH。

_GGUF_T = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _rd_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _skip_val(f, t):
    if t == 8:                       # string
        return _rd_str(f)
    if t == 9:                       # array
        et, cnt = struct.unpack("<IQ", f.read(12))
        for _ in range(cnt):
            _skip_val(f, et)
        return None
    if t == 7:                       # bool
        return bool(struct.unpack("<B", f.read(1))[0])
    sz = _GGUF_T.get(t)
    if sz is None:
        raise ValueError(f"未知 GGUF 元数据类型 {t}")
    return struct.unpack("<" + {1: "B", 2: "H", 4: "I", 8: "Q"}[sz], f.read(sz))[0]


def gguf_meta(path, want=("general.architecture", "general.name")):
    """读 GGUF 头里的若干 KV（读到就早停；这些键都在最前面）"""
    out, found = {}, set()
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return out
            struct.unpack("<I", f.read(4))          # version
            f.read(8); f.read(8)                    # n_tensors, n_kv
            while True:
                key = _rd_str(f)
                (t,) = struct.unpack("<I", f.read(4))
                val = _skip_val(f, t)
                if key in want:
                    out[key] = val
                    found.add(key)
                    if found >= set(want):
                        break
    except Exception:
        pass
    return out

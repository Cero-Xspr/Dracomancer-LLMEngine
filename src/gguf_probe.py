#!/usr/bin/env python3
"""gguf_probe —— 加载**之前**判断一个 GGUF 能不能被这个 llama.cpp 构建吃掉。

动机（2026-09-13 冒烟实测）：BitNet b1.58 的官方包加载失败，原因是它的权重类型
`GGML_TYPE_IQ4_NL_4_4` **已被当前 llama.cpp 移除**（ggml.h 里枚举被注释、ggml.c 里
type_name 直接写着 "REMOVED, use IQ4_NL with runtime repacking"、blck_size=0）。
这类失败**完全可以在读 GGUF 头时静态判定**，不需要下载完、不需要试跑。

三个静态检查（按"能不能加载"的因果强度排序）：
  ① **类型被移除**：ggml 的类型表里 blck_size==0 或名字含 "REMOVED" ⇒ 必然加载失败。
     这是唯一一类"必死"的静态判据，BitNet 就死在这里。
  ② **架构未实现**：GGUF 的 general.architecture 不在 llama-arch.cpp 的表里 ⇒ 必然失败。
  ③ 后端对某类型的支持（如 Vulkan 没有 TQ1_0）：**不是"必死"** —— llama.cpp 的调度器会把
     不支持的算子落到 CPU 上跑（混合执行），所以只影响速度，不阻止加载。列为提示。

★ 诚实边界：静态检查**看不到**数值/语义问题。例如 falcon-h1 在 Vulkan 上"同一 prompt
  结果依赖请求顺序"是运行时行为；ZAYA 那 7 个"形状对、语义错"的 bug 更是只有对账才抓得到。
  所以本工具的结论只有三档：必死 / 可能能跑 / 需要实测。
"""
from __future__ import annotations

import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_schema import _rd_str, _skip_val      # GGUF 头解析（同一份实现）

# llama.cpp 源码树里三个"事实来源"。★ 顺序有意：先源码树（有我们自写架构），后官方目录。
# ★ 路径走环境变量（与 draco.py 同一套约定）：公开后别人的目录不同，硬编码等于"只有我的机器能跑"。
_DH = os.environ.get("DRACO_HOME", "/media/xiao_/OverSys1")
_LOCAL = os.environ.get("DRACO_LOCAL_BASE", f"{_DH}/npu-direct/llama.cpp-b10819")
_OFFICIAL = os.environ.get("DRACO_LLAMA_ROOT", f"{_DH}/llamacpp-b10819")
GGML_C_SOURCES = (f"{_LOCAL}/ggml/src/ggml.c", f"{_OFFICIAL}/ggml/src/ggml.c")
ARCH_SOURCES = (f"{_LOCAL}/src/llama-arch.cpp", f"{_OFFICIAL}/src/llama-arch.cpp")
VULKAN_SOURCES = (f"{_LOCAL}/ggml/src/ggml-vulkan/ggml-vulkan.cpp",)


def _first_existing(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────── ggml 类型表 ───────────────────────────

def _strip_comments(txt):
    """去掉 `//` 行注释 —— ★ 必须做：被移除的枚举在 ggml.h 里是**注释掉的**
    （`// GGML_TYPE_IQ4_NL_4_4 = 36,`），直接正则会把注释里的名字也抓进来。"""
    return "\n".join(ln.split("//")[0] for ln in txt.splitlines())


def _enum_type_ids(hdr=None):
    """ggml.h 的 `GGML_TYPE_X = N,` → {id: 'X'}（活着的枚举；注释行不算）。"""
    hdr = hdr or _first_existing((h.replace("ggml/src/ggml.c", "ggml/include/ggml.h")
                                  for h in GGML_C_SOURCES))
    hdr = hdr or os.environ.get("DRACO_GGML_H")
    if not hdr:
        return {}
    txt = _strip_comments(open(hdr, encoding="utf-8", errors="replace").read())
    return {int(n): sym for sym, n in
            re.findall(r"GGML_TYPE_([A-Z0-9_]+)\s*=\s*(\d+)", txt)}


def ggml_type_table(src=None):
    """→ {id: (name, blck_size)}，**以 ggml.h 的活枚举为准**，名字优先取 ggml.c 的 type_name。

    ★ 两个坑（我都踩了）：
      1. ggml.c 里活类型用**符号**下标 `[GGML_TYPE_Q4_K] = {`，只有被移除的才用数字下标
         `[36] = {` ⇒ 只认数字下标会得到一张"全是已移除"的假表，把每个正常模型都判成必死。
      2. ggml.h 里被移除的枚举是**注释掉**的 ⇒ 匹配前必须先去注释。
    """
    hdr_cands = [c.replace("ggml/src/ggml.c", "ggml/include/ggml.h") for c in GGML_C_SOURCES]
    ids = _enum_type_ids(_first_existing(hdr_cands))
    src = src or _first_existing(GGML_C_SOURCES)
    by_num, by_sym = {}, {}
    if src:
        txt = open(src, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\[(\d+|GGML_TYPE_[A-Z0-9_]+)\]\s*=\s*\{", txt):
            key, body = m.group(1), txt[m.end():m.end() + 800]
            nm = re.search(r'\.type_name\s*=\s*"([^"]*)"', body)
            blk = re.search(r"\.blck_size\s*=\s*([A-Za-z0-9_]+)", body)
            if not nm:
                continue
            bl = blk.group(1) if blk else "?"
            bl = 256 if bl == "QK_K" else (int(bl) if bl.isdigit() else -1)
            rec = (nm.group(1), bl)
            (by_num if key.isdigit() else by_sym)[int(key) if key.isdigit() else
                                                  key[len("GGML_TYPE_"):]] = rec
    table = {}
    for tid, sym in ids.items():
        # 名字：ggml.c 的 type_name 优先（可能写着 REMOVED），否则用枚举名（去掉 GGML_TYPE_ 前缀）
        name, blk = by_num.get(tid) or by_sym.get(sym) or (sym, -1)
        table[tid] = (name, blk)
    # ggml.c 里有、但 ggml.h 枚举里没有的（= 已移除，只剩数字下标）
    for tid, rec in by_num.items():
        table.setdefault(tid, rec)
    return table


def dead_types(src=None):
    """被移除/不可用的类型 {id: 名字}（blck_size==0 或名字含 REMOVED）。"""
    return {i: n for i, (n, b) in ggml_type_table(src).items()
            if b == 0 or "REMOVED" in n}


def arch_table(srcs=ARCH_SOURCES):
    """llama-arch.cpp 支持的名字集合 + 金丝雀（含 '-'/'.' 的名字）。

    ★ 金丝雀的理由：我第一版正则 `[a-z0-9_]+` 不允许 `-`/`.`，把 falcon-h1 / gpt-oss /
    kimi-linear 这类名字截断成前缀，**27 个架构被误判成"不支持"**。所以这里检查
    "解析结果里必须存在含 -/. 的名字"，否则出声。
    """
    src = _first_existing(srcs)
    if not src:
        return set(), []
    txt = open(src, encoding="utf-8", errors="replace").read()
    names = set(re.findall(r'LLM_ARCH_[A-Z0-9_]+\s*,\s*"([a-z0-9_.\-]+)"', txt))
    canary = sorted(n for n in names if "-" in n or "." in n)
    if not canary:
        print(f"[probe] 警告：{src} 解析出 {len(names)} 个架构但**没有含 -/. 的名字** "
              f"（真实表里必然有 falcon-h1/gpt-oss 等）⇒ 正则可能又漏字符", file=sys.stderr)
    return names, canary


def vulkan_type_ids(src=None):
    """Vulkan 后端显式实现的类型 id 集合（只影响速度：不支持的算子会落回 CPU）。"""
    src = src or _first_existing(VULKAN_SOURCES)
    if not src:
        return set()
    txt = open(src, encoding="utf-8", errors="replace").read()
    ids = set()
    # ★ 归一成大写：ggml.c 的 type_name 是小写（"tq2_0"），ggml-vulkan.cpp 用的是符号名
    #   （GGML_TYPE_TQ2_0 → "TQ2_0"）。不归一就永远匹配不上，警告会**静默为空**。
    for m in re.finditer(r"GGML_TYPE_([A-Z0-9_]+)", txt):
        ids.add(m.group(1).upper())
    return ids


# ─────────────────────────── GGUF 头里的张量类型 ───────────────────────────

def tensor_types(path, limit_kv=None):
    """读 GGUF 的 (架构, 名字, 张量类型直方图)。只读头部，不需要完整文件。

    返回 (meta dict, {type_id: count}, {type_id: 示例张量名})。
    """
    meta, hist, sample = {}, {}, {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("不是 GGUF 文件")
        struct.unpack("<I", f.read(4))              # version
        nt, nkv = struct.unpack("<QQ", f.read(16))
        for _ in range(nkv):
            k = _rd_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            v = _skip_val(f, t)
            if k in ("general.architecture", "general.name"):
                meta[k] = v
        for _ in range(nt):
            name = _rd_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            f.read(8 * nd)                           # dims
            (tt,) = struct.unpack("<I", f.read(4))
            f.read(8)                                # offset
            hist[tt] = hist.get(tt, 0) + 1
            sample.setdefault(tt, name)
    return meta, hist, sample


def classify(path):
    """静态判定一个 GGUF。返回 dict（verdict 是 'dead' / 'arch_unsupported' / 'maybe'）。"""
    out = {"path": path, "verdict": "maybe", "reasons": [], "types": {}, "warn": []}
    meta, hist, sample = tensor_types(path)
    arch = meta.get("general.architecture", "?")
    name = meta.get("general.name", os.path.basename(path))
    dead = dead_types()
    tt = ggml_type_table()
    archs, _canary = arch_table()
    vk = vulkan_type_ids()

    out.update(arch=arch, name=name, size=os.path.getsize(path))
    for tid, cnt in sorted(hist.items(), key=lambda kv: -kv[1]):
        tname, blck = tt.get(tid, (f"<id {tid} 不在本引擎的类型表里>", -1))
        out["types"][tname] = {"count": cnt, "sample": sample.get(tid), "id": tid,
                               "vulkan": tname.upper() in vk}
    for tid in hist:
        if tid in dead:
            out["verdict"] = "dead"
            out["reasons"].append(
                f"权重用了已被本引擎移除的类型 {dead[tid]!r}（id={tid}，例：{sample.get(tid)}）"
                f" ⇒ 加载必失败。ggml.c 的原文就写着原因。")
        elif tid not in tt:
            out["verdict"] = "dead"
            out["reasons"].append(f"权重类型 id={tid} 不在本引擎的类型表里（例：{sample.get(tid)}）"
                                  f" ⇒ 加载必失败。")
    if out["verdict"] != "dead" and arch not in archs:
        out["verdict"] = "arch_unsupported"
        out["reasons"].append(f"架构 {arch!r} 不在本引擎的 llama-arch.cpp 表里（{len(archs)} 个）"
                              f" ⇒ 必然报 unknown architecture。")
    for tname, info in out["types"].items():
        # 已被移除的类型不再给"落回 CPU"的提示（它压根加载不了，说了误导）
        if info["id"] in dead:
            continue
        if not info["vulkan"] and info["id"] in tt:
            out["warn"].append(f"{tname}（{info['count']} 个张量）在 Vulkan 后端没有显式实现"
                               f" ⇒ 上 iGPU 时这些算子会落回 CPU（只影响速度，不是错误）")
    return out


def format_report(r):
    v = {"dead": "❌ 必死", "arch_unsupported": "❌ 架构未实现", "maybe": "✅ 可能可以（需实测）"}[r["verdict"]]
    L = [f"{v}  {r['name']}  ({r['arch']}, {r['size']/1e9:.2f} GB)"]
    for x in r["reasons"]:
        L.append(f"     · {x}")
    for x in r["warn"]:
        L.append(f"     ⚠ {x}")
    types = ", ".join(f"{k}×{v2['count']}" for k, v2 in r["types"].items())
    L.append(f"     类型：{types}")
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法：python3 gguf_probe.py <model.gguf> [...]")
        print("\n本机引擎事实：")
        tt, dead = ggml_type_table(), dead_types()
        archs, canary = arch_table()
        print(f"  ggml 类型表 {len(tt)} 项，其中已移除 {len(dead)} 项：{list(dead.values())}")
        print(f"  llama-arch 架构 {len(archs)} 个（含 -/. 的 {len(canary)} 个，金丝雀通过）")
        sys.exit(0)
    for p in sys.argv[1:]:
        print(format_report(classify(p)))
        print()

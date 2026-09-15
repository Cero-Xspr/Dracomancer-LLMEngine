#!/usr/bin/env python3
"""draco — Dracomancer 的简易启动器 / 聊天 CLI（Ollama 风格）

为什么需要它：直接照 README 拼命令行很难搞 —— 模型路径、每条后端各一套 flag
（`-ngl`/`-t`）、上下文、温度、以及一堆坑（`llama-cli` 不加 `-st` 会假装死循环、
NPU 侧要 `--l` 对齐上下文……）。这个工具把这些都固化成选项。

设计
----
引擎用 **`llama-server`**（llama.cpp 自带，带 **Web UI**），CLI 只负责：
选模型 / 选后端 / 设参数 / 交互式聊天（流式）。好处：
  · 聊天模板、流式、多轮历史都由 server 处理，不用我们碰；
  · 顺带就有了 Web 界面（`draco serve` 打开 http://127.0.0.1:PORT ）；
  · 绕开了 `llama-cli` 的交互模式/`--server-base` 那些坑。
只用 Python 标准库（urllib），不引入依赖。

后端
----
  cpu   → llamacpp-b10819/cpu/     全部层在 CPU（`-ngl 0`）
  igpu  → llamacpp-b10819/vulkan/  Radeon 890M，全部层上 GPU（`-ngl 99`）
  npu   → FastFlowLM（官方 NPU 运行时，独立于 llama.cpp）。2026-09-13 起可用于
          聊天/serve：`-b npu -m <tag子串>`，模型走 FLM 自己的 tag 命名空间
          （llama3.2:1b、gpt-oss:20b…，用 `flm list` 看 ✅）。
          我们自研的 mlir-air fused_decode 内核仍是延迟基线（合成权重），未接进来。

用法
----
  python3 draco.py list                    # 看有哪些模型、能否跑、实测性能
  python3 draco.py chat                    # 交互式选择后聊天
  python3 draco.py chat -m ling -b igpu    # 直接指定（名字支持前缀匹配）
  python3 draco.py chat -m qwen -b igpu -t 8 -c 4096 --temp 0.6 -sys "你是助手"
  python3 draco.py serve -m ling -b igpu   # 起服务，打印 Web UI 地址
  python3 draco.py perf                    # 本机实测性能表（tok/s 与能耗）
"""
from __future__ import annotations

import argparse
import json

# ★★ 必须导入 readline —— 它同时修两个看着不相干的问题：
#   (a) **方向键/Home/End 变成 `^[[D` 这种转义序列**：没有 readline 时用的是内核的
#       canonical 行规则，它**不认识**光标移动，转义序列被当成普通字符回显。
#   (b) **中文等宽字符删不掉**：内核行规则按**字节**退格，而 CJK 是 3 字节、显示占 2 列，
#       于是按一次退格只删掉 1 个字节、终端却擦掉 1 列 ⇒ 看起来"要删两次"、
#       而且原地残留半个字符（隐藏字节）—— 但送给模型的是完整字节流，
#       所以模型读到的其实是对的（和你观察到的一致）。
#   readline 走的是 locale 的宽字符语义（本机 zh_CN.UTF-8），两者一起解决。
import readline  # noqa: F401  （副作用导入：启用行编辑/历史/宽字符）
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# 适配档案的 schema/校验只有一份实现（draco 读注册表 + release 的提交校验器共用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_schema import (  # noqa: E402
    PROFILE_SCHEMA_VERSION, validate_profile, load_profiles, profile_for, gguf_meta,
    engine_view,
)

# ─────────────────────────── 路径与后端定义 ───────────────────────────

# 路径都可以用环境变量覆盖 —— 公开出去之后别人的目录结构必然不同，硬编码绝对路径
# 等于把"只有我的机器能跑"写进代码里。（默认值仍是本机的，本机行为不变。）
#   DRACO_GGUF_DIRS  冒号分隔的多个模型目录
#   DRACO_LLAMA_ROOT 官方 llama.cpp 发布包目录（cpu/ vulkan/ 等子目录）
#   DRACO_LOCAL_BASE 我们自研构建/源码树所在的 llama.cpp 目录（含 build-vk、build-dbg）
#   DRACO_FLM_DIR    FastFlowLM 便携版目录（npu 后端用）
_DEFAULT_HOME = "/media/xiao_/OverSys1"
GGUF_DIRS = [d for d in os.environ.get(
    "DRACO_GGUF_DIRS", f"{_DEFAULT_HOME}/gguf").split(os.pathsep) if d]
LLAMA_ROOT = os.environ.get("DRACO_LLAMA_ROOT", f"{_DEFAULT_HOME}/llamacpp-b10819")

_LOCAL_BASE = os.environ.get("DRACO_LOCAL_BASE",
                             f"{_DEFAULT_HOME}/npu-direct/llama.cpp-b10819")

# FLM 便携版位置（wrapper 脚本 + 捆绑的 NPU runtime）。★ 模型缓存在 ~/.config/flm/models，
#   大模型可以用符号链接指到大盘（本机 gpt-oss:20b 就是这样链到 flm-models/ 的）。
_FLM_DIR = os.environ.get("DRACO_FLM_DIR", f"{_DEFAULT_HOME}/npu-direct/fflm")


def local_build_dir():
    """挑本地构建目录：优先带 Vulkan 的 build-vk（zaya 能上 iGPU），退而用 build-dbg。"""
    for d in ("build-vk/bin", "build-dbg/bin"):
        p = os.path.join(_LOCAL_BASE, d)
        if os.path.exists(os.path.join(p, "llama-server")):
            return p
    return os.path.join(_LOCAL_BASE, "build-dbg/bin")


def local_has_vulkan(d):
    return os.path.exists(os.path.join(d, "libggml-vulkan.so"))


# ─────────────────── 模型适配注册表（models.d/*.json）───────────────────
# 社区适配机制的本地一半：每个 JSON 文件 = 一份"模型档案"（纯数据、白名单字段、
# 永不执行代码），按 GGUF 的 general.architecture（+ 可选名字子串）匹配。
#
# ★ schema 与校验逻辑**只在 adapt_schema.py 里实现一份** —— draco 启动时读注册表、
#   release/validate_submission.py 校验社区提交，两边共用同一份代码。
#   分开写两份必然漂移（我在 AGENT_ADAPTER.md 里就是这么警告的）。
#   加载/匹配的"出声"行为（坏文件绝不静默跳过）也在那边。
def _find_models_d():
    """注册表目录的候选搜索（★ 公开仓库的布局是 src/draco.py + 根目录 models.d，
    写死"同目录"会让它读不到档案并**退回内置兜底** —— 实测就这么静默降级过）。"""
    cands = [os.environ.get("DRACO_MODELS_D"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.d"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models.d")]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[1]          # 都没有时返回同目录（load_profiles 会出声警告）


_MODELS_D = _find_models_d()

# llama.cpp 支持的架构 —— ★ **从源码里读**，不手写。
# 我第一版手写了一份名单，当场就把 qwen35 漏判成"不支持"（它其实在 llama-arch.cpp:41）。
# 名单这种东西必须能自证，否则就是下一个坑。找不到源码时退回一份保守的内置表。
_ARCH_TABLE_FALLBACK = {
    "llama", "qwen2", "qwen3", "qwen35", "qwen2moe", "qwen3moe", "qwen35moe",
    "granitehybrid", "bailingmoe3", "gemma3", "phi3", "lfm2", "olmo2",
    "glm4moe", "nemotron_h_moe", "mistral", "gemma2",
}


BACKENDS = {
    "cpu": dict(
        dir=f"{LLAMA_ROOT}/cpu",
        ngl="0",
        desc="CPU（全部层，多线程）",
        threads_default=8,
    ),
    "igpu": dict(
        dir=f"{LLAMA_ROOT}/vulkan",
        ngl="99",
        desc="iGPU Radeon 890M（Vulkan，全部层上 GPU）",
        threads_default=8,
    ),
    # ★★ 我们**自己的构建**（源码树 llama.cpp-b10819 的 build-dbg）。
    #   `cpu`/`igpu` 指向的是官方预编译二进制，里面**没有**我们自写的架构实现
    #   （zaya 等）；而 SUPPORTED_ARCHS 是从**源码树**里读的 —— 所以列表会显示
    #   "cpu/igpu 可跑"，实际却会 'unknown architecture' 退出。这两者必须区分开。
    "local": dict(
        dir=local_build_dir(),
        ngl="0",                       # 运行时按架构/是否有 Vulkan 决定（见 Server）
        desc="本地构建（含自写架构：zaya 等；CPU + Vulkan 双后端）",
        threads_default=8,
    ),
    # ★★ NPU 走的是 **官方 FastFlowLM（FLM）**，不是 llama-server —— 启动命令、
    #   就绪探测、API 细节都和 llama.cpp 不同（见 Server / stream_chat 里的 npu 分支）。
    #   FLM 的模型是自己的 tag 命名空间（llama3.2:1b / gpt-oss:20b / qwen3:8b …），
    #   与 GGUF 发现无关，用 `flm_models()` 列出、`-m <tag 子串>` 选择。
    "npu": dict(
        dir=_FLM_DIR,
        ngl="0",
        desc="NPU（FastFlowLM serve，OpenAI 兼容）",
        threads_default=8,             # NPU 上线程数无意义，仅为接口兼容
    ),
    # ★★ 自研引擎（m5/m6 内核）的桥：draco_engine_server.py 把引擎包成 OpenAI 兼容服务。
    #    与 llama.cpp 完全独立 —— 这是 Dracomancer 本体的直接出口（schema v2 的
    #    engines.dracomancer 段终于有消费者了）。
    "dengine": dict(
        dir=os.path.dirname(os.path.abspath(__file__)),
        ngl="0",
        desc="自研引擎（m6 内核桥，OpenAI 兼容）",
        threads_default=8,             # ★ 实测：4~8 等价（135M 217~227 / ZAYA 25~27 t/s）；**全核(20) 是灾难**（135M 掉到 204、ZAYA 掉到 23）——每算子 OpenMP 同步开销
    ),
}


def flm_models():
    """`flm list` → {tag: downloaded}。输出行形如 `  - llama3.2:1b ✅` / `  - qwen3:8b ⏬`。
    只在 npu 后端被用到时才调用（不想让 list/perf 也去拉 NPU 运行时）。"""
    import re
    exe = os.path.join(_FLM_DIR, "flm")
    if not os.path.exists(exe):
        return {}
    try:
        r = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=120)
    except Exception:
        return {}
    out = {}
    for ln in (r.stdout or "").splitlines():
        m = re.match(r"\s*-\s+(\S+)\s+(\S+)", ln)
        if m:
            out[m.group(1)] = (m.group(2) == "✅")
    return out


class FlmModel:
    """npu 后端的模型对象：和 Model 鸭子类型兼容（Server/loop_chat 只用到这几个字段）。"""

    def __init__(self, tag):
        self.path = tag                # FLM 的"路径"就是它的 tag
        self.tag = tag
        self.name = f"{tag}（FLM·NPU）"
        self.short = tag
        self.arch = "flm"
        self.supported = True
        self.size = 0                  # 大小从 flm list 拿不到，显示里按 0 处理

    def backends(self):
        return ["npu"]

    def line(self):
        return f"{self.short[:34]:34s} {'（FLM）':>9s}  {'npu':14s}  npu"


def pick_flm(sub):
    """按子串选一个**已下载**的 FLM 模型；无匹配时列出已下载的。"""
    ms = flm_models()
    got = sorted(t for t, ok in ms.items() if ok)
    if not got:
        raise SystemExit(
            f"NPU（FLM）本地没有任何已下载模型。用 `{_FLM_DIR}/flm pull <tag>` 先拉取；\n"
            f"  看哪些可用：`{_FLM_DIR}/flm list`（✅=已下载）。")
    if sub:
        hits = [t for t in got if sub.lower() in t.lower()]
        if len(hits) == 1:
            return FlmModel(hits[0])
        if len(hits) > 1:
            raise SystemExit(f"'{sub}' 匹配到多个 NPU 模型：{', '.join(hits)}")
        raise SystemExit(f"NPU 上没有匹配 '{sub}' 的已下载模型。已下载：{', '.join(got)}")
    if len(got) == 1:
        return FlmModel(got[0])
    print("NPU 已下载的模型：")
    for i, t in enumerate(got):
        print(f"  [{i}] {t}")
    sel = input("序号（回车=0）: ").strip()
    try:
        return FlmModel(got[int(sel)] if sel else 0)
    except (ValueError, IndexError):
        raise SystemExit("无效选择")

# 只能用本地构建跑的架构（官方二进制没有这些实现）
_NEEDS_LOCAL = {"zaya"}

# 适配注册表：启动时读一次（文件小、纯 JSON，不拖慢启动）
PROFILES, _REG_BUILTIN = load_profiles(_MODELS_D)

# 本机实测（2026-09-12，同一份语料/协议；来源见 STAGE1_NPU.md 的能耗与 iGPU 两节）。
# 数字是「这台机器此刻」的值，跨机不可比 —— 只用来帮你选后端，不是承诺。
MEASURED = {
    ("Llama 3.2 1B Instruct", "cpu"): (78.5, 679),
    ("Llama 3.2 1B Instruct", "igpu"): (89.8, 563),
    ("Ling 3.0 Tiny", "cpu"): (47.1, 1041),
    ("Ling 3.0 Tiny", "igpu"): (58.5, 813),
    ("Qwen3.6 35B A3B Reap 48pct", "cpu"): (16.8, 3355),
    ("Qwen3.6 35B A3B Reap 48pct", "igpu"): (25.7, 1922),
    # ZAYA：J/token 还没测（-1 = 未测，列表里显示 "—"）；tok/s 是本机实测
    #   cpu  19.4~23.2（取决构建/负载）  igpu 24.3（build-vk + -ngl 99）
    ("ZAYA1 8B", "cpu"):  (21.0, -1),
    ("ZAYA1 8B", "igpu"): (24.3, -1),
    # Granite 4.0 h-tiny（2026-09-15，同会话交替 3 轮中位；AC + 平台档 low-power）
    #   ★ iGPU 21.9 远优于 CPU 9.4；dengine 是自研引擎（纯 CPU/AVX-512，无 GPU 路径）
    #   能耗未测（-1）；dengine 那行的 tok/s 是修掉"每 token 复制 125MB 词嵌入"之后的值
    ("Granite 4.0 h-tiny", "igpu"):    (21.9, -1),
    ("Granite 4.0 h-tiny", "cpu"):     (9.4,  -1),
    ("Granite 4.0 h-tiny", "dengine"): (12.2, -1),
    # Llama 3.2 1B（2026-09-15，省电档；dengine 接 autotune 后 OMP=6，贪心口径）
    ("Llama 3.2 1B Instruct", "dengine"): (33.0, -1),
    # Qwen3.5 2B f16（2026-09-16，省电档；248k f16 head 是大头，~1GB/token）
    ("Master", "dengine"): (16.0, -1),
}
MEASURED_NPU = ("Llama-3.2-1B（合成权重、仅延迟）", 56.6, 281)

# llama.cpp 支持的架构 —— ★ **从源码里读**，不手写。
# 我第一版手写了一份名单，当场就把 qwen35 漏判成"不支持"（它其实在 llama-arch.cpp:41）。
# 名单这种东西必须能自证，否则就是下一个坑。找不到源码时退回一份保守的内置表。
_ARCH_TABLE_FALLBACK = {
    "llama", "qwen2", "qwen3", "qwen35", "qwen2moe", "qwen3moe", "qwen35moe",
    "granitehybrid", "bailingmoe3", "gemma3", "phi3", "lfm2", "olmo2",
    "glm4moe", "nemotron_h_moe", "mistral", "gemma2",
}


def supported_archs(verbose=False):
    """从 llama-arch.cpp 抓 `{ LLM_ARCH_X, "name" }` 里的 name 集合。

    ★ 二进制目录（llamacpp-b10819）与源码树（npu-direct/llama.cpp-b10819）**不是同一个路径**。
    我第一版只查了二进制目录下的 src/，open() 失败后**静默退回**内置兜底表（17 条），
    而我的测试用例恰好都在这 17 条里，于是"看起来是对的" —— 实际源码里有 122 条。
    所以这里：(a) 搜多个候选路径；(b) 退回兜底时**必须出声**，不允许静默。

    ★★ 2026-09-13 修：正则原来是 `[a-z0-9_]+`，**不允许 `-` 和 `.`** —— 于是
    `falcon-h1`/`gpt-oss`/`kimi-linear`/`minimax-m2` 这类名字被截断成
    `falcon`/`gpt`/`kimi`/`minimax`，模型被误判成"llama.cpp 不支持"（实测 27 个架构受影响）。
    这是适配流程第一次跑真实外部模型（BitNet）时暴露出来的。
    光修正则不够 —— 加了**金丝雀**：解析结果里若一个含 `-`/`.` 的名字都没有，
    就说明正则又被改坏了，必须出声。（真实表里恒有这类名字。）
    """
    import re
    cands = [
        f"{LLAMA_ROOT}/src/llama-arch.cpp",
        f"{_LOCAL_BASE}/src/llama-arch.cpp",
    ]
    for src in cands:
        try:
            txt = open(src, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        names = set(re.findall(r'LLM_ARCH_[A-Z0-9_]+\s*,\s*"([a-z0-9_.\-]+)"', txt))
        canary = [n for n in names if "-" in n or "." in n]
        if not canary:
            print(f"[draco] 警告：{src} 解析出 {len(names)} 个架构，但**没有任何含 '-'/'.' 的名字**"
                  f"（真实表里必然有 falcon-h1 / gpt-oss 等）⇒ 正则可能又漏字符了",
                  file=sys.stderr)
        if len(names) >= 50:                      # 合理下限；太少说明正则/文件不对
            if verbose:
                print(f"[draco] 架构表来自 {src}（{len(names)} 个，含 '-'/'.' 的 {len(canary)} 个）",
                      file=sys.stderr)
            return names
        print(f"[draco] 警告：{src} 只解析出 {len(names)} 个架构，疑似格式变化，"
              f"改用兜底表", file=sys.stderr)
    print(f"[draco] 警告：找不到 llama-arch.cpp（试过 {len(cands)} 个路径），"
          f"改用 {len(_ARCH_TABLE_FALLBACK)} 条兜底表 —— 可能误报某模型不可用",
          file=sys.stderr)
    return set(_ARCH_TABLE_FALLBACK)


SUPPORTED_ARCHS = supported_archs()


# ─────────────────────────── 模型发现 ───────────────────────────

class Model:
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        m = gguf_meta(path)
        self.arch = (m.get("general.architecture") or "?")
        self.name = (m.get("general.name") or os.path.basename(path)).strip()
        # 名字规范化：GGUF 里有的叫 "Smollm2 135M 8k Lc100K Mix1 Ep2"，太啰嗦
        self.gguf_name = self.name       # ★ 原名保留给"选择匹配"用（display_name 只管显示）
        self.short = self.name
        self.supported = self.arch in SUPPORTED_ARCHS
        # 适配档案（models.d/*.json 命中与否）——纯数据，见 load_registry()
        self.profile = profile_for(self.arch, self.name, self.size, PROFILES)
        # ★ 档案可用 display_name 覆盖 GGUF 里那个可能没意义的 general.name
        #   （实测 Falcon-H1 的 general.name 就是 "Original"，列表里只能显示 "Original"）
        dn = draco_view(self).get("display_name") if self.profile else None
        if dn:
            self.short = self.name = dn

    @property
    def ppl_ok(self):
        return self.supported

    def needs_local_build(self):
        """官方预编译二进制跑不了、必须用我们自己的本地构建。"""
        if self.arch in _NEEDS_LOCAL:
            return True
        return bool(self.profile and draco_view(self)["requires_local_build"])

    def backends(self):
        b = []
        if self.supported:
            b += ["cpu", "igpu"]
        return b

    def line(self):
        sz = f"{self.size/1e9:.2f} GB"
        v = draco_view(self) if self.profile else {}
        if v.get("status") == "broken":
            sup = "—（档案标注不可用）"
        elif not self.supported:
            sup = "—（llama.cpp 不支持该架构）"
        elif self.needs_local_build() or v.get("backend_hint") == "local":
            sup = "local"          # 官方二进制没有该架构的实现，只有本地构建有
        else:
            sup = v.get("backend_hint") or "cpu/igpu"
        return f"{self.short[:34]:34s} {sz:>9s}  {self.arch:14s}  {sup}"


def discover():
    ms = []
    for d in GGUF_DIRS:
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(".gguf"):
                    ms.append(Model(os.path.join(root, fn)))
    ms.sort(key=lambda m: m.size)
    return ms


# ★ 自研引擎（Darco）的适配器映射：按 GGUF 名匹配 —— 只有这份表里的模型能走 dengine。
#   新增模型/架构必须先在 draco_engine_server.py 里写适配器并**过数值对账**，再登记到这里
#   （见 models.d/*.json 的 engines.dracomancer 段）。
_DENGINE_ADAPTERS = (("zaya", "zaya"), ("smollm2", "smol"), ("ling", "ling"),
                     ("granite 4.0 h", "granite"), ("llama 3.2", "llama"))
# ★ 有些模型 general.name 无意义（falcon-h1 的叫 "Original"），名字匹配不可用 ⇒ 按架构兜底。
_DENGINE_ADAPTERS_ARCH = {"falcon-h1": "falcon", "qwen35": "qwen35"}


def dengine_adapter(model):
    """该模型在 Darco 里的适配器名；没有则 None（= 不能走 dengine）。"""
    name = getattr(model, "gguf_name", model.name).lower()
    by_name = next((e for k, e in _DENGINE_ADAPTERS if k in name), None)
    if by_name:
        return by_name
    return _DENGINE_ADAPTERS_ARCH.get(getattr(model, "arch", ""), None)


def draco_view(model):
    """档案在 **draco 现在驱动的那个引擎** 视角下的扁平视图。"""
    return engine_view(getattr(model, "profile", None) or {}, "llama_cpp")


def profile_status(model):
    return draco_view(model).get("status", "untested")


def broken_reason(model, engine="llama_cpp"):
    """档案把它标成 broken 时给出理由（否则 None）。engine 按视角取。"""
    prof = getattr(model, "profile", None) or {}
    v = engine_view(prof, engine)
    if v.get("status") != "broken":
        return None
    src = prof.get("_file", "档案")
    return f"{src}：{v.get('notes') or '（档案未写 notes）'}"


def default_backend(model):
    """默认后端：**Darco（自研引擎）优先** —— 主线是引擎不是启动器（用户 2026-09-14 定）。

    顺序：
      ① 有 Darco 适配器 且 档案 engines.dracomancer.status == "works"
         → 用该段的 backend_hint（通常 dengine）；
      ② 否则用 llama_cpp 视角的 backend_hint（cpu/igpu/local）；
      ③ 都没有 → igpu。
    没有适配器（未过对账的架构）**绝不硬闯** Darco，回落是对的。
    """
    prof = getattr(model, "profile", None) or {}
    if dengine_adapter(model):
        dv = engine_view(prof, "dracomancer")
        if dv.get("status") == "works":
            hint = dv.get("backend_hint")
            return hint if hint in BACKENDS else "dengine"
    return profile_backend(model) or "igpu"


def profile_backend(model):
    """档案里 backend_hint 指定的**默认后端**（用户没给 -b 时生效）。

    ★ 之前 backend_hint 只用在 `list` 的显示列上 —— 于是 falcon-h1 的档案写着
    "推荐 cpu"，`chat -m falcon` 却仍然默认 iGPU（实测发现的接线漏洞）。
    现在它真的决定默认后端；用户 `-b` 仍然优先。
    """
    hint = draco_view(model).get("backend_hint")
    return hint if hint in BACKENDS else None


# ═══════════════════ 速度-能效倾向（--prefer / -P） ═══════════════════
# 用户 2026-09-15：「可以做一个可调参数去调整速度-能效倾向」。
# 这台机器上"更快"与"更省"**不是同一个后端**，所以需要显式倾向：
#   · 吞吐：iGPU ≈ 1.21~1.75× CPU（随模型增大而变大）；NPU 最慢（56.6 vs 89.8 tok/s）
#   · 能耗：NPU 281 < iGPU 563 < CPU 679 mJ/token（Llama 3.2 1B 口径，见 STAGE1_NPU.md）
# 倾向只改**两个有实测支撑的旋钮**：后端 与 线程数。别的（量化格式、kernel 融合、
# LM head、NPU 整数 madd）不是"调参"能动的 —— 那是改代码，不该假装是旋钮。
PREFER_CHOICES = ("speed", "balanced", "eco")
PREFER_DOC = {
    "speed":    "要吞吐 —— 按实测 tok/s 挑最快的后端（线程取后端默认）",
    "balanced": "默认 —— 自研引擎（Darco）优先 / 档案推荐，不做额外调",
    "eco":      "要能效 —— 按实测 J/token 挑最省的后端，线程取 4（实测 4~8 吞吐等价）",
}
# 无实测数据时的**能耗先验**：直接来自本机三方能耗定案（CPU 679 / iGPU 563 / NPU 281 mJ/token）。
# local 与 dengine 都是"CPU 上的自己的实现"，能耗按 CPU 算。
_ENERGY_RANK = {"npu": 0, "igpu": 1, "cpu": 2, "local": 2, "dengine": 2}
# 无实测数据时的**吞吐先验**（iGPU 实测 1.21~1.75× CPU，取保守的 1.2）。
# dengine/local 与 cpu 取平（1.0）：实测自研引擎与 llama.cpp 在 CPU 上同量级
# （ZAYA 22~27 vs 21 tok/s）—— 既然是同量级就写 1.0，**不编一个"快 5%"的系数**；
# 平局由 tie-break 决定，而 tie-break 按 balanced 次序 ⇒ 优先自研引擎。
_SPEED_PRIOR = {"npu": 0.6, "igpu": 1.2, "cpu": 1.0, "local": 1.0, "dengine": 1.0}


def _tune_best(m, backend):
    """tune 缓存里该 (模型, 后端) 的最佳 J/token → (jtok, tps, 配置说明) 或 None。"""
    try:
        fp = _model_fingerprint(m)
    except OSError:
        return None
    best = None
    for k, e in (_load_tune_cache().get("entries") or {}).items():
        if not k.startswith(f"{fp}|{backend}|") or not e.get("jtok"):
            continue
        if best is None or e["jtok"] < best[0]:
            best = (e["jtok"], e.get("tps"), k.split("|", 2)[2])
    return best


def backend_metrics(m, backend):
    """(tok/s, **mJ/token**, 来源串)。优先级：tune 实测 > MEASURED 表 > 空（用先验）。

    ★ 单位：tune 缓存里存的是 **J/token**（`_measure_config` 用 W/(tok/s)），而 MEASURED 表是
      **mJ/token** —— 必须在这里统一成 mJ，否则打印出来是"0 mJ/tok"（我第一版就踩了，
      和早先 `cmd_perf` 把 mJ 标成 J 是同一类单位错）。
    """
    t = _tune_best(m, backend)
    if t:
        return t[1], t[0] * 1000.0, f"tune 实测（{t[2]}）"
    v = MEASURED.get(_measured_key(m, backend))
    if v:
        tps, mj = v
        return tps, (None if mj is None or mj < 0 else float(mj)), "档案性能表"
    return None, None, ""


def _measured_key(m, backend):
    """MEASURED 表的查表键。★ 不能直接用 m.name：档案的 display_name 带架构后缀
    （"Ling 3.0 Tiny（bailingmoe3）"），而表里的键是干净名（"Ling 3.0 Tiny"）——
    实测两者**查不中**，于是 Ling 的实测吞吐/能耗一直没被用上（-P speed/eco 静默退化成先验）。
    这里按 全名 → 去掉括号后缀 → GGUF 原名 依次试。"""
    nm = m.name or ""
    cands = [nm, nm.split("（")[0].split("(")[0].strip(), getattr(m, "gguf_name", "") or ""]
    for c in cands:
        if c and (c, backend) in MEASURED:
            return (c, backend)
    return (nm, backend)


def rank_backends(m, prefer, cands=None):
    """按倾向给候选后端排序 → [(backend, 依据串)]。

    ★ 两条纪律（都是接线时踩出来的）：
      1. **不自动挑 npu**：FLM 是另一套模型命名空间（tag），要用户显式 `-b npu`；
         这里只排序候选，绝不会"顺手"把模型换成 NPU 上的另一个模型。
      2. **有实测就用实测，没实测才用先验，并且把"这是先验"写在依据里** ——
         绝不把先验系数包装成"实测更快"。MEASURED 表里的第二列是 **mJ/token**。
    """
    cands = list(cands if cands is not None else m.backends())
    # ★ Darco（自研引擎）与 local（本地构建）也要进候选 —— 它们是我们**自己的链**；
    #   否则 -P speed/eco 会把有 Darco 适配器的模型"悄悄"推给 llama.cpp（接线时实测到的反直觉）。
    #   dengine 目前仍跑在 CPU 上（iGPU 未接线）⇒ 能耗先验按 CPU 算，通常排不过 igpu，
    #   这正是"为什么不是自研引擎"必须在输出里说清楚的地方。
    if dengine_adapter(m) and engine_view(getattr(m, "profile", None) or {},
                                          "dracomancer").get("status") == "works":
        cands.append("dengine")
    if m.needs_local_build():
        cands.append("local")
    cands = [b for b in dict.fromkeys(cands) if b in BACKENDS]
    if not cands:
        cands = ["cpu"]
    if prefer not in PREFER_CHOICES:
        prefer = "balanced"
    rows = [(b,) + backend_metrics(m, b) for b in cands]        # (backend, tps, mJ/tok, 来源)
    bal = ["dengine", "local", "cpu", "igpu"]
    bal_rank = lambda b: bal.index(b) if b in bal else 9

    if prefer == "speed":
        # 实测 tok/s 优先；同一后端族（CPU 上跑的 local/dengine/cpu）没有实测时，
        # 用本机最佳实测 × 先验系数估一个量级；连一个实测都没有时直接用先验系数排序。
        ref = max([r[1] for r in rows if r[1]] or [0])
        def score(r):
            if r[1]:
                return r[1]
            return (ref * _SPEED_PRIOR.get(r[0], 1.0)) if ref else _SPEED_PRIOR.get(r[0], 1.0)
        rows.sort(key=lambda r: (-score(r), bal_rank(r[0])))
    elif prefer == "eco":
        # J/token 越小越省；没测过能耗的按本机能耗序先验（NPU<iGPU<CPU）
        def escore(r):
            if r[2]:
                return r[2]
            return 1e6 * (1 + _ENERGY_RANK.get(r[0], 3))        # 未测 ⇒ 排在所有实测之后
        rows.sort(key=lambda r: (escore(r), bal_rank(r[0])))
    else:
        rows.sort(key=lambda r: bal_rank(r[0]))
    out = []
    for b, tps, jtok, src in rows:
        bits = []
        if tps:
            bits.append(f"{tps:.1f} tok/s")
        if jtok:
            bits.append(f"{jtok:.0f} mJ/tok")   # mJ/token（见 backend_metrics 的单位说明）
        bits.append(src if src else
                    f"无实测（先验：吞吐×{_SPEED_PRIOR.get(b, 1.0):.2f}、能耗序 {_ENERGY_RANK.get(b, 3)}）")
        out.append((b, "，".join(bits)))
    return out


def choose_backend(m, prefer="balanced", backend_arg=None, threads_arg=None):
    """解析"用哪个后端/几条线程" → (backend, threads, notes[])。

    用户显式 `-b` 永远优先；否则按倾向排序取第一名。
    线程：eco 取 4（实测 4~8 **吞吐等价**，少线程少发热 —— 功耗未单独实测，故不声称"更省电"）。
    """
    if backend_arg:
        b = backend_arg
        return b, threads_arg or BACKENDS[b]["threads_default"], [f"用户指定 -b {b}"]
    ranks = rank_backends(m, prefer)
    b, why = ranks[0]
    th = threads_arg or BACKENDS[b]["threads_default"]
    notes = [f"倾向 {prefer}：{PREFER_DOC[prefer]}"]
    if len(ranks) > 1:
        notes.append("候选排序：" + " > ".join(x for x, _ in ranks))
    notes.append(f"选定 {b} —— {why}")
    if threads_arg is None and prefer == "eco" and BACKENDS[b]["threads_default"] > 4:
        th = 4
        notes.append(f"eco 线程取 4（后端默认 {BACKENDS[b]['threads_default']}；实测 4~8 吞吐等价）")
    return b, th, notes


def pick_model(ms, want):
    """支持名字前缀/子串匹配（不区分大小写），也支持序号"""
    if want is None:
        return None
    w = want.lower().strip()
    if w.isdigit():
        i = int(w)
        if 0 <= i < len(ms):
            return ms[i]
    hits = [m for m in ms if w in getattr(m, "gguf_name", m.name).lower()
            or w in m.name.lower() or w in os.path.basename(m.path).lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"没有匹配 '{want}' 的模型。用 `draco.py list` 看清单。")
    if len(hits) > 1:
        # 取最小的那个（避免歧义时误载 19 GB 的）
        hits.sort(key=lambda m: m.size)
        print(f"[提示] '{want}' 匹配到 {len(hits)} 个，取最小的：{hits[0].name}")
        return hits[0]


# ─────────────────────────── 服务端进程管理 ───────────────────────────

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def install_cleanup(srv):
    """让 SIGTERM/SIGHUP 也能收掉子进程。

    ★ 实测踩到：Ctrl-C 走 KeyboardInterrupt → `finally` 会清理；但**外部 SIGTERM 不触发
    `finally`**，python 直接退出、llama-server 变孤儿继续占着显存和端口。所以显式接管。
    （又一次印证：收进程要么自己留句柄，要么显式接管信号 —— 别指望模式匹配。）
    """
    def _h(signum, _frame):
        srv.stop()
        raise SystemExit(0)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _h)
        except (ValueError, OSError):
            pass


def _pdeathsig():
    """让**内核**在父进程消失时给子进程发 SIGTERM（PR_SET_PDEATHSIG）。

    ★ 为什么不能只靠信号处理器：我实测 `draco.py serve` 的父进程在收到 SIGTERM 之前就已经
    退出了（SIGKILL / 崩溃 / 父 shell 被带走都会这样），llama-server 于是变成孤儿继续占着
    端口和显存 —— 信号处理器根本没机会跑。`PR_SET_PDEATHSIG` 由内核保证，覆盖所有退出路径。
    仅在 Linux 有效，失败就算了（不能因为清理机制反而让正常路径挂掉）。
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


class Server:
    """`llama-server` 子进程。★ 句柄自己留着按 PID 收，绝不做模式匹配 kill。"""

    def __init__(self, model: Model, backend: str, ctx: int, threads: int,
                 extra=None, verbose=False):
        if backend not in BACKENDS:
            raise SystemExit(f"未知后端 {backend}；可用：{', '.join(BACKENDS)}")
        # ★ 架构需要本地构建时自动切换（官方二进制里没有 zaya 这类自写架构的实现）。
        #   名单来自两处：内置 _NEEDS_LOCAL + 注册表档案的 requires_local_build。
        # ★ 只有"官方预编译二进制"那几个后端才需要自动切 —— local 和 dengine 都是
        #   **我们自己的实现**，不该被覆盖。之前这里把 `-b dengine` 也换成了 local，
        #   导致 `-m zaya -b dengine` 实际跑的是 llama.cpp（2026-09-14 用户发现）。
        if model.needs_local_build() and backend not in ("local", "dengine"):
            src = (model.profile or {}).get("_file", "内置档案")
            print(f"[draco] {model.arch} 只有**本地构建**里有实现（官方二进制没有；"
                  f"档案 {src}）→ 后端 {backend} 自动换成 local")
            backend = "local"
        if not model.supported:
            raise SystemExit(
                f"'{model.name}'（架构 {model.arch}）llama.cpp b10819 不支持，"
                f"cpu/igpu 都跑不了。\n"
                f"  ZAYA 这类需要我们自己写内核（见 STAGE1_NPU.md）。"
            )
        self.model, self.backend = model, backend
        prof = model.profile or {}
        view = draco_view(model)
        reason = broken_reason(model)
        if reason:
            # 档案已标注"在这个引擎上不可用"（负结果）——给出理由，比让 llama.cpp
            # 抛一句 'unknown architecture' 有用得多。
            raise SystemExit(f"'{model.name}' 在当前引擎上不可用 —— {reason}")
        b = BACKENDS[backend]
        self.port = free_port()
        # ── 自研引擎（dengine）分支：桥接服务器 ──
        #   ★ 守卫用 **dracomancer 视角**的 status —— 档案说这条引擎坏了就拒，
        #     理由直接来自 notes（2026-09-13 smol 就是这样被拦下的：m6-smol 内核对账未过）。
        if backend == "dengine":
            reason = broken_reason(model, "dracomancer")
            if reason:
                raise SystemExit(f"'{model.name}' 在自研引擎上不可用 —— {reason}")
            exe = os.path.join(b["dir"], "draco_engine_server.py")
            if not os.path.exists(exe):
                raise SystemExit(f"找不到 {exe}")
            self.url = f"http://127.0.0.1:{self.port}"
            eng_adapter = dengine_adapter(model)
            if eng_adapter is None:
                raise SystemExit("自研引擎（Darco）目前只接了 SmolLM2 / ZAYA1 / Ling"
                                 "（其余模型/架构待逐个过数值对账后再接线；"
                                 "清单见 draco._DENGINE_ADAPTERS 与各档案的 engines.dracomancer 段）")
            self.cmd = [sys.executable, exe, "--model", model.path,
                        "--port", str(self.port), "--engine", eng_adapter,
                        "--threads", str(threads), "--ctx", str(ctx)]
            if extra:
                self.cmd += extra
            if verbose:
                print("  $ " + " ".join(self.cmd))
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if verbose else subprocess.DEVNULL,
                preexec_fn=_pdeathsig)
            return
        # ── NPU（FastFlowLM）分支：不是 llama-server，命令行完全不同 ──
        #   `flm serve <tag> --host 127.0.0.1 -p <port>`；模型加载完成**之后**才开 HTTP，
        #   所以"/v1/models 返回 200"就是就绪（见 wait_ready）。
        if backend == "npu":
            exe = os.path.join(b["dir"], "flm")
            if not os.path.exists(exe):
                raise SystemExit(f"找不到 {exe}")
            self.url = f"http://127.0.0.1:{self.port}"
            self.cmd = [exe, "serve", getattr(model, "tag", model.path),
                        "--host", "127.0.0.1", "-p", str(self.port)]
            if extra:
                self.cmd += extra       # 透传给 flm（如 --pmode turbo）
            if verbose:
                print("  $ " + " ".join(self.cmd))
            env = dict(os.environ)
            env.pop("FLM_PORT", None)   # 防止环境里残留端口覆盖我们的 -p
            self.proc = subprocess.Popen(
                self.cmd, env=env,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if verbose else subprocess.DEVNULL,
                preexec_fn=_pdeathsig)
            return
        exe = os.path.join(b["dir"], "llama-server")
        if not os.path.exists(exe):
            raise SystemExit(f"找不到 {exe}")
        # ★ 自写架构（zaya）：本地构建若带 Vulkan 就把层放上 iGPU —— 实测生成 24~27 tok/s
        #   vs CPU 19.4（同一构建），且数值上与 CPU 同量级（l_out-79 cos 0.984，等同 CPU-vs-精确参考
        #   的噪声量级）。想要纯 CPU 用 `--extra "-ngl 0"` 覆盖。
        ngl = b["ngl"]
        if model.needs_local_build() and local_has_vulkan(b["dir"]):
            ngl = "99"
        # 注册表可显式指定 ngl（档案数据 > 后端默认；用户 --extra 仍可再覆盖，见下）
        p_ngl = view["launch"].get("ngl")
        if p_ngl is not None:
            ngl = str(p_ngl)
        cmd = [
            exe, "-m", model.path,
            "-c", str(ctx), "-t", str(threads), "-ngl", ngl,
            "--host", "127.0.0.1", "--port", str(self.port),
            "-np", "1", "--no-warmup",
        ]
        # ★★★ ZAYA 必须 `-ub 1`（逐 token prefill）—— 这条现在来自 models.d/zaya.json 的
        #   launch.extra_args，不再是硬编码（里程碑 18 定案：批量 prefill 会被 Q8_K 量化台阶
        #   放大成 1e-3 级跳变，-ub 16 起直接胡言乱语；完整根因见 STAGE1_NPU.md 里程碑 18）。
        #   顺序：档案参数在前、用户 --extra 在后 ⇒ llama.cpp 对同名 flag 取后者，用户可覆盖。
        cmd += view["launch"].get("extra_args") or []
        cmd += (extra or [])
        self.cmd = cmd
        if verbose:
            print("  $ " + " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.STDOUT if verbose else subprocess.DEVNULL,
            text=True,
            preexec_fn=_pdeathsig if hasattr(os, "fork") else None,
        )
        self.url = f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"服务端提前退出（码 {self.proc.returncode}）")
            # llama-server 有 /health；FLM 没有（404）——退回 /v1/models。
            # FLM 是"模型加载完才开 HTTP"，所以 200 即就绪；llama-server 的 /health
            # 也是加载完才 200，语义一致。
            for ep in ("/health", "/v1/models"):
                try:
                    with urllib.request.urlopen(self.url + ep, timeout=2) as r:
                        if r.status == 200:
                            return time.time() - t0
                except urllib.error.HTTPError:
                    continue          # 404（FLM 的 /health）→ 试下一个端点
                except Exception:
                    break             # 连不上 → 直接等下一轮，别把两个端点都试一遍
            time.sleep(0.25)
            if sys.stderr.isatty():
                sys.stderr.write(f"\r  等待模型装载… {int(time.time()-t0)}s")
                sys.stderr.flush()
        raise SystemExit("等待服务端就绪超时")

    def warmup(self):
        """服务端就绪后补一次**一次性的**小请求，把"进程第一次解码"这个坑占掉。

        ★ 2026-09-13 实测（Falcon-H1-0.5B，混合 SSM/注意力架构，CPU 与 iGPU 都一样）：
          同一个 prompt、贪心、固定 seed，**新起进程的第一次解码**给出的答案
          与之后所有次都不同：
              第 1 次: "Once upon a time, in a land far, far away"
              第 2 次起: "Once upon a time in a sunlit meadow"   ← 稳定
          先用一个**无关 prompt** 占掉第一次解码后，同一 prompt 的后续回答就一致了。
          等于说"第一条消息的答案取决于它是不是进程里的第一条"——对用户是诡异的。
          实测矩阵（每次全新 server，同一个 prompt B 连问两次比 B#1 / B#2）：
              · 不预热                                       → 不一致（B#1 冷）
              · 预热体 {max_tokens:1}                         → 仍然不一致
              · 预热体 {max_tokens:24, temp:0}（无 seed/rep）  → 仍然不一致
              · 预热体 {max_tokens:24, temp:0, seed:42, rep:1.1} → **一致** ✓
          ⇒ 关键不是"发过请求"，而是**预热请求的形状要与真实请求一致**（采样参数齐全、
            别只发 1 个 token）。所以下面这个 body 刻意与 `stream_chat` 对齐 ——
            我第一版只发了 `{max_tokens:1}`，实测**无效**（selfcheck 仍报不一致）。
          · 不是后端问题（CPU/iGPU 同样）；不是 `--no-warmup`（去掉它、用 llama-server
            自带热身，现象一样）；也不是跨请求污染（同一 prompt 连问 5 次都一致）。
        副作用：启动多一次极小的 prefill（本机 <100 ms）＋几个 token；换来"同一 prompt
        的结果与顺序无关"，也让 `selfcheck` 的指纹有意义（否则第一轮永远偏）。
        """
        try:
            # ★ 形状要像真请求：带 seed / repeat_penalty，并真的生成一段（max_tokens 别是 1）
            body = json.dumps({
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "max_tokens": 32, "temperature": 0.0, "seed": 0, "repeat_penalty": 1.1,
            }).encode()
            for _ in range(2):     # 连发两次：实测单发在部分形状下仍不足以消掉"第一次解码"
                req = urllib.request.Request(
                    self.url + "/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=120) as r:
                    r.read()
        except Exception as e:
            print(f"[draco] 提示：预热请求失败（不影响使用）：{e}", file=sys.stderr)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


# ─────────────────────────── 聊天（流式 SSE） ───────────────────────────

def stream_chat(url, messages, temp, seed, max_tokens, system=None, think=None,
                repeat_penalty=1.1, hold=None):
    """POST /v1/chat/completions，逐块 yield (文本增量, timing)。

    ★ 必须同时处理 `reasoning_content`：这类模型（Ling、Qwen3.5 系等）在"思考模式"下
    把内容放进 `reasoning_content` 而 `content` 是 null。我只读 content 的第一版
    因此**一个字都收不到，服务端却报生成了 4061 个 token** —— 表现是"聊天没输出但很快"。
    yield 的增量形如 ("content"|"reasoning", 文本, timing)。
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    req_body = {
        "messages": msgs, "stream": True,
        "temperature": temp, "seed": seed,
        # ★ 不限时不发这个键：llama-server 把 null 当不限，但 FLM 会 500
        #   （json.exception.type_error.302 "type must be number, but is null"）。
        #   两个服务端对"缺键"的默认都是不限 ⇒ 省掉分支，统一不发。
        "repeat_penalty": repeat_penalty,
    }
    if max_tokens > 0:
        req_body["max_tokens"] = max_tokens
    if think is not None:                     # 关闭/开启思考（多数模板认这个开关）
        req_body["chat_template_kwargs"] = {"enable_thinking": think}
    if hold in ("line", "token", "dup"):      # ★ 思考流式粒度（只有自研引擎的桥接认这个字段）
        req_body["draco_think_hold"] = hold
    elif hold:
        # ★★ 这里曾经踩过：`hold` 插在 `repeat_penalty` 前面 ⇒ 位置调用把 1.1 传成 hold，
        #    桥接收到浮点数、对 float 调 .strip() 抛 AttributeError ⇒ **客户端只看到空回复**
        #    （selfcheck 因此"两轮逐字一致"地全空、指纹相同、还没有 timings）。
        #    ⇒ 两道防线：① 签名把 hold 放到**最后**（不改既有位置参数）；② 只接受三种合法值。
        print(f"[警告] 忽略非法的思考粒度取值 {hold!r}（应为 line/token/dup）", file=sys.stderr)
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    timing = {}
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            if "timings" in j and j["timings"]:
                timing = j["timings"]
            # FLM 不给 llama-server 的 `timings`，速度在**最后一个** chunk 的 usage 里
            # （prefill_speed_tps / decoding_speed_tps）。映射成同一套显示字段。
            u = j.get("usage") or {}
            if "decoding_speed_tps" in u:
                timing = {
                    "predicted_n": u.get("completion_tokens"),
                    "predicted_per_second": u.get("decoding_speed_tps"),
                    "prompt_per_second": u.get("prefill_speed_tps"),
                }
            for ch in j.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    yield "reasoning", d["reasoning_content"], timing
                if d.get("content"):
                    yield "content", d["content"], timing
    yield "content", "", timing


# ─────────────────────────── 子命令：list ───────────────────────────

def cmd_list(args):
    ms = discover()
    print("可用的 GGUF 模型")
    print("=" * 78)
    print(f"{'名称':34s} {'大小':>9s}  {'架构':14s}  {'可跑后端'}")
    print("-" * 78)
    for i, m in enumerate(ms):
        print(f"[{i}] {m.line()}")
    print()
    print("★ NPU 走 FastFlowLM（独立于 GGUF）：`draco.py chat -b npu -m <tag子串>`。")
    print(f"  本机已下载（`{_FLM_DIR}/flm list` 里 ✅ 的）：", end="")
    fm = flm_models()
    got = sorted(t for t, ok in fm.items() if ok)
    print("、".join(got) if got else "（无）")
    print("  gpt-oss:20b 的权重有 14GB：需空闲内存 ≥16GB 再加载（当前机器满载会顶到交换）。")
    print("  实测（llama3.2:1b，8 列 NPU）：decode 56.6 tok/s、281 mJ/token（最省电）；见 STAGE1_NPU.md。")
    print()
    print("用 `draco.py perf` 看本机实测性能（tok/s 与能耗）。")


def cmd_perf(args):
    print("本机实测性能（2026-09-12，同协议；绝对值为「此刻此机」，跨机不可比）")
    print("=" * 74)
    print(f"{'模型':30s} {'后端':6s} {'tok/s':>8s} {'mJ/token':>10s}")
    print("-" * 74)
    for (name, be), (ts, mj) in MEASURED.items():
        mjs = f"{mj:10.0f}" if mj >= 0 else f"{'—（未测）':>10s}"
        print(f"{name:30s} {be:6s} {ts:8.1f} {mjs}")
    n, ts, mj = MEASURED_NPU
    print(f"\nNPU（{n}）: {ts:.1f} tok/s, {mj} mJ/token")
    print("  ★ NPU 最省电但最慢（比 CPU 慢 1.39×）；iGPU 两个轴同时优于 CPU。")
    print("  ★ prefill 上 iGPU 快 2.5~2.9×（batch 算子更吃矩阵单元）。")


# ─────────────────────────── 子命令：caps ───────────────────────────
# 能力探针：**加载之前**静态判断每个模型能不能被这条链吃掉，并汇入档案状态。
# 与 selfcheck 的分工：caps 管"能不能加载"（静态、秒级、零副作用），
# selfcheck 管"跑起来行为对不对"（动态、要装载）。两者互补，别指望 caps 看数值问题。

def cmd_caps(args):
    import gguf_probe as GP
    tt, dead = GP.ggml_type_table(), GP.dead_types()
    archs, canary = GP.arch_table()
    print("引擎事实（从将要执行的那份源码里读出来，不手写）")
    print("=" * 78)
    print(f"  ggml 类型表 {len(tt)} 项，其中**已移除** {len(dead)} 项：")
    for i, n in sorted(dead.items()):
        print(f"      id={i:2d}  {n}")
    print(f"  llama-arch 架构 {len(archs)} 个（含 '-'/'.' 的 {len(canary)} 个 —— 金丝雀通过）")
    print()
    ms = discover()
    if args.model:
        one = pick_model(ms, args.model)
        ms = [one] if one else []
    print(f"{'模型':32s} {'架构':14s} {'静态判定':16s} {'档案状态':10s} 后端")
    print("-" * 78)
    VERD = {"dead": "❌ 必死", "arch_unsupported": "❌ 架构未实现", "maybe": "✅ 可能可以"}
    for m in ms:
        try:
            r = GP.classify(m.path)
            v = VERD[r["verdict"]]
        except Exception as e:
            r, v = {"reasons": [str(e)], "warn": []}, "⚠ 读头失败"
        vv = draco_view(m)
        st = vv.get("status", "untested") if m.profile else "—"
        be = vv.get("backend_hint") or ("local" if m.needs_local_build() else "cpu/igpu")
        print(f"{m.short[:32]:32s} {m.arch[:14]:14s} {v:16s} {st:10s} {be}")
        for x in r["reasons"]:
            print(f"      · {x}")
        if args.verbose:
            for x in r["warn"]:
                print(f"      ⚠ {x}")
    print()
    print("提示：静态判定只看'加载得了吗'。数值/顺序/后端差异这类问题必须跑 selfcheck。")
    if _REG_BUILTIN:
        print("⚠ 注意：注册表用的是**内置兜底**（models.d 没读到），档案状态可能不准。")


# ─────────────────────────── 子命令：tune ───────────────────────────
# 参数自调优（我的建议 #1）：在一小组**离散**配置上实测，目标函数用 **J/token**（不是 tok/s）——
# 本项目自己的实测结论就是"NPU/iGPU 是能效赢、速度不一定赢"，所以排序要按能耗。
# 结果**按 (模型指纹, 后端, 配置) 缓存**（每次探测要付一次装载+评测，不能每次重来）。
# ★ 功率传感器按 name+label 找，不按 hwmon 编号（编号会变，旧脚本就是这样读错的）。

TUNE_CACHE = os.path.expanduser("~/.cache/draco/tune.json")


def _model_fingerprint(m):
    st = os.stat(m.path)
    return f"{os.path.basename(m.path)}:{st.st_size}:{int(st.st_mtime)}"


def _load_tune_cache():
    try:
        with open(TUNE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"_note": "draco tune 的实测缓存；键=模型指纹|后端|配置", "entries": {}}


def _save_tune_cache(c):
    os.makedirs(os.path.dirname(TUNE_CACHE), exist_ok=True)
    with open(TUNE_CACHE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)


def _measure_config(model, backend, extra, tokens, threads, min_seconds=6.0, max_iters=4):
    """跑一组配置 → (tok/s, J/token 总口径, J/token 边际口径, 原始数据)。

    ★★ 为什么这么啰嗦（全是实测教训）：
      · 短窗口（~2s）测能耗在这台机上**不可信**：同一个配置两次跑出 18.4W 与 24.0W，
        排名会在两次运行间翻过来。而 tok/s 是稳的。噪声来源是桌面/浏览器的**秒级突发负载**，
        不是传感器（传感器 46Hz 够快）。所以：
          – 采样前**静置 1.5s**（紧邻的装载/启动余温会污染第一组）；
          – 只保留**负载段**样本，静置段单独留作**空闲基线**；
          – 窗口不足 min_seconds 就**重复请求**把窗口拉长（默认 ≥6s）；
          – 用**中位数**而不是均值（突发会把均值拽高）；
          – 同时给两个口径：**总功率**（含底噪，跨机不可比）与**边际功率**（减去本组自己
            测到的空闲基线，更接近"这个模型每 token 花多少"）。
      · 结论：**按 tok/s 排序随时可信；按 J/token 排序需要机器安静 + 足够长的窗口。**
    """
    from hwprobe import PowerSampler
    srv = Server(model, backend, 2048, threads, extra=extra, verbose=False)
    try:
        srv.wait_ready()
        srv.warmup()
        body = json.dumps({
            "messages": [{"role": "user", "content": "Write a short paragraph about the sea."}],
            "max_tokens": tokens, "temperature": 0.0, "seed": 0, "repeat_penalty": 1.1,
        }).encode()
        with PowerSampler("PPT") as ps:
            time.sleep(1.5)                      # 静置：让上一组的余温散掉
            n0 = ps.samples                      # 静置段结束的样本下标
            t0, toks, iters = time.time(), 0, 0
            while iters == 0 or (time.time() - t0 < min_seconds and iters < max_iters):
                req = urllib.request.Request(srv.url + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"},
                                             method="POST")
                with urllib.request.urlopen(req, timeout=900) as r:
                    payload = json.loads(r.read().decode("utf-8", "replace"))
                u = payload.get("usage") or {}
                tim = payload.get("timings") or {}
                toks += u.get("completion_tokens") or tim.get("predicted_n") or 0
                iters += 1
            wall = time.time() - t0
        load_vals, idle_vals = ps.vals[n0:], ps.vals[:n0]
        med = lambda v: (sorted(v)[len(v) // 2] if v else None)
        w_load, w_idle = med(load_vals), med(idle_vals)
        tps = toks / wall if wall > 0 else 0
        jtot = (w_load / tps) if (w_load and tps) else None
        jmar = ((max(0.0, w_load - w_idle) / tps) if (w_load and w_idle is not None and tps)
                else None)
        return tps, jtot, jmar, {
            "wall_s": round(wall, 2), "tokens": toks, "iters": iters,
            "power_load_median_w": None if w_load is None else round(w_load, 1),
            "power_idle_median_w": None if w_idle is None else round(w_idle, 1),
            "power_load_samples": len(load_vals),
            "timings": tim,
        }
    finally:
        srv.stop()


def cmd_tune(args):
    if args.backend == "npu":
        raise SystemExit("tune 目前只覆盖 llama.cpp 这条链（npu 走 FLM，旋钮不同）")
    ms = discover()
    m = pick_model(ms, args.model)
    if m is None:
        raise SystemExit("要指定模型：draco.py tune -m <模型>")
    reasons = broken_reason(m)
    if reasons:
        raise SystemExit(f"'{m.name}' 在当前引擎上不可用 —— {reasons}")
    # tune 的默认后端：balanced 时仍优先档案推荐（"我们已知最好"的那条，行为不变）；
    # speed/eco 时让**倾向**来排（否则 -P 在 tune 上是哑的）。
    _prefer = getattr(args, "prefer", None) or "balanced"
    backend, threads, _n = choose_backend(
        m, _prefer,
        backend_arg=args.backend or (profile_backend(m) if _prefer == "balanced" else None),
        threads_arg=args.threads)

    # 配置空间：**小**且离散。默认只在最有价值的三条轴上扫。
    base_extra = list(draco_view(m)["launch"].get("extra_args") or [])
    if backend == "dengine":
        # ★ dengine 的旋钮**不是** llama.cpp 的启动参数（桥接只认 --threads/--ctx）——
        #   硬塞 -ub/-fa 会让模型桥接的 argparse 报错退出（`-ub 1` 反而是 llama.cpp 侧才需要的）。
        #   所以这里清空档案里的 llama.cpp 参数，只扫"线程数"这一条真旋钮。
        base_extra = []
        space = [("基线（档案参数）", base_extra + [])]
    else:
        space = [
            ("基线（档案参数）", base_extra + []),
            ("prefill 逐 token -ub 1", base_extra + ["-ub", "1"]),
            ("prefill 小批 -ub 4", base_extra + ["-ub", "4"]),
            ("关 flash-attn", base_extra + ["-fa", "0"]),
        ]
    if args.threads is None:
        space.append((f"线程 {max(2, threads//2)}", base_extra + [], max(2, threads // 2)))

    cache = _load_tune_cache()
    fp = _model_fingerprint(m)
    try:
        from hwprobe import power_state
        print(f"  {power_state()}")
    except Exception:
        pass
    print(f"tune：{m.short}  后端={backend}  模型指纹={fp}")
    print(f"  配置数 {len(space)}，每组生成 {args.tokens} token；缓存 {TUNE_CACHE}")
    rows = []
    for item in space:
        label, extra = item[0], item[1]
        th = item[2] if len(item) > 2 else threads
        key = f"{fp}|{backend}|{th}|{' '.join(extra)}"
        ent = cache["entries"].get(key)
        if ent and not args.force:
            print(f"  · {label:24s} [缓存] {ent['tps']:.1f} tok/s  {ent['jtok']:.3f} J/tok"
                  f"  （{ent.get('when','')}）")
        else:
            print(f"  · {label:24s} 实测中…", end="", flush=True)
            try:
                tps, jtok, jmar, raw = _measure_config(m, backend, extra, args.tokens, th)
                w = raw.get("power_load_median_w")
            except SystemExit as e:
                print(f" 失败：{e}")
                cache["entries"][key] = {"tps": 0, "jtok": None, "error": str(e),
                                         "when": time.strftime("%Y-%m-%d %H:%M")}
                continue
            ent = {"tps": round(tps, 2), "watts": w, "jtok": None if jtok is None else round(jtok, 4),
                   "jtok_marginal": None if jmar is None else round(jmar, 4), "extra": extra,
                   "threads": th, "raw": raw, "when": time.strftime("%Y-%m-%d %H:%M")}
            cache["entries"][key] = ent
            print(f"\r  · {label:24s} {tps:6.1f} tok/s  "
                  + (f"{w:5.1f} W  总 {jtok:.3f}" + (f" / 边际 {jmar:.3f}" if jmar else "")
                     + " J/tok" if w else "（没读到功率传感器）"))
        rows.append((label, ent))
    _save_tune_cache(cache)

    ok = [r for r in rows if r[1].get("jtok")]
    print()
    if ok:
        ok.sort(key=lambda r: r[1]["jtok"])
        print(f"{'配置':26s} {'tok/s':>8s} {'W(负载)':>8s} {'J/tok 总':>9s} {'J/tok 边际':>10s}  相对")
        print("-" * 74)
        best = ok[0][1]
        for label, e in ok:
            rel = e["jtok"] / best["jtok"]
            jm = e.get("jtok_marginal")
            print(f"{label:26s} {e['tps']:8.1f} {e.get('watts') or 0:8.1f} {e['jtok']:9.3f}"
                  f" {(jm if jm is not None else float('nan')):10.3f}"
                  f"   {('★ 最优' if rel == 1.0 else f'{rel:.2f}×')}")
        print()
        print("★ 两个口径：**总**=整机 PPT 功率/吞吐（含空闲底噪）；**边际**=减去本组自己测到的"
              "空闲基线（更接近'这个模型每 token 花多少'）。")
        print("★ 可信度：**按 tok/s 排序随时可信；按 J/token 排序需要机器安静**。实测同一配置两次"
              "可差 30%（18.4W vs 24.0W），所以能耗结论别跨次比较、也别在后台跑东西时下。")
        print(f"  最优配置：{ok[0][0]}  extra_args = {json.dumps(best.get('extra') or [])}"
              f"  threads = {best.get('threads')}")
        print("  把它写进档案：models.d/<arch>.json 的 engines.llama_cpp.launch.extra_args")
    else:
        print("没有拿到可比的能耗数据（检查功率传感器 / 生成是否成功）。")


# ─────────────────────────── 子命令：selfcheck ───────────────────────────
# 社区适配机制的验证闸门（用户侧一半）：以**固定协议**跑一个模型 + 后端组合，
# 产出 (a) 可复现性指纹 (b) 健康问句通过情况 (c) 速度，和一个可直接贴进 issue 的
# JSON 块（`--json`）。bot 侧的校验脚本（release/validate_submission.py）会重放它。
#
# ★ 指纹的定位要诚实：贪心解码 + 固定 seed 也**不保证跨机逐位一致**（AVX512 vs AVX2
#   的累加次序都不同），所以指纹是「同机同后端可复现」的诊断 + 给维护者比对用的参考，
#   不是通过/失败的判据。判据是健康问句 + 维护者复跑。

def _strip_think(txt):
    """去掉思考段，只留最终答案。返回 (答案, 状态)：
      "ok"        —— 思考已闭合（或本来就没有），答案是后面的部分
      "thinking"  —— 有 <think> 但没闭合 ⇒ **仍在上思考、预算不足**，不能把思考当答案
      "no_think"  —— 没有思考标记，全文即答案
    ★ 我第一版在"剥完为空"时 `or txt.strip()` 兜底 ⇒ 把未闭合的思考块当答案，
      于是思考文本里出现的数字被当成了正确回答（**假通过**）。这种静默错是本项目一直在防的。
    """
    import re
    has_open = bool(re.search(r"<think>", txt, re.I))
    has_close = bool(re.search(r"</think>", txt, re.I))
    if has_open and not has_close:
        return "", "thinking"
    out = re.sub(r"<think>.*?</think>", "", txt, flags=re.S | re.I)
    out = re.sub(r"^.*?\[/?think\]", "", out, flags=re.S | re.I)
    return out.strip(), ("ok" if has_open else "no_think")


_SELFCHECK_PROMPTS = [
    # 可复现性用（内容不重要，重要的是固定）
    ("repro-1", "The quick brown fox", 24),
    ("repro-2", "Once upon a time", 24),
    ("repro-3", "1, 2, 3,", 24),
]
_SELFCHECK_SANITY = [
    # 健康问句（仅对 instruct 模型有意义；base 模型天然可能失败 → 只算 WARN）
    ("What is the capital of France? Answer with just the city name.", ["Paris", "paris"]),
    ("What is 1+1? Answer with just the number.", ["2"]),
    ("Count from 1 to 5 using digits, separated by commas.", ["3"]),  # 中间那个数最稳
]


def _complete_once(url, messages, temp, seed, max_tokens, rep=1.1, think=False):
    """一次非交互补全：返回 (文本, timing)。复用 stream_chat 的协议处理。"""
    buf, timing = [], {}
    for kind, delta, tim in stream_chat(url, messages, temp, seed, max_tokens,
                                        None, think, rep):
        if kind == "content" and delta:
            buf.append(delta)
        if tim:
            timing = tim
    return "".join(buf), timing


def cmd_selfcheck(args):
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    # ── 解析模型/后端（与 chat 同一套规则）──
    if args.backend == "npu":
        m = pick_flm(args.model)
        srv_model, backend = m, "npu"
        threads = args.threads or BACKENDS["npu"]["threads_default"]
    else:
        ms = discover()
        m = pick_model(ms, args.model) or next((x for x in ms if x.supported), None)
        if m is None:
            raise SystemExit("没有可用模型")
        backend, threads, notes = choose_backend(
            m, getattr(args, "prefer", None) or "balanced",
            backend_arg=args.backend, threads_arg=args.threads)
        srv_model = m
        for n in notes:
            print(f"  · {n}")

    prof = getattr(srv_model, "profile", None) or {}
    print(f"selfcheck：{srv_model.name}  后端={BACKENDS[backend]['desc']}"
          + (f"  档案={prof.get('_file')}" if prof.get("_file") else ""))
    srv = Server(srv_model, backend, args.ctx, threads,
                 extra=(args.extra.split() if args.extra else None), verbose=args.verbose)
    install_cleanup(srv)
    result = {
        "tool": "draco selfcheck", "schema_version": 1, "date": t_start,
        "model": {"name": srv_model.name, "arch": getattr(srv_model, "arch", "?"),
                  "size_gb": round(getattr(srv_model, "size", 0) / 1e9, 2),
                  "id": getattr(srv_model, "tag", None) or srv_model.path},
        "backend": backend,
        "profile_file": prof.get("_file"),
        "host": {"uname": " ".join(os.uname())[:120]},
    }
    try:
        dt = srv.wait_ready()
        srv.warmup()              # ★ 必须在量指纹之前：否则第一轮永远是"进程第一次解码"
        result["load_seconds"] = round(dt, 1)
        print(f"就绪（{dt:.1f}s，已预热）\n")

        # ── (a) 可复现性：同一组固定 prompt 跑两轮（贪心），比指纹 ──
        #   ★ 两轮的**全文**都进 JSON —— 不一致时得能看出"哪一条、怎么不一样"，
        #     否则这个字段对排障没用（实测第一次跑 falcon-h1 就报了不一致）。
        fps = []
        for rnd in (1, 2):
            texts = []
            for _, p, n in _SELFCHECK_PROMPTS:
                txt, _ = _complete_once(srv.url, [{"role": "user", "content": p}],
                                        0.0, 42, n)
                texts.append(txt)
            fps.append(texts)
        stable = fps[0] == fps[1]
        import hashlib
        fp = hashlib.sha256("\x1e".join(fps[0]).encode("utf-8")).hexdigest()[:16]
        result["reproducibility"] = {
            "stable": stable, "fingerprint": f"sha256:{fp}",
            "round1": fps[0], "round2": fps[1],
            "diff_index": [i for i in range(len(fps[0])) if fps[0][i] != fps[1][i]],
        }
        print(f"[1] 可复现性：{'✅ 两轮逐字一致' if stable else '⚠ 两轮不一致'}"
              + ("" if stable else "（可能是后端非确定性，也可能是**该架构在此后端上对请求顺序敏感**"
                                   " —— 实测 falcon-h1 在 iGPU/Vulkan 不一致、同一模型在 CPU 一致）")
              + f"  指纹={fp}")
        for i, (tag, t) in enumerate(zip(("repro-1", "repro-2", "repro-3"), fps[0])):
            mark = "" if stable or fps[0][i] == fps[1][i] else "   ← 与本轮不同"
            print(f"      {tag}: {t[:48]!r}{mark}")
            if mark:
                print(f"      {'':7s}  第二轮: {fps[1][i][:48]!r}")

        # ── (b) 健康问句（instruct 模型才有意义）──
        #   ★ 两个坑（ZAYA 上实测出来的）：
        #     ① 推理模型会把答案放在 <think>...</think> **之后** ⇒ 预算太小会截在思考里，
        #        判据却去 greps 可见文本 ⇒ 假失败（我用 32 token 时 ZAYA 0/3，其实是没轮到答案）；
        #     ② 判据要把思考块剥掉再匹配，否则思考里引用了问题也会误判"通过"。
        sanity = []
        for p, expect in _SELFCHECK_SANITY:
            txt, _ = _complete_once(srv.url, [{"role": "user", "content": p}],
                                    0.0, 42, 512)          # 给推理留足预算
            vis, st = _strip_think(txt)
            ok = st != "thinking" and any(e in vis for e in expect)
            sanity.append({"prompt": p, "expect_any": expect, "got": txt[:200],
                           "visible": vis[:64], "think_state": st, "pass": ok})
            mark = "✅" if ok else ("⚠" if st == "thinking" else "❌")
            note = "（仍在上思考：预算不足，未给答案）" if st == "thinking" else ""
            print(f"[2] 健康问句：{mark}  {p[:36]!r} → {(vis or txt)[:60]!r}{note}")
        result["sanity"] = sanity
        n_ok = sum(1 for s in sanity if s["pass"])
        n_th = sum(1 for s in sanity if s.get("think_state") == "thinking")
        print(f"    小结 {n_ok}/{len(sanity)}"
              + (f"，另有 {n_th} 条预算不足（思考未结束）" if n_th else "")
              + "（base/非 instruct 模型可能天然失败，仅参考）")

        # ── (c) 速度：固定 6 次请求的 server 侧计时中位 ──
        tps, pps = [], []
        for _, p, n in _SELFCHECK_PROMPTS:
            _, tim = _complete_once(srv.url, [{"role": "user", "content": p}],
                                    0.0, 42, n)
            if tim.get("predicted_per_second"):
                tps.append(tim["predicted_per_second"])
            if tim.get("prompt_per_second"):
                pps.append(tim["prompt_per_second"])
        med = lambda v: sorted(v)[len(v) // 2] if v else None
        result["speed"] = {"decode_tps_median": round(med(tps), 1) if tps else None,
                           "prompt_tps_median": round(med(pps), 1) if pps else None,
                           "samples": len(tps)}
        if tps:
            print(f"[3] 速度：decode 中位 {med(tps):.1f} tok/s"
                  + (f"，prompt 中位 {med(pps):.1f} tok/s" if pps else "")
                  + f"（{len(tps)} 次采样）")
        else:
            print("[3] 速度：服务端未给计时字段，跳过")
        result["command"] = " ".join(srv.cmd)
    finally:
        srv.stop()

    print("\n—— —— ——")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON 已写 {args.json}（贴进 issue 的「selfcheck 结果」一项）")
    else:
        print("（加 --json <路径> 可导出可贴进 issue 的 JSON）")



# ─────────────────────────── 子命令：chat ───────────────────────────

BANNER = r"""
   ____                              Dracomancer
  |  _ \ _ __ __ _  ___ ___  _ __ ___   _ __   __ _ _ __   ___ ___
  | | | | '__/ _` |/ __/ _ \| '_ ` _ \ | '_ \ / _` | '_ \ / __/ _ \
  | |_| | | | (_| | (_| (_) | | | | | || | | | (_| | | | | (_)  __/
  |____/|_|  \__,_|\___\___/|_| |_| |_||_| |_|\__,_|_| |_|\___\___|
"""


def apply_profile_defaults(args, model):
    """把注册表档案的采样默认值套到 chat 参数上（只补用户没显式给的）。

    ★ 判断"没显式给"的办法是与 argparse 默认值比对 —— 简陋但对这四个标量够用；
    若以后默认值变了，这里的字面量要跟着改（就在 argparse 定义旁边）。"""
    sp = (getattr(model, "profile", None) or {}).get("sampling") or {}
    if not sp:
        return
    if args.temp == 0.7 and "temp" in sp:
        args.temp = sp["temp"]
    if args.repeat_penalty == 1.1 and "repeat_penalty" in sp:
        args.repeat_penalty = sp["repeat_penalty"]
    if not args.max_tokens and "max_tokens_default" in sp:
        args.max_tokens = sp["max_tokens_default"]
    args.think_default = sp.get("think_default", False)
    # ★ 档案也可给"推荐的思考流式粒度"（sampling.think_hold）—— 开箱即用，不必让用户记环境变量。
    #   命令行没有 /hold 之前用它作为初始值（用户仍可用 /hold 覆盖）。
    args.think_hold = sp.get("think_hold") or None


def cmd_family(args):
    """家族级合规闸门（实现在 family_gate.py；这里只做 CLI 包装，避免逻辑两份）。

    与 selfcheck 的分工：selfcheck 答「这个模型现在跑起来对不对」（单模型、单后端）；
    family 答三个结构性问题：跨后端是否同答 / 跨量化是否一致 / 长上下文是否退化。
    """
    import family_gate
    argv = [args.model]
    if args.backends:
        argv += ["--backends", args.backends]
    argv += ["--ctx", str(args.ctx)]
    if args.json:
        argv += ["--json", args.json]
    old = sys.argv
    sys.argv = ["draco family"] + argv
    try:
        return family_gate.main()
    finally:
        sys.argv = old


def cmd_chat(args):
    # ── NPU 后端：模型走 FLM 的 tag 命名空间，不走 GGUF 发现 ──
    if args.backend == "npu":
        m = pick_flm(args.model)
        apply_profile_defaults(args, m)
        threads = args.threads or BACKENDS["npu"]["threads_default"]
        print(f"\n装载 {m.tag}（FastFlowLM·NPU） ctx=FLM 自管（KV 容量 131k）")
        srv = Server(m, "npu", args.ctx, threads,
                     extra=(args.extra.split() if args.extra else None), verbose=args.verbose)
        install_cleanup(srv)
        try:
            dt = srv.wait_ready()
            srv.warmup()          # 占掉"进程第一次解码"（见 Server.warmup 的实测记录）
            if sys.stderr.isatty():
                sys.stderr.write("\r" + " " * 40 + "\r")
            print(f"就绪（{dt:.1f}s）。输入消息回车发送；/help 看命令，/bye 退出。\n")
            loop_chat(srv, args)
        finally:
            srv.stop()
        return

    ms = discover()
    m = pick_model(ms, args.model)
    backend = args.backend
    prefer = getattr(args, "prefer", None) or "balanced"
    if m is None:
        print(BANNER)
        print("选择模型：")
        for i, x in enumerate(ms):
            mark = "" if x.supported else "   ← llama.cpp 不支持"
            print(f"  [{i}] {x.line()}{mark}")
        sel = input("\n序号（回车=最小可用）: ").strip()
        m = pick_model(ms, sel) if sel else next((x for x in ms if x.supported), None)
        if m is None:
            raise SystemExit("没有可用的模型")
        if not m.supported:
            raise SystemExit(f"'{m.name}' 架构 {m.arch} llama.cpp 不支持")
        if backend is None:
            if m.arch in _NEEDS_LOCAL or m.needs_local_build():
                hint = "local（该架构只有本地构建有实现）"
            elif profile_backend(m):
                hint = f"{profile_backend(m)}（模型档案推荐的默认后端）"
            else:
                hint = "cpu/igpu"
            b = input(f"后端 [{hint}]（回车=默认）: ").strip().lower()
            backend = b if b in BACKENDS else None
    # ★ 后端/线程的最终裁决走 choose_backend（含 --prefer 倾向）
    backend, threads, notes = choose_backend(
        m, prefer, backend_arg=backend, threads_arg=args.threads)

    # ★ 支持性检查必须在"装载…"提示**之前** —— 否则会先打"装载 X"再报"X 跑不了"，
    #   自相矛盾（我实测看到过）。档案标 broken 的情况同理（也走这条更可读的理由）。
    _broken = broken_reason(m)
    if _broken:
        raise SystemExit(f"'{m.name}' 在当前引擎上不可用 —— {_broken}")
    if not m.supported:
        raise SystemExit(
            f"'{m.name}'（架构 {m.arch}）llama.cpp b10819 不支持，cpu/igpu 都跑不了。\n"
            f"  这类模型需要我们自己的内核（见 STAGE1_NPU.md）。"
            f"用 `draco.py list` 看哪些能跑。")
    apply_profile_defaults(args, m)
    for n in notes:
        print(f"  · {n}")
    print(f"\n装载 {m.name}（{m.size/1e9:.2f} GB） 后端={BACKENDS[backend]['desc']} "
          f"ctx={args.ctx} threads={threads}"
          + (f"  [档案 {m.profile.get('_file')}]" if m.profile else ""))
    srv = Server(m, backend, args.ctx, threads,
                 extra=(args.extra.split() if args.extra else None), verbose=args.verbose)
    install_cleanup(srv)
    try:
        dt = srv.wait_ready()
        if sys.stderr.isatty():
            sys.stderr.write("\r" + " " * 40 + "\r")
        print(f"就绪（{dt:.1f}s）。输入消息回车发送；/help 看命令，/bye 退出。\n")
        loop_chat(srv, args)
    finally:
        srv.stop()


HELP = """斜杠命令：
  /bye 或 /exit    退出（也可 Ctrl-D）
  /clear           清空对话历史
  /params          显示当前参数
  /temp <x>        改温度（0 = 贪心）
  /maxtok <n>      改单次最多生成 token（默认 2048，0 = 不限）
  /think on|off|auto  开/关思考模式（默认=模板决定）
  /hold line|token|dup  思考流式粒度（line=按行挂起(默认)/token=完全逐 token/dup=直出+补发末行）
  （line=按行挂起(默认) / token=完全逐 token 直出 / dup=直出+收尾补发末行）—— 见 README
  /rep <x>         重复惩罚（默认 1.1；1.0 = 关）
  /system <文本>   设置/清除系统提示
  /help            本帮助"""


def loop_chat(srv, args):
    msgs, temp, seed = [], args.temp, args.seed
    maxtok, system = args.max_tokens or 2048, args.system
    # ★ 默认**关掉思考模式**：这类模型开着思考会先烧掉 ~2000 token（56 t/s 下 ≈36 s）
    #   再给答案，对聊天界面是坏体验（Ollama 式界面不会让你等这个）。/think on 可开。
    #   不支持该开关的模板会忽略这个 kwargs（已用 Llama-3.2 验证无害）。
    #   注册表档案可用 sampling.think_default 覆盖这个默认。
    think = getattr(args, "think_default", False)
    hold = getattr(args, "think_hold", None)   # 档案推荐值；/hold 可覆盖（仅 dengine 生效）
    rep = args.repeat_penalty
    in_think = [False]
    while True:
        try:
            user = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not user:
            continue
        if user.startswith("/"):
            c, _, rest = user[1:].partition(" ")
            c, rest = c.lower(), rest.strip()
            if c in ("bye", "exit", "q"):
                print("再见。")
                return
            elif c == "help":
                print(HELP)
            elif c == "clear":
                msgs.clear(); print("历史已清空。")
            elif c == "params":
                print(f"  温度={temp}  重复惩罚={rep}  seed={seed}  最多生成={maxtok or '不限'}  "
                      f"ctx={args.ctx}  后端={srv.backend}  模型={srv.model.name}")
                print(f"  系统提示={system or '(无)'}")
                if srv.backend == "dengine":
                    print(f"  思考流式粒度={hold or '(引擎默认 line)'}   （/hold line|token|dup 可改）")
                think_txt = "开" if think else "关"
                print(f"  思考模式={think_txt or '(模板默认)'}   （/think on|off|auto）")
            elif c == "temp":
                try:
                    temp = float(rest); print(f"  温度 → {temp}")
                except ValueError:
                    print("  用法：/temp 0.7")
            elif c == "maxtok":
                try:
                    maxtok = int(rest); print(f"  最多生成 → {maxtok or '不限'}")
                except ValueError:
                    print("  用法：/maxtok 256")
            elif c == "system":
                system = rest or None; print(f"  系统提示 → {system or '(已清除)'}")
            elif c == "rep":
                try:
                    rep = float(rest); print(f"  重复惩罚 → {rep}")
                except ValueError:
                    print("  用法：/rep 1.1（1.0 = 关闭）")
            elif c in ("hold", "粒度"):
                if rest not in ("line", "token", "dup"):
                    print("  用法：/hold line（按行挂起，默认）| token（完全逐 token 直出）| "
                          "dup（直出 + 收尾补发末行）")
                else:
                    hold = rest
                    print(f"  思考流式粒度 → {rest}（仅自研引擎 dengine 生效，其它后端忽略）")
            elif c == "think":
                v = rest.lower()
                think = None if v in ("", "auto") else (v in ("on", "1", "true", "yes"))
                print(f"  思考模式 → {'模板默认' if think is None else ('开' if think else '关')}")
            else:
                print(f"  未知命令 /{c}；/help 看帮助")
            continue

        msgs.append({"role": "user", "content": user})
        print()
        buf, rbuf, t0, t_last, ntok, timings = [], [], time.time(), None, 0, {}
        dim = "\x1b[2m" if sys.stdout.isatty() else ""
        rst = "\x1b[0m" if sys.stdout.isatty() else ""
        try:
            for kind, delta, tim in stream_chat(
                    srv.url, msgs, temp, seed, maxtok, system, think, rep, hold):
                if delta:
                    if t_last is None:
                        t_last = time.time()          # 首 token 到达
                    if kind == "reasoning":
                        if not in_think[0]:
                            sys.stdout.write(f"{dim}[思考] "); in_think[0] = True
                        sys.stdout.write(delta)
                    else:
                        if in_think[0]:
                            sys.stdout.write(f"{rst}\n"); in_think[0] = False
                        sys.stdout.write(delta)
                    sys.stdout.flush()
                    (rbuf if kind == "reasoning" else buf).append(delta)
                    ntok += 1
                if tim:
                    timings = tim
            if in_think[0]:
                sys.stdout.write(rst); in_think[0] = False
        except urllib.error.HTTPError as e:
            # ★ 把服务端的正文打出来：桥接/llama-server 的报错正文里才有真原因
            #   （实测：ZAYA 上下文超限时只显示 "HTTP Error 400" 是没法排查的）
            try:
                body = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
            print(f"\n[错误] 请求失败：HTTP {e.code}\n  {body}")
            msgs.pop()
            continue
        except urllib.error.URLError as e:
            print(f"\n[错误] 请求失败：{e}")
            msgs.pop()
            continue
        print()
        # ★ 撞到上限要**说出来**：否则用户以为"模型只能输出这么多"。
        lim = (timings.get("predicted_n") or ntok) if timings else ntok
        if maxtok and lim >= maxtok:
            print(f"{dim}[已达单次上限 max_tokens={maxtok}（回答可能在半句被截断）"
                  f"；用 /maxtok 4096 调大，或 /maxtok 0 设为不限]{rst}")
        reply, reason = "".join(buf), "".join(rbuf)
        # ★★ 必须把 reasoning_content 一起带回历史（2026-09-15 用户报"ZAYA 第一问会思考，
        #    接下来就不思考了"）。原因在模板里：ZAYA/Bailing 这类模板对"没有 reasoning_content 的
        #    assistant 历史"会渲染成**空 think 块**（`<think>\n</think>\n\n{content}`），
        #    模型看到"上一轮我没思考直接答了"就跟着模仿 —— 于是第二问起思考归零。
        #    实测（同一模型、同一问题）：历史只带 content → 思考 0 字符；带 reasoning_content → 思考 565 字符。
        if reply or reason:
            m = {"role": "assistant", "content": reply}
            if reason:
                m["reasoning_content"] = reason
            msgs.append(m)
        # 速度：优先用 server 给的 timings，否则用本地计时
        if timings.get("predicted_per_second"):
            gps = timings["predicted_per_second"]
            pps = timings.get("prompt_per_second") or 0
            n = timings.get("predicted_n", ntok)
            print(f"  [{n} tok, {gps:.1f} t/s"
                  + (f", prompt {pps:.1f} t/s" if pps else "") + "]")
        elif t_last and ntok:
            dt = time.time() - t_last
            if dt > 0:
                print(f"  [{ntok} 块, ~{ntok/dt:.1f} t/s]")


# ─────────────────────────── 子命令：serve ───────────────────────────

def cmd_serve(args):
    # ── NPU 后端：同 cmd_chat 的分支理由 ──
    if args.backend == "npu":
        m = pick_flm(args.model)
        srv = Server(m, "npu", args.ctx, 8,
                     extra=(args.extra.split() if args.extra else None), verbose=args.verbose)
        install_cleanup(srv)
        try:
            dt = srv.wait_ready()
            srv.warmup()
            print(f"就绪（{dt:.1f}s）  模型={m.tag}")
            print(f"\n  OpenAI 兼容 API : {srv.url}/v1/chat/completions", flush=True)
            print(f"  FLM 自带 Web UI : {srv.url}/", flush=True)
            print("\nCtrl-C 退出（会一并关掉服务端子进程）\n", flush=True)
            while True:
                time.sleep(1)
                if srv.proc.poll() is not None:
                    print("服务端已退出。")
                    return
        except KeyboardInterrupt:
            print("\n关闭服务端…")
        finally:
            srv.stop()
        return

    ms = discover()
    m = pick_model(ms, args.model) or next((x for x in ms if x.supported), None)
    if m is None:
        raise SystemExit("没有可用模型")
    backend, threads, notes = choose_backend(
        m, getattr(args, "prefer", None) or "balanced",
        backend_arg=args.backend, threads_arg=args.threads)
    print(f"装载 {m.name}（{m.size/1e9:.2f} GB） 后端={BACKENDS[backend]['desc']}")
    for n in notes:
        print(f"  · {n}")
    srv = Server(m, backend, args.ctx, threads,
                 extra=(args.extra.split() if args.extra else None), verbose=args.verbose)
    install_cleanup(srv)
    try:
        dt = srv.wait_ready()
        print(f"就绪（{dt:.1f}s）")
        print(f"\n  Web UI : {srv.url}/", flush=True)
        print(f"  OpenAI 兼容 API : {srv.url}/v1/chat/completions", flush=True)
        print("  提示：用浏览器打开 Web UI；curl 需要 --compressed（内置 UI 要 gzip）",
              flush=True)
        print("\nCtrl-C 退出（会一并关掉服务端子进程）\n", flush=True)
        while True:
            time.sleep(1)
            if srv.proc.poll() is not None:
                print("服务端已退出。")
                return
    except KeyboardInterrupt:
        print("\n关闭服务端…")
    finally:
        srv.stop()


# ─────────────────────────── main ───────────────────────────

def main():
    # ★ 重定向到文件/管道时 Python 默认块缓冲 —— `serve` 的地址会一直不刷出来，
    #   看起来像"卡住了"（我实测后台跑时日志整整空了 1 分半）。改成行缓冲。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="draco", description="Dracomancer 启动器 / 聊天 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python3 draco.py chat -m ling -b igpu --temp 0.6")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("list", help="列出模型与可跑后端")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("perf", help="本机实测性能表")
    p.set_defaults(fn=cmd_perf)

    p = sub.add_parser("caps", help="能力探针：加载之前静态判定每个模型能不能跑")
    p.add_argument("-m", "--model", help="只查一个模型（可省）")
    p.add_argument("-v", "--verbose", action="store_true", help="也打印后端/类型提示")
    p.set_defaults(fn=cmd_caps)

    p = sub.add_parser("tune", help="参数自调优：实测一小撮配置，按 J/token 排序（带缓存）")
    p.add_argument("-m", "--model", help="模型名（必填）")
    p.add_argument("-b", "--backend", choices=sorted(BACKENDS), help="后端（默认取档案推荐）")
    p.add_argument("-t", "--threads", type=int, help="线程数（默认取后端默认）")
    p.add_argument("--tokens", type=int, default=320, help="每组生成多少 token（默认 320）")
    p.add_argument("--force", action="store_true", help="忽略缓存重测")
    p.add_argument("-P", "--prefer", choices=PREFER_CHOICES,
                   help="速度-能效倾向（默认取环境变量 DRACO_PREFER，否则 balanced）："
                        "speed=挑最快后端 / eco=挑最省电后端且线程取 4 / balanced=不额外调")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("selfcheck",
                       help="固定协议体检：可复现指纹 + 健康问句 + 速度（issue 用）")
    p.add_argument("-m", "--model", help="模型名（前缀/子串，或 list 里的序号）")
    p.add_argument("-b", "--backend", choices=sorted(BACKENDS), help="后端")
    p.add_argument("-t", "--threads", type=int, help="CPU 线程数")
    p.add_argument("-c", "--ctx", type=int, default=4096, help="上下文长度（默认 4096）")
    p.add_argument("--extra", help="额外传给 llama-server 的 flag（原样透传）")
    p.add_argument("-v", "--verbose", action="store_true", help="打印服务端命令与日志")
    p.add_argument("--json", help="把结果写成 JSON 文件（可贴进 issue）")
    p.add_argument("-P", "--prefer", choices=PREFER_CHOICES,
                   help="速度-能效倾向（默认取环境变量 DRACO_PREFER，否则 balanced）："
                        "speed=挑最快后端 / eco=挑最省电后端且线程取 4 / balanced=不额外调")
    p.set_defaults(fn=cmd_selfcheck)

    p = sub.add_parser("family",
                       help="家族级合规闸门：跨后端一致 / 跨量化 / 长上下文退化曲线")
    p.add_argument("-m", "--model", required=True, help="模型名（前缀/子串）")
    p.add_argument("--backends", help="逗号分隔（默认 cpu,igpu,dengine）")
    p.add_argument("-c", "--ctx", type=int, default=2048, help="上下文长度（默认 2048）")
    p.add_argument("--json", help="把结果写成 JSON（可贴进 issue）")
    p.set_defaults(fn=cmd_family)

    for name, fn, h in (("chat", cmd_chat, "交互式聊天"), ("serve", cmd_serve, "起 HTTP 服务（带 Web UI）")):
        p = sub.add_parser(name, help=h)
        p.add_argument("-m", "--model", help="模型名（前缀/子串，或 list 里的序号）")
        p.add_argument("-b", "--backend", choices=sorted(BACKENDS), help="后端")
        p.add_argument("-t", "--threads", type=int, help="CPU 线程数")
        p.add_argument("-c", "--ctx", type=int, default=4096, help="上下文长度（默认 4096）")
        p.add_argument("--extra", help="额外传给 llama-server 的 flag（原样透传）")
        p.add_argument("-v", "--verbose", action="store_true", help="打印服务端命令与日志")
        p.add_argument("-P", "--prefer", choices=PREFER_CHOICES,
                       help="速度-能效倾向（默认取环境变量 DRACO_PREFER，否则 balanced）："
                            "speed=挑最快后端 / eco=挑最省电后端且线程取 4 / balanced=不额外调")
        if name == "chat":
            p.add_argument("--temp", type=float, default=0.7, help="温度（0=贪心）")
            p.add_argument("--seed", type=int, default=-1, help="随机种子（-1=随机）")
            p.add_argument("--max-tokens", type=int, default=0, help="单次最多生成（0=不限）")
            p.add_argument("--system", "-sys", help="系统提示")
            p.add_argument("--repeat-penalty", type=float, default=1.1,
                           help="重复惩罚（默认 1.1；低温度下防重复循环，1.0=关）")
        p.set_defaults(fn=fn)

    a = ap.parse_args()
    # ★ 倾向默认值：环境变量 DRACO_PREFER（便于 shell 里一次性设定，如 `export DRACO_PREFER=eco`）
    _envp = (os.environ.get("DRACO_PREFER") or "").strip().lower()
    if _envp in PREFER_CHOICES and getattr(a, "prefer", None) is None:
        a.prefer = _envp
    if not a.cmd:
        ap.print_help()
        return 0
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())

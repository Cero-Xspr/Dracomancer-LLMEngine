#!/usr/bin/env python3
"""archspec_spike —— 可行性 spike：**声明式架构描述 + 通用求值器**能不能复现真模型的数值？

要回答的问题（决定"自动兼容万物"值不值得投）：
  1. 一个**声明式**的架构描述（没有一行手写的融合内核）能不能跑出正确的数值？
  2. 描述里**必须显式写出**的"语义事实"有多少？其中多少是**形状看不出来**的
     （= 自动生成器不可能从 GGUF 推出来，只能由人或 agent 写清楚）？
  3. 出错的代价是什么形态？（能不能自检出来）

做法：用 numpy 写一个**通用求值器** + 一份 llama 架构的声明式描述，跑 SmolLM2-135M，
与 llama.cpp 的 logits 对账。不强求性能（135M 用 numpy 秒级），只求**语义正确**。

对账口径：logits cos 与 top-1。llama.cpp 用 Q4_K/Q5_0/Q8_0 权重（激活量化成 Q8_K），
我们用自己的反量化 + f32 矩阵乘 ⇒ 天然有 ~1e-3 级差，所以判据是 **cos ≳0.99 且 top-1 一致**，
不是逐位相等（与 ZAYA 对账同一套判据）。

用法：
  python3 archspec_spike.py <smol.gguf> [--tokens 1,2,3] [--ref /tmp/spec_ref] [--quant-act]

═══════════════════ 2026-09-13 spike 结论（决定"自动兼容"投不投）═══════════════════
**结论：可行，但形态必须是「声明式描述 + 通用求值器 + 强制逐算子对账」，不能是"让算法猜架构"。**

一、**能跑对**：一份声明式描述 + 一个 numpy 通用求值器（**零 llama 专属代码**）跑 SmolLM2-135M，
    与 llama.cpp 对账（自然句子提示，8 token）：
      · **logits cos = 0.999962**，top-1 **一致**（198 == 198）
      · 逐层 cos：l_out-0 0.999843、l_out-1 0.999894、l_out-5 0.999623（全程 ≥0.9996）
      · **两次运行完全相同**（不存在未初始化内存之类的抖动）
      · 逐算子核对：`rms_norm` 输出逐位相同；rope 后 K 逐对角度与源码公式一致（差 ~0.001 rad = 缓存 f16 精度）
    （残余 4e-5 与"求值器用精确 f32 反量化权重、llama.cpp 用 5bit 权重 + 量化激活"同量级，
      是**构造性**差异，不是语义错。）

二、**语义事实必须显式写：8 条，全部"形状看不出来"**——rope 用哪种旋转（llama=NORM 连续成对、
    ZAYA=NEOX 半分裂）、旋转维数、KV 怎么重复、attn scale 的分母、norm eps、残差顺序、
    lm_head 是否共享词表、FFN 激活形式。**其中 3 条 GGUF 里根本没有**（rope 类型只能从
    "架构→引擎"的表里拿）⇒ **自动生成器不可能从权重推出来**。

三、**错法的代价 = 静默算错**，本 spike 亲手抓到 **2 个真 bug**（都在 rope 一步，都是"形状合法"）：
      ① rope 必须**逐头**做；我第一版把整个 q[576]/k[192] 当一个 64 维向量转 ⇒ 只转了第 0 个头。
         症状：K 整体 cos 0.9826（3 头只对 1 头）、注意力输出偏 12%、逐层 cos 掉到 0.68。
         ★ 而"只看第 0 个头"的逐对检查**完全看不出** —— 要靠整体 cos + 逐层漂移。
      ② 头数靠"x.size % head_dim == 0"**猜**：k 有 3 个头也算整除，被切成 9 段 21 元素，
         余下元素落进 `np.empty_like` 的**未初始化内存** ⇒ 同一命令两次跑出不同结果。
   两条都说明：这个机制的安全阀是**对账**，不是生成。

四、**对账抓手现成**：llama.cpp 自己 via `cb_eval` 就能 dump `attn_norm / kqv_out / cache_k_l / l_out`
   （ZAYA 那套方法直接复用），所以"逐算子对账"的成本很低 —— 这是投它的最大理由。

五、**投法建议**：先做**声明式描述覆盖 3~4 个已有架构**的表达力验证（本 spike 已证 llama 可行），
   把"语义事实"做成**必填项**（缺了就报错，而不是猜默认值），再考虑通用求值器的性能实现。
   不要先写映射器，也不要让任何自动环节决定语义。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")
import gguf  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# 一、声明式描述：**语义事实**与**结构步骤**分开写
# ═══════════════════════════════════════════════════════════════════════
# ★ 这个分割就是 spike 的结论所在：
#   · "结构"（哪些张量、按什么顺序乘）**可以从 GGUF 的张量名+形状推出来**；
#   · "语义"（rope 用哪种旋转、KV 怎么重复、残差在哪、norm 的 eps…）**推不出来**，
#     必须由描述显式给出。标 shape_invisible=True 的那些是**永远无法从形状推断**的。

SEMANTICS_LLAMA = {
    "rope_type": {
        "value": "NORM",
        "why": "llama 架构用 ggml 的 GGML_ROPE_TYPE_NORMAL = 连续成对 (i,i+1) 旋转；"
               "而 NEOX 是半分裂 (i, i+d/2)。ZAYA 就是 NEOX。",
        "shape_invisible": True,
        "how_found": "读 ggml-cpu/ops.cpp 的 rotate_pairs(n_dims, 1, ...) 与 "
                     "llama-model.cpp:2901 `case LLM_ARCH_LLAMA: return LLAMA_ROPE_TYPE_NORM`",
    },
    "rope_dims": {
        "value": "n_rot（=GGUF 的 rope.dimension_count，未给则 head_dim）",
        "why": "只转前 n_rot 维；剩余通道原样复制。",
        "shape_invisible": True,
        "how_found": "GGUF 的 llama.rope.dimension_count",
    },
    "kv_repeat": {
        "value": "repeat_interleave 风格：query head h 用 kv head h // (n_head/n_head_kv)",
        "why": "GQA 的重复方式有两种（连续块 or 取模交错），数值不同。",
        "shape_invisible": True,
        "how_found": "llama.cpp 源码 + ZAYA 上踩过的坑（轴序写反 ⇒ 形状对、静默算错）",
    },
    "attn_scale": {"value": "1/sqrt(head_dim)", "why": "注意是 head_dim 不是 n_rot",
                   "shape_invisible": True, "how_found": "源码"},
    "norm_eps": {"value": "GGUF 的 attention.layer_norm_rms_epsilon",
                 "why": "norm 里的 eps 直接决定数值", "shape_invisible": True,
                 "how_found": "GGUF"},
    "residual_order": {"value": "pre-norm：x += attn(norm(x))；x += ffn(norm(x))",
                       "why": "post-norm 也合法，结果不同", "shape_invisible": True,
                       "how_found": "源码"},
    "lm_head": {"value": "tied：复用 token_embd（GGUF 里没有 output.weight）",
                "why": "另有一份 output.weight 也合法", "shape_invisible": True,
                "how_found": "GGUF 张量清单里没有 output.weight"},
    "ffn_act": {"value": "silu(gate) * up",
                "why": "gelu/relu2/无门控都合法", "shape_invisible": True, "how_found": "架构约定"},
}

# 结构步骤：求值器按顺序执行；权重绑定写成 (角色, 张量名模板)
LAYER_STEPS = [
    ("rms_norm",   {"src": "x", "w": "blk.{i}.attn_norm.weight", "out": "xn"}),
    ("mul_mat",    {"w": "blk.{i}.attn_q.weight", "src": "xn", "out": "q"}),
    ("mul_mat",    {"w": "blk.{i}.attn_k.weight", "src": "xn", "out": "k"}),
    ("mul_mat",    {"w": "blk.{i}.attn_v.weight", "src": "xn", "out": "v"}),
    # ★ 头数**显式声明**：q 用 n_head，k/v 用 n_head_kv。第一版我按"x.size % head_dim == 0"
    #   去猜 —— k 有 3 个头（192 维）也被"整除"成 9 段（hd=21），剩下 3 个元素写进
    #   np.empty_like 的**未初始化内存** ⇒ 同一命令两次跑出不同结果。形状合法、静默错。
    ("rope",       {"src": "q", "pos": "pos", "out": "q", "heads": "n_head"}),
    ("rope",       {"src": "k", "pos": "pos", "out": "k", "heads": "n_head_kv"}),
    ("attn_gqa",   {"q": "q", "k": "k", "v": "v", "cache": True, "out": "a"}),
    ("mul_mat",    {"w": "blk.{i}.attn_output.weight", "src": "a", "out": "o"}),
    ("add",        {"a": "x", "b": "o", "out": "x"}),
    ("rms_norm",   {"src": "x", "w": "blk.{i}.ffn_norm.weight", "out": "xn2"}),
    ("mul_mat",    {"w": "blk.{i}.ffn_gate.weight", "src": "xn2", "out": "g"}),
    ("mul_mat",    {"w": "blk.{i}.ffn_up.weight", "src": "xn2", "out": "u"}),
    ("silu_mul",   {"g": "g", "u": "u", "out": "h"}),
    ("mul_mat",    {"w": "blk.{i}.ffn_down.weight", "src": "h", "out": "d"}),
    ("add",        {"a": "x", "b": "d", "out": "x"}),
]


# ═══════════════════════════════════════════════════════════════════════
# 二、通用求值器（op 只管张量，不含任何 llama 专属逻辑）
# ═══════════════════════════════════════════════════════════════════════

class Ctx:
    """声明式求值上下文：命名张量 + 几何 + 语义 + 逐 token 的 KV 缓存。"""

    def __init__(self, geo, sem):
        self.geo, self.sem = geo, sem
        self.t = {}
        self.kcache, self.vcache = [], []      # 每层一段

    def get(self, name, il):
        return self.t[self._rm(name, il)]

    def put(self, name, il, val):
        self.t[self._rm(name, il)] = val

    @staticmethod
    def _rm(name, il):
        return name.replace("{i}", str(il))


def q8_round(x):
    """把激活按 32 元素一块量化成 Q8（scale=max|x|/127），模仿 llama.cpp 的
    `vec_dot_type`：它做量化权重×量化激活的点积，我这边是精确 f32 —— 这个差就是主噪声源。"""
    y = x.astype(np.float32).copy()
    n = y.size
    for b in range(0, n, 32):
        blk = y[b:b + 32]
        d = np.abs(blk).max() / 127.0
        if d > 0:
            y[b:b + 32] = np.round(blk / d) * d
    return y


def _rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean() + eps) * w


def _rope(x, pos, n_rot, theta_base, mode):
    """两种旋转：NORM=连续成对；NEOX=半分裂。★ 差别就在这里，形状完全一样。"""
    d = x.size
    n = min(n_rot, d)
    half = n // 2
    inv = 1.0 / (theta_base ** (np.arange(half, dtype=np.float64) * 2.0 / n))
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    y = x.astype(np.float64).copy()
    if mode == "NORM":
        for j in range(half):                  # 连续成对 (2j, 2j+1)
            a, b = y[2 * j], y[2 * j + 1]
            y[2 * j], y[2 * j + 1] = a * c[j] - b * s[j], a * s[j] + b * c[j]
    elif mode == "NEOX":
        for j in range(half):                  # 半分裂 (j, j+half)
            a, b = y[j], y[j + half]
            y[j], y[j + half] = a * c[j] - b * s[j], a * s[j] + b * c[j]
    else:
        raise ValueError(mode)
    return y.astype(np.float32)


class Evaluator:
    """把 (语义, 结构步骤) 解释成计算。**没有一行 llama 专属代码**。"""

    def __init__(self, wget, geo, sem):
        self.wget, self.geo, self.sem = wget, geo, sem

    quant_act = False                           # 打开则模仿 llama.cpp 的激活量化

    def _mm(self, wname, x):
        W = self.wget(wname)                    # [out, in]
        if self.quant_act:
            x = q8_round(x)
        return W @ x                            # x: [in] → [out]

    def run(self, ctx, il, pos, step, args):
        g, s = self.geo, self.sem
        op = step
        # 步骤里的 {i} 要替换成层号（权重名与张量名都要）
        args = {k: (v.replace("{i}", str(il)) if isinstance(v, str) else v)
                for k, v in args.items()}
        gv = lambda k: ctx.get(args[k], il)
        if op == "rms_norm":
            ctx.put(args["out"], il, _rms_norm(gv("src"), self.wget(args["w"]), g["eps"]))
        elif op == "mul_mat":
            ctx.put(args["out"], il, self._mm(args["w"], gv("src")))
        elif op == "rope":
            # ★★★ 必须**逐头**旋转！我第一版把整个 q[576]/k[192] 当成一个 64 维向量转，
            #   于是只有第 0 个头被转了、其余头原样保留 —— 形状完全合法，数值静默错。
            #   症状（就是靠它抓出来的）：K 整体 cos 0.9826（3 个头只对 1 个）、
            #   注意力输出偏 12%；而"只看第 0 个头"的逐对角度检查完全看不出问题。
            mode = s["rope_type"]["value"].split()[0]
            x = gv("src")
            nh = int(g[args["heads"]])                  # 头数由描述给出，不猜
            hd = x.size // nh
            assert hd * nh == x.size, f"head 数 {nh} 与向量长度 {x.size} 不整除"
            out = x.copy()                              # ★ 不用 empty：宁可保留原值也不要未初始化内存
            for h in range(nh):
                out[h * hd:(h + 1) * hd] = _rope(x[h * hd:(h + 1) * hd], pos,
                                                 g["n_rot"], g["rope_base"], mode)
            ctx.put(args["out"], il, out)
        elif op == "silu_mul":
            gate = gv("g")
            ctx.put(args["out"], il, (gate / (1 + np.exp(-gate))) * gv("u"))
        elif op == "add":
            ctx.put(args["out"], il, gv("a") + gv("b"))
        elif op == "attn_gqa":
            hd, nh, nkv = g["head_dim"], g["n_head"], g["n_head_kv"]
            q = gv("q").reshape(nh, hd)
            k = gv("k").reshape(nkv, hd)
            v = gv("v").reshape(nkv, hd)
            while len(ctx.kcache) <= il:
                ctx.kcache.append([]); ctx.vcache.append([])
            ctx.kcache[il].append(k); ctx.vcache[il].append(v)
            K = np.stack(ctx.kcache[il])        # [T, nkv, hd]
            V = np.stack(ctx.vcache[il])
            rep = nh // nkv
            # ★ KV 重复方式：连续块（h//rep）。写成 h%rep 形状也一样，但数值全错
            Kx = K[:, [h // rep for h in range(nh)], :]     # [T, nh, hd]
            Vx = V[:, [h // rep for h in range(nh)], :]
            out = np.zeros((nh, hd), np.float64)
            for h in range(nh):
                sc = (Kx[:, h, :] @ q[h]) * g["scale"]
                sc = sc - sc.max()
                w = np.exp(sc); w /= w.sum()
                out[h] = (w[:, None] * Vx[:, h, :]).sum(0)
            ctx.put(args["out"], il, out.reshape(-1).astype(np.float32))
        else:
            raise ValueError(f"求值器不认识的操作 {op}")

    def forward(self, tokens, tok_embd, trace=None):
        ctx = Ctx(self.geo, self.sem)
        for pos, tid in enumerate(tokens):
            ctx.t = {"x": tok_embd[tid].copy()}
            for il in range(self.geo["n_layer"]):
                for step, args in LAYER_STEPS:
                    self.run(ctx, il, pos, step, args)
                if trace is not None:                      # 逐层对账用
                    trace[il] = ctx.get("x", il).copy()
            h = _rms_norm(ctx.get("x", self.geo["n_layer"] - 1),
                          self.wget(self.geo["out_norm"]), self.geo["eps"])
        logits = tok_embd @ h                  # tied
        return logits


# ═══════════════════════════════════════════════════════════════════════
# 三、权重装载（我们自己的反量化）
# ═══════════════════════════════════════════════════════════════════════

def load_weights(path):
    r = gguf.GGUFReader(path)
    f = {t.name: t for t in r.fields.values()}
    cache = {}

    def wget(name):
        if name in cache:
            return cache[name]
        t = next((x for x in r.tensors if x.name == name), None)
        if t is None:
            raise KeyError(f"GGUF 里没有张量 {name}")
        raw = np.frombuffer(t.data, dtype=np.uint8)
        w = gguf.quants.dequantize(raw, t.tensor_type).astype(np.float32)
        shp = tuple(int(x) for x in t.shape)          # 文件顺序 = ggml ne 顺序
        if len(shp) == 2:                             # ne=[in,out] → [out,in]
            w = w.reshape(shp[1], shp[0])
        else:
            w = w.reshape(shp)
        cache[name] = w
        return w

    hp = {}
    for k, t in f.items():
        try:
            v = t.contents()
        except Exception:
            continue
        if not isinstance(v, (list, bytes)):
            hp[k] = v
    geo = {
        "n_layer": int(hp["llama.block_count"]),
        "n_embd": int(hp["llama.embedding_length"]),
        "n_head": int(hp["llama.attention.head_count"]),
        "n_head_kv": int(hp["llama.attention.head_count_kv"]),
        "n_rot": int(hp.get("llama.rope.dimension_count", 0) or 0),
        "rope_base": float(hp.get("llama.rope.freq_base", 1e4)),
        "eps": float(hp["llama.attention.layer_norm_rms_epsilon"]),
        "out_norm": "output_norm.weight",
    }
    geo["head_dim"] = geo["n_embd"] // geo["n_head"]
    if not geo["n_rot"]:
        geo["n_rot"] = geo["head_dim"]
    geo["scale"] = 1.0 / np.sqrt(geo["head_dim"])
    return wget, wget("token_embd.weight"), geo, hp


# ═══════════════════════════════════════════════════════════════════════
# 四、与 llama.cpp 对账
# ═══════════════════════════════════════════════════════════════════════

GD = "/media/xiao_/OverSys1/npu-direct/hybrid/zaya_gdump"
BUILD = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"


def llama_cpp_logits(model, tokens, outdir):
    """用我们的 gdump 夹具跑同一串 token，取最后一个 token 的 logits。

    ★ 夹具本身与架构无关（只 decode + dump logits），所以能拿来当通用参照。
    """
    subprocess.run(["rm", "-rf", outdir], check=False)
    env = dict(os.environ, LD_LIBRARY_PATH=BUILD)
    r = subprocess.run([GD, model, ",".join(map(str, tokens)), outdir], env=env,
                       capture_output=True, text=True, cwd=os.path.dirname(GD), timeout=900)
    lp = os.path.join(outdir, "logits.bin")
    if not os.path.exists(lp):
        raise SystemExit(f"参照跑失败：{r.stdout[-500:]} {r.stderr[-500:]}")
    return np.fromfile(lp, np.float32)


def per_layer_compare(model, toks, trace, outdir):
    """用夹具 dump 的 l_out-{il} 做逐层对账，找**第一个发散层**。

    ★ 这套方法是从 ZAYA 那边直接搬来的：绝对 cos 说明"差多少"，**逐层**才知道"从哪一跳开始差"。
    """
    import glob
    llama_cpp_logits(model, toks, outdir)               # 顺带把层间张量 dump 出来
    # ★ 文件名里的序号是**全局计数器**，不能写死 .00000 —— 按前缀建索引（我踩过两次）
    idx = {}
    for q in glob.glob(os.path.join(outdir, "l_out-*.bin")):
        base = os.path.basename(q).split(".")[0]        # l_out-<il>
        try:
            il = int(base.rsplit("-", 1)[1])
        except ValueError:
            continue
        a = np.fromfile(q, np.float32)
        if a.size == 576:                                # 只收隐状态那一份
            idx.setdefault(il, q)
    if not idx:
        print("  （夹具没 dump 到 l_out，跳过逐层）"); return
    print(f"  {'层':<5}{'cos':>10}{'max|Δ|':>10}{'|ref|max':>10}")
    first = None
    for il in sorted(idx):
        ref = np.fromfile(idx[il], np.float32).astype(np.float64)
        mine = trace[il].astype(np.float64)
        c = float(mine @ ref / (np.linalg.norm(mine) * np.linalg.norm(ref) + 1e-30))
        d = float(np.abs(mine - ref).max())
        mark = ""
        if first is None and c < 0.999:
            first, mark = il, "   ← 第一个发散"
        if il < 6 or mark:
            print(f"  {il:<5}{c:>10.6f}{d:>10.3f}{np.abs(ref).max():>10.2f}{mark}")
    print(f"  ⇒ 第一个发散层：{first if first is not None else '无（全层 cos>0.999）'}"
          f"（共 {len(idx)} 层有 dump）")



# ═══════════════════════════════════════════════════════════════════════════
# C3：声明式规格的**完整性校验**（缺字段必须报错，不许静默走默认）
#   为什么要有它：spike 的整套主张是"语义事实写全了就能自动组合"。可如果缺字段时**悄悄取默认**，
#   得到的会是"能跑但算错"的结果 —— 那比报错危险得多（本项目已经被这类静默错误坑过多次）。
#   所以：每条事实必须带 (value, why, shape_invisible, how_found) 四件套；value 为 None 视为**未确定**。
# ═══════════════════════════════════════════════════════════════════════════
REQUIRED_FACT_KEYS = ("value", "why", "shape_invisible", "how_found")


def validate_spec(semantics, name):
    """校验一份语义事实表；不完整就抛 SystemExit（带清楚的理由，不返回 False）。"""
    bad = []
    for k, v in semantics.items():
        if not isinstance(v, dict):
            bad.append(f"{k}: 不是 {{value/why/shape_invisible/how_found}} 结构")
            continue
        for req in REQUIRED_FACT_KEYS:
            if req not in v:
                bad.append(f"{k}: 缺 {req}")
        if v.get("value") is None:
            bad.append(f"{k}: 值**未确定**（{str(v.get('why') or '')[:60]}）")
    if bad:
        raise SystemExit(f"声明式规格 {name} 不完整（{len(bad)} 处）：\n  - " + "\n  - ".join(bad)
                         + f"\n⇒ 缺字段**必须**报错，不许静默走默认（C3 纪律）。")
    return True


# ── SSM（granite-hybrid / Mamba 系）：这些事实是**已知需要、但尚未测定**的 ──
#    每条都写清"要看什么"，value=None 表示未确定 ⇒ validate_spec 会拒绝该族。
#    线索来源：我们实现 granite-hybrid 时踩过的点（见 STAGE1_NPU.md 与 granite 档案）。
SEMANTICS_SSM = {
    "layer_type_indexing": {
        "value": None,
        "why": "混合架构里哪些层是 SSM、哪些是注意力：靠 `blk.N.ssm_*` 张量存在性判断，"
               "还是靠 `ssm.layer_indices`/`layer_types` 之类的超参？",
        "shape_invisible": True,
        "how_found": "待定：读 llama-model.cpp 的 granite-hybrid 分支 + 对照 GGUF 里该类键是否存在",
    },
    "ssm_conv_width": {
        "value": None,
        "why": "因果卷积核宽度（状态深度 = width-1 帧），决定状态缓冲大小与复位语义",
        "shape_invisible": False,
        "how_found": "待定：`ssm.conv_kernel` 超参 + `ssm_conv1d.weight` 形状",
    },
    "ssm_state_reset": {
        "value": None,
        "why": "新序列开始时是否清零（对应我们引擎里 pos==0 按参考语义复位 conv/dw 状态）",
        "shape_invisible": True,
        "how_found": "待定：ggml 的 ssm 状态是每次 llama_decode 传入的；零初始化由调用方决定",
    },
    "ssm_dt_rank": {
        "value": None,
        "why": "dt（时间步）投影的秩，影响 `ssm_dt.bias` 的形状与 softplus 的应用位置",
        "shape_invisible": False,
        "how_found": "待定：`ssm.time_step_rank` + dt 张量形状",
    },
    "ssm_gate_clamp": {
        "value": None,
        "why": "门控/clamp 的上下界（不同实现有 -1..inf 或 0..1 之类），改数值不改形状",
        "shape_invisible": True,
        "how_found": "待定：读参考实现（llama.cpp 的 mamba/granite 分支）",
    },
}

# ── KDA（bailingmoe3）线性注意力：同样"已知需要、尚未测定" ──
SEMANTICS_KDA = {
    "kda_layer_indexing": {
        "value": None,
        "why": "哪些层是 MLA、哪些是 KDA（Ling 是 3/7/11/15/19/23 共 6 层 MLA，其余 KDA）",
        "shape_invisible": True,
        "how_found": "待定：llama-model.cpp 的 bailingmoe3 分支 + 张量名存在性",
    },
    "kda_decay_param": {
        "value": None,
        "why": "衰减 A 的取法：是 `-exp(A_log)` 还是别的（我们实现 Ling 时按 `-exp(A_log)` 对上了，"
               "但这是**语义事实**，必须写进规格而不是埋在代码里）",
        "shape_invisible": True,
        "how_found": "已实测可得（ling_proto 对账过）⇒ 需回填成正式条目",
    },
    "kda_head_dim": {
        "value": None,
        "why": "KDA 的头维与头数（Ling: 16×128），决定 delta-net 状态 S 的形状",
        "shape_invisible": False,
        "how_found": "待定：张量形状 + 超参",
    },
    "kda_beta_activation": {
        "value": None,
        "why": "beta 是否过 sigmoid、是否带偏置（改数值不改形状）",
        "shape_invisible": True,
        "how_found": "待定：参考实现",
    },
    "kda_norm_placement": {
        "value": None,
        "why": "输出侧 norm（Ling 的 `o_norm`）作用在哪一步、是否按头",
        "shape_invisible": True,
        "how_found": "待定：参考实现",
    },
}


def selftest_validate():
    """自证校验器真的会拦下来（避免"校验器自己不报错"这种假通过）。"""
    validate_spec(SEMANTICS_LLAMA, "SEMANTICS_LLAMA")          # 完整 ⇒ 应通过
    for nm, sp in (("SEMANTICS_SSM", SEMANTICS_SSM), ("SEMANTICS_KDA", SEMANTICS_KDA)):
        try:
            validate_spec(sp, nm)
        except SystemExit:
            print(f"  ✓ {nm} 被正确拦下（{len(sp)} 条待定）")
            continue
        raise SystemExit(f"✗ {nm} 竟然通过了校验 —— 校验器有 bug")
    broken = dict(SEMANTICS_LLAMA)
    broken["rope_type"] = {k: v for k, v in broken["rope_type"].items() if k != "how_found"}
    try:
        validate_spec(broken, "broken")
    except SystemExit:
        print("  ✓ 删掉一条事实的 how_found 也被拦下")
    else:
        raise SystemExit("✗ 删字段竟然通过了")
    print("  ✅ 校验器自证通过：完整规格通过、缺字段/未确定都被拦")


if __name__ == "__main__" and os.environ.get("SELFTEST_SPEC"):
    selftest_validate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", default="1,818,3823,8864")
    ap.add_argument("--ref", default="/tmp/archspec_ref")
    ap.add_argument("--quant-act", action="store_true",
                    help="把激活按 Q8 量化（模仿 llama.cpp 的 vec_dot_type），用来判定残余差是不是量化噪声")
    a = ap.parse_args()
    toks = [int(x) for x in a.tokens.split(",")]

    wget, emb, geo, hp = load_weights(a.model)
    print("=== 声明式描述：几何从 GGUF 推出，语义**必须显式写出** ===")
    print(f"  几何: {json.dumps({k: v for k, v in geo.items() if k != 'out_norm'})}")
    inv = [k for k, v in SEMANTICS_LLAMA.items() if v["shape_invisible"]]
    print(f"  语义事实 {len(SEMANTICS_LLAMA)} 条，其中**形状看不出来**的 {len(inv)} 条：")
    for k in inv:
        print(f"    · {k:16s} = {SEMANTICS_LLAMA[k]['value'][:52]}")
        print(f"      └ 怎么定的：{SEMANTICS_LLAMA[k]['how_found'][:64]}")

    print(f"\n=== 跑 {len(toks)} 个 token（{geo['n_layer']} 层，numpy f32）===")
    ev = Evaluator(wget, geo, SEMANTICS_LLAMA)
    ev.quant_act = bool(a.quant_act)
    trace = {}
    mine = ev.forward(toks, emb, trace=trace)

    ref = llama_cpp_logits(a.model, toks, a.ref)
    n = min(mine.size, ref.size)
    A, B = mine[:n].astype(np.float64), ref[:n].astype(np.float64)
    cos = float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B)))
    t1a, t1b = int(A.argmax()), int(B.argmax())
    print(f"\n=== 对账（最后一个 token）===")
    print(f"  声明式求值器 top1 = {t1a}   llama.cpp top1 = {t1b}   {'一致 ✓' if t1a == t1b else '不一致 ✗'}")
    print(f"  logits cos = {cos:.6f}   max|Δ| = {np.abs(A - B).max():.4f}")
    print(f"  判据：cos≳0.99 且 top-1 一致（口径与 ZAYA 对账相同；激活/权重量化天然带 1e-3 级差）")
    print("\n=== 逐层对账（找第一个发散层）===")
    per_layer_compare(a.model, toks, trace, a.ref + "_layers")
    verdict = ("✅ 语义对齐：声明式描述 + 通用求值器**能**复现真模型数值（残余差在本例是构造性的："
               "求值器用**精确 f32 反量化权重**，llama.cpp 用 5bit 权重 + 量化激活，天然有 % 级差）"
               if cos > 0.99 else
               "❌ 语义没对上：cos 低于量化能解释的量级 ⇒ 还有语义事实写错了（见逐层第一个发散层）")
    print(f"\n  {verdict}")
    if cos > 0.99 and t1a != t1b:
        print(f"  ⚠ top-1 不一致（{t1a} vs {t1b}）但 cos={cos:.4f} —— 这是**刀尖效应**："
              f"接近平局的分布下，量化级差就能翻 argmax。**别用单个 argmax 当判据**"
              f"（ZAYA 那边同样的教训）。要更严的判据就得换精确 fp32 参考。")


if __name__ == "__main__":
    main()

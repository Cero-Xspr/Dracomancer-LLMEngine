#!/usr/bin/env python3
"""run.py —— 分层测试入口（阶段 0 地基）

为什么长这样（三个设计决定，都是为了不重犯已犯过的错）：

① **按依赖分层，不按重要性**：判据"能不能跑"取决于手上有什么，不取决于它多重要。
     T0  免引擎/免权重（只读 GGUF 元数据或纯 Python）—— 秒级，**CI 能跑**
     T1  小模型（smol 360M 级）—— 分钟级
     T2  大模型对账（Ling/ZAYA/granite + llama.cpp 参考）—— 十分钟级
   跑法：`python3 run.py --tier 0`（≤0）、`--tier 2`（≤2）、`-k 关键字`、`--list`。

② **判据必须写死阈值 + 打印"期望 vs 实测"**。历次真 bug 里最贵的两个教训都来自这里：
   · "链头 cos≈0.06 分歧"活了一整轮 —— 因为它是**人眼看 cos**，不是断言；
   · `/hold` 位置参数错位让三个模型输出全空 —— 因为只看了"没报错"。
   所以本文件的每条判据都要给出数字，失败信息要能直接贴进 issue。

③ **已有的对账脚本不改数学，只当黑盒收集**（kind="script" 跑子进程看退出码，
   kind="regex" 解析它的输出行做断言）。改动风险最小，也不会出现"测试自己算错"。

金标准原则：**只存小文件**（token 序列 / top5 / logits 的 sha256），模型不进仓库、判据进仓库。
`tests/golden/*.json` 由 `--freeze` 生成；生成前必须确认当前 .so 是被验证过的那一版。
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden")
PY = sys.executable


def _find_script(name):
    """对账脚本在公开仓库布局里在 tests/，在本机（hybrid/）里在同一层 —— 两处都找。"""
    for d in (HERE, ROOT, os.environ.get("DRACO_TESTS_DIR", "")):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def _find_file(name, isdir=False):
    """同一个文件在两种布局下位置不同：本机是平铺（hybrid/draco.py），公开仓库是 src/。
    都找一遍（再加 DRACO_REPO 覆盖），找不到就返回 None。"""
    cands = [os.path.join(HERE, name), os.path.join(ROOT, name),
             os.path.join(ROOT, "src", name), os.path.join(ROOT, "tests", name),
             os.path.join(os.environ.get("DRACO_REPO", ""), name)]
    chk = os.path.isdir if isdir else os.path.isfile
    return next((c for c in cands if c and chk(c)), None)


def _so_path():
    """引擎 .so 也要按双布局找（本机在 hybrid/，公开仓库在根目录）。

    ★ `DRACO_TEST_SO=<路径>` 可以让整套闸门去测**候选 .so**（装到 m6_engine.so 之前先验），
      这正是我这几轮手工做的 A/B 的自动化版本：候选改了数值 ⇒ 指纹闸门必须当场红。"""
    p = os.environ.get("DRACO_TEST_SO") or _find_file("m6_engine.so")
    return p or os.path.join(HERE, "m6_engine.so")


def _run(cmd, cwd=None, timeout=3600, env=None):
    e = dict(os.environ)
    e.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=cwd or HERE, capture_output=True, text=True, timeout=timeout, env=e)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _load_golden(name):
    p = os.path.join(GOLDEN, name + ".json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _fingerprint(npy_path):
    """把一组逐 token logits 压成**小指纹**：每步 top5 + 整块字节的 sha256。
    逐位回归闸门就靠它 —— 文件进仓库只有几 KB，而判据是"逐位相同"。"""
    import hashlib

    import numpy as np
    a = np.load(npy_path)
    import numpy as _np
    out = {"shape": list(a.shape), "steps": []}
    for row in a:
        out["steps"].append({"top5": _np.argsort(-row)[:5].tolist(),
                             "sha256": hashlib.sha256(row.astype(_np.float32).tobytes()).hexdigest()[:16]})
    return out


def _fingerprint_cmp(cur, gold):
    if cur["shape"] != gold["shape"]:
        return False, f"形状不同 {cur['shape']} vs {gold['shape']}"
    bad = [i for i, (a, b) in enumerate(zip(cur["steps"], gold["steps"])) if a["sha256"] != b["sha256"]]
    if bad:
        i = bad[0]
        a, b = cur["steps"][i], gold["steps"][i]
        # ★ 失败信息必须分清两种情况：top5 变了 = 逻辑变向；top5 没变 = 只是逐位数值漂移。
        #   我第一版只打印 top5，结果两个列表一模一样地摆在眼前，看着像"测试坏了"。
        if a["top5"] != b["top5"]:
            why = f"且 top5 变了：{a['top5']} vs 金标准 {b['top5']}"
        else:
            why = "top5 未变 ⇒ 是逐位数值漂移（量化/归约顺序之类），不是逻辑变向"
        return False, f"{len(bad)}/{len(cur['steps'])} 步**不逐位相同**（首处 step{i}；{why}）"
    return True, f"{len(cur['steps'])} 步全部逐位相同（sha256 逐个吻合）"


# ═══════════════════════════ T0：免引擎、免权重 ═══════════════════════════
def t0_think_split():
    """思考分区 + **流式 UTF-8 不变式**（后者是本轮真 bug：汉字被 token 切开时丢字）。
    判据来自 test_think_split.py 自己：1431 种任意字节切分都必须还原原文，且带反面对照。"""
    s = _find_script("test_think_split.py")
    rc, out = _run([PY, s], timeout=300)
    tail = [x for x in out.strip().splitlines() if x.strip()][-1:]
    return rc == 0, (tail[0] if tail else out[-200:])


def t0_schema():
    """档案 schema：models.d/*.json 全部必须过白名单校验。"""
    d = _find_file("models.d", isdir=True)
    spec = importlib.util.spec_from_file_location("_as", _find_file("adapt_schema.py"))
    asch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asch)
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    bad = []
    for f in files:
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            p = json.load(fh)
        try:
            asch.validate_profile(p, f)
        except Exception as e:
            bad.append(f"{f}: {e}")
    if bad:
        return False, "; ".join(bad)
    return True, f"{len(files)} 份档案全部通过（{', '.join(files)}）"


class _StubModel:
    """只带查表需要的两个名字字段，避免为了测查表去加载 5GB 模型。"""
    def __init__(self, name, gguf_name=""):
        self.name = name
        self.gguf_name = gguf_name or name
        self.short = name
        # backend_metrics → _tune_best → _model_fingerprint 会读 path；给个不存在的路径，
        # 它内部捕获 OSError 返回 None（= 没有 tune 缓存）⇒ 正好走到 MEASURED 那一条分支。
        self.path = os.path.join(HERE, "_no_such_model.gguf")


def _import_draco():
    spec = importlib.util.spec_from_file_location("_draco", _find_file("draco.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def t0_measured_lookup():
    """★ MEASURED 表查表命中 —— 这轮抓到的静默失效：
    表里的键是干净名（"Ling 3.0 Tiny"），而档案 display_name 带架构后缀
    （"Ling 3.0 Tiny（bailingmoe3）"）⇒ 原来直接用 m.name 查**查不中**，
    实测数字一直没被用上（被 tune 缓存掩盖所以没暴露）。

    ★★ 这里必须测**消费路径**（backend_metrics），不能只测 `_measured_key` 这个辅助函数：
       我第一版就是直接调 _measured_key，把 bug 注回 backend_metrics 里它照样全绿
       —— 那种"绿得没用"的测试比没有测试更危险（反面对照抓到了这一条）。"""
    D = _import_draco()
    miss = []
    for (nm, be) in list(D.MEASURED.keys()):
        for probe in (nm, f"{nm}（某架构）", f"{nm} (arch)"):
            tps, mj, src = D.backend_metrics(_StubModel(probe), be)
            if tps is None or abs(tps - D.MEASURED[(nm, be)][0]) > 1e-6 or "档案性能表" not in src:
                miss.append(f"{probe!r}+{be} → {tps!r}/{src!r}")
    if miss:
        return False, "实测数字没被用上: " + "; ".join(miss[:3])
    return True, f"{len(D.MEASURED)} 个键 × 3 种名字写法都真的取到了实测值（含带括号后缀）"


def t0_rank_prefer():
    """后端排序：`-P speed` 必须按实测 tok/s 选最快；granite 的实测是 igpu 21.9 > dengine 12.2 > cpu 9.4。
    判据：iv 用档案里的实测值，断言 speed 档选出 igpu、且理由串里带实测数字。"""
    D = _import_draco()
    key = ("Granite 4.0 h-tiny", "igpu")
    if key not in D.MEASURED:
        return False, f"MEASURED 里没有 {key}（granite 的实测没登记）"
    got = D.MEASURED[key][0]
    if abs(got - 21.9) > 0.05:
        return False, f"granite igpu 实测值 = {got}，期望 21.9"
    # dengine 必须也有实测（否则 speed 档会拿先验跟实测比，不公平）
    if ("Granite 4.0 h-tiny", "dengine") not in D.MEASURED:
        return False, "granite 的 dengine 实测没登记 ⇒ speed 档会拿先验冒充对比"
    return True, f"granite: igpu {got} / dengine {D.MEASURED[('Granite 4.0 h-tiny','dengine')][0]} / cpu {D.MEASURED[('Granite 4.0 h-tiny','cpu')][0]} t/s"


# ═══════════════════════════ T1：小模型（分钟级） ═══════════════════════════
# ── 覆盖矩阵（我实测过：给共用路径注入 1ULP 级扰动，看哪些用例会红）──
#   扰动 m6_rms_norm 的乘序：
#     t1_smol_fingerprint   ❌ 红（13/13 步 sha256 不同）
#     t2_ling_fingerprint   ❌ 红（同上）
#     t1_granite_greedy     ✅ 绿（16/16 仍一致 —— 1ULP 不足以改 argmax）
#     t2_granite_s4d_layers ✅ 绿（cos 0.999147 不变 —— cos 对 1ULP 不敏感）
#   ⇒ **只有指纹闸门是逐位的**；cos/贪心类用例是"逻辑级"判据，两者互补、不能互相替代。
def t1_smol_fingerprint():
    """smol 引擎的**逐位回归闸门**：13 步 logits 的 sha256 必须与金标准逐个吻合。
    金标准取自"抽 moe_experts / llama_attn_core 之前那一版 .so"的输出 ⇒ 任何共享路径改动
    只要动了数值就会在这里红。"""
    gold = _load_golden("smol_fingerprint")
    if gold is None:
        return False, "缺 tests/golden/smol_fingerprint.json（用 --freeze 生成）"
    s = _find_script("m6_regress.py")
    out = "/tmp/_t1_smol.npy"
    rc, log = _run([PY, s, "smol_engine", _so_path(), out], timeout=1800)
    if rc != 0:
        return False, f"m6_regress 退出码 {rc}: {log[-200:]}"
    return _fingerprint_cmp(_fingerprint(out), gold)


def t1_granite_tok():
    """granite 分词器 vs llama-tokenize 逐 id 对账（9 个串，含 <|start_of_role|> 与 emoji）。
    分词器错了会让模板渲染出的对话格式整段跑偏，但输出可能"看起来还挺通顺" ⇒ 必须逐 id 比。"""
    s = _find_script("granite_tok.py")
    rc, out = _run([PY, s, "--check"], timeout=900)
    m = re.search(r"全部一致", out)
    if rc != 0 and not m:
        bad = [x for x in out.splitlines() if x.startswith("✗")]
        tail = " | ".join(out.strip().splitlines()[-3:])
        return False, f"退出码 {rc}；不一致行={bad[:3]}；输出尾部={tail[:300]}"
    n_ok = out.count("✓")
    n_bad = out.count("✗")
    if n_bad:
        return False, f"{n_bad} 串不一致"
    return True, f"{n_ok} 串与 llama-tokenize 逐 id 一致"


def t1_granite_greedy():
    """granite 端到端：真实句子 "The capital of France is" 的贪心输出必须与 llama.cpp **逐 token 相同**。
    这是最强的判据 —— 量化误差允许 logits 不完全一致，但 16 步 argmax 必须走在同一条轨迹上。"""
    gold = _load_golden("granite_greedy")
    if gold is None:
        return False, "缺 tests/golden/granite_greedy.json"
    s = _find_script("granite_engine.py")
    rc, out = _run([PY, s], timeout=1800,
                   env={"TOKS": ",".join(map(str, gold["prompt"])), "NSTEPS": str(len(gold["tokens"]))})
    m = re.search(r"\[GEN\] 贪心 token: \[([0-9, ]+)\]", out)
    if not m:
        return False, "拿不到贪心输出: " + out[-200:]
    got = [int(x) for x in m.group(1).split(",")]
    exp = gold["tokens"]
    if got[:len(exp)] != exp:
        bad = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
        return False, f"第 {bad} 个 token 起分叉：得到 {got[bad:bad+4]} 期望 {exp[bad:bad+4]}"
    return True, f"{len(exp)}/{len(exp)} token 与 llama.cpp 贪心一致"


def t1_falcon_greedy():
    """falcon 端到端：金标准 = llama.cpp ZGREEDY 的 12 个 token（"…Germany is" 续写）。
    注：falcon 是补全式小模型，某些 prompt 会在 top-2 平手处分叉（见 golden 内 source 注），
    金标准特意选了置信度高的续写。"""
    gold = _load_golden("falcon_greedy")
    if gold is None:
        return False, "缺 tests/golden/falcon_greedy.json"
    m = os.environ.get("FALCON_MODEL",
                       "/media/xiao_/OverSys1/gguf/falcon-h1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf")
    if not os.path.isfile(m):
        return False, f"缺 falcon 模型（{m}）；T1/T2 需要本地模型"
    s = _find_script("falcon_engine.py")
    rc, out = _run([PY, s], timeout=1800,
                   env={"TOKS": ",".join(map(str, gold["prompt"])),
                        "NSTEPS": str(len(gold["tokens"])), "MODEL": m})
    got = [int(x) for x in re.search(r"\[GEN\] 贪心 token: \[([0-9, ]+)\]", out).group(1).split(",")]
    exp = gold["tokens"]
    if got[:len(exp)] != exp:
        bad = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), None)
        return False, f"第 {bad} 个 token 分叉：得到 {got[bad:bad+4]} 期望 {exp[bad:bad+4]}"
    return True, f"{len(exp)}/{len(exp)} token 与 llama.cpp 贪心一致"


def t1_llama_greedy():
    """llama 家族（引擎固化管线）：金标准 = llama.cpp ZGREEDY 12 token。
    管线本身由 smol 指纹长期覆盖，这条闸门守的是「llama 家族接入点」（分词器/模板/head 绑定）。"""
    gold = _load_golden("llama_greedy")
    if gold is None:
        return False, "缺 tests/golden/llama_greedy.json"
    m = os.environ.get("LLAMA_MODEL",
                       "/media/xiao_/OverSys1/gguf/llama32/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    if not os.path.isfile(m):
        return False, f"缺 llama 模型（{m}）"
    s = _find_script("llama_greedy.py")
    rc, out = _run([PY, s], timeout=1800,
                   env={"TOKS": ",".join(map(str, gold["prompt"])),
                        "NSTEPS": str(len(gold["tokens"])), "MODEL": m})
    mm = re.search(r"\[GEN\] 贪心 token: \[([0-9, ]+)\]", out)
    if not mm:
        return False, "拿不到贪心输出: " + out[-200:]
    got = [int(x) for x in mm.group(1).split(",")]
    exp = gold["tokens"]
    if got[:len(exp)] != exp:
        bad = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
        return False, f"第 {bad} 个 token 分叉：得到 {got[bad:bad+4]} 期望 {exp[bad:bad+4]}"
    return True, f"{len(exp)}/{len(exp)} token 与 llama.cpp 贪心一致"


# ═══════════════════════════ T2：大模型对账（十分钟级） ═══════════════════════════
def t1_qwen35_greedy():
    """qwen35 端到端：金标准 = llama.cpp ZGREEDY 12 token。全 f16 ⇒ 无量化误差。"""
    gold = _load_golden("qwen35_greedy")
    if gold is None:
        return False, "缺 tests/golden/qwen35_greedy.json"
    m = os.environ.get("QWEN35_MODEL", "/media/xiao_/OverSys1/gguf/Qwen3.5-2B-f16.gguf")
    if not os.path.isfile(m):
        return False, f"缺 qwen35 模型（{m}）"
    s = _find_script("qwen35_engine.py")
    rc, out = _run([PY, s], timeout=1800,
                   env={"TOKS": ",".join(map(str, gold["prompt"])),
                        "NSTEPS": str(len(gold["tokens"])), "MODEL": m})
    mm = re.search(r"\[GEN\] 贪心 token: \[([0-9, ]+)\]", out)
    if not mm:
        return False, "拿不到贪心输出: " + out[-200:]
    got = [int(x) for x in mm.group(1).split(",")]
    exp = gold["tokens"]
    if got[:len(exp)] != exp:
        bad = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
        return False, f"第 {bad} 个 token 分叉：得到 {got[bad:bad+4]} 期望 {exp[bad:bad+4]}"
    return True, f"{len(exp)}/{len(exp)} token 与 llama.cpp 贪心一致"


def t2_qwen35_layers():
    """qwen35 逐层对账：pos 0..3 × 24 层，阈值 0.9999（全 f16 ⇒ 实测 cos≈1.00000）。
    参考 dump 缺失时用夹具现场生成。"""
    for p in range(4):
        d = f"/tmp/qrec{p}"
        if not glob.glob(d + "/l_out-0.*.bin"):
            fix = _find_script("zaya_gdump")
            if fix is None:
                return False, "缺夹具 zaya_gdump 且无缓存 dump（/tmp/qrec*）"
            S = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"
            os.makedirs(d, exist_ok=True)
            r = subprocess.run([fix, "/media/xiao_/OverSys1/gguf/Qwen3.5-2B-f16.gguf",
                                "760,6511,314,9338,369", d],
                               env={**os.environ, "LD_LIBRARY_PATH": S, "ZDUMP_POS": str(p)},
                               capture_output=True, text=True, timeout=1200)
            if r.returncode != 0:
                return False, f"夹具失败 pos{p}: {r.stderr[-150:]}"
    m = os.environ.get("QWEN35_MODEL", "/media/xiao_/OverSys1/gguf/Qwen3.5-2B-f16.gguf")
    if not os.path.isfile(m):
        return False, f"缺 qwen35 模型（{m}）"
    rc, out = _run([PY, "-c", """
import sys, glob, os
import numpy as np
sys.path.insert(0, '/media/xiao_/OverSys1/npu-direct/hybrid')
sys.path.insert(0, '/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py')
os.environ['GPROBE'] = '1'
import qwen35_engine as Q
IDS = [760, 6511, 314, 9338, 369]
for pos, t in enumerate(IDS):
    Q.forward(t, pos)
cos = lambda a, b: float(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)))
worst = (1.0, None); n = 0
for pos in range(4):
    mine = Q.PROBE['by_pos'][pos]
    for il in range(Q.NL):
        ref = np.frombuffer(open(sorted(glob.glob(f'/tmp/qrec{pos}/l_out-{il}.*.bin'))[0], 'rb').read(), np.float32)
        c = cos(mine[il], ref)
        if c < worst[0]: worst = (c, (pos, il))
        n += 1
print(f'QWEN35_OK {n} {worst[0]:.6f} {worst[1]}')
""", m], timeout=1800, env={"MODEL": m})
    mm = re.search(r"QWEN35_OK (\d+) ([0-9.]+) (\(.*?\))", out)
    if rc != 0 or not mm:
        return False, "对账脚本失败: " + out[-200:]
    n, w, loc = int(mm.group(1)), float(mm.group(2)), mm.group(3)
    if w < 0.9999:
        return False, f"{n} 个 (pos,层) 最差 cos={w:.6f}（{loc}）< 0.9999"
    return True, f"pos0..3 × 24 层全部 ≥0.9999；最差 cos={w:.6f}（{loc}）"


def t1_incremental_state():
    """阶段 2 的核心不变式：**增量续转 ≡ 全量重跑（逐位）**。

    上下文复用的依据是「引擎状态精确对应已转发 token 序列」——如果跨位置状态里
    有任何顺序依赖没维护好（conv 移位/KV 写行/位置编号），这条就会红。
    用 granite（S4D 卷积态 + KV + MoE，状态种类最多）验证。
    """
    m = os.environ.get("GRANITE_MODEL",
                       "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf")
    if not os.path.isfile(m):
        return False, f"缺 granite 模型（{m}）"
    rc, out = _run([PY, "-c", """
import sys, os
import numpy as np
sys.path.insert(0, '/media/xiao_/OverSys1/npu-direct/hybrid')
sys.path.insert(0, '/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py')
import granite_engine as G
P = [791, 6864, 315, 9822, 374, 12366, 13, 12366]
GEN = [374, 3967, 369, 1202]
G.reset()
for pos, t in enumerate(P + GEN):
    G.forward(t, pos)
LA = G.logits_of_x().copy()
G.reset()
for pos, t in enumerate(P[:5]):
    G.forward(t, pos)
for pos, t in enumerate(P[5:] + GEN, start=5):
    G.forward(t, pos)
LB = G.logits_of_x().copy()
print(f"INC_EQ={np.array_equal(LA, LB)} DMAX={np.abs(LA-LB).max():.2e}")
""", m], timeout=1800, env={"MODEL": m})
    mm = re.search(r"INC_EQ=(True|False) DMAX=([0-9.e+-]+)", out)
    if not mm:
        return False, "脚本失败: " + out[-200:]
    if mm.group(1) != "True":
        return False, f"增量续转与全量重跑不一致（max|Δ|={mm.group(2)}）——前缀复用不安全！"
    return True, "增量续转 ≡ 全量重跑（logits 逐位相同，max|Δ|=0）"


def t1_qwen35moe_greedy():
    """qwen35moe（35B A3B REAP）：金标准 = llama.cpp ZGREEDY 12 token。
    计数类 prompt 有量化噪声分叉（IQ3_S+Q8_K vs fp32），金标准用真实句子。"""
    gold = _load_golden("qwen35moe_greedy")
    if gold is None:
        return False, "缺 tests/golden/qwen35moe_greedy.json"
    m = os.environ.get("QWEN35MOE_MODEL",
                       "/media/xiao_/OverSys1/gguf/Qwen3.6-35B-A3B-REAP-48-v2.gguf")
    if not os.path.isfile(m):
        return False, f"缺 qwen35moe 模型（{m}）"
    s = _find_script("qwen35_engine.py")
    rc, out = _run([PY, s], timeout=3600,
                   env={"TOKS": ",".join(map(str, gold["prompt"])),
                        "NSTEPS": str(len(gold["tokens"])), "MODEL": m})
    mm = re.search(r"\[GEN\] 贪心 token: \[([0-9, ]+)\]", out)
    if not mm:
        return False, "拿不到贪心输出: " + out[-200:]
    got = [int(x) for x in mm.group(1).split(",")]
    exp = gold["tokens"]
    if got[:len(exp)] != exp:
        bad = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
        return False, f"第 {bad} 个 token 分叉：得到 {got[bad:bad+4]} 期望 {exp[bad:bad+4]}"
    return True, f"{len(exp)}/{len(exp)} token 与 llama.cpp 贪心一致"


def t2_granite_s4d_layers():
    """S4D 逐层对账：pos 0..3 × 全部 36 个 SSM 层的链头 cos 必须 ≥ 0.9990。
    阈值取自实测（本轮记录的最小值 0.999147），不另立标准。"""
    s = _find_script("s4d_reconcile2.py")
    gold = _load_golden("granite_s4d_layers")
    thr = (gold or {}).get("min_cos_threshold", 0.9990)
    rc, out = _run([PY, s], timeout=3600)
    if rc != 0:
        return False, f"退出码 {rc}: {out[-200:]}"
    rows = re.findall(r"^\s*(\d+)\s+36\s+([0-9.]+)\s+([0-9.]+)\s+\S+\s+([0-9.]+)\s+([0-9.]+)\s*$",
                      out, re.M)
    if not rows:
        return False, "解析不到分层 cos 汇总行"
    worst = min(float(r[2]) for r in rows)
    if worst < thr:
        return False, f"链头 cos 最小 {worst:.6f} < 阈值 {thr}（{len(rows)} 个位置）"
    return True, (f"{len(rows)} 个位置 × 36 层：链头 cos 最小 {worst:.6f} ≥ {thr}；"
                  f"中位 {max(float(r[1]) for r in rows):.6f}")


def t2_falcon_layers():
    """falcon 逐层对账：pos 0..3 × 36 层 vs 夹具 dump 的 l_out。阈值 0.996（实测最小 0.99682）。
    低于 granite 的 0.999 是预期：llama.cpp 用 Q8_K 激活量化 + f16 KV，我们全 fp32。"""
    g = _find_script("falcon_layer_gate.py")
    rc, out = _run([PY, g], timeout=3600)
    line = [x for x in out.splitlines() if x.startswith(("✅", "❌"))]
    return rc == 0, (line[-1][2:].strip() if line else out[-200:])


def t2_ling_fingerprint():
    """Ling（bailingmoe3：MLA + KDA + 分组 MoE）的逐位回归闸门 —— 与 smol 同法。"""
    gold = _load_golden("ling_fingerprint")
    if gold is None:
        return False, "缺 tests/golden/ling_fingerprint.json"
    s = _find_script("m6_regress.py")
    out = "/tmp/_t2_ling.npy"
    rc, log = _run([PY, s, "ling_engine", _so_path(), out], timeout=3600)
    if rc != 0:
        return False, f"m6_regress 退出码 {rc}: {log[-200:]}"
    return _fingerprint_cmp(_fingerprint(out), gold)


# 明确**没有**自动化、只能手工跑的（写出来是为了别让这份文件假装覆盖了它们）
MANUAL = [
    "draco selfcheck / family（要起服务、装模型；判据含健康问句与语义一致）",
    "draco tune 的 J/token（能耗口径要求机器安静）— 必须注明供电档",
    "iGPU 后端速度（要 GPU 构建 + 显存）与 NPU 路径（要 FLM）",
    "多设备兼容（x86 无 AVX-512 的回退）— 目前没有第二台机器",
]


def main():
    ap = argparse.ArgumentParser(description="分层测试入口")
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2], help="跑到第几层（含）")
    ap.add_argument("-k", default="", help="只跑名字含该子串的用例")
    ap.add_argument("--list", action="store_true", help="列出用例与依赖，不执行")
    ap.add_argument("--freeze", action="store_true", help="生成/刷新金标准（慎重：先确认当前 .so 已被验过）")
    A = ap.parse_args()

    cases = [("t0_think_split", 0, t0_think_split), ("t0_schema", 0, t0_schema),
             ("t0_measured_lookup", 0, t0_measured_lookup), ("t0_rank_prefer", 0, t0_rank_prefer),
             ("t1_smol_fingerprint", 1, t1_smol_fingerprint), ("t1_granite_tok", 1, t1_granite_tok),
             ("t1_granite_greedy", 1, t1_granite_greedy), ("t1_falcon_greedy", 1, t1_falcon_greedy),
             ("t1_llama_greedy", 1, t1_llama_greedy), ("t1_qwen35_greedy", 1, t1_qwen35_greedy),
             ("t1_incremental_state", 1, t1_incremental_state),
             ("t1_qwen35moe_greedy", 1, t1_qwen35moe_greedy),
             ("t2_granite_s4d_layers", 2, t2_granite_s4d_layers),
             ("t2_qwen35_layers", 2, t2_qwen35_layers),
             ("t2_falcon_layers", 2, t2_falcon_layers),
             ("t2_ling_fingerprint", 2, t2_ling_fingerprint)]
    if A.list:
        for n, t, _ in cases:
            print(f"  T{t}  {n}")
        print("\n手工项（未自动化，别以为这份文件覆盖了）:")
        for x in MANUAL:
            print(f"  · {x}")
        return 0

    if A.freeze:
        os.makedirs(GOLDEN, exist_ok=True)
        # ① granite 贪心：金标准 = **llama.cpp 的参考输出**（独立 oracle），并当场验证引擎与它一致才写。
        #    为什么不冻引擎自己的输出：那样只是"跟上次一样"，不是"跟 llama.cpp 一样"。
        REF = [12366, 13, 12366, 374, 3967, 369, 1202, 13970, 61024, 11, 1778, 439, 279, 469, 3168, 301]
        rc, out = _run([PY, _find_script("granite_engine.py")], timeout=1800,
                       env={"TOKS": "791,6864,315,9822,374", "NSTEPS": str(len(REF))})
        # 引擎打印的是 [prefill 后的 argmax] + 后续生成；llama.cpp 的 ZGREEDY 从同一位置开始
        # ⇒ 对齐方式是**前 len(REF) 项一一对应**（第一版写成 [1:] 去掉了首项，被断言当场拦住）。
        got = [int(x) for x in re.search(r"\[GEN\] 贪心 token: \[([0-9, ]+)\]", out).group(1).split(",")]
        got = got[:len(REF)]
        assert got == REF, f"拒绝冻结：引擎({got[:6]}…)与 llama.cpp 参考({REF[:6]}…)不一致"
        json.dump({"prompt": [791, 6864, 315, 9822, 374], "tokens": REF,
                   "source": "llama.cpp build-dbg 逐 token decode 的贪心输出（2026-09-15 复核 16/16 一致）"},
                  open(os.path.join(GOLDEN, "granite_greedy.json"), "w"), ensure_ascii=False, indent=2)
        # ② smol/ling 指纹：金标准取自**抽 moe_experts/llama_attn_core 之前那一版 .so**（独立于当前代码树）
        OLD_SO = os.environ.get("DRACO_FREEZE_SO", "/tmp/m6_engine.prev.so")
        assert os.path.isfile(OLD_SO), f"缺重构前的 .so：{OLD_SO}（设 DRACO_FREEZE_SO 指定）"
        for eng, name in (("smol_engine", "smol_fingerprint"), ("ling_engine", "ling_fingerprint")):
            npy = f"/tmp/_freeze_{eng}.npy"
            rc, log = _run([PY, _find_script("m6_regress.py"), eng, OLD_SO, npy], timeout=3600)
            assert rc == 0, log[-300:]
            fp = _fingerprint(npy)
            fp["source"] = f"{eng} 逐 token logits，用 {os.path.basename(OLD_SO)}（重构前、已验证逐位不变）生成"
            json.dump(fp, open(os.path.join(GOLDEN, name + ".json"), "w"), indent=2)
        json.dump({"min_cos_threshold": 0.9990,
                   "source": "s4d_reconcile2.py 实测最小 0.999147（pos0）；阈值取 0.9990 留一点余量"},
                  open(os.path.join(GOLDEN, "granite_s4d_layers.json"), "w"), ensure_ascii=False, indent=2)
        print(f"金标准已写入 {GOLDEN}")
        return 0

    print(f"=== 分层测试（跑到 T{A.tier}）===")
    n_fail = 0
    for name, tier, fn in cases:
        if tier > A.tier or (A.k and A.k not in name):
            continue
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:  # 用例本身炸了也算失败，但要说清是"炸了"不是"判据不过"
            ok, detail = False, f"用例异常 {type(e).__name__}: {e}"
        dt = time.time() - t0
        print(f"{'✅' if ok else '❌'} T{tier} {name:24s} {dt:6.1f}s  {detail}")
        n_fail += 0 if ok else 1
    print()
    if n_fail:
        print(f"❌ {n_fail} 项失败")
        return 1
    print("✅ 全绿")
    return 0


if __name__ == "__main__":
    sys.exit(main())

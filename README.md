# Dracomancer — 模型适配注册表

Dracomancer 是一个自研的 **GGUF 量化推理引擎**（端侧优先：内存受限、没有独显的机器）。
这个仓库当前承载它的**模型适配注册表**——让引擎不必把「世界上很多很多模型」逐个硬编码，
而是用一份份**参数档案**（纯数据）+ 社区/agent 协作来覆盖。

> **现状（诚实版）**：引擎本体（C/C++ 内核 + 启动器）**尚未公开**，仍在内部打磨。
> 本仓库先公开适配机制的 **schema、注册表与提交流程**，好让适配工作可以现在就跑起来。
> 引擎发布后，`draco.py` 会读同一批档案，无需迁移。

## 目录

| 路径 | 作用 |
|---|---|
| `AGENT_ADAPTER.md` | **入口**：给 agent（或人）的适配指南——决策树、schema、自检、提交流程 |
| `models.d/` | 注册表本体：每个 `*.json` 是一份模型档案 |
| `src/draco.py` | 启动器 / 聊天 CLI：`list` `chat` `serve` `perf` `selfcheck` `caps` `tune` |
| `src/adapt_schema.py` | 档案 schema 的**唯一**实现（只依赖标准库） |
| `src/gguf_fast.py` | **只读 GGUF 解析器**（gguf-py 的兼容子集）：元数据解析快 40~50×，见下 |
| `src/gguf_probe.py` | **能力探针**：加载之前静态判定（类型被移除 / 架构未实现 / 可能可以） |
| `src/hwprobe.py` | 功率传感器按 name+label 定位（不按 hwmon 编号——编号会变） |
| `tests/` | 自带对账/单测：`gguf_fast_check.py`（逐字节对账 gguf-py）、`test_think_split.py`（思考分区不变式）、`backend_ab.py`（后端单轮对拍） |
| `release/validate_submission.py` | 提交校验器（schema/白名单 + 可选 GGUF 匹配核对） |
| `.github/` | 「模型适配」issue 表单 + 自动校验 bot（只做 schema 级过滤，不替代人工复核） |

### 三个诊断命令

| 命令 | 管什么 | 要装载模型吗 |
|---|---|---|
| `draco.py caps` | **能不能加载**（静态：类型被移除/架构未实现） | 不用，秒级 |
| `draco.py selfcheck` | **跑起来行为对不对**（指纹/健康问句/速度） | 要 |
| `draco.py tune` | **参数哪个好**（小配置空间实测，按 tok/s 与 J/token 双口径排序，带缓存） | 要 |

三者互补：`caps` 看不了数值问题，`tune` 的能耗口径需要机器安静，`selfcheck` 是最终判据。

### 路径配置

默认路径是本机布局；别人的机器用环境变量覆盖，无需改代码：
`DRACO_GGUF_DIRS`（模型目录，冒号分隔）、`DRACO_LLAMA_ROOT`（llama.cpp 发布包）、
`DRACO_LOCAL_BASE`（自研构建/源码树）、`DRACO_FLM_DIR`（FastFlowLM，npu 后端）。

### 速度-能效倾向：`-P / --prefer`

这台机器上**"更快"和"更省"不是同一个后端**（实测：iGPU 吞吐 ≈1.2~1.75× CPU，
而 NPU 最省电却最慢），所以倾向是个显式选项，不是一个"自动最优"的黑箱：

| 值 | 选什么 | 依据 |
|---|---|---|
| `balanced`（默认） | 自研引擎优先 / 档案推荐后端 | 主线是引擎，不做额外调 |
| `speed` | 按实测 tok/s 挑最快的后端 | 有实测就用实测；没有才用先验（iGPU≈1.2×CPU），并在输出里写明"这是先验" |
| `eco` | 按实测 J/token 挑最省的后端，线程取 4 | 没有能耗实测时按本机能耗序（NPU<iGPU<CPU） |

```bash
python3 src/draco.py chat -m zaya -P eco      # 要省电
export DRACO_PREFER=speed                     # 也可以一次性设定
```

倾向只改**两个有实测支撑的旋钮**（后端、线程数）。量化格式、kernel 融合、LM head、
NPU 整数 madd 这些真正的杠杆不是"调参"能动的——那是改代码，不做成假旋钮。
每次启动都会把决策依据打出来（选定谁、候选怎么排、哪条是实测哪条是先验）。

### 并行化的粒度：短提示词看不出来的坑

引擎里"每头一次串行 softmax/加权求和"这种写法在短提示词下毫无症状，但它是 **O(上下文长度)** 的：
真实聊天上下文累积到几百 token 时，它就从"可以忽略"变成"主导项"。实测（360M、8 线程、同一进程内交替 A/B、
测最后 24 个 token 的中位）：上下文 32 token 时改进 1.16×，256 token 时 1.51×，768 token 时 **2.55×**；
端到端（413 token 上下文，OpenAI 接口）**61.7 → 111.5 tok/s**，而数值**逐位不变**
（逐行/逐头互相独立 ⇒ 并行度不改变加法次序；判据是 8 个 token 的末层隐状态 max|Δ|=0）。

教训：性能结论必须标**上下文长度**，否则"优化"和"没优化"在短 prompt 下看起来一样 ——
这和上面那条"tok/s 必须标生成长度/线程数"是同一类错误。

### 装载速度：元数据解析是隐藏的大头

`gguf.GGUFReader` 在 ZAYA1-8B（5.19 GB / 262k 词表 / 1283 张量）上要 **13.6 s**，
而且**磁盘读 0 字节**——cProfile 指认元凶是每个字符串元素一次 numpy 封装
（词表三件套 78 万元素 ⇒ 830 万次 numpy 调用）。`src/gguf_fast.py` 用 `mmap` + `struct`
直接切，**0.29 s（47×）**，并且逐字节对账 `gguf-py` 完全一致（`tests/gguf_fast_check.py`）。
配合张量零拷贝（mmap 基址页对齐 ⇒ 文件偏移已 64 对齐的张量直接交给内核，不搬）
与"适配器不再 import transformers"（2.83 s → 0.04 s），端到端：

| 模型 | 改前 | 改后 |
|---|---|---|
| ZAYA1-8B（5.19 GB） | ~39 s | **3.05 s** |
| Ling 3.0 Tiny（4.58 GB） | ~13 s | **1.62 s** |
| SmolLM2-360M（0.25 GB） | 5.68 s | **0.29 s** |

（同机 llama.cpp 装载 ZAYA 约 6 s，作参照。）

## 三档适配（为什么是"参数文件"，不是"代码"）

| 档 | 什么时候用 | 产物 |
|---|---|---|
| **Tier 0** | 引擎认识的已知架构的新权重/新量化 | 不用提交 |
| **Tier 1** | 已知架构，但需要特定启动参数/采样默认/后端选择 | `models.d/*.json` |
| **Tier 2** | 引擎不认识的新架构（需要写计算图） | **适配档案**（张量清单/hparams/结构伪代码/证据），由维护者实现代码 |

核心分工：**参数文件管配置，适配档案管探路，代码归维护者，验证闸门说了算。**
档案是纯数据的白名单字段，**永不执行代码**——这是安全边界，也是能自动校验的前提。

## 档案长什么样

```json
{
  "schema_version": 1,
  "match": { "arch": "zaya" },
  "requires_local_build": true,
  "launch": { "extra_args": ["-ub", "1"] },
  "backend_hint": "local",
  "sampling": { "think_default": false },
  "notes": "为什么是这些值、踩过什么坑（最有价值的部分）",
  "source": { "author": "you", "date": "2026-09-13" }
}
```

一个真实例子见 [`models.d/zaya.json`](models.d/zaya.json)：ZAYA1-8B 是 CCA（卷积注意力）
+ MoE 的混合架构，必须逐 token prefill——**批量 prefill 会被量化台阶放大的误差毁掉 prompt**
（`-ub 16` 起输出乱码）。这条结论是实测出来的，档案里就写着它，别人不必重踩。

## 提交

用 issue 表单「模型适配」提交，附上自检结果。bot 会自动校验 schema 与白名单，
维护者复核参数合理性后合入 `models.d/`。

## 许可证

**MIT** —— 见 [`LICENSE`](LICENSE)。

## 参考硬件

适配数据来自一台 Ryzen AI 9 H365（Strix）笔记本，22 GB 内存、无独显（Radeon 890M 核显）、
AMD XDNA2 NPU。性能数字是「此刻此机」，跨机不可比——档案里请写**相对结论**
（哪个后端更好、安全上界在哪），而不是绝对 tok/s。

实测口径：能耗用 APU 封装功率（hwmon 的 PPT，按 name+label 找，不按编号——编号会变），
采样前静置 1.5 s、只取负载段、窗口拉到 ≥6 s、用中位数；同一配置两次跑可差 30%，
所以**能耗结论只在机器安静时成立**，而 tok/s 随时可信。

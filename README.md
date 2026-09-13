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
| `src/gguf_probe.py` | **能力探针**：加载之前静态判定（类型被移除 / 架构未实现 / 可能可以） |
| `src/hwprobe.py` | 功率传感器按 name+label 定位（不按 hwmon 编号——编号会变） |
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

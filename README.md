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
| `adapt_schema.py` | 档案 schema 的**唯一**实现（独立可跑，只依赖标准库） |
| `models.d/` | 注册表本体：每个 `*.json` 是一份模型档案 |
| `release/validate_submission.py` | 提交校验器（schema/白名单 + 可选 GGUF 匹配核对） |
| `.github/ISSUE_TEMPLATE/` | 「模型适配」issue 表单 |
| `.github/workflows/` | 自动校验 bot（只做 schema 级过滤，不替代人工复核） |

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

**待定**（未声明即保留所有权利）。在明确之前，请勿假设任何使用许可。

## 参考硬件

适配数据来自一台 Ryzen AI 9 H365（Strix）笔记本，22 GB 内存、无独显（Radeon 890M 核显）、
AMD XDNA2 NPU。性能数字是「此刻此机」，跨机不可比——档案里请写**相对结论**
（哪个后端更好、安全上界在哪），而不是绝对 tok/s。

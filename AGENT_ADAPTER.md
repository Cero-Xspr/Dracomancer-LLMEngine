# AGENT_ADAPTER.md —— 给做模型适配的 agent（和人）读的指南

> 你（agent）的任务：让一个 Dracomancer 暂时跑不动的模型跑起来（甚至跑得好），
> 然后把成果按本文件的格式提交回项目。
> 全程**只产出数据文件和运行结果**，不产出可执行代码——代码由维护者写。

## 0. 三条铁律

1. **先试再说**。大量"新模型"其实是已知架构的新权重/新量化，draco 直接就能跑。
   跑 `python3 draco.py list`，能列出且后端列不是"—"的，属于 **Tier 0：不需要提交任何东西**
   （除非你有调优发现，见 Tier 1）。
2. **只交数据，不交代码**。本机制接受的产物是 JSON 档案 + selfcheck 结果。
   任何"我改了 draco.py/llama.cpp"的方案走 GitHub PR 流程，不走本机制。
3. **结论必须带证据**。没有 `draco.py selfcheck --json` 结果的提交会被 bot 直接退回。

## 1. 决策树：这个模型属于哪一档

```
draco.py list 能看到它吗？
├─ 看不到（GGUF 损坏/不在扫描目录）→ 修路径或换权重，不用提交
├─ 后端列是 "—"（llama.cpp 不支持该架构）
│   → Tier 2：真·新架构。产出「适配档案」（§4），交 issue 等维护者实现 C++。
│     同时可以交一份 Tier 1 档案草案（你探明的参数），维护者实现时会用上。
└─ 能看到、能跑
    ├─ 默认参数就很好 → Tier 0，什么都不用交
    └─ 你发现了更好的参数/修复了模板问题/踩到了坑 → Tier 1：写一份 JSON 档案（§3）
```

## 2. 怎么探参数（Tier 1 的工作方法）

对同一模型逐项试，**一次只改一项**，每项都用 selfcheck 记录前后对比：

| 探什么 | 怎么探 | 判据 |
|---|---|---|
| 后端 cpu vs igpu | `draco.py selfcheck -b cpu` vs `-b igpu` | 速度 + 健康问句都持平甚至更好才推荐换 |
| ubatch（prefill 精度） | `--extra "-ub 1/4/16/64"` 各跑 selfcheck | 健康问句开始失败的上界；档案里写**安全值** |
| 上下文/线程 | `-c` / `-t` 扫一遍 | 速度拐点；内存受限设备宁小勿爆 |
| 采样默认 | 手动 chat 试 temp/重复惩罚 | 不陷重复循环、不改坏事实性 |
| 思考模式默认 | `/think on` 对比 | 开着是否先烧上千 token 才给答案 |

⚠ 如果某个参数改变后**健康问句从通过变失败**，那不是"调优"，是"正确性红线"——
把安全值写进档案，并在 notes 里记录失败现象（这比速度数字有价值得多）。

## 3. Tier 1：档案 JSON（schema v1）

存成 `models.d/<arch 或模型名>.json`，字段白名单**外的任何键都会被拒绝**：

```json
{
  "schema_version": 1,
  "match": {
    "arch": "zaya",
    "name_contains": "可选：general.name 子串（不区分大小写）",
    "max_size_gb": 6.0
  },
  "requires_local_build": false,
  "launch": {
    "extra_args": ["-ub", "1"],
    "ngl": "99"
  },
  "sampling": {
    "temp": 0.7,
    "repeat_penalty": 1.1,
    "max_tokens_default": 2048,
    "think_default": false
  },
  "backend_hint": "local",
  "notes": "人类可读：为什么是这些值、踩过什么坑（重点写！）",
  "source": { "author": "你的 GitHub ID", "date": "2026-09-13",
              "issue": null, "selfcheck_ref": null }
}
```

| 字段 | 含义 | 约束 |
|---|---|---|
| `match.arch` | GGUF 的 `general.architecture` **精确匹配**，必填 | 用 `draco.py list` 里显示的架构名 |
| `requires_local_build` | 官方预编译二进制没有该架构实现，必须用项目本地构建 | 只有维护者确认后才为 true |
| `launch.extra_args` | 追加到 llama-server 命令行；用户 `--extra` 在其后可覆盖 | 字符串数组，如 `["-ub","1"]` |
| `launch.ngl` | 覆盖后端默认 `-ngl` | `"0"` / `"99"` |
| `sampling.*` | chat 的默认采样（只补用户没显式给的） | 数值型 + `think_default` 布尔 |
| `backend_hint` | **默认后端**（用户没给 `-b` 时生效）+ list 显示的建议 | `cpu`/`igpu`/`local`/`npu` |

## 4. Tier 2：新架构的「适配档案」

llama.cpp 不认识的架构需要维护者写 C++ 计算图——**这是你来探路，不是你来写码**。
把下面每一项探明，放进 issue（这能把实现时间砍半，都是我们实测最费时间的部分）：

1. **架构标识**：`general.architecture` 的确切值 + GGUF 来源（HF repo/出处）；
2. **张量清单**：全部张量名与形状（`python3 -c "import gguf,...; ..."` 或
   llama.cpp 的 `llama-gguf` 工具能 dump）；标出哪些是逐层的、哪些是全局的；
3. **hparams**：GGUF 元数据里所有 `*.context_length`/`*.embedding_length`/
   `*.attention.*`/`*.rope.*`/`*.expert_*` 键的实际值；
4. **结构描述**：伪代码写出**一个 token 的前向**（层序、每层的算子顺序、
   注意力变种、MoE 路由方式、各种 norm 的位置）。写得越像代码越好；
5. **状态结构**：有没有跨 token 状态（卷积态/递推态/延迟值）？各多宽？
6. **chat 模板**：tokenizer 里的 Jinja 模板能否渲染？特殊 token 的类型对不对
   （有 bug 就记录 token id 和现象）；
7. **证据**：你自己机器上能拿到的任何对账（比如用 transformers/其它推理器跑
   同一 prompt 的 logits/top-1 对比）。

## 5. selfcheck：提交前必须跑

```bash
python3 draco.py selfcheck -m <模型> -b <后端> [--extra "..."] --json /tmp/sc.json
```

它做三件事：**可复现指纹**（固定 prompt 贪心解码跑两轮）、**健康问句**
（3 道事实题，base 模型可能天然失败，仅参考）、**速度**。
`--json` 的输出整段贴进 issue 的对应字段。

> ⚠ `draco.py` 随**引擎本体**发布，本仓库暂时没有它。在它发布前，请用你手上任何
> 推理器（llama.cpp 的 `llama-cli`/`llama-server`、Ollama、或该模型的官方 runner）
> 跑**同样的协议**并把原始输出贴进 issue：
> ① 同一组固定 prompt 贪心解码两次，比较两次文本是否逐字一致；
> ② 上面那 3 道健康问句的原样回答；
> ③ decode/prefill 速度。
> 协议的意义在于**可比**，不在于由谁跑。（本仓库只放 schema 与流程，不发引擎。）

诚实声明：指纹**不保证跨机一致**（不同 CPU 指令集的累加次序不同），
它是"同机同后端可复现"的诊断 + 维护者比对的参考，不是通过判据。

## 6. 提交

- 用仓库的 **issue 模板「模型适配」**（`.github/ISSUE_TEMPLATE/`），
  把档案 JSON 和 selfcheck JSON 填进对应字段；
- bot 会自动：校验 schema/白名单 → 重放 selfcheck（模型公开时）→ 打 label；
- 维护者审绿色 label 的提交，合入 `models.d/`；
- **不要**在 issue 里贴权重文件、二进制、base64 块——bot 会直接关闭。

## 7. 维护者侧（我们）

- 校验/重放脚本：`release/validate_submission.py`（本地可独立运行，CI 里同一份）；
- CI 骨架：`release/workflow_validate.yml`（仓库建好后挪到 `.github/workflows/`）；
- 合入前必须维护者本机复跑 selfcheck；`requires_local_build: true` 只能由维护者写。

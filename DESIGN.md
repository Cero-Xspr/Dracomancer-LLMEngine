# Dracomancer 引擎设计（脱敏版）

本文描述引擎的分层架构、验证方法论与性能档位。数字均来自参考机
（Ryzen AI 9 H365，DDR5，无独显）的实测，跨机不可比——请读**相对结论**。

## 分层架构

```
draco.py（启动器/CLI/服务）          ← 档案驱动，不含任何模型硬编码
  └─ src/draco_engine_server.py      ← OpenAI 兼容服务 + 各架构适配器
       └─ src/<family>_engine.py     ← 每架构一个 Python 驱动（装载 + 描述符 + 前向编排）
            └─ m6_engine.c           ← 通用层引擎：算子接口 + 整层融合驱动
                 └─ m5/m5_kern*.so   ← 手写量化 gemv 内核（AVX-512）
                    m5_kern_scalar.so ← 标量回退内核（任何 x86-64 可跑）
```

- **m5 内核**：单格式量化 gemv（Q8_0/Q4_K/Q5_K/Q6_K/Q5_0/F16/F32 + IQ 系），
  整调用与行区间两个入口；全部与 numpy 反量化参考做逐 (类型×形状) 三方对账。
- **m6 层驱动**：把「归一化 → 投影 → 注意力/递推 → 残差」融合成每 token 一次 C 调用，
  消灭 Python 编排开销；OpenMP 并行策略由 `autotune.py` 按机器拓扑 + 内核微基准探测。
- **Python 驱动**：装载（`gguf_fast` 零拷贝 mmap）、描述符构造、会话状态管理、分词器。
- **适配器**：服务侧按架构分派；档案（models.d/*.json）决定参数与后端偏好。

## 支持的模型家族（8）

| 家族 | 结构 | 状态 |
|---|---|---|
| llama（SmolLM2 / Llama-3.2） | 标准 decoder | ✅ 引擎固化管线 |
| zaya（ZAYA1-8B） | CCA 卷积注意力 + MoE | ✅ |
| bailingmoe3（Ling） | MLA + KDA 线性注意力 + 分组 MoE | ✅ |
| granitehybrid（Granite 4.0） | Mamba-1/S4D + GQA + MoE | ✅ |
| falcon-h1 | 每层 GQA ∥ Mamba-2 并行 | ✅ |
| qwen35 / qwen35moe | GDN + 全注意力（MRoPE）+ MoE | ✅ |
| bitnet | 登记（b1.58 内核待实现） | 📋 |

每个家族接入都要过三层证据：**逐层对账**（vs llama.cpp 夹具 dump 的逐层 cos 阈值）、
**算子级无状态验证**（单热/合成块提取精确列）、**端到端贪心**（与 llama.cpp 逐 token 一致）。
判据与实测值存于 `tests/golden/*.json`。

## 验证方法论

`tests/run.py` 分层执行（T0 免模型秒级可进 CI / T1 小模型 / T2 大模型对账）：

- **逐位指纹**：改共享代码路径后，跨模型 logits 必须 sha256 逐位一致；
- **增量复用不变式**：前缀续转 vs 全量重跑 logits 逐位相同；
- **UTF-8 流式不变式**：任意字节切分（1431 种）都必须无损还原；
- **贪心金标准**：与 llama.cpp 逐 token 对齐（金标准由脚本当场断言 engine==ref 后落盘）；
- **反面对照纪律**：每个闸门都要验证「注入 bug 会红」——绿的没用比没有更危险。

## 性能档位

| 档 | 条件 | 说明 |
|---|---|---|
| AVX-512（默认） | CPU 支持 avx512f | 手写内核，同机 SmolLM2-135M 183 t/s 量级 |
| 标量回退 | 无 AVX-512 | 慢 ~16× 但正确（端到端贪心与 llama.cpp 一致已验证）；IQ 系格式暂缺 |
| llama.cpp cpu / igpu / npu | 安装对应后端 | igpu 在低功耗态常最快（宽波并行吃满 DDR）；npu 走 FastFlowLM |

带宽要点（MoE 感知口径）：低功耗态 CPU 单线程纯读 ~22.5 GB/s、MoE 式散聚集几乎不亏；
引擎有效带宽 11 GB/s（纯读屋顶的 49%），iGPU 20 GB/s（89%）——iGPU 优势来自取数效率
而非带宽总量。性能数字必须记录 governor/EPP/当前频率与供电状态；用户实测为最终裁决。

## 服务特性

- OpenAI 兼容 `/v1/chat/completions`（流式/非流式），模型自带 jinja 模板渲染
- **上下文增量复用**：多轮对话只 prefill 增量（第二轮 TTFT 实测 2.5ms → 0.6ms）
- **多会话状态槽**：交替会话各自保留状态快照（LRU 2 槽）
- 思考流式分区（`/think`、粒度可调）、UTF-8 流式无损

## 已知边界（诚实版）

- 单请求串行（整代锁）；多并发排队
- 标量回退档不支持 IQ 系量化格式（fail loudly）
- 训练/微调/多模态不在范围内

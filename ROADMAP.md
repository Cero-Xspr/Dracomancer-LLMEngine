# Dracomancer 路线图（草稿 v1，2026-09-15）

> 用法：每项都写清 **为什么 / 做什么 / 怎么算做完**。优先按"风险 × 收益"排，不按新旧排。
> 图例：🔥 现在就该做 · ⭐ 重要但不急 · 💤 可以等（外部条件未到）

---

## A. 正确性风险（最高优先：静默算错比慢致命）

### A1 ✅ 已完成（2026-09-15 结案）：gemv 两个入口的等价性
- **结论**：`gemv_any`（整调用，内含 OMP）与 `gemv_range_any`（行区间，无 OMP）**逐位等价**。
  用**引擎真实调用参数**录制并重放验证：smol 6 组 + Ling 18 组 + ZAYA 10 组，共 **34 个组合全部逐位相同**
  （含 MoE 批量路径/胖并行区用的**部分行区间**，如 224..256；也含 ZAYA 路由器 MLP 的 n_out=17 这类小形状）。
- **工具**：`tests/gemv_audit_replay.py`（C 侧 hook 记录每次 gemv 调用 → 按真实参数重放比对）。
  取参数靠 `m6_engine.so` 的 `m6_audit_*` 导出（`M6_AUDIT=1` 开启，不开时只有一次已缓存的分支判断）。
  旧的 `gemv_entry_audit.py`（按 GGUF 形状推参数）已被它取代 —— 那版会漏掉"引擎传的参数与形状映射不同"的情形。
- **悬案的真相**：曾以为"两个入口不等价"，其实是我给 range 入口传的行区间写成 `0, NOPE`(=128)
  而 n_out 是 `KL`(=512) ⇒ 只算前 128 行、其余留陈旧数据。**教训：先查参数与单位，再怀疑引擎。**
- **MoE 多实现一致性（已完成）**：`tests/moe_paths_compare.py` 用**同一层、同一输入**比较四个实现
  —— 纯 Python 逐专家循环（语义参考）/ `m6_bailing_moe`（融合路由）/ `m6_moe_batch4` / `m6_moe_batch3`。
  实测（Ling 层 1，128 专家选 8、组 8/4、norm_w、w_scale 2.5）：三条 C 路径与参考的差异都是
  **max|Δ|=5.96e-08、cos=1.00000000** ⇒ 只有 1~2 ULP 的累加顺序/FMA 差异，**无语义分歧**
  （脚本把判据写成"cos>0.999999 且 ≤4 ULP"，更大的偏差就会被判为实质分歧）。

### A2 🔥 引擎上下文与启动器一致性
- 已修一部分（ZAYA 的 `ZMAXT` 现在跟随 `-c`；桥接报错显示正文）。
- **还需**：`draco` 启动时把"引擎真实 ctx"回读上来并**在超出时明确提示**（而不是等第二轮才 400）；
  NPU(FLM)/iGPU 的 ctx 语义也不同（FLM 自管 131k）—— 统一成一句"本次会话真实可用上下文 = N"。
- **做完标准**：`draco chat -c 8192` 在每种后端都给出真实可用的 ctx 数字，超限时第一条消息就提示。

---

## B. 性能（已有实测方法学，按杠杆排序）

### B1 ❌ 前提被推翻（2026-09-15，用可信参考复测）：**不做** int8/VNNI
- **关键复测**：把参考换成**引擎自己的 `m5_gemv`**（生产已验证）后，同一张 50MB Q8_0 张量
  （token_embd 49152×960）**单线程就有 40.4 GB/s** —— 这已经接近单核带宽实际可达（20~40GB/s）。
  我写的 VNNI 版只有 **30.4 GB/s（0.75×，更慢）**，而且数值全错（见下）。
- ⇒ **B1 的前提错了**：Q8_0 大矩阵上我们的内核**没有指令效率问题**，它已经是带宽受限。
  之前看到的"单核 10 GB/s"是在**小矩阵**（1.7~2MB 的 attn/ffn 权重）上测的 —— 那里吃亏的是
  **每次调用的固定开销**（正是 B3），不是每字节的指令数。⇒ **把 B3 提上来，B1 暂停**。
- ⚠️ 两次自写实现（浮点重写、VNNI）的数值都错（与引擎参考 cos 0.058~-0.61），连"标量整数参考"
  也对不上 —— 说明我在**读块布局或量化**这一层就有错，且三个自写实现各有各的错。
  教训两次叠加：**参照物必须是生产验证过的代码**（这次改对了），而且**别用自写实现做自证**。
- 保留下来的东西：`tests/i8dot_check.py`（可信的验证框架：引擎内核当参考 + 判据"量化级偏差"）；
  `tests/i8dot_bench.c` 里那条 VNNI 路径**留作反例**（谁要再提 int8 就先用它打掉）。
- 若以后要再捡起 B1：只有在**某个格式在某形状上确实低于带宽屋顶**时才值得做，且必须先跑
  `i8dot_check.py` 把参考对齐（本次连参考层都没对齐就往下做，白费了两轮）。

### B1' 🔥 取而代之：把"单核/小形状"的真实瓶颈问清楚（接 B3）
- **2026-09-15 第一步（可行性微基准，`m5/i8dot_bench.c`）—— 负面结论，但方向没否掉**：
  · 本机 ISA 齐全：`avx512_vnni` / `avx512bw` / `avx512_bf16`（已确认）；
  · 现有四格式（Q5_0/Q6_K/Q8_0/Q4_K）**全是**"整数→float 转换 + FMA"，所以"改整数点积"有出发点；
  · **朴素整数化反而更慢**：真实 Q8_0（49152×960，50MB）浮点 25.4 GB/s vs 整数 21.8 GB/s ——
    每 32 权重块为"权重 +128 变无符号"多付 2 条 maddubs/madd 修正指令，把省下的转换指令吃回去了。
  ⇒ **必须走 VNNI**（`_mm512_dpbusd_epi32`，一条指令 64 个 MAC，两个 32 块塞进同一寄存器、
    修正项折叠进掩码），并按 llama.cpp 的 Q8_0 结构组织；朴素版保留为反例基准。
  · **第二次实测（VNNI 版）**：速度有正向信号 —— 浮点 26.8 / maddubs 23.5 / **VNNI 30.9 GB/s**
    （50MB 张量）⇒ VNNI 比现状快约 **15%**，方向成立；
  · ⚠️⚠️ 但**本微基准的数值列整体不可信**：四个实现两两不一致（浮点 vs 标量整数 cos=0.059），
    连我临时重写的"浮点路径"都是错的 ⇒ 本轮的数值结论**全部作废**（只有计时可用）。
  **下一步（必须按这个顺序）**：① 把参考换成**引擎自己的 Q8_0 内核**（`m5_kern6.so` 的 `m5_gemv`，
    生产已验证），删掉自己重写的那份；② 用它对 VNNI 版量偏差（判据=偏差小且可解释，因为激活被量化了）；
    ③ 速度与数值都过关后，再作为可选分支接进 `gemv_any`（默认关）；④ 逐模型对账后逐个格式开
    （Q8_0 → Q6_K → Q5_0 → Q4_K）。
- **为什么**：我们的单核内核即使数据全在 L2 也只有 ~10GB/s（nibble→float→FP32 FMA）；
  llama.cpp 把激活量化成 Q8_K 走 VNNI 整数点积（一条指令 4 个 MAC）≈18GB/s。这是小模型/省电模式下的主要差距。
- **代价（必须先讲清）**：会改数值（激活量化 = ZAYA `-ub 1` 那类风险）⇒ 必须
  **逐模型对账后再逐个开**，默认不全局打开；`-P eco` 这类场景可作为可选档。
- **做什么**：先做 Q8_0（最简单、Ling 的 k_b/v_b 就是它）→ Q5_0（smol 主力）→ Q6_K/Q4_K（ZAYA/Ling 主力）。
- **做完标准**：每个格式有"整数路径 vs 现在路径"的 cos/逐位判据；模型级 `selfcheck` 指纹不变或差异可解释。

### B2 💤 长上下文的剩余串行项
- ZAYA 三处注意力已经并行（1.27×），但 Tn=768 时 attn 段仍有 11.7ms：里面还有 ck/cv1/cv2 小 gemv 串行、
  qres/kr 循环、MoE 路由。**先量再改**（`DRACO_ENG_PROF=1` + `bench_ab.py`）。
- **做完标准**：每个剩余项都有"占总时间 x%"的数字，改完有交替 A/B 的提速数字。

### B3 🔄 进行中：每算子固定开销 —— 账已算清，瓶颈定位到"每算子一次 fork/join"
- **量出来的账**（smol-360M，T=8，`gemv_callcost.py` + 引擎自带剖面）：
  · smol 路径上 gemv **几乎全走"分段调用"**（整调用 0 次/token；我原来的"测整调用"harness 会高估开销）；
  · **ffn 段 3.89ms/token 已在带宽屋顶**（132MB @ ~34GB/s）；**attn 段 2.23ms 里有约 0.7ms 非流式开销**；
  · 每 token 约 64 个并行区（32 层 × {attn, ffn}）⇒ 0.7ms ÷ 64 ≈ **10µs/次**，即"每算子一次 fork/join"
    （GOMP 被动等待下唤醒 8 个线程就是这个量级）。
- **被否掉的假设**：以为"分段太碎导致每段都有准备开销" ⇒ 实测把分段上限从 32 降到 4：
  总时间 **0.83×**、attn **0.55×（更慢）** —— 因为 8 线程配 4 段会空闲一半线程。
  **结论：要减的是"并行区个数"，不是"段数"。**
- **下一步选项（按预期收益）**：① 持久化线程池 + 自旋栅栏（llama.cpp 的做法，直接吃掉那 10µs/次）；
  ② 把同一层里独立的算子塞进同一个并行区（例如 attn 的 q/k/v 已经在一个区里了，能否把
  attn 与 ffn 的并行区合并？受数据依赖限制，能合的是"残差 add + norm"这类）；
  ③ 接受现状（0.7ms/8ms ≈ 9%），把精力放回 C1/C2。
  ⇒ **先做 ① 的最小可用版**（只给最热的两个 op 换池子），用交替 A/B 验证，再决定是否推广。
- **2026-09-15 结案：B3 也关掉 —— 引擎已经在 DRAM 带宽屋顶上，没有可捡的。**
  ① 零代码实验：`OMP_WAIT_POLICY=active`（+`GOMP_SPINCOUNT=10000`）交替 A/B 实测 **无收益**
     （1.015× / 0.992×，噪声内）⇒ 那 10µs/次**不是线程唤醒延迟**；
  ② 重算账：整个 token 流 268.6MB、Tn=32 用 6.39ms ⇒ **有效带宽 42 GB/s**，
     而 DDR5-5600 双通道实际可达 ~50% ≈ 45 GB/s ⇒ **已在 DRAM 屋顶**；
  ③ 我之前引用的"各格式渐近 30~31 GB/s"是**缓存带宽**（基准把同一个 1.6MB 矩阵反复读，数据留在 L2/L3），
     拿它当 DRAM 上限去算"attn 有 0.7ms 开销"**是算错了** —— 按真实字节数算，attn 段是 56MB/2.02ms
     = 27.7 GB/s 的有效率，与 ffn 段同级，都在屋顶附近。
  ⇒ **结论：B1（指令效率）与 B3（固定开销）都关掉。** 性能上真正剩下的杠杆是
     **减少字节**（更省位的量化）或**换介质**（iGPU/NPU 的独立带宽），不是微优化内核。
     用户可见的下一步改去做 C1（`draco tune` 的 J/token 回归）与 C2（selfcheck 家族级闸门）。
- 已通过"合并并行区"吃掉一部分（smol/llama 4→1、ZAYA 3→1、Ling 48→1 个区）。
- **剩余**：持久化线程池 + 自旋栅栏（llama.cpp 的做法）；我们的线程扩展性已经不差，收益主要在少线程场景。
- **做完标准**：T=4/8 下每 token 调用次数与固定开销占比有数字，且 A/B 有提速。

### ⚠️⚠️ 全局纪律升级（2026-09-16）：`platform_profile` 字段**不足以**代表真实姿态
- **事故**：早前一批 dengine 速度（qwen35 16 / llama1B 33）是 AC 充电 boost 虚高——
  `platform_profile` 明明显示 low-power，但 governor/EPP/充电状态都会独立影响实际频率；
  用户实测 qwen35 只有 5 tok/s。真省电档（AC 满电、governor=powersave、EPP=power、~2GHz）
  重测：qwen35 **4.6**、llama1B **14**（granite 前后一致 12，说明虚高只发生在部分时段）。
- **新规**：①速度数字必须记录 governor+EPP+当前频率，不只 platform_profile；
  ②**用户实测是最终裁决**——我测的数与用户手上的数不一致时，默认我错；
  ③正确性对账（cos/贪心）与频率无关，不受影响（qwen35 贪心序列重测逐 token 不变）。
## ⚠️ 全局纪律（2026-09-15 用户提醒后新增）：**每条性能/能耗数字必须带供电状态**
- 用户指出「上一轮我离电了」—— 一查：同一份 Q8_0 gemv、同一个内核，**AC 上 40.4 GB/s，
  电池上只有 19.2~24.4**（1.7~2.1×）。我一度把这当成"测量噪声"，其实是**电池下平台 PPT 上限低得多**。
- ⇒ `hwprobe.power_state()` 已做成唯一实现，并由 `draco tune` / `bench_ab.py` / `gemv_callcost.py`
  **强制打印**（电池下还会加一句"跨供电不可比"的警告）。
- ⇒ 凡是"某内核 x GB/s""某配置 y J/token"的结论，**必须注明 AC 还是电池**；跨供电比较无效。

### B4 ⏩ 有条件继续（2026-09-15 接电后 AC 复测，修正上一轮结论）
**AC（Charging 91%、平台档 performance）复测同一张 50MB Q8_0 张量**：
| | CPU（引擎 m5_kern6，单线程） | iGPU（Q8_0 OpenCL） | 比值 |
|---|---|---|---|
| AC | 25.1 GB/s | **46.3 GB/s**（lws=64） | **1.85×** |
| 电池 | 24.4 / 19.2 GB/s | 26.5 / 35.0 GB/s | 1.02~1.82×（不稳） |
| 小形状 320×960（两次都测） | 19.4 GB/s | **4.2 GB/s** | **0.22×（iGPU 惨输）** |
- **结论修正**：B4 不是"暂停"，而是**形状相关**：
  ① **超大形状值得卸载**（AC 下 1.85×，且与电池那次 1.82× **比值一致** ⇒ 这是真实的内核水平差，
     不是噪声）；
  ② **中小形状绝不能上 iGPU**（0.22×，内核启动延迟 ~77µs 收不回来）——而它们才是每 token 调用 200+ 次的主力。
  ⇒ 若做 M2/M3，**目标固定为"只卸载 head 与 MoE 专家矩阵"**。
- 教训补充：**比值也可能随供电状态变化**（电池那次 CPU/iGPU 打平）⇒ 任何 A/B 结论都要标明供电，
  并且**同一份数据要在 AC 上复测一次**才算定论（这次就是 AC 复测把结论从"暂停"改回"有条件继续"）。
- ⚠️ 仍要注意：这台机只有这一个 GPU 且在渲染桌面，行内有重度 compute 需避让（曾触发过一次 hard recovery，
  桌面未受影响）。

### B4（早期记录，含 GPU 事故）
**M1 实测（Q8_0 的 OpenCL gemv，`igpu/gemv_q8_0.py`，与引擎 CPU 内核同形状对拍、正确性 cos=1.0000001）**：
| 形状 | iGPU | CPU（同 run） | 结论 |
|---|---|---|---|
| 320×960（0.33MB，attn_v） | **4.2 GB/s** | 19.4 GB/s | iGPU **输 4.5×** —— 77µs 里主要是**内核启动延迟** |
| 49152×960（50MB，词嵌入/head） | **35.0 GB/s** | 19.2 GB/s | iGPU 1.82×，**但 CPU 基线不可信** |
· **CPU 基线不稳**：同一张 50MB 张量、同一个内核，两小时内两次测到 19.2 与 **40.4** GB/s（2.1× 差）。
  想用交替 A/B 定论时**我的 OpenCL context 崩了**（`CS has cancelled because the context is lost`
  / hard recovery）⇒ **判决未成立**。
· **重测（2026-09-15，电池 98% 放电中，测前/测后都确认）**：CPU **24.4** GB/s vs iGPU **26.5**（lws=128）
  ⇒ **基本打平（1.02~1.09×，噪声内）**，而且两边都随供电/功耗态大幅摆动（CPU 40.4(AC) → 19.2/24.4(电池)；
  iGPU 35.0 → 26.5）。⇒ **上一轮那个"iGPU 1.82×"不可复现**，B4 的判决仍不成立（不是"CPU 更快"，
  而是"在电池下测不出差别"）。**要定论必须在 AC + 安静机器上做交替 A/B**（等用户方便插电时再说）。
· **事故记录（重要）**：这台机**只有这一个 GPU，它同时在渲染桌面**。这次 hard recovery 只炸掉了我的
  CL context，**桌面会话完好**（gnome-shell 47h 未断、wayland/Xwayland 正常）。但这是真实风险：
  **在显示 GPU 上跑重度 compute 可能波及交互**；真要做必须在用户不依赖桌面时做，且不要放循环里反复触发。
- **判定：B4 暂停**（按 M1 的过关线「明显超过 CPU」）：
  ① 调用次数最多的中小形状**明确输**（启动延迟）② 只有超大形状（head/专家矩阵）有希望，
  而其 CPU 基线两测相差 2.1×，需要在**安静机器**上重测才谈得上结论；
  ③ 该机 iGPU 的能效优势目前是**llama.cpp 的内核**跑出来的（C1 的 J/token 也印证），我们还拿不出等价的内核。
- **若将来重启 B4**，第一目标固定为"**只卸载超大形状**"（head + MoE 专家），且：安静机器 + 交替 A/B +
  不在用户用桌面时跑；中小形状留给 CPU（连启动延迟都收不回来）。

### B4（更早的第一步记录，保留）带宽有空间，但内核差得远
- **测出来两件事**（`igpu/igpu_spike2.py`，OpenCL 直连，无需 pyopencl）：
  ① **iGPU 纯读带宽 68.5 GB/s（16MB）/ 81.5 GB/s（64MB）** —— 而我们 CPU 引擎的**整 token 有效带宽
     只有 42 GB/s** ⇒ 带宽侧确实有 1.4~1.9× 空间（这推翻了记忆里「iGPU 25.4GB/s 略输 CPU」的旧结论，
     那是内核吞吐，不是带宽）；
  ② **但现有 CL gemv 内核只跑出 6.9~13.6 GB/s**（IQ3_S；8 专家 3.6MB → 13.6，64 专家 28.8MB → 6.9）
     ⇒ 只有可用带宽的 10~20%：**瓶颈是内核（反量化指令成本），不是带宽**。正确性 cos=1.0000001 ✓。
- ⇒ **B4 的真瓶颈不是「把引擎搬到 iGPU」，而是「给我们的热格式写高效 CL 内核」**：现在这套直接搬过去
  只会更慢。而且这次测的是 IQ3_S（查表型、最慢的格式之一），**热格式 Q4_K/Q6_K/Q5_0/Q8_0 这个 spike
  里根本没测**。
- **里程碑拆解（逐档可验证）**：
  M1 单格式（建议 Q8_0：最规整，int8 + 每 32 一块 f16 尺度）CL gemv，靶子 = 该形状的 CPU 数字
     （Q8_0 320×960 → 27~28 GB/s；head 49152×960 单线程 → 40 GB/s）；**打不过 CPU 就停在这里**；
  M2 过一个 MoE 专家块（`igpu_moe.py` 已有雏形），核对与 CPU 一致；
  M3 接成可选后端（默认关），跑 `family_gate.py` + `draco tune`（后者直接回答「iGPU 在能效上是否也赢」）；
  M4 再扩其余格式与 ZAYA/Ling 的专用算子（CCA/KDA/MLA）。

### B4（旧描述，保留）Darco 的 iGPU 路径
- **为什么**：现在 `-P speed/eco` 会把**有 Darco 适配器的模型推给 llama.cpp**（Darco 只有 CPU 链）；
  而 iGPU 在这台机上同时胜出速度与能效（实测 CPU 679 / iGPU 563 mJ/token）。
- **做完标准**：`-b dengine` 能选 iGPU，且与 CPU 路径有 cos 对账 + 端到端 A/B。

---

## C. 适配与协作机制（发行前的护城河）

### C1 ✅ 已完成（2026-09-15）：`draco tune` 的 J/token 回归（smol / Ling / ZAYA）
- 先修了两个前置问题：① `tune` 的配置空间里有 `-ub`/`-fa` 这类**llama.cpp 专属参数**，
  dengine 后端硬塞会让桥接 argparse 直接退出 ⇒ 现在按后端分流（dengine 只扫线程数）；
  ② `backend_metrics` 的**单位错**：tune 缓存是 J/token、MEASURED 表是 mJ/token，
  混用会打印出"0 mJ/tok" ⇒ 统一成 mJ/token。
- 结果：**smol** `-ub 1` 0.106 J/tok 最优（但只比基线快 5%，噪声内，故不改档案）；
  **Ling** 总口径说 `-fa 0` 好、边际口径说基线好 ⇒ **结论不可用**（工具自己警告能耗排序要机器安静）；
  **ZAYA** 基线与 `-ub 1` 完全一致（档案本来就带 `-ub 1`，互为验证）⇒ 档案参数确认最优。
- ⇒ 收益：`-P eco` 从"能耗序先验"升级为**本机 tune 实测**（已实测确认：决策理由会显示
  「tune 实测（8|-ub 1）」）。**教训**：能耗结论只在"同一轮、机器安静"时可比；跨轮不可比。
- **为什么**：`-P eco` 现在只有"能耗序先验"（NPU<iGPU<CPU），没有本机实测；
  tune 跑过之后倾向决策才从"先验"升级为"实测"。至今只跑过 falcon-h1。
- **做完标准**：至少 3 个代表模型（小 dense / MoE / 混合 SSM）各有 J/token 数字，写进 `models.d/*.json`。

### C2 ✅ 已完成（2026-09-15）：家族级合规闸门 `family_gate.py`
- 定位：`selfcheck` 答"这个模型现在跑起来对不对"；家族闸门答**三个结构性问题**
  ① 跨后端一致（cpu/igpu/dengine 对同样问题是否同答 —— 单跑一个后端看不出来的"静默坏掉"）
  ② 跨量化一致（同架构不同量化不该改答案）③ 长上下文退化曲线（0/256/512/1024 token）。
- 判据（合格线）：① 各后端健康问句全过 + ② 跨后端**逐字一致** + ③ 长上下文不报错；
  长上下文"答案退化"给 **WARN 而不是 FAIL**（小模型天然会掉，大模型掉了才要查上下文处理）。
- 实测结果（2026-09-15）：
  · **ZAYA：PASS** —— dengine vs llama.cpp(local) **逐字一致**（3/3），且长上下文曲线**完全平**
    （0/256/512/1024 都答 "2"）。这是自研引擎在 CCA+MoE 上的又一次独立验证。
  · **Ling：PASS** —— dengine vs igpu 一致，长上下文曲线同样平。
  · **smol-360M：PASS**（三后端逐字一致）；但长上下文从 256 token 起答案退化 ⇒ **WARN**
    （360M 玩具模型的预期行为，作为数据点记录）。
- 校准过程中修了两个自己的问题：填充文本按 20 token/句估 ⇒ 1024 档超上限被打 400
  （现在按 25/句，且 400 视为"这档测不到"而非失败）；把"小模型长上下文变傻"当 FAIL 也是不对的。
- 保留下一步：`draco.py family` 的 CLI 包装（现在是独立脚本，功能等价）。

### C2（旧描述，保留）`selfcheck` 升级为家族级合规闸门
- **为什么**：现在是单模型体检。社区提交适配时，需要的是"这一族/这条引擎实现是否可信"的自动闸门。
- **做什么**：家族级判据（同架构不同量化之间的一致性、跨后端一致性、长上下文退化曲线）+
  失败时给出可复现命令。
- **做完标准**：一个新模型接进来，一条命令就能给出"可以去提 issue 的证据包"。

### C3 🔄 进行中（2026-09-15 第一小步：**完整性校验 + 把 SSM/KDA 的待定事实显式列出来**）
- **已完成**：`archspec_spike.py` 新增 `validate_spec()` —— 每条语义事实必须带
  `(value, why, shape_invisible, how_found)` 四件套，`value=None` 视为**未确定**，一律 `SystemExit` 报错。
  **纪律：缺字段必须报错，不许静默走默认**（静默默认＝"能跑但算错"，比报错危险得多）。
- **已完成**：`SEMANTICS_SSM`（granite-hybrid/Mamba 系）与 `SEMANTICS_KDA`（bailingmoe3）各列出 5 条
  **已知需要、尚未测定**的事实（层类型索引方式、卷积宽度/状态复位语义、dt 秩、门控 clamp、衰减参数取法、
  beta 激活、norm 位置…），每条都写清"要看什么"，`value=None` ⇒ 现在这两族**跑不起来**（会被拦），
  这正是"没覆盖"该有的样子。
- **自证**：`selftest_validate()` 证明拦得住 —— 完整规格通过；SSM/KDA 被拦；删掉一条事实的 `how_found`
  也被拦（避免"校验器自己不报错"的假通过）。
- **已完成（第二小步，提交 72a9bf2）**：10 条事实**测定回填**（KDA 5 条取自**已对账过**的 ling_proto
  —— vs llama.cpp cos 0.998814；SSM 5 条取自 GGUF KV + llama.cpp 算子表 + 本地 granite-h-tiny 实测）
  ⇒ 两族现在**能通过校验**；保留"删掉一条 how_found 也会被拦"的探测。
- **已完成（第三小步）**：补上 `LAYER_STEPS_SSM` / `LAYER_STEPS_KDA`（层内步骤数据），
  并加 `validate_steps()`：每个 op 名合法、每条 `semantics` 引用都能在事实表里找到。
  ⇒ **描述层三族齐全且可校验**。新增 `unimplemented_ops()` **自动列出求值器缺口**：
  `ssm_conv`、`ssm_scan`、`kda_delta` —— 这就是 C3 的剩余工作清单（而不是靠人记）。
- **数值层进度（2026-09-15）**：
  ① ✅ **`ssm_conv`**（因果深度可分离 conv1d，SSM 与 KDA 共用）：纯函数 `ssm_conv_step()` +
     **KAT 已知答案测试**（随机权重/序列，与朴素参考**逐位相同**；含状态推进与 `pos==0` 复位，
     且测试自证"复位确实改变数值"以免测试无效）。已登记进 `IMPLEMENTED_OPS`。
     坑：偏置必须**最后**加 —— 我先写成"从 b 起累加"，差 **1 ULP**，而判据是逐位相同
     （对账用：差 1 ULP 说明顺序没对齐，不是"精度问题"）。
  ② ✅ **`ssm_scan`**（SSM 递推，Mamba2 风格）：`dt=softplus(dt_pre+dt_bias)`（clamp）→
     `dA=exp(dt*a)`、`x'=dt*x` → `h = dA*h + x'⊗b` → `y = Σ h*c + d*x`。
     KAT **内部一致性**通过：与朴素逐帧参考**逐位相同**、状态复位确实影响输出（|Δ|=0.121）、
     a≤0 时 50 步稳态有界（|h|max=0.21）。
     ⚠️ **公式本身尚未与 llama.cpp 对账**（a 的取法、softplus 的 clamp 界、D 跳连的位置都要读
     `build_mamba2_layer` 逐行核对后回填）—— 这是**有意标注**：B1 那次"四个自写实现互相打架"
     就是自写参考自证的代价，所以这里只声称"内部一致"，不声称"与参考一致"。
  ③ ✅ **`kda_delta`**（delta-net 递推）—— **逐字移植自已对账过的 `ling_proto.kda_step`**
     （vs llama.cpp cos 0.998814、top-5 一致）⇒ 三族里第一个有**外部参考**的算子。
     语义：g=sigmoid(gate·ssm_a)·GATE_LB（逐通道衰减）→ S 衰减 → delta=(v−S·k)·beta →
     S += k⊗delta（rank-1 写入，i=key 维、j=value 维）→ o=(S·q)·hd^-0.5 →
     per-head-dim RMS(o_norm) → ×out_gate。
     **四项语义性质测试**（不靠自写参考自证）：① β=0 且 g=0 ⇒ 状态完全不变 ✓
     ② delta 规则的定义性质：‖S·k − v‖ 一步归零（**因为 ‖k‖=1**，这正是索引约定正确的证据 ——
     写错维度就不会是 0）✓ ③ 强衰减(g=-30) 后 S == k⊗v（max|Δ|=0）✓ ④ 读出与缩放公式一致 ✓
- ⇒ **C3 数值层 3/3 完成**：`unimplemented_ops()` 已返回**空**。
  （`unimplemented_ops()` 现在会自动报 `ssm: [ssm_scan]`、`kda: [kda_delta]`）
- **装置总验收（2026-09-15）**：`python3 archspec_spike.py --selftest` 一条命令跑完
  ①语义事实完整性 ②三族层内步骤校验 ③三个算子 KAT ④缺口清单（自动）—— 可进 CI 当闸门。
  实测全过：`ssm_conv`/`ssm_scan`/`kda_delta` KAT 通过、三族步骤通过、缺口为**空**。
- **剩余（C3 收尾，明确且不小）**：三个算子还**没串进求值器的 op 分派**（现在只是各自的纯函数 + KAT），
  所以还不能端到端跑一族的模型。要做：
  ① ✅ **权重布局断言已做**（`check_conv_layout` / `check_vec_layout`，已接进 `--selftest`）：
     `ssm_conv1d.weight` 可能是 [C,width] 也可能是 [width,C]，`ssm_a` 可能是 [nh]/(1,nh)/(nh,1) ——
     **拿错不报错、只会静默算错**，所以加载时必须显式确认，确认不了就带着尺寸报错，**绝不猜**。
     自证：合法轴序通过；错形状（(8,5)/(3,4,8)/(4,)/(16,4) 等）一律被拦。
  ② ✅ **完成**：三算子接进 `Evaluator` 分派 + `Ctx.has()` + `Evaluator._state(key,shape,pos)`
     （pos==0 清零 = 语义事实 ssm_state_reset）；`ssm_scan` **强制声明 `split`**（切分不许猜）、
     `ssm_conv` 强制过布局断言（WC 轴序会被认出并转置）。
     **合成权重烟测已通过**（上一轮它卡在我自造的张量形状自相矛盾上；这次**先把形状账在纸上推清**——
     conv 通道 = x:4 + dt:4 + B:32 + C:32 = 72、`ssm_out` 取 [H,NH]——再写代码，一次通过）：
     SSM 族声明被完整执行 3 个 token，末状态全有限、形状正确；并验证 pos==0 复位真的改变输出。
     烟测已**接回 `--selftest`** 闸门。
     ⇒ 当前可声称：**SSM 族的声明式描述是可执行的**（KDA 族的接线同构、执行路径同一段代码）。
  ② 权重加载补 SSM/KDA 张量名（`ssm_conv1d`/`ssm_a`/`ssm_dt`/`ssm_d`/`ssm_beta`/`ssm_norm`…），
     并对**权重布局**（如 `ssm_conv1d.weight` 是 [C,width] 还是 [width,C]）做显式断言 —— 拿错不报错才最危险；
  ③ ✅ **KDA 族真权重对账完成（2026-09-15）**：`tests/kda_reconcile.py` —— 按 SEMANTICS_KDA 语义
     **独立重写**的求值路径（fp32 权重 + numpy）vs `ling_proto.kda_step`（已对账 llama.cpp
     cos 0.998814 的参考），Ling 层 0/1/2 三层 **cos=1.00000000**（max|Δ| ≤ 2.8e-08）。
     过程中抓到一个自己的 bug（第一版三个 conv 分支全错用 `conv_q` ⇒ cos 0.056）——
     **逐段二分**（conv → delta → 读出）一轮定位。
     ⇒ **「声明式描述能复现真模型」在 KDA 族上被验证**（描述 → 语义 → 独立实现 → 与已对账参考一致）。
  ③a 🔄 **granite-h-tiny（SSM 族）的发现与待办**：**它是 Mamba-1/S4D 结构，不是 Mamba-2**——
     `ssm_a` 形状 (1, 48) 是 **[1, dt_rank]**（A 作用在 dt 秩上，每 state 列共享），
     而声明式求值器的 `ssm_scan` 实现的是 Mamba-2（A={d_state, n_head}，作用在 state 维）。
     `ssm_in` 6448 = dt_rank 48 + inner 3072 + 2·state·group 佐证（llama.cpp mamba-base 同款）。
     ⇒ 不能拿现算子直接对账。
     **✅ S4D 语义链已在真实权重上走通（2026-09-15）**：完整链 =
     in_proj(6448) → 切 [z 3072 | xBC 3328 | dt 48] → conv(3328 通道、4 tap、**只作用在 xBC**) →
     **silu** → 再切 [B 128 | C 128 | x 3072] → softplus(dt)·A（A={1,n_head} ⇒ **标量 dA 分支**）→
     状态 → 读出 → **y += x·ssm_d（按头，D 跳连乘的是 scan 输入段 x）** → ssm_norm → ssm_out。
     实测：输出 (1536,) 全有限、dA∈[0.699,1.000]（<1 衰减正确）。
     语义事实已写进 `archspec_spike.py`（S4D 节）。
     **对账进展（2026-09-15，未完成但重大推进）**：
     ① 给 `zaya_gdump.cpp` 加了 `mamba2_y_add_d` / `mamba_out` 两个名字（granite 图里仅有的 cb 名），
        编译并成功 dump granite 40 层；
     ② **链尾验证通过**：`swiglu(z, dump_y_add_d) → rms_norm → ssm_out` vs `mamba_out-0`
        **cos=0.99968** ⇒ 链尾语义（swiglu 位置、norm、投影）确认；
     ③ **✅✅ 链头对账已完成（2026-09-15）—— 原「cos≈0.06 分歧」撤回**：
        **分歧是我自己比较脚本里的索引错，不是引擎或模型的问题**。dump 的内存序是 ne0 最快 =
        `[dim(64) 最快, head(48)]`，所以 `flat.reshape(48,64)` 才是 `[head, dim]`；
        我当初写成 `flat.reshape(64,48).T`，等于把参考值自己转置了两次 ⇒ 余弦自然接近正交。
        正确读法下的实测（`s4d_reconcile2.py`，逐位置递推，pos 0..3 × 全部 36 个 SSM 层）：
        **链头 cos 中位 0.999990~0.999994、最小 0.999147**（残差是 Q4_K 把激活量化成 Q8_K 的
        固有误差）；链尾 0.99951~0.99966。**⇒ S4D 全链（conv 历史帧顺序 / A·dt 衰减 /
        状态更新次序 / y 展开顺序 / swiglu / grouped RMSNorm / ssm_out）与 llama.cpp 逐层一致。**
        敏感度对照（`s4d_chainhead_probe.py`）证明链尾对 y 是敏感的（同范数随机 y 只给 0.0069、
        全 1 给 −0.0827）⇒ "链尾对上"确实有信息量。
        过程中真抓到一个 bug：`granite_engine.py` 的 `y = y_hk.T.reshape(-1)` 是错的，
        必须 `y_hk.reshape(-1)`（h 主序）；对照实测 h 主序链尾 0.9998 vs 转置 −0.0436。
        **⇒ 不需要改 llama.cpp 的图代码、不需要重编译库** —— 此前"必须给夹具补链头锚点"的结论
        建立在那次假分歧上，一并撤回（夹具只重编译过一次：加 mamba 的两个 cb 名）。
     ④ 已确认的新语义：**y = swiglu_split(z, y_add_d)**——z 段是门（silu(z)·y），
        位置在 D 跳连之后、norm 之前。另**修正切分顺序**：in_proj 输出切成
        `[z(3072) | xBC(3328) | dt(48)]`，xBC 内部再切成 **`[x(3072) | B(128) | C(128)]`**
        （此前本文写成「B|C|x」，按 mamba-base.cpp 238-244 行的 view 偏移，**x 段在前**）。
     过程小坑：xBC 切片一度写成 2·G·ST(=256)（正确是 DI+2·G·ST=3328）、
     D 跳连对象试错了两版（乘层输入 z 不成立，乘 scan 输入 x 按头才形状自洽且语义合理）。
     `ssm_a` 形状 (1, 48) 是 **[1, dt_rank]**（A 作用在 dt 秩上，每 state 列共享），
     而声明式求值器的 `ssm_scan` 实现的是 Mamba-2（A={d_state, n_head}，作用在 state 维）。
     `ssm_in` 6448 = dt_rank 48 + inner 3072 + 2·state·group 佐证了这一点（llama.cpp mamba-base 同款）。
     ⇒ **不能拿现算子直接对账**；正确路径是二选一：(a) 给求值器加 S4D 分支（A 取 [1,dt_rank]、
     Δ=B/C 分组共享），或 (b) 先用 **Ling（KDA 族）**完成真权重对账（它没有这个结构分歧），
     S4D 对账排后。**「先纸上对账再写代码」再次避免了一轮白干**。
  ④ ✅ **`ssm_scan` 公式已与 llama.cpp 逐行核对**（来源：`models/mamba-base.cpp` 的图 +
     `ggml-cpu/ops.cpp` 的 `ggml_compute_forward_ssm_scan_f32`），**修正三处凭常识写错的地方**：
     ①内核**无 clamp**（旧版凭印象加了 ±4）；②**A 形状 {d_state, n_head}**，d_state>0 时
     `dA = exp(dt*a)` **逐 state 元素**（旧版写成逐头标量）；③内核**没有 D 跳连**
     （`y = Σ h·c` 而已，旧版加了 `d*x`；Mamba 论文的 D 项在 llama.cpp 里不在 ssm_scan 内）。
     KAT 同步更新后全绿（被测与朴素参考逐位相同；复位 |Δ|=0.169；A≤0 稳态有界）。
     ⇒ 「先核对再对账」的顺序被证明正确：带旧公式去对 granite-h-tiny 会白对一轮才发现这三处。
  ⇒ 这四步做完，"声明式描述能复现真模型"才算被验证过（当前只验证了"描述完整 + 算子自身正确"）。

### C3（旧描述，保留）声明式表达力覆盖混合 SSM/KDA
- archspec spike 已验证 llama 密集 + MoE（cos 0.999962 / 0.9998），但 granite-hybrid、bailingmoe3(KDA) 还没进。
- **做完标准**：把这两族的"形状看不出来的语义事实"补进 `SEMANTICS_*`，并让缺失字段**报错而不是默认**。

### C4 🔄 进行中（2026-09-15）：granite-hybrid 的 Darco 适配
- **账已盘**（张量 40 层 = SSM 36 + GQA 注意力 4 [5/15/25/35] + MoE 64 选 6 + 共享专家全 40 层）：
  · **可复用**：attn（m6_llama_attn_op）、MoE（m6_bailing_moe：granite 是 softmax 路由 + top6 +
    norm_w、无分组无偏置、**无 `expert_weights_scale`（⇒ 不缩放）** ⇒ n_group=1/probs_b=null 同构）
  · **新写**：S4D 段（36 层核心）—— 原型已固化在 `granite_engine.py`，语义来自 llama.cpp 逐行核对，
    **并已逐层对账通过**（见 C3 ③a），可以下沉 C
- **新查清的三条 granite 专属语义**（都会影响数值，写在这里免得重踩）：
  · **注意力层不套 RoPE（NoPE）**：GGUF `rope.scaling.finetuned=0` ⇒ llama.cpp 的 granite-hybrid
    把 `rope_pattern` 全填 false ⇒ `has_rope(il)==false` ⇒ 图里 `inp_pos=nullptr`、
    `build_attention_layer` 整段 `if (hparams.has_rope(il))` 被跳过。**不做 rope 才对**。
  · **无任何 bias**：attn/ffn 一个 bias 张量都没有（`create_tensor_qkv` 传 flags=0）。
  · 残差：分支输出 **先 ×res_scale(0.22) 再加**（attn/SSM 分支和 MoE 分支各一次），
    层输入 `attn_norm` 是 **RMSNorm 的输出**（夹具里 dump 的 `attn_norm-{il}` 就是 SSM/attn 的输入）。
- **对账卡点已解除**：原以为要改 llama.cpp 图代码 —— 实为我的索引假分歧（见 C3 ③a）。
- **✅✅ 已完成并登记（2026-09-15）**：`m6_granite_s4d_op` / `m6_granite_attn_op` / `m6_granite_moe`
  + `m6_granite_forward_token`（40 层图，含两处残差缩放与共享专家）已落地；
  `granite_engine.py` 当驱动、`granite_tok.py` 当分词器；`_DENGINE_ADAPTERS` 已登记
  （key 用 GGUF 原名 `'granite 4.0 h'`，不是 `'granite-h'` —— 匹配的是 `gguf_name` 而非文件名）；
  `models.d/granite.json` 已写。
  **两层证据**：① 算子级 S4D 逐层对账 cos 0.99999（pos 0..3 × 36 层）；
  ② 端到端真实句子贪心 **16/16 token 与 llama.cpp 完全一致**；
  ③ 经 `draco_engine_server.py --engine granite` 实测答出 "The capital of France is Paris."。
- **✅ 性能已修（2026-09-15）：6.42 → 12.45 tok/s（1.94×）**，根因是**驱动里的一行**
  `row = bytes(_EMB.data)[a:b]` —— `bytes()` 每个 token 把**整个 125MB 词嵌入复制一遍**再按字节
  偏移切片（占端到端的约 58%！）。gguf_fast 的 `.data` 本来就是 mmap 上的只读 uint8 **二维**视图
  `(n_tokens, bytes_per_row)`，直接 `[tid]` 取行即可（按字节偏移切是切「行」维度，会切出空数组）。
  同会话交替 A/B（都带负载记录）：**dengine 中位 12.16 vs llama.cpp 中位 10.3 tok/s**
  ⇒ 从"慢 1.5×"变成**至少打平（区间重叠）**。
- **★ 修正（2026-09-15，用户指出后复测）："打平"只在 CPU 后端成立。** llama.cpp 的 **iGPU 后端**
  （官方 vulkan 包或自建 build-vk，`-ngl 99`）稳定 **21.4~21.9 t/s**，是 CPU 后端的 2.3×，且几乎
  不受 CPU 负载影响。所以三方对照是：**dengine(CPU) 中位 8.1 / llama.cpp CPU 中位 9.4 /
  llama.cpp iGPU 中位 21.9**。⇒ 我上一轮把 llama.cpp 的基线测错了（没给 `-ngl`，拿的是纯 CPU 数）。
  **dengine 目前是纯 CPU（AVX-512），没有 GPU 路径**（draco.py 的 `_SPEED_PRIOR` 注释里早就写着
  "dengine 目前仍跑在 CPU 上（iGPU 未接线）"）⇒ **按绝对速度，这台机上跑 granite 最快的方式是
  llama.cpp + iGPU**。已在 `models.d/granite.json` 把 granite 的 llama_cpp 默认后端改成 `igpu`。
  dengine 要不要接 GPU 是另一个量级的工程（且早前的 iGPU 内核实验结论是"带宽受限、仅重载时胜出"，
  与这里 llama.cpp 走 fp16 拿到 2.3× 的机制不同）——记为待议，不要顺手开工。
- **★ 后端政策（实测后的现状，别搞混）**：`draco chat/serve -m granite` 默认（balanced）仍然
  **走 dengine**（"自研引擎优先"是有意设计）；要 iGPU 得显式 `-b igpu` 或 `-P speed`。
  已把实测灌进 `MEASURED`（granite: igpu 21.9 / dengine 12.2 / cpu 9.4，能耗列 -1=未测），
  于是 `-P speed` 实测确认会选 igpu（日志："选定 igpu —— 21.9 tok/s，档案性能表"）。
- **★ 顺带修掉一个静默失效**：`MEASURED` 表原来用 `m.name` 直接查，而档案的 display_name 带架构
  后缀（`Ling 3.0 Tiny（bailingmoe3）`）⇒ **查不中**，Ling 的实测吞吐/能耗一直没被用上
  （被 tune 缓存掩盖了，所以没暴露）。改成 `_measured_key()` 依次试 全名 → 去括号后缀 → GGUF 原名。
  教训与 C1 的单位错同类：**"表里有数据"不等于"数据被用上了"**，加条目后必须回读一次命中。
- **D3 长上下文（granite/dengine）✅ 2026-09-15**：`--ctx 4096` 下实测两档
  —— 624 token（暗号在开头）答对 7391；**1326 token（暗号埋在 1/3 深度）答对 5482**，
  无 400/500、prefill ~16.7 t/s。⇒ 不只是"不报错"，而是**真的用上了上下文**；
  也确认 `--ctx` 透传生效（没有 ZAYA 那种"ctx 硬写 1024"的老毛病）。
- **修后的真实分段账**（AC / 平台档 low-power / load ~10；`m6_prof_t[8..23]` 是这段的专用槽）：
  wall 75.9ms = MoE 32.8（**43%**：gate+up 17.6 + down 11.2 + silu 0.9 + 加权 1.3 + 路由 1.4）
  + 分支 27.3（S4D 36 层：in_proj 14.0 / ssm_out 6.6 / scan 4.3 / conv 0.8 / norm 0.2，attn 4 层在内）
  + 共享专家 8.4 + 其余 ~7（lm_head 与 Python 侧）⇒ **分段与墙钟闭合**（此前差 58% 就是那个复制）。
- **方法学教训（这轮最贵的）**：同一份 MoE 代码，两次分块实测分别是 **194.9ms 与 38.6ms**（5×）——
  单次采样在负载波动下完全不可信；必须交替 A/B + 记录 load。另：**孤立微基准会骗人**
  （"gemv_any 整调用比行区间慢 15×"在端到端复现不出来，据此做的融合改动 A/B 无差异 ⇒ 已撤回）；
  内核微基准只能用来判断"有没有并行加速"这类**相对**性质（实测：行区间入口在 1 线程与 8 线程下
  耗时相同 ⇒ 完全没有并行加速，并行度必须由调用方切行提供；整调用在 8 线程下 2.7×）。
- **顺带抓到的引擎级坑（已修，对所有架构成立）**：驱动**不设 OMP_NUM_THREADS** 时默认 20 逻辑核，
  granite 从 7.8 → **1.06 tok/s（7.4× 退化）**；Ling/smol 一直靠 autotune 设这个。
  `granite_engine.py` 现在先跑 `autotune.plan` 再 dlopen m6（**必须在首次 dlopen 之前**，
  libgomp 初始化时就把该变量读走了）。新适配器一律先过 autotune。

### C4（旧描述，保留）更多模型适配
- granitehybrid、qwen35/qwen35moe…；每个都要过数值对账才登记进 `_DENGINE_ADAPTERS`。
- **做完标准**：登记即"已对账"，档案里带证据。

---

## D. 体验（用户可见，随时可做）

### D1 ✅ 已完成（2026-09-15）
- 装载 39s→3.05s（ZAYA）；思考按行流式（原来是按段挂起 ⇒ 思考憋到最后才蹦出来）；
  历史带 `reasoning_content`（否则第二轮起模型模仿"不思考"）；`-P speed/balanced/eco`；
  超长提示词给 400 而不是"连接被断开"；`DRACO_ENG_PROF=1` 分段耗时。

### D2 ⭐ ZAYA 的"思考开关 × 质量"取舍要写清楚并定默认
- 实测：think off → 17×23 答 **497（错）**；think on → 391（对）但思考会烧满 max_tokens（2048 tok ≈ 85s）。
- 现在档案默认 think off（快）。**待你拍板**：默认值放哪边？或按问题类型自适应？
- **做完标准**：档案里写清取舍 + 默认值 + 一行命令切换；README 里也提一句。

### D2b ✅ 完成（2026-09-15）：粒度可调 + 已接进 `/hold` 运行时切换
- 两种用法：环境变量 `DRACO_THINK_HOLD=line|token|dup`（启动时）与**请求字段 `draco_think_hold`**
  （每请求；draco 的 `/hold` 命令走这条）。桥接把粒度从"类属性（import 时固化）"改成实例参数、
  缺省取环境变量 ⇒ 两种都支持。
- 实测（同一问题、ZAYA 开思考、走真实 HTTP 接口数分片）：`line` → **3 个分片**；
  `token`/`dup` → **131 个分片**；且三种模式下 思考 382 / 回答 5 字符**完全一致**（只改粒度、不丢字）。
- 仍待做（小）：铺到 `serve`（Web UI 侧）；考虑在档案里给模型一个推荐的默认粒度。
- 已加 `DRACO_THINK_HOLD=line|token|dup`（环境变量，桥接启动时读）：
  `line`（默认）按行挂起，思考延迟一行、收尾能把末行捞成白字回答；
  `token` 完全直出（**实测 ZAYA：8 分片 → 136 分片，首个思考分片 5.6s → 1.5s**），
  代价=不写 `</think>` 的模型回答会一直是暗色；`dup` 在 token 基础上收尾把末行再发一次当 content
  （思考实时 + 回答白字，代价=那一小段显示两次）。
- **为什么必须三选一**：把某段判成"回答"要**事后**才知道（模型可能接着写 `</think>`），
  所以"零延迟"与"零重复的正确分区"在流式协议下不可兼得 —— 交给用户挑，不替他决定。
- 待办：把它接到 draco 的 `/` 命令或档案字段（现在只有环境变量）。

### D4 ✅ 完成（2026-09-15）：granite-hybrid 首次全链体检 + 家族闸门判据升级
- granite-hybrid 在 llama.cpp 两个后端（cpu/igpu）体检：健康问句 3/3、长上下文曲线平（0~1024 都对）。
- 首轮家族闸门 FAIL ⇒ **抓到判据误报**：cpu 答 'Here is the count…: 1, 2, 3, 4, 5.'、
  igpu 答 '1, 2, 3, 4, 5' —— **语义相同、措辞不同**（基线模型话痨），不该 FAIL。
- 判据升级：跨后端比较从"逐字相同"改为**逐字 → 尾部数字序列语义**两级
  （`semantic_agree()`；wording_diff 显式记录在 JSON 里，不静默吞）。升级后 granite PASS。
- 教训：**"逐字一致"这个强判据对基线/话痨模型会误报** —— 一致性判据要分"语义级一致"
  （合格线）与"逐字级一致"（加分项），且措辞差必须显式记录、不许静默。

### D3 💤 Web UI（serve）的长上下文/思考显示
- 桥接已经把 reasoning/content 分开送，但 Web UI 是否按两种颜色显示、超长上下文的提示是否清楚，还没系统看过。

---

## E. 发布前（外部条件：iGPU 之后再搞）

### E1 💤 多设备/多环境兼容测试
- 按你定的顺序："发行前的兼容性测试放到 iGPU 之后"。
### E2 💤 脱敏设计说明（替代 `STAGE1_NPU.md`），你说过这份不上传。
### E3 ⭐ 凭据卫生：fine-grained token 记得轮换/设过期（一直没落盘，只用在一次性 URL）。

---


### 🚨 已修回归（2026-09-15，`/hold` 改动引入）：位置参数错位导致空回复
- **症状**：selfcheck 三个模型的指纹**全是同一个** `02f4d0509942c618`、"两轮逐字一致"（因为两轮都是**空串**）、
  健康问句 0/3、且"服务端未给计时字段"。
- **根因**：我把 `hold` 插在 `stream_chat` 签名的 `repeat_penalty` **之前** ⇒ 既有**位置调用**
  （selfcheck 的 `_complete_once`）把 `1.1` 当成 hold 发出 ⇒ 桥接收到**浮点数**、对 float 调
  `.strip()` 抛 AttributeError ⇒ 生成器中途崩、连接关闭 ⇒ 客户端只看到空内容。
- **修法（两道防线）**：① `stream_chat` 的 `hold` 挪到签名**最后**（不改既有位置参数）；
  ② 只在值属于 `line/token/dup` 时才发该字段；③ 桥接侧 `str(hold)` 类型兜底
  （**可选字段的坏值不该让桥接崩掉**）。
- **验证**：smol 指纹回到 `586d5dcc2ea1c54c`、ZAYA `2c6ba54842e108bd`、Ling `36a05801c12bda28`，
  健康问句 3/3，timings 恢复；家族闸门重跑 ZAYA 仍 **PASS**（dengine vs llama.cpp 逐字一致 + 长上下文曲线平）。
- **教训（重要）**：改共享函数的签名（尤其插参数）会**静默**改掉既有位置调用的语义 ——
  这次的烟测我只看"命令跑起来了、`/hold` 提示打出来了"，**没有断言答案内容**，所以放过了它。
  ⇒ 规矩：碰流式/请求路径后，**必须断言真实答案**（指纹/健康问句），不能只看"没报错"。

## 阶段 1 续 🔄 qwen35moe（Qwen3.6-35B-A3B-REAP-48-v2，9.4GB）—— 适配进行中（2026-09-16）

**已完成**：驱动器泛化（arch 自适应 qwen35/qwen35moe；MoE=133×8 + 带标量 sigmoid 门的共享专家；
F32 张量 code 9；无 recurrent_layers 键时按 full_attention_interval 推导——同 llama.cpp fallback）。
**层 0 全部分段已验证**：GDN 0.9986 / moe_out 0.99984 / shexp 0.9994（门 0.07048 vs 参考真门
0.0705）/ ffn_out 0.99984。**已修 C bug**：m6_granite_shexp 的门写成了 silu（应为 sigmoid）——
症状是 shexp 输出与参考**精确反相关**（cos −0.9994），靠"逐元素 out/shg 恒为 −0.1818"
（= silu(−2.579)）抓到。

**遗留（精确交接，2026-09-16 深夜）**：两个 bug 已修（①shexp 门 silu→sigmoid；
②**_ffn 漏加 moe_out**——m6_granite_moe 写 M["mo"]，shexp 覆盖 out 后忘了相加，
FFN 只剩 shexp；手工对账时手动相加把引擎缺陷掩盖了）。修复后 **pos0 层 0~30 cos 全 1.000**，
层 31~39 余 0.989~0.997（从最后一个全注意力层 31 起）。内核矩阵已排除内核嫌疑
（blk.0/15/35 全部张量逐一对账 0 失败，含 Q8_0/Q5_K/Q4_K/IQ4_XS）。
**✅ 又修一 bug（2026-09-16）**：**head 绑定**——qwen35moe 有独立 output.weight，
驱动却硬编码用词嵌入 ⇒ 隐层对（cos 0.997）而 logits 全错（ref 首 token 排到 12 万名外）。
修复后：**真实句子贪心 12/12 与 llama.cpp 一致**（"…Germany is"）；计数 prompt 在
第 3 token 分叉——判定为量化噪声（IQ3_S 专家 + Q8_K 激活 vs 我们 fp32，pos3 逐层
min 0.9935 且层 0~2 pos3 = 1.000 ⇒ 状态递推无 bug）。
**剩余收尾（下轮）**：①贪心金标准落盘（France/Germany prompt 已验证 12/12）②登记
_ADAPTERS_ARCH + models.d ③T1/T2 用例 ④层 31~39 的 0.989~0.997 归档为 Q8_K 激活差异预期。

## 阶段 3 ✅ 诊断完成（2026-09-16，离电+省电档）：iGPU 2.3× 的来源 = 低功耗下 CPU 取不动 DDR

**MoE 感知带宽账（granite-h-tiny，活跃字节 = 常驻 570MB（含 tied head 232MB）+ 专家 6/64×3.66GB=343MB）**：

| 后端 | t/s | 有效带宽 | 相对自身屋顶 |
|---|---|---|---|
| dengine（CPU AVX-512） | 12.0 | **11.0 GB/s** | 纯读屋顶的 **49%** |
| llama.cpp CPU（-t8） | 9.4 | 8.6 GB/s | 38% |
| llama.cpp iGPU（-ngl 99） | 21.9 | **20.0 GB/s** | **~89%**（同一块 DDR） |

**纯读实测（省电档+离电）**：CPU 单线程顺序 22.5 GB/s、MoE 式散聚集（24×8MB 块）21.1 GB/s
——**散读几乎不亏**（第一次 B4 的"散读差"结论作废：那是采样读法被骗，读进了缓存）。

**判决**：
① 2.3× 的机制 = 低功耗下 CPU 核取 DDR 的能力被压到 ~22.5GB/s（单线程），而两个 CPU 实现
   只跑到 8.6~11GB/s（算力/同步吃掉一半），iGPU 的宽波并行 + 显存控制器把同一块 DDR
   吃到 20GB/s（89%）⇒ **iGPU 赢在"取数效率"，不是带宽总量**（iGPU 没有独立显存）。
② dengine 的 CPU 头顶：理论 22.5×活跃字节 ≈ 24 t/s（若内核做到 100% 内存界）；按 70~80%
   现实折算 ~17~19 t/s ⇒ **CPU 上还有 ~1.5× 的合法优化空间**（dequant 与访存重叠/大核优先）。
   顺带：AC/性能模式下 CPU 屋顶 42GB/s ⇒ CPU 理论可反超 iGPU（它也受同一 DDR 限制）——
   iGPU 优势主要存在于低功耗态。
③ 早期 B4 的"iGPU 纯读 68.5~81.5 GB/s"是**性能模式 + OpenCL 多队列**的数；省电档不适用。

**✅ 访存重叠实验已完成（2026-09-17）——预取假设被否**：暖/冷 A/B（同 6 专家块反复 vs
64 块全轮转 28MB，6 线程，Q4_K 512×1536 专家形状）= **1.05×（19.3 vs 20.4 GB/s）**。
⇒ 内核在 MoE 形状上是**计算吞吐受限**（解量化+点积管线 ~20GB/s 封顶），数据在缓存还是
DRAM 都一样 ⇒ **软预取/权重重叠不会带来收益**，「访存重叠 1.5×」假设撤回。
引擎 11 GB/s 与内核 20 GB/s 的差 = 非 gemv 开销（routing/silu/加权/conv/rms/残差 + 每算子
fork/join）+ tied head（232MB/token，granite ~10ms）+ Python 侧编排。
真要提速的路径变成：①tied head 量化或降精度（granite 专属大项）②非 gemv 逐元素算子融合进
gemv 出口 ③每算子 fork/join 合并（ZAYA 时代的老结论仍成立）。均记为可选专项，非发行阻塞。

**★ 速度结论修正（2026-09-17，用户实测驱动）**：性能模式重测 granite（dengine 26.2/20.7
vs llama.cpp CPU 20.4/30.1，交替两轮）⇒ **性能模式下两者打平**（中位 23 vs 25，交替胜负）；
省电档「dengine 更快」只适用省电档。用户实测 qwen35 2B dengine = llama.cpp 的 85%
（性能模式，含非剪枝版）⇒ 采纳记录。iGPU 21.9 也是省电档数，性能模式下需重测。
**所有速度结论必须按（供电 × EPP 档）双条件记录。**
falcon-h1 乱码（中→英跳变+�）判定：temp=0 同解码路径完全干净（贪心+serve 双验）；
temp 0.6/0.7 基本连贯 ⇒ 0.5B 模型在采样/长生成下进入字节级 token 循环（� 为模型输出的
真实无效 UTF-8），非解码 bug。qwen35(moe) think_block=True 已改（即兴 think 显示为思考区）。

- **来源**：IFM/K2-Horizon-MoVA-36B-A4B（HF），官方 GGUF 仅 BF16 74.9GB；社区量化齐
  （ngquocvinh IQ1_M 8.7GB ~ Q8 39.8GB）。本机磁盘剩 18G ⇒ IQ1_M/Q2_K_S 可下。
- **结构（128MB 头部 Range 下载 + 手写解析，未下全量）**：arch=`k2-horizon`，48 层，
  H=2560，dense FFN 6144，GQA 32/8×128，rope 128@1e7，上下文 512k，词表 250624（gpt2 BPE、
  pre=k2-horizon、`<|ifm|begin_of_text|>` 族特殊 token）。
  · **纯全注意力（无 SSM）+ 两个路由系统**：FFN MoE 100 选 8（专家 FF 768，**前 3 层 dense**、
    共享专家 1 个 FF 768、**expert_weights_norm=1 且 expert_weights_scale=2.5**、
    **expert_gating_func=2**——语义需查新 llama.cpp 枚举）
  · **MoVA（创新点）= V 投影也是 MoE**：value_expert_count=64、value_expert_used_count=4
    ⇒ 引擎没有的"值混合"算子，本次适配的唯一真新算子
- **前置条件**：本地 b10819 **不认识 k2-horizon** ⇒ 必须先升级 llama.cpp 源码树并重建
  build-dbg（对账 oracle 才存在），再下量化。**未下载全量**（等 oracle 就绪，避免白占 9~17GB）。
- **工作量预估（诚实）**：Tier-2 新算子（MoVA）+ gating/scale 新语义 ⇒ 以 qwen35moe 为基准
  （一个会话轮内 3 个 bug 全由层探针流水线定位），MoVA 预计 2~3 个会话轮——这正是
  「兼容过程效率」的真实度量场景：驱动/夹具/金标准/闸门全套现成，新算子是唯一变量。
- 顺带收获：手写 GGUF 头 Range 解析器（/tmp/parse_k2.py，含完整类型表）——以后评估任何
  新模型都可以**只拉 128MB 就看清结构**，不用先下全量。

## 阶段 4 🔄 AVX2/标量回退（2026-09-16 开工）：Q8_0 已验证，K 系布局待查

- **已完成**：`m5/m5_kern_scalar.c`（标量回退内核：m5_gemv/m5_gemv_range/m5_scalar_supported，
  布局对照本仓 gguf-py quants.py 抄写）已编译；验证脚本 /tmp/scalar_verify.py
  （35B 模型全部 (类型,形状) 组合，标量 vs AVX-512 内核 vs numpy 三方对账）。
- **Q8_0 全形状 cos 1.000 ✓**；**K 系（Q4_K/Q5_K/Q6_K）标量版与 gguf-py 不一致**
  （个别形状还输出全零/nan）——布局从 quants.py 抄的，错误细节待查。
  **块级对账已做（2026-09-16 深夜，单热向量法）**：Q6_K 行 0 的 **元素 0 正确**（0.00810 ✓，
  d/sc/q 全对）、**元素 16 错**（+0.0004 vs −0.00601）⇒ 错在 ql/qh 的**第二半段索引映射**，
  不是数学。numpy 广播链需要精确重推：`ql.reshape(n,-1,1,64)>>[0,4]` 实为
  (n,4,2,**64**)→(n,16,32)（16 行×32，不是我假设的 8 行×32）——行分解应为
  g2=r/4, sel2=(r/2)%2, half=r%2（ql 字节 = qlB[g2*64+half*32+k]），qh 同理重推
  （`qh.reshape(n,-1,1,32)>>[0,2,4,6]` = (n,2,4,32)→(n,8,32)，g3=r/4, shl=(r%4)*2 可能已对）。
  **修复后合成块验证（2026-09-17）**：构造编码合成块让 gguf-py 吐出真实映射——
  Q4_K 位置 p 的字节 = (p/64)*32 + p%32、nibble = (p/32)%2 ⇒ **与修后的 C 一致**；
  但整张量矩阵仍 0.428 ⇒ 剩余嫌疑=C 块间/行 stride 或验证矩阵本身。
  **下一脚已做（2026-09-17）**：①C 合成块直验 ✓（位置 0..7 = 1..8，与 gguf-py 映射一致）；
  ②**真实 Q4_K 行 0 单热全验 ✓（256 元素 0 失配）**——标量 Q4_K 解量化在真实数据上正确！
  ⇒ 之前矩阵的 0.428 是**矩阵脚本自身的假象**（该轮疑似未设 MODEL/或全张量路径待隔离）。
  **下轮**：全行（2048 元素）+ 多行单热隔离矩阵脚本问题 → Q5_K/Q6_K 同法逐行验 →
  kernsel 分派 → 强制标量跑 T1 → 阶段 4 收口。
- F16/F32 平凡 ✓ 未单独跑。IQ 系（IQ3_S 等）标量版未实现（fail loudly），qwen35moe
  在纯标量机器上暂不可跑——接受（发行说明写清），或后续补 IQ3_S。
- 分派机制（Python 侧 kernsel：无 AVX-512 ⇒ 全槽指向标量 .so）未做——等 K 系修通一起。
- 验证脚本里 n_in/n_out 方向踩坑：GGUF ne=(n_in, n_out)，ne0=n_in 最快——写死进注释。

- **机制**：`_SLOTS[key] = {ids, snap}`，key = system+首问哈希（同会话首问跨轮不变；撞键安全——
  前缀不匹配就回退全量）。快照/恢复 = 整块 memcpy STATES（granite ~70MB 毫秒级），
  LRU 2 槽。有 STATES 的引擎（smol/llama/granite/falcon/qwen35）自动启用；zaya/ling 回退单会话。
- **实测（granite serve，A/B 两会话交替）**：A1 cached=0 → B1 cached=30 → B2 cached=47 →
  **A2 切回 cached=40**，答案全对（Euro）——交替会话不再互相打掉缓存。
- smol_engine 补了 STATES 清单（kc/vc；tl 由适配器 reset 兜底）。
- ✅ 并发压测已脚本化：`tests/concurrency_e2e.py`（并发 N 会话：全部完成/答案无串线/
  缓存命中；granite 3 会话实测 22.4s 全过）。仍为手工 e2e（要起服务），不进自动套件。
- ✅ AVX-512 守卫（E 组第一小步）：无 AVX-512 的机器在装载前得到干净报错与 llama.cpp
  替代建议，而不是 SIGILL 假段错误（`_require_avx512`，读 /proc/cpuinfo）。
- ✅ zaya/ling STATES 化（2026-09-17，**字节快照方案 (b) 落地**）：适配器各加
  `snap_state()/restore_state()`（ct.string_at/memmove，字段与尺寸严格镜像各自 reset()；
  zaya 含 cca_ps×4 缓冲+tlen+EDA 的 Z.RH；ling 含 MLA kcache/vcache+tlen+KDA conv_state/S），
  槽逻辑统一为适配器多态（state_arrays 数组槽 / snap_state 字节槽）。8 家族全部获得多会话槽。
  *验证（ling serve 双会话交替）*：恢复机制工作（cached>0、无崩溃、A2 回答正确）；
  测试中 B2 提到 France 是**测试脚本伪影**（B1 思维链被 16 token 截断、整段被当 assistant
  历史喂回，模型自然接 France 话题）——非引擎状态串染。
  *遗留观察*：ling A2 的 cached 只有 17（预期更深）——怀疑 Ling 模板对多轮历史的渲染含
  条件分支（如 thinking on/off 措辞）导致与原始生成序列早分叉，待查；不影响正确性（前缀
  不匹配就回退全量）。

## 阶段 2 ✅ 第一批（2026-09-16）：上下文增量复用 + 整代串行锁

- **机制**：引擎状态精确对应 `_CTX_CACHE["ids"]`（prompt+已生成 token，生成时同步延长）。
  新请求与缓存求最长公共前缀 L：只要新序列更长就**不 reset，直接从位置 L 续转增量**；
  任何不匹配 ⇒ 全量 reset（永远正确，只是退回旧速度）。模板怎么拼历史都无所谓——
  比较的是原始 token id。
- **实测（granite serve，两轮对话）**：第二轮 prompt_cached=47 / 实际只 prefill 21（usage 新增
  `prompt_cached` 字段如实上报）、TTFT 2.5ms → 0.6ms、墙钟 9.4s → 1.5s，答案正确（Berlin）。
- **并发**：`_ENG_LOCK` 罩住**整代**（包装生成器 `with _ENG_LOCK: yield from _gen_locked(...)`，
  yield 挂起不释放，GeneratorExit 断连也会释放）。并发第二请求排队而非交错——单状态引擎
  交错 forward 必然互写状态；客户端中途卡住会占住引擎（与 llama-server 排队同性质，已知取舍）。
- **安全性证据**：新增 T1 用例 `t1_incremental_state`（granite：增量续转 vs 全量重跑，
  logits **逐位相同 max|Δ|=0**）——这条不变式红了就说明前缀复用不安全。
- **边界语义**：与新缓存完全同头的短/等长 prompt（L==len(ids)）⇒ 状态已越过前缀且无
  rewind ⇒ 全量重跑；跨会话请求 ⇒ 前缀不匹配 ⇒ 全量重跑（无回归）。

## 阶段 1 ✅ 家族 #7（2026-09-16）：Qwen3.5 2B（qwen35）接入 Darco——对账首次到达 cos=1.00000

- **接入**：m6_gdn_attn × 18 + m6_full_attn × 6（两个整层驱动都是 m4f 时代的现成件）+ dense FFN；
  新代码只有驱动器 qwen35_engine.py。
- **对账（全 f16 ⇒ 无量化误差，判据首次能到顶）**：pos 0..3 × 全部 24 层 **cos 全 1.00000**；
  端到端贪心 **12/12**。**全注意力层的 MRoPE 分段 [11,11,10,0] 在文本输入上退化为普通成对 rope**
  ——用 dump 的 Qcur-post-rope 判定，不用读 MRoPE 源码猜。
- **★ 抓到的坑（对账假分歧第三形态：布局错位）**：ssm_beta/ssm_alpha 的 GGUF ne=[2048,16] ne0 最快
  ⇒ 预转置必须 reshape(16,2048).T；我填成 reshape(2048,16) ⇒ beta/alpha 全部错位。
  症状极其隐蔽：端到端仍能答对 Paris 级问题、逐层 cos 0.987~0.99（不是崩也不是正交）。
  定位路径：qkv_mixed 锚点先证 gemv/权重无罪（kernF 与 numpy 双双 cos=1.000000）→
  numpy 全链复刻 pos0（beta/alpha/conv/递推/门控 rms）cos=1.000000 范数逐位一致 ⇒
  数学无错 ⇒ 只剩"喂给 C 的布局" ⇒ 改 reshape 后层 0 范数逐位一致（1.746）。
  **教训：numpy 复刻链是"数学 vs 接线"二分的裁决器——数学对 ⇒ 查布局。**
- 其它：pre=qwen35 正则（数字单字符）；无 bos 键（falcon_tok 容错）；25 块含 1 MTP 不进主图；
  本地这份是 Base 底模 ⇒ 聊天输出思维链式长文（引擎忠实，换 Instruct 权重即可）。
- 性能：dengine ~16 tok/s（省电档）；head 248k×2048 f16 ≈ 1GB/token 是大头（ZAYA 同款）。

## 阶段 1 续：qwen35（Qwen3.5-2B-f16）—— 侦察已完成，精确开工说明（2026-09-15）

**模型**：general.name="Master"（无意义名 ⇒ 适配器按架构 qwen35 匹配），25 层 = 19 GDN + 6 全注意力
（recurrent_layers [1,1,1,0]×6+[0]，即第 3/7/11/15/19/23 层是全注意力），H=2048 FF=6144，
head_dim 256（全注意力层）、GQA 8/2，**rope 分段 [11,11,10,0]**，SSM：d_inner 2048 / dt_rank(=n_v_heads)
16 / n_group(=n_k_heads) 16 / state 128 / conv 4，**全 f16**（kernF，code 7），vocab 248047，
**无 bos 键**（falcon_tok 已修为容错），pre=qwen35（正则待查 llama-vocab.cpp）。

**已就位的复用件**：
  · `m6_gdn_attn`（GdnW）——注释就是按 2B 维度写的（6144/16/2048）；**GdnW 填法与陷阱在
    `m4f_v5.py` 225-660 行**（beta_wt/alpha_wt 要预转置成 float；码打包在 _c1/_c2/_c3 字段——
    ctypes 未知关键字会被静默忽略 ⇒ 码恒 0；conv 重排成 (NQKV,4)）
  · 夹具锚点已加：linear_attn_qkv_mixed / z / Qcur_normed / Kcur_normed / gate_reshaped /
    attn_residual / attn_post_norm / post_ffn / h_nextn（**已重编译**）
  · **参考 dump 已生成**：`/tmp/qrec0`（pos0，tokens 760,6511,314,9338,369，251 个文件）；
    分支参考 = attn_residual-{il} − l_out-{il-1}（层0 减 model.input_embed）
  · 分词器 falcon_tok 已兼容无 bos 键的 GGUF

**待做（新代码集中在全注意力层）**：
  ① 19 个 GDN 层：m6_gdn_attn 直接用，逐层对账（attn_norm 输入 → attn_residual 差分参考）
  ② 6 个全注意力层：**qk-norm + 分段 rope + head_dim 256** 是新语义——看 qwen35.cpp 的
     build_layer_attn（Qcur_full→reshaped→normed→rope sections），大概率要新写
     m6_qwen35_attn_op（可复用 llama_attn_core 的头并行骨架 + qk-norm 前置 + 分段 rope 表）
  ③ dense FFN 复用 m6_dense_op；管线是「attn → 残差 → post_attention_norm? → FFN → 残差」
     —— 注意 qwen35 有 **post_attention_norm**（granite/falcon 没有），顺序照图核对
  ④ f16 权重的 emb/head 读法照 smol（kernF code 7）；head 是 248k×2048 f16 ≈ 1GB/token
     —— 性能上 head 会是大头（ZAYA 同款问题）
  ⑤ 对账 → 贪心金标准（夹具 ZGREEDY，注意 ids 不要带方括号）→ 登记（_DENGINE_ADAPTERS_ARCH
     或档案 match.arch=qwen35 + name Master 无意义）→ T1 金标准

## 阶段 1 ✅ 家族 #6（2026-09-15）：Llama 3.2 1B（llama 架构）接入——复用率最高的一个

- **零新代码**：引擎固化管线就是 llama 形状（smol 同管线）；只写了适配器（_load_llama）+
  GGUF 直构分词器（复用 falcon_tok，pre=llama-bpe ⇒ llama3 正则）+ 档案。
- **对账**：端到端贪心 **12/12 token 与 llama.cpp 一致**（一次通过）；管线数值由 smol 的
  逐位指纹闸门长期覆盖。serve 实测答对 Paris、第二问无状态污染（33.5 t/s）。
- **要点**：llama 3.2 模板渲染文本自带 bos ⇒ encode 不前置（免双 BOS，falcon_tok 加了
  add_bos 参数）；tied embeddings（head=词嵌入）；metadata 无 rope.scaling ⇒ 纯 rope@500k。
- **家族意义**：llama 架构 = llama/qwen2/qwen3/gemma/mistral 同构管线，这条接入点打通后
  同族模型是 Tier 0（改档案不改代码）。
- **顺带**：smol_engine 补上了 autotune（granite 同款 OMP 退化：llama32-1B 默认 20 线程
  18 t/s → OMP=6 33 t/s，1.9×；线程数不改数值已再次验证）；smol 指纹逐位不变。
- 本地还有 Qwen3.5-2B-f16（qwen35，GDN 注意力，m6_gdn_attn 现成）与 Qwen3.6-35B-A3B
  两个量化版（qwen35moe）——qwen35 是下一个家族的现成候选（注意 2B 是 f16，~4GB）。

## 阶段 1 ✅ 首个家族（2026-09-15）：Falcon-H1 接入 Darco 并登记

- **为什么快**：走 llama.cpp 的 mamba2 图，A={1,n_heads} 标量衰减 ⇒ **S4D 算子原样复用**；
  dense FFN 复用 m6_dense_op；注意力在 llama_attn_core 加了个 use_neox 参数（4 类算子只新写了 0.5 个）。
  印证了"架构覆盖的边际成本在降"的判断。
- **对账**：逐层 pos0..3×36 层 cos 最小 0.99682（阈值 0.996，比 granite 宽是预期：Q8_K 激活+f16 KV）
  ；无状态分支验证 attn 0.99996/ssm 0.99979；**rope 风格用数据判定**（NEOX 0.9999 vs 连续成对 0.89）
  ；两个 prompt 端到端贪心 **12/12 token 与 llama.cpp 一致**。serve 端到端答对 Paris（66 t/s）。
- **新语义**：NEOX rope + freq_base 1e11；**无 ssm_norm**（C 侧 norm=NULL 跳过）；
  ffn_norm 张量名无 .weight 后缀；无任何 scale；tokenizer=falcon-h1 pre ⇒ llama3 正则+add_bos。
- **★ 本次最大教训（对账假分歧的第二种形态）**：夹具 zaya_gdump 的 ids 参数带了方括号
  （python 打印格式），`atoi("[17")=0` ⇒ **参考整个跑在错误的 prompt 上**，逐层对账"全错"，
  折腾了好几轮才在"第一步 rms 就正交"处定位到是 prompt 不同。修法：解析加固 + 金标准生成时
  **脚本当场断言 engine==ref** 才落盘。加上之前那次数组索引假分歧 ⇒ 对账对不上时的排查顺序：
  ①比较代码的索引/形状 ②两边输入是否真的是同一序列 ③才轮到被测引擎。
- falcon 的 general.name 是 "Original" ⇒ draco 新增 `_DENGINE_ADAPTERS_ARCH` 按**架构**匹配适配器。

## 阶段 0 ✅ 完成（2026-09-15）：分层测试套件地基

- `run.py`（分层入口：T0 免模型 / T1 小模型 / T2 大模型对账；`--tier/-k/--list/--freeze`；
  `DRACO_TEST_SO=<候选.so>` 让整套闸门去测**候选构建**——这几轮手工 A/B 的自动化版本）。
- **判据写死阈值 + 打印期望 vs 实测**，不许人眼看 cos（"链头假分歧"活一轮的根因）。
- 已有对账脚本**不改数学**，当黑盒收编（跑子进程看退出码 / 解析输出行做断言）。
- 金标准只存**小文件**进仓库（`tests/golden/`：token 序列 / top5+sha256 指纹 / cos 阈值），
  模型不进仓库；且取自**独立来源**（smol/Ling 用重构前那版 .so，granite 用 llama.cpp 参考输出），
  不是"冻引擎自己的输出"。
- **验收**：全量 T0–T2 **47 秒全绿**（T0 0.2s / T1 14s / T2 32s）；
  **干净 clone（无模型、无 .so）T0 全绿 0.6s** ⇒ CI 跑得动（`release/workflow_tests.yml`）。
- **覆盖矩阵（实测出来的，不是声称的）**：给共用路径 `m6_rms_norm` 注入 1ULP 级扰动后 ——
  `t1_smol_fingerprint` ❌、`t2_ling_fingerprint` ❌（13/13 步 sha256 不同）；
  而 `t1_granite_greedy` 与 `t2_granite_s4d_layers` **仍然绿**（1ULP 不改 argmax、cos 也不敏感）
  ⇒ **只有指纹闸门是逐位的**，cos/贪心类用例是逻辑级判据，两者互补不可互相替代。
- **两个自找的教训（都当场被反面对照抓住）**：
  ① 假测试：第一版 `t0_measured_lookup` 直接调 `_measured_key` 这个**辅助函数**，
     把 bug 注回它的**消费方** `backend_metrics` 后测试照样全绿 —— 绿的没用比没有更危险，
     改成测消费路径后才真的会红。
  ② 反面对照要**瞄对代码路径**：我先把扰动注进 granite 的 S4D 再去测 smol 的指纹，当然不红。
- **未自动化、只能手工跑的（写出来是为了别让这份文件假装覆盖）**：`selfcheck/family`（要起服务）、
  `tune` 的 J/token（要求机器安静 + 必须标供电档）、iGPU/NPU 后端速度、多设备兼容。

## 附：本项目已验证的"方法学纪律"（每轮都要守）
1. **性能结论必须标 (上下文长度, 生成长度, 线程数, 热态)** —— 短提示词下看不出长上下文的坑。
2. **必须交替 A/B 取中位**（本机噪声 5~10%），单次对比会把噪声当收益。
3. **被数据否掉的假设要撤回**（例：rope 角度表缓存 A/B 仅 0.995×，已撤，撤回后 .so 逐字节一致）。
4. **数值改动必须给判据**：能逐位就逐位，不能就 cos + top-k + 范数比；**"跑起来了"不算判据**。
5. **替换运行中的 .so 用原子换名**（用户可能正开着会话）。
6. **归因给模型之前先跑参考实现做对照**（llama.cpp 同提示词/同采样）。

---

## 夜班 2026-09-19：qwen35moe 1.7× + qwen35-2B Q8_0 1.52× + 投机解码定案

### B-新 ✅ qwen35moe 提速（贪心 5.4→9.6 t/s，AC×性能/8线程；同机 llama.cpp 11.1 ⇒ 追到 86%）
- **解剖**（内建剖面 m6_prof_t）：MoE 专家段 52%（gate+up 2/3、down 1/3）、GDN 14%、head 4%。
  IQ3_S 核是**算力受限**（~2.4 GB/s/核 ≈ 纯读 25%），不是带宽问题 —— 与 B1/B3 结论自洽。
- **修了什么**（每行点积仍由同一内核按原序计算 ⇒ 全部逐位不变，全套件 17 用例绿）：
  ① `moe_experts` 行分块（`MOE_CHUNK`，默认 64）：8 专家整矩阵 dynamic 任务 → 细任务，
     大小核混部负载均衡，端到端配对 1.05-1.10×（bench_moe 隔离 1.25-1.29×）。
  ② GDN/全注意力/head 的**大 gemv 行分段**（`g_gdn_seg`，m6_set_gdn_seg 可 A/B）：
     此前 qkv(6144×2048)、gate、ssm_out、词表投影(248320×2048) 全部单线程 ——
     每层 ~20MB 只有 1 核在读。同进程配对 1.40-1.47×（与 Ling MLA 同款修法）。
- **kern12 整数快路径**（IQ3_S i32gather，`DRACO_IQ3_FAST=1` opt-in）：bench 1.12-1.45×，
  但整数累加经 40 层残差放大 ⇒ 逐层 cos 0.9829 < 0.985（mode12/13 都红）⇒ 按闸门纪律不上默认。
- **新实验台**：`src/bench/bench_moe.{c,py}` 同进程交错 A/B + 逐位校验。
- **★ 新发现（待查）**：引擎固有运行间不确定性 —— 同状态两次 forced-logits max|Δ|~1.4e-5
  （chunk=0 也复现；疑某并行归约顺序不定）。不影响现有闸门，但该家族**不能加逐位指纹闸门**。

### B-新 ✅ qwen35-2B：f16 → Q8_0（llama-quantize 现成工具 7 秒出活）
- f16 每 token 读 4GB 本就贴屋顶 ⇒ 唯一大杠杆是减字节。Q8_0（2.08GB）实测 14.6→22.2 t/s（1.52×），
  **48 贪心 token 与 f16 逐个一致**。文件在 `gguf/Qwen3.5-2B-Q8_0.gguf`，与 f16 并存。

### B-新 ✅❌ 投机解码定案（2026-09-19，带数字的"不轻言否决"）
- **需求侧实测**（ngram/prompt-lookup 离线模拟，6 类提示词 × KEY∈{4,6} × K∈{4,8}）：
  复读/改写类每步平均接受 4.2-5.3 个（K=8）；**聊天类 ≈ 0**（新内容无从抄）。
- **供给侧结构**：GDN/S4D/KDA 层 qkv(t) 依赖 state(t-1) ⇒ 递归层内跨位置**严格串行**，
  权重摊薄结构性不可行；qwen35moe 40 层仅 10 层全注意力可批 ⇒ 验证加速上限 ~1.07×。
- **结论**：**本引擎的旗舰模型（混合递归架构）上投机解码结构性不赚**——不是"没优化好"。
  解锁三条件：纯稠密模型 + 同词表小草稿 + skinny-GEMM 批内核（zoo 现无一满足）。
  MTP 头（GGUF 里有）也绕不开验证批处理这堵墙。
- **副产品**：离线接受率模拟法（贪心流与草稿无关 ⇒ 录流离线算）本身可复用。

### ⭐ iGPU/NPU 重评估（分析，未开工）
- **2026-09-19 实测（AC×性能，llama-bench tg64）**：Vulkan iGPU **23.8** vs llama.cpp CPU 15.0 vs
  dengine 9.6 t/s ⇒ **iGPU 仍是该模型最大杠杆（对 dengine 2.5×）**。`draco chat -b igpu` 已可用。
  （注意口径：llama-bench 比 llama-cli 单 prompt 偏快，11.1(cli) vs 15.0(bench)；对比时同口径。）
- dengine 原生 GPU 内核的冷静账：IQ3_S 核在 CPU 上是**算力受限**（25% 纯读），搬到 iGPU 若按同思路
  直译大概率仍算力受限；真工作是把热格式改写成**查表/向量友好**的 GPU 内核（LUT+VBMI2 方向），
  不是简单移植 —— 这修正了此前"带宽受限、写内核就能赢"的旧假设（B4 节）。
- NPU：FLM 官方路线对支持的模型仍是能效王（0.29 J/token）；自研 NPU 路线前置阻塞未变。

### 🔥 B-新（2026-09-19 凌晨实测）：**prefill 悬崖** —— 下一个大项目的头号候选
- **数字**（qwen35moe，pp256，AC×性能）：dengine **13.7 t/s**（256 tok = 18.7s）vs llama.cpp CPU **89.7**（6.5×）vs
  Vulkan iGPU **259.7**（19×）。聊天历史 500 token ⇒ 我们 TTFT ~36s（decode 追平之后，这才是体验上的真差距）。
- **根因**：我们的 prefill = 逐 token decode 循环（每 token 全套 gemv）；llama.cpp 有**位置批处理 matmul** +
  GDN/Mamba 类的**分块并行扫描**（delta rule 的状态递推满足结合律：S_t = g_t·S_{t-1} + k_t⊗δ_t 可按块合并）。
- **prefill 算子分布**（192 tok，包裹计时）：MoE 43% / GDN 32% / head 10% / 全注意力 8% / shexp 4%。
- **项目范围**：①GDN 分块扫描（结合律合并，参照 granite ssm_scan 的 llama.cpp 逐行核对法，oracle 现成）
  ②批处理内核（[T×in]×[in×out] skinny-GEMM，IQ3_S/Q6_K 热格式优先）③full-attn 层的批 KV 追加。
  做完标准：pp256 ≥ llama.cpp CPU 的 80%（~70 t/s），贪心续接与逐 token 路径一致（阈值闸门）。
- **副产品**：skinny-GEMM 一旦存在，投机解码的验证批处理墙也被拆掉（见上节）—— 一石二鸟。

#### prefill 悬崖的攻坚设计（2026-09-19 凌晨分析，供立项）
- **设计真相**：prefill 的钱都花在**重复读权重**上（每 token 一遍全套 gemv）。llama.cpp 的路子：
  ①每层投影（qkv/gate/in_proj）对**全部 T 个位置一次 matmul**（权重只读一遍）；
  ②状态类算子（ssm_scan/gdn）在内核里**逐 token 走状态但只碰激活**（128×128 小状态在缓存里，不碰权重）；
  ③MoE/FFN 在 T 个位置上批处理（专家按 union 去重后每专家一次 GEMM）。
- **我们对应的三件套**：①skinny-GEMM 内核（[T×in]×[in×out]，热格式 Q6_K/IQ3_S/Q8_0 优先）
  ②GDN 的「批投影 + 逐 token 状态扫描」两段式（扫描只碰 q/k/v/β/g 激活与 S 状态）
  ③MoE 按位置批处理（需要 GDN 输出全位置可得 ⇒ 引出 chunked/associative scan，delta rule
  S_t = g_t(I−β_t k_tk_tᵀ)S_{t-1} + β_t k_tv_tᵀ 满足结合律；llama.cpp 89.7 t/s 说明全管线可行，直接读它的
  qwen35moe.cpp + ggml ssm/scan 实现做 oracle）。
- **里程碑**：M1 两段式 GDN（MoE 仍逐 token）预期 13.7→~25 t/s；M2 MoE 位置批处理 → 冲 70-90；
  验收 = pp256 ≥ llama.cpp CPU 80% 且贪心续接与逐 token 路径一致。副产品：投机解码验证墙同步拆除。
- **教训（流程）**：用 pkill -f 收自己的测试服务会把外层 bash 包装器一起匹配杀掉（本次整个命令块被截断、
  追加与提交都没执行）——测试服务一律用 nohup 时记下 PID 再按 PID kill。

#### prefill M1 ✅ 已完成（2026-09-19 上午）：两段式 GDN + 批投影 + 批注意力
- **落地**：`m5/m5_gemm.c`（Q8_0/Q5_K/Q6_K/F16 批投影，反量化对账 gguf-py 逐位一致）、
  `m6_gdn_scan.c`（激活态扫描，对账 numpy 参考 S/ON/tail 逐位级）、`qwen35_prefill.py`（编排）、
  `prefill_chunk_gate.py`（闸门）。服务端 ≥64 token 增量自动走 chunk（`DRACO_CHUNK_PREFILL=0` 关）。
- **等价性**：qwen35 与 qwen35moe 均 **末位 cos=1.000000 + 贪心续接 12 token 逐 token 一致**（T2 闸门入套件）。
- **速度**（AC×性能/8线程）：qwen35moe T=192 热态交替 **1.44×**（11.3→16.3 t/s）；服务端 154 tok 实测 15.0 t/s。
- **调试教训**（三个连环坑）：①我自己 rms_mean 对 2D 输入错加一维（`ms/shape[1]` 变逐元素）——
  引擎 xn 反推比对抓出；②全注意力 q/k rms 是**均值式**（n2/HD+eps），GDN 头内 rms 是**和式**
  （fmax(sqrt(nq))）——Kr 恰差 √256=16× 抓出；③引擎调试哨兵 buf[39000]=1 + buf[30000](qkv 全量)/
  buf[38240](xn) 是现成的真值出口，比外部 oracle（gguf-py 对该文件的字节偏移与引擎未必一致）更可信。
- **剩余**：chunk 内 MoE 仍逐 token（~60% 时间）⇒ M2 专家并集批处理是下一刀（预期再 2-3×）。

#### prefill M2 🔄 数值已通、性能待内核（2026-09-19 上午）
- **已落地**：`m6_moe_tok.c`（专家并集分组 + 按专家 gather → 批 GEMM → silu → 按 token 确定性
  scatter）+ `m5_gemm.c` 补全 IQ3_S/IQ4_NL/Q4_K/IQ4_XS（反量化全部对账 gguf-py **逐位一致**）。
  端到端 cos=1.000000 + 贪心续接一致 ✓，`DRACO_MOE_BATCH=1` 可开（默认关）。
- **性能结论（诚实）**：实测 1.20× < M1 的 1.44×，**反而慢**。原因：m5_gemm 是「材料化反查表」
  （先把权重行反量化成 float 再点积），每 (层,专家) 反量化 1.35MB 的成本没有被足够摊薄——
  逐 token 的 m6_granite_moe 是**块内融合反量化**（dequant 在寄存器里直接 fma，不落内存）。
- **真解（下一步内核项目）**：融合式 GEMM = rows_iq3_s 的寄存器级 dequant 结构 + 每 16 值
  v16 寄存器复用 × token 分块（T'=8~16 个 racc 复用 zmm 寄存器），IQ3_S 优先（专家 43% 占比），
  其次 Q6_K/Q5_K（投影）。做完标准：pp256 ≥ 25 t/s（M1 是 16.3）。
- **调试教训**：①OMP 嵌套调用 GEMM + 共享 wbuf/输出 scatter = 数据竞争（cos 每轮漂移）与
  病态变慢（43s）；②浮点归约不开 -ffast-math 不会自动向量化——批内核的点积必须手写 zmm；
  ③REAP 文件是混合量化（深层层专家 IQ4_XS/Q4_K，其余 IQ3_S）——GEMM 必须全格式支持。

#### prefill M2 ✅ 转正（2026-09-19 下午）：融合反量化 GEMM 内核
- **m5_gemm_fused.c**：IQ3_S 寄存器级反量化 × 8-token 分块（v16 算一次、8 个 racc 复用）。
  **两个关键实测**：①单层累加（dg 折进 v16）+ 不加 unroll pragma = 0.52ms/矩阵；加
  `#pragma GCC unroll 8`（谓词展开）反而 5ms——谓词展开把寄存器压爆，**手写内核别加它**；
  ②材料化 GEMM（0.71ms）其实也只反量化一次，输在「标量 dequant + store/reload」而非付两次。
- **m6_moe_tok v3**：全 IQ3_S 专家走融合 GEMM（行主序、无转置），其余格式 DQ+dotf 回退。
- **端到端**（T=192 热态交替 ×2）：MOE_BATCH=0 → 14.1-15.4 t/s；=1 → **21.1-22.7 t/s**（2.15-2.20×
  vs 逐 token；T=128 时 2.33-2.55×）。数值：cos=0.999998 + 贪心续接 12 token 逐 token 一致 ✓。
  **已转默认**（DRACO_MOE_BATCH=0 回退）。
- **pp256 估算**：T=192 chunk 21-23 t/s ⇒ 256 token 约 25-30s→12s 区间，仍低于 llama.cpp CPU
  89.7——下一刀是把融合 GEMM 推广到 Q6_K/Q5_K 投影（当前投影走材料化 m5_gemm，占 chunk ~24%）。

#### 融合 GEMM 全格式 ✅（2026-09-19 下午续）
- **m5_gemm_fused.c** 新增 Q8_0/Q6_K/Q5_K/Q4_K/IQ4_XS 融合变体 + `m5_gemm_auto` 统一分发
  （dq16 每 16 值组寄存器反量化）。逐位对账 m5_gemm：**全部 Δ=0**。隔离加速：Q6_K 2.7×、
  Q5_K 2.2-2.3×、IQ4_XS 1.35-1.5×、Q8_0 ~1×、Q4_K ~1×。
- **接线**：qwen35_prefill._gemm 自动路由融合（F16/IQ4_NL 留材料化）；m6_moe_tok 换 auto 入口
  （深层层 IQ4_XS/Q4_K 专家也批处理化）。
- **端到端**（T=192 热态交替）：全融合 **20.9-24.7 t/s（2.31×）**；调试坑：①Q8_0 反量化误用
  cvtepu8（int8 要 cvtepi8）；②dq16 的位域必须用**块内位置 p=base&255**（Q6_K ql/qh/sc、
  Q5_K 组号都是块内语义，全局 base 会错位到后面块）；③Q4_K sc/m 6-bit 解包下标按 kern7 逐式抄。

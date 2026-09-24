# MI50 (gfx906) 上 ROCm 10 的「双栈混用」配方与实测结论

## 背景
ROCm 10 对 gfx906（MI50/Pro VII）：编译 ✓、枚举 ✓、hipMalloc ✓，**HIP kernel launch ✗
（invalid image，全内核静默）**；rocBLAS 只带 gfx1150 Tensile。Vulkan/RADV 路径不受影响
（我们的引擎已全绿：k2_gate PASS 2.73e-07、decode 6.69 t/s @50W 钳）。

## 实测结论（2026-09-24）
1. **不是 CO 版本问题**：-mcode-object-version=4/5/default 全部同样失败。
2. **SONAME 断代是硬墙**：6.3 栈 = libamdhip64.so.6 + comgr.so.2 + hsa.so.1；
   ROCm10 链 .so.7 + comgr.so.3 ⇒ 单换两个库无效（NEEDED 名字对不上）。
   **正确形态 = 整套 6.3 用户态栈 + patchelf 改写 DT_NEEDED + rpath**。
3. **patchelf 配方生效**（ldd 全部解析到 overlay ✓），且：
   - gfx1150 内核 + 6.3 栈 + 现代 clang 编译 → **iGPU PASS**（两代 CO 元数据兼容）
   - gfx906 内核 + 6.3 栈 → **invalid device function**（模块载入成功、kernel 符号查找失败）
   ⇒ 最后一块拼图 = **用 6.3 时代工具链编译 gfx906 内核**（rocm-llvm LLVM18 = 325MB）。
4. power1_cap(50W) 是 **SMU 持久化**的，重启不丢；perf=low+sclk 钉档会诱发
   powerplay 'dpm not enabled' → ring reset fail → BACO atom init -22 → 复位线程挂死
   （TB3 上自动复位不可靠，恢复=重启）——**安全配方只用 control=on + cap + perf=auto**。

## overlay 位置
`/media/xiao_/OverSys1/npu-direct/rocm63-mix/`（debs + root/opt/rocm-6.3.0，共 ~68MB）

## patchelf 配方（对任何 ROCm10 编的 HIP 二进制）
```bash
OV=/media/xiao_/OverSys1/npu-direct/rocm63-mix/root/opt/rocm-6.3.0
patchelf --replace-needed libamdhip64.so.7 libamdhip64.so.6 \
        --replace-needed libamd_comgr.so.3 libamd_comgr.so.2 \
        --set-rpath "$OV/lib" <binary>
LD_LIBRARY_PATH=$OV/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib ./<binary>
```

## 待办（若继续）
- [ ] 下载 6.3 工具链（rocm-llvm 325MB + hip-dev + hipcc + rocm-device-libs ≈ 327MB）
- [ ] 老 hipcc 编 gfx906 kernel → patchelf → launch 验证
- [ ] 通过后：rocBLAS Tensile gfx906（149MB deb 里只取 gfx906 目录）→ llama.cpp HIP 路线

## 终审补充（2026-09-24 晚，工具链已下并实测）
6.3 工具链（rocm-llvm 325MB 等 4 包）已解进 overlay（root/ 共 1.2GB），用老 hipcc 实测：
- **老工具链 + 老栈原生组合（同代发行版配对）→ 仍 invalid device function** ⇒ 排除
  一切版本代差假设（CO 版本/SONAME/工具链时代 三层全灭）。
- **裸 `--offload-arch=gfx906`（嵌入串无 sramecc/xnack 尾巴）→ 仍败** ⇒ 排除特征串不匹配假设。
- **strace 决定性负结果**：/dev/kfd 328 个 ioctl **零失败**（ALLOC/MAP/EVENT 全成功），
  但 **CREATE_QUEUE 从未被调用** ⇒ 死在用户态 comgr↔agent 符号/目标解析层，队列都没排到。
- iGPU 同栈同编译器 PASS ⇒ comgr 读 CO 本身没问题，**只有 gfx906 的 agent 目标推导出错**。
- KFD topology：`name`=“vega20”（kernel7 新命名，非 LLVM 三元组）、properties 无特征串、
  gfx_target_version=90006 ⇒ 嫌疑：老 rocr 从新内核 sysfs/驱动推 agent target 时拿不到
  与镜像匹配的串。
**剩余步骤（下一夜，需 ltrace/gdb 或读 rocr6.3 源码）**：对
`comgr_copy_symbols/comgr_validate_code_object` 打点看返回码；或追 rocr 的
agent-target 推导（hsakmt topology→amdgpu asic 映射）。overlay+工具链全部就位可复用。

## ★★ 真根因（2026-09-24 终章）：多 GPU 可见性 bug，一行环境变量修复
前面五层（CO 版本/SONAME/工具链时代/特征串/KFD 推导）全是弯路——决定性 2×2：

| 栈 | 双卡可见（默认） | HIP_VISIBLE_DEVICES=1 |
|---|---|---|
| ROCm10 原生 | ✗ invalid image | ✅ **PASS (y=42)** |
| 6.3 overlay 老栈 | ✗ invalid device function | ✅ PASS |

**根因 = clr 加载器在多 GPU 可见时符号/agent 查串**（ltrace 指纹：
`hsa_executable_symbol_get_info(kind=10 VARIABLE_IS_CONST) = 0x1001 ×225`、
KERNEL_OBJECT(22)×15 却成功——正是“在两个 agent 间遍历错配”的形态）。
新老栈症状不同（image vs device function）但同病灶。

### 最终配方（HIP-on-MI50，ROCm10 原生即可，无需 overlay）
```bash
HIP_VISIBLE_DEVICES=1 ./your_hip_binary     # 隔离到 MI50（索引按枚举序，本机 1=vega20）
# 或 ROCR_VISIBLE_DEVICES=1；iGPU 侧进程用 HIP_VISIBLE_DEVICES=0，两进程互不干扰
# 注意：进程内设备序会被过滤重排——按 gcnArchName/枚举结果选卡，别写死索引
```
- **rocBLAS 仍缺 gfx906 Tensile 库**（只有 gfx1150 目录）——BLAS 混用下一步
  （149MB rocblas_4.3 deb 里抽 gfx906，预期同样受 VISIBLE 过滤保护）。
- 6.3 overlay + 老工具链（1.2GB）保留：Tensile 混用/考古备用，非必需。
- 本机 bug 归属：kernel7.0 + ROCm10 clr 双卡枚举路径（AMD 上游可报 issue 的级别）。

## ✅ A 结案（同夜）：rocBLAS gfx906 混用 PASS
配方（三个变量，全环境变量、零 root）：
```bash
HIP_VISIBLE_DEVICES=1 \
ROCBLAS_TENSILE_LIBPATH=/media/xiao_/OverSys1/npu-direct/rocm63-mix/tensile_gfx906_slim \
./your_blas_program   # 链系统 librocblas.so.5（ROCm10）
```
- 数据源：rocblas_4.3 deb（149MB，Size=149099742 校验过）内 `library/*gfx906*` +
  `TensileLibrary_lazy_gfx906.dat` → 已抽成 **tensile_gfx906_slim（156 文件/142MB）**；
  3.7GB 全架构解包目录已删（deb 保留可重解）。
- **实测：sgemm512 status=0、C[0]=512.00 精确、burst 2866-3298 GFLOPS（860MHz/50W 钳下）**
  —— rocBLAS5.6 读 4.3 时代 Tensile 数据无 schema 拒收。
- 坑：hipMemset(A,1,…) 是按字节填 → float=1.4e-39 非规格化 → C[0]=0 假 FAIL；
  用宿主 1.0f 数组 memcpy 才是真初始化（初版测试自摆乌龙）。
- 至此 HIP 三件套全通：内核 launch ✓（单卡过滤）+ rocBLAS ✓ + 配方可复用；
  下一步自然是 llama.cpp-HIP/attention 走 hipBLAS 实测。

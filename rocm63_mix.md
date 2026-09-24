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

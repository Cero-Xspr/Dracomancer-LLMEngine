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

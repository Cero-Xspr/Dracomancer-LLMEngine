#!/bin/bash
# bench_hetero_bw.sh — 异构聚合带宽实验：iGPU 推理并发下测 CPU 纯读带宽
# 结论(2026-09-20, Ryzen AI 9 H365, LPDDR5X-8000 128bit 总线):
#   CPU 单独读 52.8 GiB/s；iGPU 推理 33.4 t/s(≈64GiB/s 权重流) 且并发下不降速；
#   并发时 CPU 读 32.1 GiB/s ⇒ 聚合 ≈96GiB/s ≈ 1.8× CPU 单独，直逼总线 128GB/s。
#   ⇒ 异构聚合假说实测成立：CPU+iGPU 并发访问共享内存池有效。
VK=/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-vk
MODEL=/media/Data-1/gguf/Qwen3.5-2B-Q8_0.gguf
BW=/media/xiao_/OverSys1/npu-direct/hybrid/m5/bwtest
echo "== 基线: CPU 单独 =="
(cd /media/xiao_/OverSys1/npu-direct/hybrid/m5 && OMP_NUM_THREADS=20 ./bwtest | tail -1)
echo "== 基线: iGPU 单独 =="
(cd $VK && ./bin/llama-bench -m $MODEL -p 0 -n 128 -ngl 99 2>/dev/null | grep tg128)
echo "== 并发: iGPU 持续推理 + CPU 读 =="
( for i in 1 2 3; do $VK/bin/llama-bench -m $MODEL -p 0 -n 128 -ngl 99 2>/dev/null | grep tg128; done ) &
IGPU=$!
sleep 20
(cd /media/xiao_/OverSys1/npu-direct/hybrid/m5 && OMP_NUM_THREADS=20 ./bwtest | tail -1)
wait $IGPU

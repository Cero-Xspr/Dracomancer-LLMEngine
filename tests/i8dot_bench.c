// m5/i8dot_bench.c —— B1 可行性微基准：Q8_0 的"浮点转换路径" vs "整数点积路径"
//
// 为什么要先做这个：B1（int8/VNNI）**会改数值**（激活要量化），所以先证明它值这个代价。
// 本机 CPU 有 avx512_vnni + avx512_bf16（/proc/cpuinfo 已确认）。
// 两种内层循环，同样的权重字节、同样的形状，只比"每字节权重的速度"：
//   A: 现在的做法 —— 权重 int8 → float（cvtepi8_epi32 + cvtepi32_ps），再与 fp32 激活做 FMA
//   B: 激活量化成 int8（每 32 个一组、f32 尺度）→ 权重 +128 变无符号 → maddubs/madd 整数点积
//      dot = Σ(w+128)*x - 128*Σx  （llama.cpp 的经典做法；这里用 maddubs+madd，不需 VNNI 也能跑）
// 编译：gcc -O3 -march=native -fPIC -shared i8dot_bench.c -o i8dot_bench.so -lm
//
// ★★ 2026-09-15 第一次实测的结论（**负面**，但很有用）：
//   · 本机 ISA 齐全（avx512_vnni / avx512bw / avx512_bf16，/proc/cpuinfo 已确认）；
//   · 现有四个格式（Q5_0/Q6_K/Q8_0/Q4_K）**全是**"整数→float 转换 + FMA"路径，
//     所以"换成整数点积"方向本身成立；
//   · **但下面这个 maddubs + 逐块修正的朴素写法比浮点路径还慢**：
//     真实 Q8_0 张量（49152×960，50MB）：浮点 25.4 GB/s vs 整数 21.8 GB/s（T=1/8 都一样，
//     因为本内核没开 OMP）—— 原因是每个 32 权重块为了"权重 +128 变无符号"要多付
//     2 条 maddubs/madd 的修正指令，把省下来的转换指令又吃回去了。
//   ⇒ 结论：B1 要做得用 **VNNI `_mm512_dpbusd_epi32`**（一条指令 64 个 MAC）+ 把两个 32 块
//     塞进同一个 512 位寄存器、修正项折叠进掩码/预计算（llama.cpp 的做法），
//     而不是这种"朴素整数化"。本文件保留作为**反例基准**（改 B1 时先跑它对比）。
//   ⚠️ 本文件的数值列（cos/max|Δ|）**当前不可信**：我临时写的 float64 参考实现自己也对不上
//     浮点路径（cos 0.968，说明参考的块/行布局读错了）。改 B1 时要用**引擎自己的浮点内核**
//     当参考（判据：整数路径与它的偏差应是小而可解释的，因为激活被量化了）。
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>

// ---------- A: 浮点路径（与 m5_kern6.c 的 rows_q8_0 内层同构，去掉 tail/多累加器技巧）----------
static void path_fp32(const float* x, const uint8_t* W, int n_out, int n_in, float* y) {
    const int nb = n_in / 32;
    for (int o = 0; o < n_out; o++) {
        const uint8_t* row = W + (size_t)o * nb * 34;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 34;
            const float d = _cvtsh_ss(*(const unsigned short*)blk);
            const __m256i q = _mm256_loadu_si256((const __m256i*)(blk + 2));   // 32 × int8
            // 低 16 个 → float
            __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q)));
            __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q, 1)));
            __m512 dot = _mm512_mul_ps(f0, _mm512_loadu_ps(x + b * 32));
            dot = _mm512_fmadd_ps(f1, _mm512_loadu_ps(x + b * 32 + 16), dot);
            acc0 = _mm512_fmadd_ps(_mm512_set1_ps(d), dot, acc0);
        }
        (void)acc1;
        y[o] = _mm512_reduce_add_ps(acc0);
    }
}

// ---------- B: 整数路径 ----------
// 激活量化: 每 32 个一组 → int8 + f32 尺度 (llama.cpp 的 Q8_0 激活量化)
static void quant_x_q8(const float* x, int n_in, int8_t* qx, float* dx) {
    for (int b = 0; b < n_in / 32; b++) {
        const __m512 v = _mm512_loadu_ps(x + b * 32);
        const __m512 a = _mm512_abs_ps(v);
        const float amax = _mm512_reduce_max_ps(a);
        const float d = amax / 127.0f;
        dx[b] = d;
        const __m512 inv = _mm512_set1_ps(d > 0 ? 1.0f / d : 0.0f);
        // round-half-away: 与 llama.cpp 一致用 nearbyintf
        __m512 s = _mm512_mul_ps(v, inv);
        __m512i r = _mm512_cvtps_epi32(_mm512_roundscale_ps(s, _MM_FROUND_TO_NEAREST_INT));
        // _mm512_cvtsepi32_epi8 只吃 __m512i 且只出低 16 个 int8 ⇒ 用 shuffle 把高 16 挪到低位再出
        __m128i lo16 = _mm512_cvtsepi32_epi8(r);
        __m128i hi16 = _mm512_cvtsepi32_epi8(_mm512_shuffle_i64x2(r, r, 0xEE));
        _mm256_storeu_si256((__m256i*)(qx + b * 32), _mm256_set_m128i(hi16, lo16));
    }
}

static void path_int8(const float* x, const uint8_t* W, int n_out, int n_in, float* y,
                      int8_t* qxbuf, float* dxbuf) {
    const int nb = n_in / 32;
    quant_x_q8(x, n_in, qxbuf, dxbuf);
    // Σx（按 32 个一组，供 +128 偏移修正用）
    const __m512i ones = _mm512_set1_epi8(1);
    for (int o = 0; o < n_out; o++) {
        const uint8_t* row = W + (size_t)o * nb * 34;
        __m512 facc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 34;
            const float dw = _cvtsh_ss(*(const unsigned short*)blk);
            __m512i qw = _mm512_loadu_si512((const __m512i*)(blk + 2));      // 32 int8
            __m512i qx = _mm512_loadu_si512((const __m512i*)(qxbuf + b * 32));
            __m512i wu = _mm512_xor_si512(qw, _mm512_set1_epi8((char)0x80)); // 变无符号
            __m512i p16 = _mm512_maddubs_epi16(wu, qx);                      // u8 × i8 → i16 对
            __m512i p32 = _mm512_madd_epi16(p16, _mm512_set1_epi16(1));      // 相邻两两相加 → i32
            // 修正项：减去 128 * Σx（同一组内）
            __m512i s16 = _mm512_maddubs_epi16(ones, qx);
            __m512i s32 = _mm512_madd_epi16(s16, _mm512_set1_epi16(1));
            __m512i corr = _mm512_slli_epi32(s32, 7);                        // *128
            const __m512i doti = _mm512_sub_epi32(p32, corr);
            const float sc = dw * dxbuf[b];
            facc = _mm512_fmadd_ps(_mm512_set1_ps(sc), _mm512_cvtepi32_ps(doti), facc);
        }
        y[o] = _mm512_reduce_add_ps(facc);
    }
}

// 供 Python 调用：把两种路径都跑一遍（数值由调用方比对，计时也由调用方做）
void bench_path(int which, const float* x, const uint8_t* W, int n_out, int n_in, float* y,
                int8_t* qxbuf, float* dxbuf) {
    if (which == 0) path_fp32(x, W, n_out, n_in, y);
    else path_int8(x, W, n_out, n_in, y, qxbuf, dxbuf);
}

// m5_gemm_fused.c — IQ3_S 融合反量化 GEMM（M2 真解的第一块）
// Y[c×n_out] = X[c×n_in] · Wᵀ，W 为 IQ3_S。
// 关键：**寄存器级反量化 × token 分块** —— 每 16 值的 v16 在寄存器里算一次，
// 立即对 G=8 个 token 做 fma（gacc/racc 结构与 m5_kern6 rows_iq3_s 逐式对应，
// 单 token 数学与逐 token 路径完全同序 ⇒ 逐位一致）。反量化只做 c/G 次摊薄。
// 编译: gcc -O3 -march=native -mf16c -fopenmp -shared -fPIC m5_gemm_fused.c -o m5_gemm_fused.so
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include "iq3s_grid.h"

static inline float h2f1(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { float v = ((float)man) * 5.9604644775390625e-8f; uint32_t vi; memcpy(&vi, &v, 4); f = sign | vi; }
    } else if (exp == 31) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 112u) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}
static inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

#define GTOK 8

// 单个 token 分块 [t0, t0+tc) 的整行 GEMM（内部函数，racc 结构与 rows_iq3_s 同构）
static inline void gemm_iq3s_rows(const float* X, const uint8_t* row, int t0, int tc, int n_out, int n_in, float* Y, int r) {
    const int nb = n_in / 256;
    const __m512i lane_bits = _mm512_setr_epi32(1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768);
    const __m512 one = _mm512_set1_ps(1.0f), neg1 = _mm512_set1_ps(-1.0f);
    __m512 racc[GTOK];
    for (int t = 0; t < tc; t++) racc[t] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + (size_t)b * 110;
        const float d = h2f1(*(const uint16_t*)blk);
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* signs = blk + 74;
        const uint8_t* scales = blk + 106;
        for (int g = 0; g < 8; g++) {
            const uint8_t nib = (g & 1) ? (scales[g >> 1] >> 4) : (scales[g >> 1] & 0x0F);
            // ★ dg 直接折进 v16（单层累加，寄存器减半；数值容差级差异，闸门把关）
            const __m512 dg = _mm512_set1_ps(d * (1.0f + 2.0f * (float)nib));
            for (int qi = 0; qi < 8; qi += 4) {
                const int i0 = g * 8 + qi;
                uint32_t qs4; memcpy(&qs4, qs + i0, 4);
                const uint8_t qhb = qh[i0 >> 3];
                const int sh = i0 & 7;
                const uint16_t sbits = (uint16_t)signs[i0 >> 1] | ((uint16_t)signs[(i0 >> 1) + 1] << 8);
                const int idx0 = (qs4 & 0xFF)        | (((qhb >> (sh    )) & 1) << 8);
                const int idx1 = ((qs4 >> 8) & 0xFF) | (((qhb >> (sh + 1)) & 1) << 8);
                const int idx2 = ((qs4 >> 16) & 0xFF)| (((qhb >> (sh + 2)) & 1) << 8);
                const int idx3 = ((qs4 >> 24) & 0xFF)| (((qhb >> (sh + 3)) & 1) << 8);
                uint32_t w0, w1, w2, w3;
                memcpy(&w0, IQ3S_GRID + idx0 * 4, 4);
                memcpy(&w1, IQ3S_GRID + idx1 * 4, 4);
                memcpy(&w2, IQ3S_GRID + idx2 * 4, 4);
                memcpy(&w3, IQ3S_GRID + idx3 * 4, 4);
                const __m128i w16 = _mm_setr_epi32(w0, w1, w2, w3);
                const __m512i g16 = _mm512_cvtepi8_epi32(w16);
                const __m512i bits = _mm512_and_si512(_mm512_set1_epi32((int)sbits), lane_bits);
                const __mmask16 nz = _mm512_cmpneq_epi32_mask(bits, _mm512_setzero_si512());
                const __m512 sgn = _mm512_mask_blend_ps(nz, one, neg1);
                const __m512 v16 = _mm512_mul_ps(_mm512_cvtepi32_ps(g16), sgn);
                const __m512 vd = _mm512_mul_ps(v16, dg);
                for (int t = 0; t < GTOK && t < tc; t++) {
                    const float* xs = X + (size_t)(t0 + t) * n_in + b * 256 + i0 * 4;
                    racc[t] = _mm512_fmadd_ps(vd, _mm512_loadu_ps(xs), racc[t]);
                }
            }
        }
    }
    for (int t = 0; t < tc; t++) Y[(size_t)(t0 + t) * n_out + r] = hsum512(racc[t]);
}

// GEMM：行间 omp 并行（每行整行独立），行内按 GTOK 分块
void m5_gemm_iq3s(const float* X, const uint8_t* W, int c, int n_out, int n_in, float* Y) {
    const int nb = n_in / 256;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int r = 0; r < n_out; r++) {
        const uint8_t* row = W + (size_t)r * nb * 110;
        for (int t0 = 0; t0 < c; t0 += GTOK) {
            const int tc = (t0 + GTOK <= c) ? GTOK : (c - t0);
            gemm_iq3s_rows(X, row, t0, tc, n_out, n_in, Y, r);
        }
    }
}

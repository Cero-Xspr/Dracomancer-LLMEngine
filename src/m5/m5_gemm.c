// m5_gemm.c — 批量投影（prefill 提速 M1）：Y[T×n_out] = X[T×n_in] · Wᵀ
// 权重每行反量化**一次**进 wbuf，然后对 T 个激活行做点积 —— 把 prefill 的
// "每 token 重读一遍权重" 摊薄成一遍反量化 + T 次纯 FMA。
// 反量化位布局与 m5_kern6/8/9.c 逐位一致（那些注释已对账 gguf-py）。
// 支持 code：0=Q8_0(34B/32) 3=Q5_K(176B/256) 4=Q6_K(210B/256) 7=F16(2B/1)
// 编译: gcc -O3 -march=native -mf16c -fopenmp -shared -fPIC m5_gemm.c -o m5_gemm.so
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
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

static inline void get_sc_m(const uint8_t* s, int j, uint8_t* d8, uint8_t* m8) {
    if (j < 4) { *d8 = s[j] & 63; *m8 = s[j + 4] & 63; }
    else {
        *d8 = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *m8 = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// 反量化一行到 out[n_in]（值与 gguf-py dequantize 逐位一致；点积顺序与 gemv 内核不同 ⇒ 容差闸门）
int m5_dequant_row(int code, const uint8_t* row, int n_in, float* out) {
    if (code == 0) {                                   // Q8_0
        const int nb = n_in / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 34;
            const float d = h2f1(*(const uint16_t*)blk);
            __m256i q = _mm256_loadu_si256((const __m256i*)(blk + 2));
            __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q)));
            __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q, 1)));
            _mm512_storeu_ps(out + b * 32,      _mm512_mul_ps(lo, _mm512_set1_ps(d)));
            _mm512_storeu_ps(out + b * 32 + 16, _mm512_mul_ps(hi, _mm512_set1_ps(d)));
        }
        return 0;
    }
    if (code == 7) {                                   // F16
        const uint16_t* s = (const uint16_t*)row;
        for (int i = 0; i < n_in; i++) out[i] = h2f1(s[i]);
        return 0;
    }
    if (code == 1) {                                   // IQ4_NL: 18B/32 = d(fp16) + 16 nibble 字节
        static const int8_t KVAL[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        const int nb = n_in / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 18;
            const float d = h2f1(*(const uint16_t*)blk);
            for (int j = 0; j < 16; j++) {
                const uint8_t q = blk[2 + j];
                out[b*32 + j]    = d * (float)KVAL[q & 0x0F];
                out[b*32 + 16+j] = d * (float)KVAL[q >> 4];
            }
        }
        return 0;
    }
    if (code == 2) {                                   // IQ3_S: 110B/256，与 m5_kern6 rows_iq3_s 同布局
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 110;
            const float d = h2f1(*(const uint16_t*)blk);
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 66;
            const uint8_t* signs = blk + 74;
            const uint8_t* scales = blk + 106;
            float* ob = out + b * 256;
            for (int g = 0; g < 8; g++) {
                const uint8_t nib = (g & 1) ? (scales[g >> 1] >> 4) : (scales[g >> 1] & 0x0F);
                const float dg = d * (1.0f + 2.0f * (float)nib);
                for (int qi = 0; qi < 8; qi += 4) {
                    const int i0 = g * 8 + qi;
                    uint32_t qs4; memcpy(&qs4, qs + i0, 4);
                    const uint8_t qhb = qh[i0 >> 3];
                    const int sh = i0 & 7;
                    const uint16_t sbits = (uint16_t)signs[i0 >> 1] | ((uint16_t)signs[(i0 >> 1) + 1] << 8);
                    for (int j = 0; j < 16; j++) {
                        const int idx = ((qs4 >> (8 * (j >> 2))) & 0xFF) | (((qhb >> (sh + (j >> 2))) & 1) << 8);
                        const float sgn = ((sbits >> j) & 1) ? -1.f : 1.f;
                        ob[i0 * 4 + j] = dg * (float)(int8_t)IQ3S_GRID[idx * 4 + (j & 3)] * sgn;
                    }
                }
            }
        }
        return 0;
    }
    if (code == 5) {                                   // Q4_K: 144B/256 = d dmin scales[12] qs[128]
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float d = h2f1(*(const uint16_t*)blk);
            const float dmin = h2f1(*(const uint16_t*)(blk + 2));
            uint8_t sc[8], m[8];
            { const uint8_t* s = blk + 4;
              for (int i = 0; i < 4; i++) { sc[i] = s[i] & 0x3F; m[i] = s[4+i] & 0x3F; }
              for (int i = 0; i < 4; i++) { sc[4+i] = (s[8+i] & 0x0F) | ((s[i] >> 2) & 0x30);
                                            m[4+i] = ((s[8+i] >> 4) & 0x0F) | ((s[4+i] >> 2) & 0x30); } }
            const uint8_t* qs = blk + 16;
            float* ob = out + b * 256;
            // ★ 64 值共享同一 32 字节流：低 nibble = 偶数组 g，高 nibble = 奇数组 g+1（kern7 同映射）
            for (int g = 0; g < 8; g += 2) {
                const uint8_t* qb = qs + (g >> 1) * 32;
                const float scf0 = d * (float)sc[g],     min0 = dmin * (float)m[g];
                const float scf1 = d * (float)sc[g + 1], min1 = dmin * (float)m[g + 1];
                for (int j = 0; j < 16; j++) {
                    ob[g*32 + j]          = scf0 * (float)(qb[j] & 0x0F)      - min0;
                    ob[g*32 + 16 + j]     = scf0 * (float)(qb[16 + j] & 0x0F) - min0;
                    ob[(g+1)*32 + j]      = scf1 * (float)(qb[j] >> 4)        - min1;
                    ob[(g+1)*32 + 16 + j] = scf1 * (float)(qb[16 + j] >> 4)   - min1;
                }
            }
        }
        return 0;
    }
    if (code == 6) {                                   // IQ4_XS: 136B/256 = d scales_h scales_l(4) qs(128)
        static const int8_t KV2[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 136;
            const float d = h2f1(*(const uint16_t*)blk);
            const uint16_t sh = *(const uint16_t*)(blk + 2);
            const uint8_t* sl = blk + 4;
            const uint8_t* qs = blk + 8;
            float* ob = out + b * 256;
            for (int g = 0; g < 8; g++) {
                const uint8_t lo = (g & 1) ? (sl[g >> 1] >> 4) : (sl[g >> 1] & 0x0F);
                const int sc = (int)((((unsigned)sh >> (2 * g)) & 3) << 4 | lo) - 32;
                const float dg = d * (float)sc;
                for (int j = 0; j < 16; j++) {
                    const uint8_t q = qs[g * 16 + j];
                    ob[g*32 + j]    = dg * (float)KV2[q & 0x0F];
                    ob[g*32 + 16+j] = dg * (float)KV2[q >> 4];
                }
            }
        }
        return 0;
    }
    if (code == 4) {                                   // Q6_K: 210B/256 = ql[128] qh[64] scales[16] d
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = (const int8_t*)(blk + 192);
            const float d = h2f1(*(const uint16_t*)(blk + 208));
            for (int p = 0; p < 256; p++) {
                const uint8_t qlb = ql[(p >> 7) * 64 + (p & 63)];
                const uint8_t nib = ((p >> 6) & 1) ? (qlb >> 4) : (qlb & 0x0F);
                const uint8_t qhb = qh[(p >> 7) * 32 + (p & 31)];
                const uint8_t two = (qhb >> (((p >> 5) & 3) * 2)) & 3;
                const int q = (int)(nib | (two << 4)) - 32;
                out[b * 256 + p] = d * (float)sc[p >> 4] * (float)q;
            }
        }
        return 0;
    }
    if (code == 3) {                                   // Q5_K: 176B/256 = d dmin scales[12] qh[32] qs[128]
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 176;
            const uint8_t* qs = blk + 48;
            const uint8_t* qh = blk + 16;
            uint8_t sc[8], m[8];
            for (int j = 0; j < 8; j++) get_sc_m(blk + 4, j, &sc[j], &m[j]);
            const float d = h2f1(*(const uint16_t*)blk);
            const float dmin = h2f1(*(const uint16_t*)(blk + 2));
            for (int p = 0; p < 256; p++) {
                const uint8_t qsb = qs[(p >> 6) * 32 + (p & 31)];
                const uint8_t nib = ((p >> 5) & 1) ? (qsb >> 4) : (qsb & 0x0F);
                const uint8_t qhb = qh[p & 31];
                const int bit = (qhb >> (2 * (p >> 6) + ((p >> 5) & 1))) & 1;
                const int q = (int)nib + (bit ? 16 : 0);
                const int g = p >> 5;
                out[b * 256 + p] = (float)q * d * (float)sc[g] - dmin * (float)m[g];
            }
        }
        return 0;
    }
    return -1;
}

// 行点积：wbuf[n_in] · X[t] —— zmm fma（顺序与 gemv 内核不同，容差闸门）
static inline float dot_row(const float* w, const float* x, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) s += w[i] * x[i];
    return s;
}

// Y[T×n_out] = X[T×n_in] · Wᵀ。wbuf ≥ n_in floats（调用方提供，避免每线程 malloc）。
// 行维并行（每个线程独占整行：dequant + T 个点积），行间无依赖。
int m5_gemm(int code, const float* X, const uint8_t* W, int T, int n_out, int n_in,
            float* Y, float* wbuf) {
    #pragma omp parallel
    {
        float* wb = wbuf + (size_t)omp_get_thread_num() * ((n_in + 15) & ~15);
        #pragma omp for schedule(dynamic, 4)
        for (int r = 0; r < n_out; r++) {
            const int rowbytes = (code == 0) ? (n_in / 32) * 34
                              : (code == 7) ? n_in * 2
                              : (code == 4) ? (n_in / 256) * 210
                              : (code == 3) ? (n_in / 256) * 176
                              : (code == 2) ? (n_in / 256) * 110
                              : (code == 1) ? (n_in / 32) * 18
                              : (code == 5) ? (n_in / 256) * 144
                              : (code == 6) ? (n_in / 256) * 136 : -1;
            if (m5_dequant_row(code, W + (size_t)r * rowbytes, n_in, wb) != 0) continue;
            for (int t = 0; t < T; t++)
                Y[(size_t)t * n_out + r] = dot_row(wb, X + (size_t)t * n_in, n_in);
        }
    }
    return 0;
}

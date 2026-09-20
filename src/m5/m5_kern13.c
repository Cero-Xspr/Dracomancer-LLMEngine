// m5_kern13.c: AVX-512 IQ2_S gemv (code 13)
// 块 = 82B × 256 值: d(f16) + qs[32] + signs[32] + qh[8] + scales[8]
//   子块 s: idx10 = qs[s] | ((qh[s>>2]>>((s&3)*2))&3)<<8
//           scale 成对: db = d·(0.5+nib[s>>1])·0.25
//           val_j = db · grid8[idx10][j] · (sign_byte[s]>>j &1 ? -1 : +1)
// GRIDSGN[1024*256*8] = 网格×符号 预展开 int8 表 (2MB), 首次调用构建
// gcc -O3 -march=native -mf16c -fopenmp -shared -fPIC m5_kern13.c -o m5_kern13.so
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include "iq2s_grid_gen.h"

static inline float h2f1(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { float v = ((float)man) * 5.9604644775390625e-8f; uint32_t vi; memcpy(&vi,&v,4); f = sign|vi; }
    } else if (exp == 31) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 112u) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static int8_t *GRIDSGN = NULL;

static void build_gridsgn(void) {
    GRIDSGN = (int8_t*)malloc((size_t)1024 * 256 * 8);
    for (int e = 0; e < 1024; e++) {
        uint16_t g16 = IQ2S_GRID[e];
        int8_t g8[8];
        for (int j = 0; j < 8; j++) {
            // 2-bit 码是原字节编码: 0→0x08(8), 1→0x19(25), 2→0x2b(43)
            const int code = (g16 >> (2 * j)) & 3;
            g8[j] = (code == 0) ? 8 : (code == 1) ? 25 : 43;
        }
        for (int sb = 0; sb < 256; sb++) {
            int8_t* dst = GRIDSGN + ((size_t)e * 256 + sb) * 8;
            for (int j = 0; j < 8; j++)
                dst[j] = (sb >> j) & 1 ? (int8_t)-g8[j] : g8[j];
        }
    }
}

static void rows_iq2s(const float* restrict x, const uint8_t* restrict W, int n_out, int n_in,
                      float* restrict y, int o0, int o1) {
    /* 逐行译码（块元数据是每行每块一份，不跨行共享——第一版曾误设跨行共享，出过垃圾数据）。
       微优化：① (idx10,sign) 合并寻址 GSB 表  ② 成对子块共享 scale ⇒ set1 减半
               ③ 下块 GRIDSGN 行预取。 */
    const int nb = n_in / 256;
    if (nb <= 0 || n_in % 256 != 0) return;
    if (GRIDSGN == NULL) build_gridsgn();
    for (int o = o0; o < o1; o++) {
        const uint8_t* row = W + (size_t)o * nb * 82;
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 82;
            const float d = h2f1(*(const uint16_t*)blk) * 0.25f;
            const uint8_t* qs = blk + 2;
            const uint8_t* sg = blk + 34;
            const uint8_t* qh = blk + 66;
            const uint8_t* scb = blk + 74;
            if (b + 1 < nb)
                _mm_prefetch((const char*)(blk + 82), _MM_HINT_T0);
            float sc[16];
            for (int i = 0; i < 8; i++) {   // 交错 nibble
                sc[2*i]     = d * (0.5f + (float)(scb[i] & 0x0F));
                sc[2*i + 1] = d * (0.5f + (float)(scb[i] >> 4));
            }
            __m256 accb = _mm256_setzero_ps();
            for (int s = 0; s < 32; s += 2) {
                const int i0 = qs[s]     | (((qh[s >> 2] >> ((s & 3) * 2)) & 3) << 8);
                const int i1 = qs[s + 1] | (((qh[(s + 1) >> 2] >> (((s + 1) & 3) * 2)) & 3) << 8);
                const __m128i q8a = _mm_loadl_epi64(
                    (const __m128i*)(GRIDSGN + ((size_t)i0 * 256 + sg[s]) * 8));
                const __m128i q8b = _mm_loadl_epi64(
                    (const __m128i*)(GRIDSGN + ((size_t)i1 * 256 + sg[s + 1]) * 8));
                const float* xp = x + b * 256 + s * 8;
                __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8a)),
                                         _mm256_loadu_ps(xp));
                t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8b)),
                                    _mm256_loadu_ps(xp + 8), t);
                accb = _mm256_fmadd_ps(_mm256_set1_ps(sc[s >> 1]), t, accb);
            }
            acc = _mm512_add_ps(acc, _mm512_insertf32x4(_mm512_castps256_ps512(accb),
                    _mm256_extractf128_ps(accb, 1), 1));
        }
        y[o] = _mm512_reduce_add_ps(acc);
    }
}

static void gemv_iq2s(const float* x, const uint8_t* W, int n_out, int n_in, float* y) {
    if (GRIDSGN == NULL) build_gridsgn();
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < n_out; o += 32) {
        const int o1 = (o + 32 < n_out) ? o + 32 : n_out;
        rows_iq2s(x, W, n_out, n_in, y, o, o1);
    }
}

int m5_gemv(int type, const float* x, const uint8_t* W, int n_out, int n_in, float* y) {
    if (type != 13) return -1;
    gemv_iq2s(x, W, n_out, n_in, y);
    return 0;
}

int m5_gemv_range(int type, const float* x, const uint8_t* W, int n_out, int n_in, float* y, int o0, int o1) {
    if (type != 13) return -1;
    if (GRIDSGN == NULL) build_gridsgn();
    rows_iq2s(x, W, n_out, n_in, y, o0, o1);
    return 0;
}

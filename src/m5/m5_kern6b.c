// m5_kern6b.c — IQ3_S 浮点内核的 gather 变体（spike）
// 与 m5_kern6.c rows_iq3_s 逐位一致：同样的 16 值/步结构、同样的累加顺序、
// 只是 4 次 memcpy 标量查表 → 一次 _mm256_i32gather_epi32(4 索引)。
// gcc -O3 -march=native -mf16c -shared -fPIC m5_kern6b.c -o m5_kern6b.so
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
        else { float v = ((float)man) * 5.9604644775390625e-8f; uint32_t vi; memcpy(&vi,&v,4); f = sign|vi; }
    } else if (exp == 31) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 112u) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}
static inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

// gather 版：与原版唯一的差别是 w0..w3 的来源（gather vs memcpy），数学完全同序
static void rows_iq3_s_gather(const float* restrict x, const uint8_t* restrict W, int n_out, int n_in, float* restrict y, int o0, int o1) {
    const int nb = n_in / 256;
    const __m512i lane_bits = _mm512_setr_epi32(1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768);
    const __m512 one = _mm512_set1_ps(1.0f), neg1 = _mm512_set1_ps(-1.0f);
    for (int o = o0; o < o1; o++) {
        const uint8_t* row = W + (size_t)o * nb * 110;
        __m512 racc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 110;
            float d = h2f1(*(const uint16_t*)blk);
            const uint8_t* qs    = blk + 2;
            const uint8_t* qh    = blk + 66;
            const uint8_t* signs = blk + 74;
            const uint8_t* scales= blk + 106;
            const float* xs = x + b * 256;
            for (int g = 0; g < 8; g++) {
                uint8_t nib = (g & 1) ? (scales[g >> 1] >> 4) : (scales[g >> 1] & 0x0F);
                __m512 dg = _mm512_set1_ps(d * (1.0f + 2.0f * (float)nib));
                __m512 gacc = _mm512_setzero_ps();
                for (int qi = 0; qi < 8; qi += 4) {
                    int i0 = g * 8 + qi;
                    uint32_t qs4; memcpy(&qs4, qs + i0, 4);
                    uint8_t qhb = qh[i0 >> 3];
                    int sh = i0 & 7;
                    uint16_t sbits = (uint16_t)signs[i0 >> 1] | ((uint16_t)signs[(i0 >> 1) + 1] << 8);
                    int idx0 = (qs4 & 0xFF)        | (((qhb >> (sh    )) & 1) << 8);
                    int idx1 = ((qs4 >> 8) & 0xFF) | (((qhb >> (sh + 1)) & 1) << 8);
                    int idx2 = ((qs4 >> 16) & 0xFF)| (((qhb >> (sh + 2)) & 1) << 8);
                    int idx3 = ((qs4 >> 24) & 0xFF)| (((qhb >> (sh + 3)) & 1) << 8);
                    __m128i vidx = _mm_setr_epi32(idx0, idx1, idx2, idx3);
                    __m128i w16g = _mm_i32gather_epi32((const int*)IQ3S_GRID, vidx, 4);
                    __m512i g16 = _mm512_cvtepi8_epi32(w16g);
                    __m512i bits = _mm512_and_si512(_mm512_set1_epi32((int)sbits), lane_bits);
                    __mmask16 nz = _mm512_cmpneq_epi32_mask(bits, _mm512_setzero_si512());
                    __m512 sgn = _mm512_mask_blend_ps(nz, one, neg1);
                    __m512 v16 = _mm512_mul_ps(_mm512_cvtepi32_ps(g16), sgn);
                    gacc = _mm512_fmadd_ps(v16, _mm512_loadu_ps(xs + i0 * 4), gacc);
                }
                racc = _mm512_fmadd_ps(dg, gacc, racc);
            }
        }
        y[o] = hsum512(racc);
    }
}

int m5_gemv(int type, const float* x, const uint8_t* W, int n_out, int n_in, float* y) {
    if (type != 2) return -1;
    for (int o = 0; o < n_out; o++) rows_iq3_s_gather(x, W, n_out, n_in, y, o, o+1);
    return 0;
}
int m5_gemv_range(int type, const float* x, const uint8_t* W, int n_out, int n_in, float* y, int o0, int o1) {
    if (type != 2 || o0 < 0 || o1 > n_out || o0 >= o1) return -1;
    rows_iq3_s_gather(x, W, n_out, n_in, y, o0, o1);
    return 0;
}

// m5_batch8_k.c: K2 的 MoE/MoVA 批量 gemv —— N 个专家一次 OMP 区, 摊薄 fork/join 与缺页风暴
//   y[e][o] = Σ_i x_e[i] · W[idx[e]][o][i]
// x_e = x + e*x_stride (gate/up 共享同一 x ⇒ x_stride=0; down 逐专家 ⇒ x_stride=n_in)
// type: 5=Q4_K (144B 块)  4=Q6_K (210B 块)
// 数学与 m5_kern7/m5_kern8 的行内核逐位一致 (同乘加顺序):
//   Q4_K: Σ (d·sc·Σq·x − dmin·m·Σx),  min 项只减 lane0 (maskz)
//   Q6_K: Σ d·sc[p>>4]·q·x
// gcc -O3 -march=native -mf16c -fopenmp -shared -fPIC m5_batch8_k.c -o m5_batch8_k.so
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include <stdlib.h>

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

// ── Q4_K: N 专家批量（行块级并行）──
static void rows8_q4_k_xe(const float* x, long x_stride, const uint8_t* W, const int* idx, long per,
                          int n_out_e, int n_in, float* y, int e, int o0, int o1,
                          const float* sx, int nb) {
    const __m256i m0F = _mm256_set1_epi8(0x0F);
    {
        const uint8_t* Wb = W + (int64_t)idx[e] * per;
        const float* xe = x + (int64_t)e * x_stride;
        float* ye = y + (int64_t)e * n_out_e;
        for (int o = 0; o < n_out_e; o++) {
            const uint8_t* row = Wb + (size_t)o * nb * 144;
            __m512 acc = _mm512_setzero_ps();
            for (int b = 0; b < nb; b++) {
                const uint8_t* blk = row + (size_t)b * 144;
                const float d = h2f1(*(const uint16_t*)blk), dmin = h2f1(*(const uint16_t*)(blk+2));
                uint8_t sc[8], m[8];
                { const uint8_t* s = blk+4;
                  for(int i=0;i<4;i++){sc[i]=s[i]&0x3F;m[i]=s[4+i]&0x3F;}
                  for(int i=0;i<4;i++){sc[4+i]=(s[8+i]&0x0F)|((s[i]>>2)&0x30);m[4+i]=((s[8+i]>>4)&0x0F)|((s[4+i]>>2)&0x30);} }
                const uint8_t* qs = blk + 16;
                const float* xp = xe + b*256;
                float scf[8], minf[8];
                for (int i = 0; i < 8; i++) { scf[i] = d * (float)sc[i]; minf[i] = dmin * (float)m[i]; }
                _mm_prefetch((const char*)(blk + 144), _MM_HINT_T0);
                for (int g = 0; g < 8; g += 2) {
                    const __m256i b8  = _mm256_loadu_si256((const __m256i*)(qs + (g>>1)*32));
                    const __m256i vlo = _mm256_and_si256(b8, m0F);
                    const __m256i vhi = _mm256_and_si256(_mm256_srli_epi16(b8, 4), m0F);
                    {
                        const __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(vlo)));
                        const __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(vlo, 1)));
                        __m512 dot = _mm512_mul_ps(f0, _mm512_loadu_ps(xp + g*32));
                        dot = _mm512_fmadd_ps(f1, _mm512_loadu_ps(xp + g*32 + 16), dot);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(scf[g]), dot, acc);
                        acc = _mm512_fnmadd_ps(_mm512_set1_ps(minf[g]),
                                _mm512_maskz_mov_ps((__mmask16)1, _mm512_set1_ps(sx[b*8+g])), acc);
                    }
                    {
                        const __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(vhi)));
                        const __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(vhi, 1)));
                        __m512 dot = _mm512_mul_ps(f0, _mm512_loadu_ps(xp + (g+1)*32));
                        dot = _mm512_fmadd_ps(f1, _mm512_loadu_ps(xp + (g+1)*32 + 16), dot);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(scf[g+1]), dot, acc);
                        acc = _mm512_fnmadd_ps(_mm512_set1_ps(minf[g+1]),
                                _mm512_maskz_mov_ps((__mmask16)1, _mm512_set1_ps(sx[b*8+g+1])), acc);
                    }
                }
            }
            ye[o] = hsum512(acc);
        }
    }
}

int m5_gemvn_q4k(const float* x, long x_stride, const uint8_t* W, const int* idx, long per,
                 int n_out_e, int n_in, int n_exp, float* y) {
    const int nb = n_in / 256;
    if (nb <= 0 || n_in % 256 != 0 || nb > 256) return -1;
    float* sx = (float*)malloc(sizeof(float) * nb * 8);
    if (!sx) return -2;
    for (int b = 0; b < nb; b++)
        for (int g = 0; g < 8; g++) {
            const float* xp = x + b*256 + g*32;
            __m512 s = _mm512_add_ps(_mm512_loadu_ps(xp), _mm512_loadu_ps(xp + 16));
            sx[b*8+g] = hsum512(s);       // 仅 x_stride==0（共享 x）时严格成立
        }
    const int BLK = 96;                    // 行块: n_exp×ceil(n_out/96) 个并行单元
    const int nob = (n_out_e + BLK - 1) / BLK;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int e = 0; e < n_exp; e++)
        for (int ob = 0; ob < nob; ob++) {
            const int o0 = ob * BLK, o1 = (o0 + BLK < n_out_e) ? o0 + BLK : n_out_e;
            if (x_stride != 0) {
                // 逐专家 x: sx 按本专家的 x 重算
                float sxl[256*8];
                for (int b = 0; b < nb; b++)
                    for (int g = 0; g < 8; g++) {
                        const float* xp = x + (int64_t)e * x_stride + b*256 + g*32;
                        __m512 s = _mm512_add_ps(_mm512_loadu_ps(xp), _mm512_loadu_ps(xp + 16));
                        sxl[b*8+g] = hsum512(s);
                    }
                rows8_q4_k_xe(x, x_stride, W, idx, per, n_out_e, n_in, y, e, o0, o1, sxl, nb);
            } else {
                rows8_q4_k_xe(x, x_stride, W, idx, per, n_out_e, n_in, y, e, o0, o1, sx, nb);
            }
        }
    free(sx);
    return 0;
}

// ── Q6_K: N 专家批量（行块级并行）──
static void rows8_q6_k(const float* x, long x_stride, const uint8_t* W, const int* idx, long per,
                       int n_out_e, int n_in, float* y, int e, int o0, int o1) {
    const int nb = n_in / 256;
    {
        const uint8_t* Wb = W + (int64_t)idx[e] * per;
        const float* xe = x + (int64_t)e * x_stride;
        float* ye = y + (int64_t)e * n_out_e;
        for (int o = 0; o < n_out_e; o++) {
            const uint8_t* row = Wb + (size_t)o * nb * 210;
            __m512 racc = _mm512_setzero_ps();
            for (int b = 0; b < nb; b++) {
                const uint8_t* blk = row + (size_t)b * 210;
                const uint8_t* ql = blk;
                const uint8_t* qh = blk + 128;
                const int8_t*  sc = (const int8_t*)(blk + 192);
                const float d = h2f1(*(const uint16_t*)(blk + 208));
                const float* xs = xe + b * 256;
                const __m128i m0F = _mm_set1_epi8(0x0F);
                const __m128i m03 = _mm_set1_epi8(0x03);
                const __m128i c32 = _mm_set1_epi8(32);
                float scf[16];
                _mm_prefetch((const char*)(blk + 210), _MM_HINT_T0);
                for (int i = 0; i < 16; i++) scf[i] = d * (float)sc[i];
                for (int sb = 0; sb < 2; sb++)
                for (int g2 = 0; g2 < 4; g2++) {
                    const __m128i ql16 = _mm_loadu_si128((const __m128i*)(ql + sb * 64 + g2 * 16));
                    const __m128i qh16 = _mm_loadu_si128((const __m128i*)(qh + sb * 32 + (g2 & 1) * 16));
                    const int base = sb * 8 + g2;
                    const int sh0 = ((base >> 1) & 3) * 2;
                    const int sh1 = (((base + 4) >> 1) & 3) * 2;
                    {
                        const __m128i q8 = _mm_sub_epi8(
                            _mm_or_si128(_mm_and_si128(ql16, m0F),
                                         _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, sh0), m03), 4)), c32);
                        const __m512 qf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q8));
                        racc = _mm512_fmadd_ps(_mm512_mul_ps(qf, _mm512_set1_ps(scf[base])),
                                               _mm512_loadu_ps(xs + base * 16), racc);
                    }
                    {
                        const __m128i q8 = _mm_sub_epi8(
                            _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql16, 4), m0F),
                                         _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, sh1), m03), 4)), c32);
                        const __m512 qf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q8));
                        racc = _mm512_fmadd_ps(_mm512_mul_ps(qf, _mm512_set1_ps(scf[base + 4])),
                                               _mm512_loadu_ps(xs + (base + 4) * 16), racc);
                    }
                }
            }
            ye[o] = _mm512_reduce_add_ps(racc);
        }
    }
}

int m5_gemvn_q6k(const float* x, long x_stride, const uint8_t* W, const int* idx, long per,
                 int n_out_e, int n_in, int n_exp, float* y) {
    const int nb = n_in / 256;
    if (nb <= 0 || n_in % 256 != 0) return -1;
    const int BLK = 96;
    const int nob = (n_out_e + BLK - 1) / BLK;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int e = 0; e < n_exp; e++)
        for (int ob = 0; ob < nob; ob++) {
            const int o0 = ob * BLK, o1 = (o0 + BLK < n_out_e) ? o0 + BLK : n_out_e;
            rows8_q6_k(x, x_stride, W, idx, per, n_out_e, n_in, y, e, o0, o1);
        }
    return 0;
}

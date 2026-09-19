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

#define GTOK 16   // 16-token 分块（G8→G16 实测 +8%，逐位一致）

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

// ===== Q8_0 / Q6_K / Q5_K 融合变体（与 m5_gemm_fused 的 IQ3_S 同模式）=====
// 每 16 值组寄存器反量化一次 → GTOK 个 token fma。

static inline int rowbytes_of(int code, int n) {
    switch (code) {
        case 0: return (n / 32) * 34;
        case 7: return n * 2;
        case 4: return (n / 256) * 210;
        case 3: return (n / 256) * 176;
        case 2: return (n / 256) * 110;
        case 1: return (n / 32) * 18;
        case 5: return (n / 256) * 144;
        case 6: return (n / 256) * 136;
        default: return 0;
    }
}

// 每 16 值组反量化（base 16 对齐；位布局与 m5_gemm.c/m5_kern6/7/8/9 一致）
static inline void get_sc_m(const uint8_t* s, int j, uint8_t* d8, uint8_t* m8) {
    if (j < 4) { *d8 = s[j] & 63; *m8 = s[j + 4] & 63; }
    else {
        *d8 = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *m8 = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

static inline void dq16(int code, const uint8_t* row, int base, __m512* v) {
    if (code == 0) {                                     // Q8_0
        const uint8_t* blk = row + (size_t)(base >> 5) * 34;
        const float d = h2f1(*(const uint16_t*)blk);
        const __m128i b8 = _mm_loadu_si128((const __m128i*)(blk + 2 + (base & 31)));
        *v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8)), _mm512_set1_ps(d));
    } else if (code == 4) {                              // Q6_K
        const int p = base & 255;                        // ★ 组内位置：所有位域都按块内 p
        const uint8_t* blk = row + (size_t)(base >> 8) * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = (const int8_t*)(blk + 192);
        const float d = h2f1(*(const uint16_t*)(blk + 208));
        const int seg = (p >> 6) & 1;
        const int qoff = ((p >> 5) & 3) * 2;
        const __m128i b8 = _mm_loadu_si128((const __m128i*)(ql + ((p >> 7) * 64) + (p & 63)));
        const __m128i nib = seg ? _mm_and_si128(_mm_srli_epi16(b8, 4), _mm_set1_epi8(0x0F))
                                : _mm_and_si128(b8, _mm_set1_epi8(0x0F));
        const __m128i qhb = _mm_loadu_si128((const __m128i*)(qh + ((p >> 7) * 32) + (p & 31)));
        const __m128i two = _mm_and_si128(_mm_srli_epi16(qhb, qoff), _mm_set1_epi8(0x03));
        const __m128i q6 = _mm_or_si128(nib, _mm_slli_epi16(two, 4));
        const __m512i q32 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(q6), _mm512_set1_epi32(32));
        *v = _mm512_mul_ps(_mm512_cvtepi32_ps(q32),
                           _mm512_set1_ps(d * (float)sc[p >> 4]));
    } else if (code == 5) {                              // Q4_K（144B/256：nibble 无 qh，6-bit sc/m 同 kern7）
        const int p = base & 255;
        const uint8_t* blk = row + (size_t)(base >> 8) * 144;
        const float d = h2f1(*(const uint16_t*)blk);
        const float dmin = h2f1(*(const uint16_t*)(blk + 2));
        const int g32 = p >> 5;
        uint8_t scj, mj;                       // ★ 只解当前组的两个标量（原 8 组全解在 16 组粒度下是 8× 冗余）
        { const uint8_t* s = blk + 4;
          if (g32 < 4) { scj = s[g32] & 0x3F; mj = s[4+g32] & 0x3F; }
          else { scj = (s[4+g32] & 0x0F) | ((s[g32-4] >> 2) & 0x30);
                 mj = ((s[4+g32] >> 4) & 0x0F) | ((s[g32] >> 2) & 0x30); } }
        const int hpar = g32 & 1;
        const __m128i b8 = _mm_loadu_si128((const __m128i*)(blk + 16 + (g32 >> 1) * 32 + (p & 31)));
        const __m128i nib = hpar ? _mm_and_si128(_mm_srli_epi16(b8, 4), _mm_set1_epi8(0x0F))
                                 : _mm_and_si128(b8, _mm_set1_epi8(0x0F));
        const __m512 lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(nib)),
                                        _mm512_set1_ps(d * (float)scj));
        *v = _mm512_sub_ps(lo, _mm512_set1_ps(dmin * (float)mj));
    } else if (code == 6) {                              // IQ4_XS（136B/256：d scales_h scales_l(4) qs(128)）
        static const int8_t KV2[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        const int p = base & 255;
        const uint8_t* blk = row + (size_t)(base >> 8) * 136;
        const float d = h2f1(*(const uint16_t*)blk);
        const uint16_t sh = *(const uint16_t*)(blk + 2);
        const uint8_t* sl = blk + 4;
        const int g32 = p >> 5;
        const uint8_t lo4 = (g32 & 1) ? (sl[g32 >> 1] >> 4) : (sl[g32 >> 1] & 0x0F);
        const int sc = (int)((((unsigned)sh >> (2 * g32)) & 3) << 4 | lo4) - 32;
        const int half = (p >> 4) & 1;
        const __m128i b8 = _mm_loadu_si128((const __m128i*)(blk + 8 + g32 * 16));
        const __m128i nib = half ? _mm_and_si128(_mm_srli_epi16(b8, 4), _mm_set1_epi8(0x0F))
                                 : _mm_and_si128(b8, _mm_set1_epi8(0x0F));
        const __m128i kv = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)KV2), nib);
        *v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(kv)), _mm512_set1_ps(d * (float)sc));
    } else if (code == 3) {                              // Q5_K
        const int p = base & 255;                        // ★ 组内位置
        const uint8_t* blk = row + (size_t)(base >> 8) * 176;
        const uint8_t* qs = blk + 48;
        const uint8_t* qh = blk + 16;
        uint8_t scj, mj;
        get_sc_m(blk + 4, p >> 5, &scj, &mj);
        const float d = h2f1(*(const uint16_t*)blk);
        const float dmin = h2f1(*(const uint16_t*)(blk + 2));
        const int hpar = (p >> 5) & 1;
        const __m128i b8 = _mm_loadu_si128((const __m128i*)(qs + ((p >> 6) * 32) + (p & 31)));
        const __m128i nib = hpar ? _mm_and_si128(_mm_srli_epi16(b8, 4), _mm_set1_epi8(0x0F))
                                 : _mm_and_si128(b8, _mm_set1_epi8(0x0F));
        const __m128i qhb = _mm_loadu_si128((const __m128i*)(qh + (p & 31)));
        const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(qhb, (p >> 6) * 2 + hpar), _mm_set1_epi8(0x01));
        const __m512i q5 = _mm512_add_epi32(_mm512_cvtepu8_epi32(nib),
                                            _mm512_slli_epi32(_mm512_cvtepu8_epi32(hi1), 4));
        const __m512 lo = _mm512_mul_ps(_mm512_cvtepi32_ps(q5),
                                        _mm512_set1_ps(d * (float)scj));
        *v = _mm512_sub_ps(lo, _mm512_set1_ps(dmin * (float)mj));
    }
}

static inline void gemm_rows_t(const float* X, const uint8_t* row, int t0, int tc,
                               int n_out, int n_in, float* Y, int r, int code) {
    __m512 racc[GTOK];
    for (int t = 0; t < tc; t++) racc[t] = _mm512_setzero_ps();
    const int NB16 = n_in / 16;
    for (int u = 0; u < NB16; u++) {
        __m512 v;
        dq16(code, row, u * 16, &v);
        for (int t = 0; t < GTOK; t++)
            racc[t] = _mm512_fmadd_ps(v, _mm512_loadu_ps(X + (size_t)(t0 + t) * n_in + u * 16), racc[t]);
    }
    for (int t = 0; t < tc; t++) Y[(size_t)(t0 + t) * n_out + r] = hsum512(racc[t]);
}

#define FUSED3(FN, CODE)                                                    \
void FN(const float* X, const uint8_t* W, int c, int n_out, int n_in, float* Y) { \
    const int rb = rowbytes_of(CODE, n_in);                                  \
    _Pragma("omp parallel for schedule(dynamic, 4)")                          \
    for (int r = 0; r < n_out; r++) {                                        \
        const uint8_t* row = W + (size_t)r * rb;                             \
        for (int t0 = 0; t0 < c; t0 += GTOK) {                               \
            const int tc = (t0 + GTOK <= c) ? GTOK : (c - t0);               \
            gemm_rows_t(X, row, t0, tc, n_out, n_in, Y, r, CODE);            \
        }                                                                    \
    }                                                                        \
}

FUSED3(m5_gemm_q80, 0)
FUSED3(m5_gemm_q6k, 4)
FUSED3(m5_gemm_q5k, 3)
FUSED3(m5_gemm_q4k, 5)
FUSED3(m5_gemm_iq4xs, 6)

// 统一分发入口：支持的码直接融合，返回 0；不支持的码返回 -1（调用方回退）
int m5_gemm_auto(int code, const float* X, const uint8_t* W, int c, int n_out, int n_in, float* Y) {
    switch (code) {
        case 0: m5_gemm_q80(X, W, c, n_out, n_in, Y); return 0;
        case 2: m5_gemm_iq3s(X, W, c, n_out, n_in, Y); return 0;
        case 3: m5_gemm_q5k(X, W, c, n_out, n_in, Y); return 0;
        case 4: m5_gemm_q6k(X, W, c, n_out, n_in, Y); return 0;
        case 5: m5_gemm_q4k(X, W, c, n_out, n_in, Y); return 0;
        case 6: m5_gemm_iq4xs(X, W, c, n_out, n_in, Y); return 0;
        default: return -1;
    }
}

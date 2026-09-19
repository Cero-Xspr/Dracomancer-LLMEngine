// m6_moe_tok.c — MoE FFN 按位置批处理（prefill M2 核心）v3
// 串行建索引 → 专家级并行（阶段间栅栏、阶段内无嵌套 OMP；每线程私有反量化行缓冲）：
//   ① gather  ② gate/up：行级 dequant + 点积（列主序写 [inter×c]）→ silu → 转置行主序
//   ③ down：行级 dequant + 点积  ④ 按 token 的确定性 scatter（k 升序 = 与逐 token 路径同序）
// work:  T*n_used*(n_in + 3*inter + n_out) floats（gather/gu/uu/act/yd 专家分段区）
// iwork: [T*n_used slot][T*n_used ilist][n_exp+1 off]
// gwbuf: ≥ nthreads * n_in floats（每线程私有反量化行）
// 编译: gcc -O3 -march=native -fopenmp -shared -fPIC m6_moe_tok.c -o m6_moe_tok.so
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef int (*dequant_fn)(int, const uint8_t*, int, float*);

// ★ 手写 zmm 点积：浮点归约不开 -ffast-math 不会被自动向量化（v2 教训：标量点积慢 4×+）
#include <immintrin.h>
static inline float dotf(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

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

void m6_moe_tok(dequant_fn DQ, float* gwbuf,
                const float* X, const int* order, const float* wtop,
                const uint64_t* ep, const int* ec,
                int T, int n_used, int inter, int n_in, int n_out, int n_exp,
                float* out, float* work, int* iwork) {
    int* slot  = iwork;                       // [T*n_used]
    int* ilist = slot + (size_t)T * n_used;   // [T*n_used]
    int* off   = ilist + (size_t)T * n_used;  // [n_exp+1]

    memset(off, 0, sizeof(int) * (n_exp + 1));
    for (int i = 0; i < T * n_used; i++) {
        const int e = order[i];
        if (e >= 0 && e < n_exp) off[e + 1]++;
    }
    for (int e = 0; e < n_exp; e++) off[e + 1] += off[e];
    if (n_exp <= 512) {
        int cursor[512];
        for (int e = 0; e < n_exp; e++) cursor[e] = off[e];
        for (int i = 0; i < T * n_used; i++) {
            const int e = order[i];
            if (e >= 0 && e < n_exp) {
                slot[i] = cursor[e] - off[e];
                ilist[cursor[e]] = i;
                cursor[e]++;
            }
        }
    }
    const int seg_t = T * n_used;
    float* gather = work;                                   // [seg_t * n_in]
    float* gu     = gather + (size_t)seg_t * n_in;          // [seg_t * inter] 列主序 (r,j)
    float* uu     = gu + (size_t)seg_t * inter;             // [seg_t * inter] 列主序 (r,j)
    float* act    = uu + (size_t)seg_t * inter;             // [seg_t * inter] 行主序 (j,r)
    float* yd     = act + (size_t)seg_t * inter;            // [seg_t * n_out]
    memset(out, 0, sizeof(float) * (size_t)T * n_out);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* wb = gwbuf + (size_t)tid * ((n_in + 15) & ~15);
        // ① gather
        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < n_exp; e++) {
            const int c = off[e + 1] - off[e];
            if (c <= 0) continue;
            const int* my = ilist + off[e];
            float* seg_g = gather + (size_t)off[e] * n_in;
            for (int j = 0; j < c; j++) {
                const int t = my[j] / n_used;
                memcpy(seg_g + (size_t)j * n_in, X + (size_t)t * n_in, sizeof(float) * n_in);
            }
        }
        // ② gate/up（列主序）→ silu → 转置行主序
        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < n_exp; e++) {
            const int c = off[e + 1] - off[e];
            if (c <= 0) continue;
            const int cg = ec[3*e], cu = ec[3*e + 1];
            const uint8_t* wg = (const uint8_t*)(uintptr_t)ep[3*e];
            const uint8_t* wu = (const uint8_t*)(uintptr_t)ep[3*e + 1];
            const int rbg = rowbytes_of(cg, n_in), rbu = rowbytes_of(cu, n_in);
            const int* my = ilist + off[e];
            float* seg_g  = gather + (size_t)off[e] * n_in;
            float* seg_gu = gu + (size_t)off[e] * inter;
            float* seg_uu = uu + (size_t)off[e] * inter;
            float* seg_ac = act + (size_t)off[e] * inter;
            for (int r = 0; r < inter; r++) {
                DQ(cg, wg + (size_t)r * rbg, n_in, wb);
                for (int j = 0; j < c; j++)
                    seg_gu[(size_t)r * c + j] = dotf(wb, seg_g + (size_t)j * n_in, n_in);
                DQ(cu, wu + (size_t)r * rbu, n_in, wb);
                for (int j = 0; j < c; j++)
                    seg_uu[(size_t)r * c + j] = dotf(wb, seg_g + (size_t)j * n_in, n_in);
            }
            for (int j = 0; j < c; j++) {
                for (int r = 0; r < inter; r++) {
                    const float gv = seg_gu[(size_t)r * c + j];
                    const float uv = seg_uu[(size_t)r * c + j];
                    seg_ac[(size_t)j * inter + r] = (gv / (1.f + expf(-gv))) * uv;
                }
            }
            // ③ down
            const int cd = ec[3*e + 2];
            const uint8_t* wd = (const uint8_t*)(uintptr_t)ep[3*e + 2];
            const int rbd = rowbytes_of(cd, inter);
            float* seg_yd = yd + (size_t)off[e] * n_out;
            for (int r = 0; r < n_out; r++) {
                DQ(cd, wd + (size_t)r * rbd, inter, wb);
                for (int j = 0; j < c; j++)
                    seg_yd[(size_t)j * n_out + r] = dotf(wb, seg_ac + (size_t)j * inter, inter);
            }
        }
        // ④ 按 token 的确定性 scatter（k 升序 = 逐 token 路径的专家累加序）
        #pragma omp for schedule(static)
        for (int t = 0; t < T; t++) {
            float* o = out + (size_t)t * n_out;
            for (int k = 0; k < n_used; k++) {
                const int i = t * n_used + k;
                const int e = order[i];
                if (e < 0 || e >= n_exp) continue;
                const float w = wtop[i];
                const float* yj = yd + ((size_t)off[e] + slot[i]) * n_out;
                for (int r = 0; r < n_out; r++) o[r] += w * yj[r];
            }
        }
    }
}

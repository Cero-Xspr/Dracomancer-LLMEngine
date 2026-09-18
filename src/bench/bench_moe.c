// bench_moe.c — MoE 专家段并行组织变体对比（每行点积不变 ⇒ 输出逐位一致，只动任务划分）
// 变体：0=现状(每专家整矩阵1个dynamic任务) 1=行分块(默认64行/块,dynamic)
//       3=单线程基线  4=行分块+kern12整数核(type 13)
// 编译: gcc -O3 -march=native -fopenmp -shared -fPIC -o bench_moe.so bench_moe.c
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>

typedef int (*range_fn)(int, const float*, const uint8_t*, int, int, float*, int, int);
static range_fn KRNG;   // kern6 m5_gemv_range
static range_fn KRNG12; // kern12 m5_gemv_range（可选）

void bm_set_kern(range_fn k6, range_fn k12) { KRNG = k6; KRNG12 = k12; }

#define MAXT 512
typedef struct { const uint8_t* w; int code; int j; int r0, r1; float* yb; } mt_t;  // yb=矩阵基址; 调用时传 yb-r0（range 约定=绝对行号写 y）

void bm_moe(const float* x0,
            const uint8_t* const* gp, const uint8_t* const* up, const uint8_t* const* dp,
            const float* wtop, int n_used, int inter, int n_in, int n_out,
            float* scratch, float* out, int variant, int chunk) {
    float* gu  = scratch;
    float* uu  = gu + (size_t)n_used*inter;
    float* act = uu + (size_t)n_used*inter;
    float* yd  = act + (size_t)n_used*inter;
    range_fn K  = (variant == 4 && KRNG12) ? KRNG12 : KRNG;
    int   code = (variant == 4 && KRNG12) ? 13 : 2;
    int ch = chunk > 0 ? chunk : inter;

    if (variant == 3) {                      // 单线程基线（silu 与并行版同式）
        for (int j = 0; j < n_used; j++) {
            K(code, x0, gp[j], inter, n_in, gu + (size_t)j*inter, 0, inter);
            K(code, x0, up[j], inter, n_in, uu + (size_t)j*inter, 0, inter);
        }
        for (int i = 0; i < n_used*inter; i++) {
            const float gv = gu[i];
            act[i] = (gv / (1.f + expf(-gv))) * uu[i];
        }
        for (int j = 0; j < n_used; j++)
            K(code, act + (size_t)j*inter, dp[j], n_out, inter, yd + (size_t)j*n_out, 0, n_out);
        for (int i = 0; i < n_out; i++) {
            float a = 0.f;
            for (int j = 0; j < n_used; j++) a += wtop[j] * yd[(size_t)j*n_out + i];
            out[i] = a;
        }
        return;
    }

    static mt_t tg[MAXT], tu[MAXT], td[MAXT];
    static int ng, nu, nd;
    ng = nu = nd = 0;
    for (int j = 0; j < n_used; j++) {
        for (int r = 0; r < inter; r += ch) {
            int r1 = r + ch < inter ? r + ch : inter;
            if (ng < MAXT) { tg[ng].w = gp[j]; tg[ng].code = code; tg[ng].j = j; tg[ng].r0 = r; tg[ng].r1 = r1; tg[ng].yb = gu + (size_t)j*inter; ng++; }
            if (nu < MAXT) { tu[nu].w = up[j]; tu[nu].code = code; tu[nu].j = j; tu[nu].r0 = r; tu[nu].r1 = r1; tu[nu].yb = uu + (size_t)j*inter; nu++; }
        }
        for (int r = 0; r < n_out; r += ch) {
            int r1 = r + ch < n_out ? r + ch : n_out;
            if (nd < MAXT) { td[nd].w = dp[j]; td[nd].code = code; td[nd].j = j; td[nd].r0 = r; td[nd].r1 = r1; td[nd].yb = yd + (size_t)j*n_out; nd++; }
        }
    }

    #pragma omp parallel
    {
        if (variant == 0) {                  // 现状：每专家整矩阵一个 dynamic 任务
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < n_used; j++) {
                KRNG(2, x0, gp[j], inter, n_in, gu + (size_t)j*inter, 0, inter);
                KRNG(2, x0, up[j], inter, n_in, uu + (size_t)j*inter, 0, inter);
            }
            #pragma omp for schedule(static)
            for (int i = 0; i < n_used*inter; i++) {
                const float gv = gu[i];
                act[i] = (gv / (1.f + expf(-gv))) * uu[i];
            }
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < n_used; j++)
                KRNG(2, act + (size_t)j*inter, dp[j], n_out, inter, yd + (size_t)j*n_out, 0, n_out);
        } else {                             // 1/4：行分块任务
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < ng; i++)
                K(tg[i].code, x0, tg[i].w, inter, n_in, tg[i].yb, tg[i].r0, tg[i].r1);   /* 绝对行号：基址不动 */
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < nu; i++)
                K(tu[i].code, x0, tu[i].w, inter, n_in, tu[i].yb, tu[i].r0, tu[i].r1);
            #pragma omp for schedule(static)
            for (int i = 0; i < n_used*inter; i++) {
                const float gv = gu[i];
                act[i] = (gv / (1.f + expf(-gv))) * uu[i];
            }
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < nd; i++)
                K(td[i].code, act + (size_t)td[i].j*inter, td[i].w, n_out, inter,
                  td[i].yb, td[i].r0, td[i].r1);
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n_out; i++) {
            float a = 0.f;
            for (int j = 0; j < n_used; j++) a += wtop[j] * yd[(size_t)j*n_out + i];
            out[i] = a;
        }
    }
}

double bm_time(void) { return omp_get_wtime(); }

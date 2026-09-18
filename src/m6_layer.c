// m6_layer.c: GDN 层注意力路径的 C 驱动 (消灭 numpy/python 编排开销)
// gcc -O3 -mavx512f -mavx512bw -mfma -fopenmp -shared -fPIC -o m6_layer.so m6_layer.c
// 注: m6_engine.c 里有一行 #include "m6_layer.c" → 两个库共用这份派发代码, 改这里两边都生效。
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define EPS 1e-6f
#define SV 128
#define N_KH 16

#include <dlfcn.h>
#include <stdlib.h>
#include <time.h>
// ---- 内建剖面 (Python 侧用 m6_prof_get 读; 关掉时开销可忽略) ----
#include <time.h>
double m6_prof_t[24]; long m6_prof_n[24];   // 0..7 原有（llama 系/zaya）；8..19 是 granite 的分段
static inline double _now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
void m6_prof_reset(void) { for (int i = 0; i < 24; i++) { m6_prof_t[i] = 0; m6_prof_n[i] = 0; } }
double m6_prof_get(int i) { return m6_prof_t[i]; }
long m6_prof_cnt(int i) { return m6_prof_n[i]; }

typedef int (*gemv_fn)(int, const float*, const uint8_t*, int, int, float*);
typedef int (*gemv_range_fn)(int, const float*, const uint8_t*, int, int, float*, int, int);
static gemv_fn G012, G3, G4, G56, GS, G50;   // G50 = Q5_0 (code 8)
static gemv_range_fn GR012, GR3, GR4, GR56, GRF, GR50;   // 行区间入口 (无内嵌 OMP)
static gemv_fn G12;                                      // IQ3_S 整数快速路径 (可选)
static gemv_range_fn GR12;
static inline void gemv_range_any(int code, const float* x, const uint8_t* buf,
                                  int n_out, int n_in, float* y, int o0, int o1);   // 定义在 MoE 段前
// GDN/全注意力的 qkv/gate/ssm_out 大 gemv 行分段并行开关（逐位不变）；GDN_SEG=0 回单线程
static int g_gdn_seg = 1;
int m6_set_gdn_seg(int v) { int o = g_gdn_seg; g_gdn_seg = v; return o; }
// MoE 专家段行分块 (moe_experts)：>0 时把每专家整矩阵任务拆成 ~g_moe_chunk 行的
// 细任务。每行点积仍由同一内核按原序计算 ⇒ 输出**逐位不变**（bench_moe 已验证
// 三种块宽逐位一致）；收益来自大核/小核混部下的负载均衡（qwen35moe 实测 MoE 段
// ~1.25-1.29×；引擎端到端配对 1.05-1.10×，64 优于 32/16）。env MOE_CHUNK 覆盖；
// 0 = 旧路径（每专家 1 个 dynamic 任务）。
static int g_moe_chunk = 64;
int m6_set_moe_chunk(int v) { int o = g_moe_chunk; g_moe_chunk = v; return o; }   // 运行时 A/B 用
// ===================== 调用记录（审计用；M6_AUDIT=1 时开） =====================
// 为什么要有它：审计"两个 gemv 入口是否等价"时，若按 GGUF 形状去推 n_out/n_in，
// 会漏掉"引擎实际传的参数与形状映射不同"的情形（2026-09-15 就是这样漏掉了 v_b 的真实组合，
// 还给出过一次 NaN 误导）。⇒ 让引擎**自己报出每次调用**，审计拿真实参数重放。
// 开销：每次 gemv 一次分支判断（记录时再加几次赋值）；不记录时约等于零。
#define M6_AUDIT_MAX 6000
// cnt = 整调用(gemv_any)次数；cnt_seg = 行区间调用(gemv_range_any)次数 —— 两者语义不同，
// 必须分开数：并行区里一次 960×960 会被切成几十段调用，混在一起会把账算错。
typedef struct { int code, n_out, n_in, o0, o1, wmod64, xmod64; long cnt, cnt_seg; } M6AuditRec;
static M6AuditRec g_audit[M6_AUDIT_MAX];
static int g_audit_n = 0;
static int g_audit_on = -1;
static inline int m6_audit_enabled(void) {
    if (g_audit_on < 0) g_audit_on = getenv("M6_AUDIT") ? 1 : 0;
    return g_audit_on;
}
static inline void m6_audit_rec(int code, const float* x, const uint8_t* W,
                                int n_out, int n_in, int o0, int o1, int is_seg) {
    if (!m6_audit_enabled() || g_audit_n >= M6_AUDIT_MAX) return;
    // 每个 (code,n_out,n_in) 组合只占一条，但**累计调用次数**（B3 要靠次数算"每 token 调用数"）
    for (int i = 0; i < g_audit_n; i++)
        if (g_audit[i].code == code && g_audit[i].n_out == n_out && g_audit[i].n_in == n_in) {
            if (is_seg) g_audit[i].cnt_seg++; else g_audit[i].cnt++;
            return;
        }
    M6AuditRec* r = &g_audit[g_audit_n++];
    r->code = code; r->n_out = n_out; r->n_in = n_in; r->o0 = o0; r->o1 = o1;
    r->wmod64 = (int)(((uintptr_t)W) & 63); r->xmod64 = (int)(((uintptr_t)x) & 63);
    r->cnt = is_seg ? 0 : 1;
    r->cnt_seg = is_seg ? 1 : 0;
}
int m6_audit_n(void) { return g_audit_n; }
int m6_audit_field(int i, int f) {
    if (i < 0 || i >= g_audit_n) return -1;
    const M6AuditRec* r = &g_audit[i];
    switch (f) {
        case 0: return r->code;   case 1: return r->n_out; case 2: return r->n_in;
        case 3: return r->o0;     case 4: return r->o1;    case 5: return r->wmod64;
        case 6: return r->xmod64; case 7: return (int)r->cnt; default: return (int)r->cnt_seg;
    }
}

void m6_init(gemv_fn d012, gemv_fn d3, gemv_fn d4, gemv_fn d56, gemv_fn dscalar) {
    G012 = d012; G3 = d3; G4 = d4; G56 = d56; GS = dscalar;
}
// 直接 dlopen 内核库 (消除 Python 回调开销); 返回 0 成功
// 追加格式 (Q5_0) 的加载入口: 可选, 独立于 m6_init_dl 以免破坏既有 5 参数调用方
int m6_init_extra_dl(const char* p50) {
    void* h = dlopen(p50, RTLD_NOW | RTLD_LOCAL);
    if (!h) return 1;
    G50  = (gemv_fn)dlsym(h, "m5_gemv");
    GR50 = (gemv_range_fn)dlsym(h, "m5_gemv_range");
    return G50 ? 0 : 2;
}
// IQ3_S 整数快速路径 (kern12) 的可选加载入口。
// 只有显式调用本函数才会接管 code 2; 不调用则 IQ3_S 完全走原 kern6 浮点路径 (逐位不变)。
// kern12 的 m5_gemv 只认 type 12/13/14, 故调用处把 code 2 改名后传入。
// IQ3S_MODE 环境变量选择变体 (在 init 时读一次, 并行区之前, 无竞争):
//   12 = 栈上展开查表, 13 = AVX512 i32gather, 14 = 查表桩(仅测速, 输出无意义)
//   轮转交错实测 13 一直 >= 12 (0.60 vs 0.51 等), 故**默认 13**。
static int g_iq3_mode = 13;
int m6_init_iq3_dl(const char* p12) {
    const char* mv = getenv("IQ3S_MODE");
    if (mv) {
        int v = atoi(mv);
        if (v == 12 || v == 13 || v == 14) g_iq3_mode = v;
    }
    void* h = dlopen(p12, RTLD_NOW | RTLD_LOCAL);
    if (!h) return 1;
    G12  = (gemv_fn)dlsym(h, "m5_gemv");
    GR12 = (gemv_range_fn)dlsym(h, "m5_gemv_range");
    if (!G12) { G12 = NULL; GR12 = NULL; return 2; }
    return 0;
}
int m6_iq3_fast_enabled(void) { return G12 != NULL; }
int m6_iq3_fast_mode(void) { return g_iq3_mode; }
int m6_init_dl(const char* p012, const char* p3, const char* p4, const char* p56, const char* ps) {
    void* h;
    {   // MoE 行分块宽（串行区读一次，无竞争）；MOE_CHUNK=0 回旧路径
        const char* mv = getenv("MOE_CHUNK");
        if (mv) g_moe_chunk = atoi(mv);
    }
    h = dlopen(p012, RTLD_NOW | RTLD_LOCAL); if (!h) return 1; G012 = (gemv_fn)dlsym(h, "m5_gemv"); if (!G012) return 1;
    GR012 = (gemv_range_fn)dlsym(h, "m5_gemv_range");   // 可选 (老 kern6 没有)
    h = dlopen(p3, RTLD_NOW | RTLD_LOCAL);   if (!h) return 2; G3   = (gemv_fn)dlsym(h, "m5_gemv"); if (!G3) return 2;
    GR3  = (gemv_range_fn)dlsym(h, "m5_gemv_range");
    h = dlopen(p4, RTLD_NOW | RTLD_LOCAL);   if (!h) return 3; G4   = (gemv_fn)dlsym(h, "m5_gemv"); if (!G4) return 3;
    GR4  = (gemv_range_fn)dlsym(h, "m5_gemv_range");
    h = dlopen(p56, RTLD_NOW | RTLD_LOCAL);  if (!h) return 4; G56  = (gemv_fn)dlsym(h, "m5_gemv"); if (!G56) return 4;
    GR56 = (gemv_range_fn)dlsym(h, "m5_gemv_range");
    h = dlopen(ps, RTLD_NOW | RTLD_LOCAL);   if (!h) return 5; GS   = (gemv_fn)dlsym(h, "m5_gemv"); if (!GS) return 5;
    GRF  = (gemv_range_fn)dlsym(h, "m5_gemv_range");
    return 0;
}
static inline void gemv_any(int code, const float* x, const uint8_t* buf, int n_out, int n_in, float* y) {
    m6_audit_rec(code, x, buf, n_out, n_in, 0, n_out, 0);
    if (code == 3) G3(code, x, buf, n_out, n_in, y);
    else if (code == 4) G4(code, x, buf, n_out, n_in, y);
    else if (code == 5 || code == 6) G56(code, x, buf, n_out, n_in, y);
    else if (code == 7) GS(code, x, buf, n_out, n_in, y);
    else if (code == 8 && G50) G50(code, x, buf, n_out, n_in, y);
    else if (code == 9) {
        // ★ F32: 原来是**逐值标量、且完全不开并行**。
        // ZAYA 的 MoE 路由 MLP 全是 F32 (down_proj [2048→256] 每层 2.1MB, 40 层共 84MB/token),
        // 微基准 (m5/m5_f32bench.c) 实测该形状 标量串行 325us vs AVX512 并行 15.2us = 21.4x。
        // 4 个独立累加器避免 FMA 依赖链; if(!omp_in_parallel) 保证被嵌在并行区里调用时仍正确
        // (GOMP 遇嵌套并行会退化为单线程, 结果不变)。
        const float* w = (const float*)buf;
        #pragma omp parallel for schedule(static) if(!omp_in_parallel())
        for (int r = 0; r < n_out; r++) {
            const float* wr = w + (size_t)r * n_in;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            int i = 0;
            for (; i + 63 < n_in; i += 64) {
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + i),      _mm512_loadu_ps(x + i),      a0);
                a1 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + i + 16), _mm512_loadu_ps(x + i + 16), a1);
                a2 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + i + 32), _mm512_loadu_ps(x + i + 32), a2);
                a3 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + i + 48), _mm512_loadu_ps(x + i + 48), a3);
            }
            for (; i + 15 < n_in; i += 16)
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + i), _mm512_loadu_ps(x + i), a0);
            float s = _mm512_reduce_add_ps(
                _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
            for (; i < n_in; i++) s += wr[i] * x[i];
            y[r] = s;
        }
    }
    else if (code == 2 && G12) G12(g_iq3_mode, x, buf, n_out, n_in, y);
    else G012(code, x, buf, n_out, n_in, y);
}
static inline float h2f1(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { float v = ((float)man) * 5.9604644775390625e-8f; uint32_t vi; memcpy(&vi,&v,4); f = sign|vi; }
    } else if (exp == 31) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 112u) << 23) | (man << 13);
    float o; memcpy(&o, &f, 4); return o;
}

typedef struct {
    const float* attn_norm;        // [2048]
    const uint8_t* qkv_buf;  int qkv_code;      // out n_qkv
    const uint8_t* gate_buf; int gate_code;     // out n_vh*128
    const float* beta_wt;          // [2048*n_vh] 预转置 (x @ W_T)
    const float* alpha_wt;         // [2048*n_vh]
    const float* ssm_a;            // [n_vh]
    const float* ssm_dt;           // [n_vh]
    const float* conv;             // [n_qkv*4]
    const float* ssm_norm;         // [128]
    const uint8_t* ssm_out_buf; int ssm_out_code;  // out 2048, in d_in
    const float* post_norm;        // [2048]
    int n_qkv; int n_vh; int d_in;              // qwen3.5 家族维度 (2B: 6144/16/2048)
} GdnW;

// h1[2048] → x2n[2048]; ssm[32*128*128] 与 tail[8192*3] 原地更新
// buf: 调用方提供的 scratch ≥ 28672 floats
void m6_gdn_attn(const float* h1, const GdnW* w, float* ssm, float* tail,
                 float* buf, float* x2n, float* x2_out) {
    const int NQKV = w->n_qkv, NVH = w->n_vh, DIN = w->d_in;
    float *xn = buf;                    // 2048
    float *qkv = buf + 2048;            // n_qkv
    float *z = buf + 2048 + NQKV;       // n_vh*128
    float *beta = z + NVH*SV;           // n_vh
    float *alpha = beta + NVH;          // n_vh
    float *cout = alpha + NVH;          // n_qkv
    float *o = cout + NQKV;             // n_vh*128
    float ms = 0.f;
    for (int i = 0; i < 2048; i++) ms += h1[i] * h1[i];
    float inv = 1.f / sqrtf(ms / 2048.f + EPS);
    for (int i = 0; i < 2048; i++) xn[i] = h1[i] * inv * w->attn_norm[i];
    // qkv + gate：两个大 gemv 之前各自单线程跑（每层 ~20MB 只有 1 核在读）。
    // 行分段并行（与 Ling MLA 同款修法）：每行点积仍由同一内核按原序算 ⇒ 逐位不变。
    const int NS = omp_get_max_threads();
    if (g_gdn_seg) {
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS; s++)
                gemv_range_any(w->qkv_code, xn, w->qkv_buf, NQKV, 2048, qkv,
                               s * NQKV / NS, (s + 1) * NQKV / NS);
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS; s++)
                gemv_range_any(w->gate_code, xn, w->gate_buf, NVH*SV, 2048, z,
                               s * (NVH*SV) / NS, (s + 1) * (NVH*SV) / NS);
        }
    } else {
        gemv_any(w->qkv_code, xn, w->qkv_buf, NQKV, 2048, qkv);
        gemv_any(w->gate_code, xn, w->gate_buf, NVH*SV, 2048, z);
    }
    if (buf[39000] != 0.f) memcpy(buf + 30000, qkv, sizeof(float) * NQKV);   // 调试: buf[39000]=哨兵
    {   // beta/alpha = [32,2048]×x 的两个 gemv。旧写法按列主序逐头点积 (128B 步长, 串行);
        // 改为 t 外层顺序扫描 (内存布局本来就是 t*NVH+i), 内层 i 向量化
        float bacc[64] = {0.f}, aacc[64] = {0.f};
        for (int t = 0; t < 2048; t++) {
            const float xt = xn[t];
            const float* bw = w->beta_wt + (size_t)t*NVH;
            const float* aw = w->alpha_wt + (size_t)t*NVH;
            for (int i = 0; i < NVH; i++) { bacc[i] += bw[i] * xt; aacc[i] += aw[i] * xt; }
        }
        for (int i = 0; i < NVH; i++) {
            beta[i] = 1.f / (1.f + expf(-bacc[i]));
            float av = aacc[i] + w->ssm_dt[i];
            float sp = av > 0 ? av + log1pf(expf(-av)) : log1pf(expf(av));
            alpha[i] = sp * w->ssm_a[i];
        }
    }
    if (buf[39000] != 0.f) {   // 调试: conv 输入快照
        for (int i = 0; i < 8; i++) buf[38200+i] = tail[i];
        for (int i = 0; i < 8; i++) buf[38208+i] = qkv[i];
        for (int i = 0; i < 8; i++) buf[38216+i] = w->conv[i];
    }
    // depthwise conv + silu; cin = [tail(3) | qkv(1)]
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NQKV; i++) {
        const float* wk = w->conv + i * 4;
        float acc = wk[0] * tail[i*3] + wk[1] * tail[i*3+1] + wk[2] * tail[i*3+2] + wk[3] * qkv[i];
        cout[i] = acc / (1.f + expf(-acc));
        tail[i*3] = tail[i*3+1];
        tail[i*3+1] = tail[i*3+2];
        tail[i*3+2] = qkv[i];
    }
    float* q = cout;             // [N_KH][128]
    float* k = cout + 2048;      // [N_KH][128]
    float* v = cout + 4096;      // [n_vh][128]
    #pragma omp parallel for schedule(static)
    for (int hh = 0; hh < N_KH; hh++) {
        float nq = 0.f, nk = 0.f;
        for (int t = 0; t < SV; t++) { nq += q[hh*SV+t]*q[hh*SV+t]; nk += k[hh*SV+t]*k[hh*SV+t]; }
        nq = 1.f / fmaxf(sqrtf(nq), EPS); nk = 1.f / fmaxf(sqrtf(nk), EPS);
        for (int t = 0; t < SV; t++) { q[hh*SV+t] *= nq; k[hh*SV+t] *= nk; }
    }
    // GDN 朴素 AR 递推 (32 v头; k/q 头 = h%16)
    const float scale = 1.f / sqrtf((float)SV);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NVH; h++) {
        const int j = h % 16;
        const float g = expf(alpha[h]);
        const float b = beta[h];
        float* S = ssm + h * SV * SV;
        const float* kh = k + j * SV;
        const float* qh = q + j * SV;
        const float* vh = v + h * SV;
        float sk[SV], oh[SV];
        for (int kk = 0; kk < SV; kk++) {           // 1) 衰减 (行连续)
            float* row = S + kk * SV;
            for (int vv = 0; vv < SV; vv++) row[vv] *= g;
        }
        for (int vv = 0; vv < SV; vv++) sk[vv] = 0.f;
        for (int kk = 0; kk < SV; kk++) {           // 2) sk += row_k · kh[k] (行连续累加)
            const float* row = S + kk * SV;
            const float kv = kh[kk];
            for (int vv = 0; vv < SV; vv++) sk[vv] += row[vv] * kv;
        }
        for (int vv = 0; vv < SV; vv++) oh[vv] = (vh[vv] - sk[vv]) * b;   // 3) delta
        for (int kk = 0; kk < SV; kk++) {           // 4) row += kh[k]·delta (行连续)
            float* row = S + kk * SV;
            const float kv = kh[kk];
            for (int vv = 0; vv < SV; vv++) row[vv] += kv * oh[vv];
        }
        for (int vv = 0; vv < SV; vv++) sk[vv] = 0.f;
        for (int kk = 0; kk < SV; kk++) {           // 5) o += row_k · qh[k] (行连续)
            const float* row = S + kk * SV;
            const float qv = qh[kk];
            for (int vv = 0; vv < SV; vv++) sk[vv] += row[vv] * qv;
        }
        for (int vv = 0; vv < SV; vv++) o[h * SV + vv] = sk[vv] * scale;
    }
    // 门控 rms: ssm_norm 按 v 广播；rms 完成后（区内隐式栅栏）同区并行做 ssm_out gemv
    float* on = buf;
    float* attn = buf + 4096;   // qkv 废弃区 (与 on 不重叠)
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int h = 0; h < NVH; h++) {
            float ms2 = 0.f;
            for (int vv = 0; vv < SV; vv++) ms2 += o[h*SV+vv] * o[h*SV+vv];
            float inv2 = 1.f / sqrtf(ms2 / SV + EPS);
            for (int vv = 0; vv < SV; vv++) {
                float zv = z[h*SV+vv];
                on[h*SV+vv] = o[h*SV+vv] * inv2 * w->ssm_norm[vv] * (zv / (1.f + expf(-zv)));
            }
        }
        if (g_gdn_seg) {
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS; s++)
                gemv_range_any(w->ssm_out_code, on, w->ssm_out_buf, 2048, DIN, attn,
                               s * 2048 / NS, (s + 1) * 2048 / NS);
        } else {
            #pragma omp single
            gemv_any(w->ssm_out_code, on, w->ssm_out_buf, 2048, DIN, attn);
        }
    }
    ms = 0.f;   /* ★ 必须重置: ms 前面用于 xn 的 rms */
    {   // ★ 确定性求和：reduction(+:ms) 的累加顺序随调度漂移（实测同状态两次
        //   forced-logits 差 ~1.4e-5 的唯一来源），改成 static 分块 + 每线程部分和
        //   固定槽 + 区外按序相加 ⇒ 并行度不变、每次运行逐位一致。
        const int NT = omp_get_max_threads();
        float part[128];
        for (int i = 0; i < NT && i < 128; i++) part[i] = 0.f;
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float p = 0.f;
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < 2048; i++) {
                x2_out[i] = h1[i] + attn[i];
                p += x2_out[i] * x2_out[i];
            }
            if (tid < 128) part[tid] = p;
        }
        for (int i = 0; i < NT && i < 128; i++) ms += part[i];
    }
    inv = 1.f / sqrtf(ms / 2048.f + EPS);
    for (int i = 0; i < 2048; i++) x2n[i] = x2_out[i] * inv * w->post_norm[i];
}
// 调试: 纯 conv 对账
void m6_conv_dbg(const float* tail, const float* qkv, const float* conv, float* cout) {
    for (int i = 0; i < 8192; i++) {
        const float* wk = conv + i * 4;
        float acc = wk[0]*tail[i*3] + wk[1]*tail[i*3+1] + wk[2]*tail[i*3+2] + wk[3]*qkv[i];
        cout[i] = acc / (1.f + expf(-acc));
    }
}

// ===================== 全注意力层 (10 层) =====================
typedef struct {
    const float* attn_norm;      // [2048]
    const uint8_t* q_buf; int q_code;    // out nh*512
    const uint8_t* k_buf; int k_code;    // out 512
    const uint8_t* v_buf; int v_code;    // out 512
    const float* q_norm;         // [256]
    const float* k_norm;         // [256]
    const uint8_t* o_buf; int o_code;    // out 2048, in 2048
    const float* post_norm;      // [2048]
    const float* rope_inv;       // [32] base^(-2t/rot)
    int nh;                      // q 头数 (35B:16, 2B:8)
} FullW;

// h1[2048]; Kr/Vc: 旧缓存 (Told,512); pos0; 输出 x2n[2048], x2[2048], kr_new[512], v_new[512]
void m6_full_attn(const float* h1, const FullW* w,
                  const float* Kr, const float* Vc, int Told, int pos0,
                  float* buf, float* scores, float* x2n, float* x2_out,
                  float* kr_new, float* v_new) {
    const int HD = 256, NKV = 2, HALF = 32;
    const int NH = w->nh;
    const int NQG = NH * 2 * HD;             // q+gate
    float *xn = buf;              // 2048
    float *QG = buf + 2048;       // NQG
    float *Qr = buf + 2048 + NQG;  // NH*256
    float *attn_out = Qr + NH*HD;  // 2048
    float ms = 0.f;
    for (int i = 0; i < 2048; i++) ms += h1[i] * h1[i];
    float inv = 1.f / sqrtf(ms / 2048.f + EPS);
    for (int i = 0; i < 2048; i++) xn[i] = h1[i] * inv * w->attn_norm[i];
    const int NS2 = omp_get_max_threads();
    if (g_gdn_seg) {
        // q/k/v 三个大 gemv 单区行分段（同 GDN 修法，逐位不变）
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS2; s++)
                gemv_range_any(w->q_code, xn, w->q_buf, NQG, 2048, QG,
                               s * NQG / NS2, (s + 1) * NQG / NS2);
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS2; s++)
                gemv_range_any(w->k_code, xn, w->k_buf, 512, 2048, kr_new,
                               s * 512 / NS2, (s + 1) * 512 / NS2);
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS2; s++)
                gemv_range_any(w->v_code, xn, w->v_buf, 512, 2048, v_new,
                               s * 512 / NS2, (s + 1) * 512 / NS2);
        }
    } else {
        gemv_any(w->q_code, xn, w->q_buf, NQG, 2048, QG);
        gemv_any(w->k_code, xn, w->k_buf, 512, 2048, kr_new);
        gemv_any(w->v_code, xn, w->v_buf, 512, 2048, v_new);
    }
    // K: 每头 RMS norm + rope (写回 kr_new)
    for (int kv = 0; kv < NKV; kv++) {
        float* kh = kr_new + kv * HD;
        float n2 = 0.f;
        for (int t = 0; t < HD; t++) n2 += kh[t] * kh[t];
        n2 = 1.f / sqrtf(n2 / HD + EPS);
        for (int t = 0; t < HD; t++) kh[t] *= n2 * w->k_norm[t];
        if (buf[39000] != 0.f) memcpy(buf + 30000 + kv * HD, kh, sizeof(float) * HD);   // 调试: rope 前
        for (int t = 0; t < HALF; t++) {
            float ang = (float)pos0 * w->rope_inv[t];
            float c = cosf(ang), s = sinf(ang);
            float a = kh[t], b = kh[t + HALF];
            kh[t] = a * c - b * s; kh[t + HALF] = a * s + b * c;
        }
        if (buf[39000] != 0.f) { for (int t = 0; t < 32; t++) buf[30800 + kv*32 + t] = w->rope_inv[t]; }
    }
    // Q: 每头取 QG[h*512 .. +256], RMS norm + rope → Qr[h*256]; gate 留在 QG
    for (int h = 0; h < NH; h++) {
        const float* qh = QG + h * 512;
        float* qo = Qr + h * HD;
        float n2 = 0.f;
        for (int t = 0; t < HD; t++) n2 += qh[t] * qh[t];
        n2 = 1.f / sqrtf(n2 / HD + EPS);
        for (int t = 0; t < HD; t++) qo[t] = qh[t] * n2 * w->q_norm[t];
        for (int t = 0; t < HALF; t++) {
            float ang = (float)pos0 * w->rope_inv[t];
            float c = cosf(ang), s = sinf(ang);
            float a = qo[t], b = qo[t + HALF];
            qo[t] = a * c - b * s; qo[t + HALF] = a * s + b * c;
        }
    }
    // attention: 2 KV 头 × 8 Q 头; K/V 序列 = 旧缓存 Told 行 + 新行
    const int T = Told + 1;
    const float scale = 1.f / sqrtf((float)HD);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) {
        const int kv = h / (NH / NKV);   // 35B: 16/2=8; 2B: 8/2=4 (原 h>>3 硬编码 35B)
        const float* qh = Qr + h * HD;
        float* sc = scores + h * T;
        for (int t = 0; t < Told; t++) {
            const float* krow = Kr + (size_t)t * 512 + kv * HD;
            float acc = 0.f;
            for (int d = 0; d < HD; d++) acc += qh[d] * krow[d];
            sc[t] = acc * scale;
        }
        {
            const float* krow = kr_new + kv * HD;
            float acc = 0.f;
            for (int d = 0; d < HD; d++) acc += qh[d] * krow[d];
            sc[Told] = acc * scale;
        }
        float mx = sc[0];
        for (int t = 1; t < T; t++) if (sc[t] > mx) mx = sc[t];
        float sum = 0.f;
        for (int t = 0; t < T; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float rsum = 1.f / sum;
        float* oh = attn_out + h * HD;
        for (int d = 0; d < HD; d++) oh[d] = 0.f;
        for (int t = 0; t < Told; t++) {
            const float p = sc[t] * rsum;
            const float* vrow = Vc + (size_t)t * 512 + kv * HD;
            for (int d = 0; d < HD; d++) oh[d] += p * vrow[d];
        }
        {
            const float p = sc[Told] * rsum;
            const float* vrow = v_new + kv * HD;
            for (int d = 0; d < HD; d++) oh[d] += p * vrow[d];
        }
    }
    // gate: a *= sigmoid(gate)
    for (int h = 0; h < NH; h++) {
        float* oh = attn_out + h * HD;
        const float* gh = QG + h * 512 + HD;
        for (int d = 0; d < HD; d++) oh[d] *= 1.f / (1.f + expf(-gh[d]));
    }
    float* attn = buf + 2048;    // 复用 QG 区
    if (g_gdn_seg) {
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < NS2; s++)
                gemv_range_any(w->o_code, attn_out, w->o_buf, 2048, NH*HD, attn,
                               s * 2048 / NS2, (s + 1) * 2048 / NS2);
        }
    } else {
        gemv_any(w->o_code, attn_out, w->o_buf, 2048, NH*HD, attn);   // 原 4096 = 35B 的 NH*HD
    }
    ms = 0.f;
    for (int i = 0; i < 2048; i++) { x2_out[i] = h1[i] + attn[i]; ms += x2_out[i] * x2_out[i]; }
    inv = 1.f / sqrtf(ms / 2048.f + EPS);
    for (int i = 0; i < 2048; i++) x2n[i] = x2_out[i] * inv * w->post_norm[i];
}

// ===================== MoE 路由专家批量 =====================
// ptrs: 3*n_exp 个专家字节指针 (gate, up, down 交替); codes: 3*n_exp 格式码; wtop: n_exp 权重
void m6_moe_batch(const float* x0, const uint64_t* ptrs, const int* codes,
                  const float* wtop, int n_exp, float* out) {
    float y512[512], y512b[512], act[512], y2048[2048];
    for (int i = 0; i < 2048; i++) out[i] = 0.f;
    for (int j = 0; j < n_exp; j++) {
        const uint8_t* g = (const uint8_t*)(uintptr_t)ptrs[3*j + 0];
        const uint8_t* u = (const uint8_t*)(uintptr_t)ptrs[3*j + 1];
        const uint8_t* d = (const uint8_t*)(uintptr_t)ptrs[3*j + 2];
        gemv_any(codes[3*j + 0], x0, g, 512, 2048, y512);
        gemv_any(codes[3*j + 1], x0, u, 512, 2048, y512b);
        for (int i = 0; i < 512; i++) act[i] = (y512[i] / (1.f + expf(-y512[i]))) * y512b[i];
        gemv_any(codes[3*j + 2], act, d, 2048, 512, y2048);
        const float w = wtop[j];
        for (int i = 0; i < 2048; i++) out[i] += w * y2048[i];
    }
}

// ===================== MoE 批量 v2: 单区域版 =====================
// n_exp 个路由专家 + 第 (n_exp) 块 = 共享专家, 3 个并行区域搞定整层 FFN:
//   区域1: 全部 gate+up gemv (行区间, 无嵌套 OMP)
//   区域2: act = silu(g)*u
//   区域3: 全部 down gemv
// 末尾: out = Σ wtop[j]*y_j + shgate*y_sh;  shgate = sigmoid(x0·sh_gate_inp)
// scratch: 调用方提供 ≥ (n_exp+1)*(512*2+2048) floats
// 返回 0 成功; -1 = GR012 缺失且含非 0/1/2 码
void m6_moe_batch2(const float* x0, const uint64_t* ptrs, const int* codes,
                   const float* wtop, int n_exp,
                   const uint64_t* sh_ptrs, const int* sh_codes, const float* sh_gate_inp,
                   float* scratch, float* out) {
    const int NE = n_exp + 1;                 // 含 shexp
    const uint64_t* P = ptrs;                 // 3*n_exp
    const uint64_t* SP = sh_ptrs;             // 3 (gate, up, down)
    float* gu  = scratch;                     // NE*512 gate 输出
    float* uu  = gu + NE*512;                 // NE*512 up 输出
    float* act = uu + NE*512;                 // NE*512
    float* yd  = act + NE*512;                // NE*2048
    const int nc = GR012 != NULL;             // range 可用
    // 区域1: gate+up
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < NE; j++) {
        const uint8_t* g = (const uint8_t*)(uintptr_t)(j < n_exp ? P[3*j+0] : SP[0]);
        const uint8_t* u = (const uint8_t*)(uintptr_t)(j < n_exp ? P[3*j+1] : SP[1]);
        const int cg = j < n_exp ? codes[3*j+0] : sh_codes[0];
        const int cu = j < n_exp ? codes[3*j+1] : sh_codes[1];
        if (nc && (cg == 0 || cg == 1 || cg == 2)) {
            GR012(cg, x0, g, 512, 2048, gu + (size_t)j*512, 0, 512);
            GR012(cu, x0, u, 512, 2048, uu + (size_t)j*512, 0, 512);
        } else {                              // 罕见: 嵌套区域由 GOMP 退化为单线程, 仍正确
            gemv_any(cg, x0, g, 512, 2048, gu + (size_t)j*512);
            gemv_any(cu, x0, u, 512, 2048, uu + (size_t)j*512);
        }
    }
    // 区域2: act = silu(g)*u
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NE*512; i++) {
        const float gv = gu[i];
        act[i] = (gv / (1.f + expf(-gv))) * uu[i];
    }
    // 区域3: down
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < NE; j++) {
        const uint8_t* d = (const uint8_t*)(uintptr_t)(j < n_exp ? P[3*j+2] : SP[2]);
        const int cd = j < n_exp ? codes[3*j+2] : sh_codes[2];
        if (nc && (cd == 0 || cd == 1 || cd == 2)) {
            GR012(cd, act + (size_t)j*512, d, 2048, 512, yd + (size_t)j*2048, 0, 2048);
        } else {
            gemv_any(cd, act + (size_t)j*512, d, 2048, 512, yd + (size_t)j*2048);
        }
    }
    // 汇总 (串行, ~20K 加法)
    float shg = 0.f;
    if (sh_gate_inp) {
        for (int i = 0; i < 2048; i++) shg += x0[i] * sh_gate_inp[i];
        shg = 1.f / (1.f + expf(-shg));
    }
    for (int i = 0; i < 2048; i++) {
        float s = 0.f;
        for (int j = 0; j < n_exp; j++) s += wtop[j] * yd[(size_t)j*2048 + i];
        out[i] = s + shg * yd[(size_t)n_exp*2048 + i];
    }
}

// ===================== MoE 批量 v3: 形状参数化 (任意 hidden/inter) =====================
// gate/up: (inter, n_in); down: (n_out, inter)。scratch 内部 malloc。
void m6_moe_batch3(const float* x0, const uint64_t* ptrs, const int* codes,
                   const float* wtop, int n_exp,
                   int inter, int n_in, int n_out, float* out) {
    float* gu  = (float*)malloc(sizeof(float) * (size_t)n_exp * inter);
    float* uu  = (float*)malloc(sizeof(float) * (size_t)n_exp * inter);
    float* act = (float*)malloc(sizeof(float) * (size_t)n_exp * inter);
    float* yd  = (float*)malloc(sizeof(float) * (size_t)n_exp * n_out);
    if (!gu || !uu || !act || !yd) { free(gu); free(uu); free(act); free(yd); return; }
    for (int j = 0; j < n_exp; j++) {
        gemv_any(codes[3*j + 0], x0, (const uint8_t*)(uintptr_t)ptrs[3*j + 0], inter, n_in, gu + (size_t)j*inter);
        gemv_any(codes[3*j + 1], x0, (const uint8_t*)(uintptr_t)ptrs[3*j + 1], inter, n_in, uu + (size_t)j*inter);
    }
    for (int i = 0; i < n_exp*inter; i++) {
        float gv = gu[i];
        act[i] = (gv / (1.f + expf(-gv))) * uu[i];
    }
    for (int j = 0; j < n_exp; j++) {
        gemv_any(codes[3*j + 2], act + (size_t)j*inter, (const uint8_t*)(uintptr_t)ptrs[3*j + 2], n_out, inter, yd + (size_t)j*n_out);
    }
    for (int i = 0; i < n_out; i++) out[i] = 0.f;
    for (int j = 0; j < n_exp; j++) {
        const float w = wtop[j];
        for (int i = 0; i < n_out; i++) out[i] += w * yd[(size_t)j*n_out + i];
    }
    free(gu); free(uu); free(act); free(yd);
}



// ===================== MoE 批量 v4: 形状参数化 + 3 个胖 OMP 区域 =====================
// 与 batch2 同思路但 inter/n_in/n_out 全参数化; 内核走 m5_gemv_range (无内嵌 OMP),
// 所以"每专家 3 次 gemv"退化成一个并行区域内的顺序调用, 区域数从 3*n_exp 降到 3。
// scratch: 调用方提供 >= (3*inter + n_out) * n_exp floats
static inline void gemv_range_any(int code, const float* x, const uint8_t* buf,
                                  int n_out, int n_in, float* y, int o0, int o1) {
    m6_audit_rec(code, x, buf, n_out, n_in, o0, o1, 1);
    gemv_range_fn R = NULL;
    if (code == 2 && GR12) { GR12(g_iq3_mode, x, buf, n_out, n_in, y, o0, o1); return; }
    if (code == 0 || code == 1 || code == 2) R = GR012;
    else if (code == 3) R = GR3;
    else if (code == 4) R = GR4;
    else if (code == 5 || code == 6) R = GR56;
    else if (code == 7) R = GRF;
    else if (code == 8) R = GR50;
    if (R) { R(code, x, buf, n_out, n_in, y, o0, o1); return; }
    // 该格式的库没有 range 导出 → 回落到带 OMP 的整调用
    // (在并行区域内嵌套并行会被 GOMP 退化为单线程, 结果仍正确)
    gemv_any(code, x, buf, n_out, n_in, y);
}

void m6_moe_batch4(const float* x0, const uint64_t* ptrs, const int* codes,
                   const float* wtop, int n_exp,
                   int inter, int n_in, int n_out, float* scratch, float* out) {
    float* gu  = scratch;                       // n_exp*inter
    float* uu  = gu + (size_t)n_exp*inter;
    float* act = uu + (size_t)n_exp*inter;
    float* yd  = act + (size_t)n_exp*inter;
    // 区域1: 全部 gate+up
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n_exp; j++) {
        gemv_range_any(codes[3*j + 0], x0, (const uint8_t*)(uintptr_t)ptrs[3*j + 0],
                       inter, n_in, gu + (size_t)j*inter, 0, inter);
        gemv_range_any(codes[3*j + 1], x0, (const uint8_t*)(uintptr_t)ptrs[3*j + 1],
                       inter, n_in, uu + (size_t)j*inter, 0, inter);
    }
    // 区域2: act = silu(gate)*up
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_exp*inter; i++) {
        const float gv = gu[i];
        act[i] = (gv / (1.f + expf(-gv))) * uu[i];
    }
    // 区域3: 全部 down
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n_exp; j++) {
        gemv_range_any(codes[3*j + 2], act + (size_t)j*inter, (const uint8_t*)(uintptr_t)ptrs[3*j + 2],
                       n_out, inter, yd + (size_t)j*n_out, 0, n_out);
    }
    // 汇总
    for (int i = 0; i < n_out; i++) out[i] = 0.f;
    for (int j = 0; j < n_exp; j++) {
        const float w = wtop[j];
        const float* yj = yd + (size_t)j*n_out;
        for (int i = 0; i < n_out; i++) out[i] += w * yj[i];
    }
}

static inline int topk_idx(const float* v, int n, int k, int* out);   // 定义在本文件后面

// ===================== MoE 专家合并 (路由无关的共用段) =====================
// n_used 个专家各跑 gate/up→silu→down 再按 wtop 加权求和。**一个 OMP 区**（原来 3 个区）。
// 抽成共用函数的原因：Bailing（sigmoid 门控 + probs_b 偏置 + noaux_tc 分组）与
// granite（softmax 门控、无偏置无分组）只差路由前端，这一段完全同构 —— 复制两份以后会改漏。
// scratch 需要 >= 3*n_used*inter + n_used*n_out（★ 是 n_out 不是 inter：yd 每专家一份完整输出）。
static void moe_experts(const float* x0, const uint64_t* p3, const int* c3, const float* wtop,
                        int n_used, int inter, int n_in, int n_out, float* scratch, float* out) {
    float* gu  = scratch;
    float* uu  = gu + (size_t)n_used*inter;
    float* act = uu + (size_t)n_used*inter;
    float* yd  = act + (size_t)n_used*inter;
    double _m0 = _now();
    const int ch = g_moe_chunk;    // ≤0 = 旧路径
    #pragma omp parallel
    {
        if (ch > 0) {
            // 行分块：扁平索引 → (专家, 行区间)。range 入口的行号是**绝对行号**，
            // y 基址不动。块宽用除法从 i 反解，免任务表。
            const int cg = (inter + ch - 1) / ch;      // gate/up 每矩阵块数
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < 2*n_used*cg; i++) {
                const int j = i / cg, r = (i % cg) * ch;
                const int r1 = r + ch < inter ? r + ch : inter;
                const int e = j >> 1;
                if (j & 1) gemv_range_any(c3[3*e+1], x0, (const uint8_t*)(uintptr_t)p3[3*e+1],
                                          inter, n_in, uu + (size_t)e*inter, r, r1);
                else       gemv_range_any(c3[3*e+0], x0, (const uint8_t*)(uintptr_t)p3[3*e+0],
                                          inter, n_in, gu + (size_t)e*inter, r, r1);
            }
        } else {
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < n_used; j++) {
                gemv_range_any(c3[3*j+0], x0, (const uint8_t*)(uintptr_t)p3[3*j+0], inter, n_in, gu + (size_t)j*inter, 0, inter);
                gemv_range_any(c3[3*j+1], x0, (const uint8_t*)(uintptr_t)p3[3*j+1], inter, n_in, uu + (size_t)j*inter, 0, inter);
            }
        }
        #pragma omp single
        { double _t = _now(); m6_prof_t[18] += _t - _m0; m6_prof_n[18]++; _m0 = _t; }
        #pragma omp for schedule(static)
        for (int i = 0; i < n_used*inter; i++) {
            const float gv = gu[i];
            act[i] = (gv / (1.f + expf(-gv))) * uu[i];
        }
        #pragma omp single
        { double _t = _now(); m6_prof_t[19] += _t - _m0; m6_prof_n[19]++; _m0 = _t; }
        if (ch > 0) {
            const int cd = (n_out + ch - 1) / ch;
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < n_used*cd; i++) {
                const int j = i / cd, r = (i % cd) * ch;
                const int r1 = r + ch < n_out ? r + ch : n_out;
                gemv_range_any(c3[3*j+2], act + (size_t)j*inter, (const uint8_t*)(uintptr_t)p3[3*j+2],
                               n_out, inter, yd + (size_t)j*n_out, r, r1);
            }
        } else {
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < n_used; j++)
                gemv_range_any(c3[3*j+2], act + (size_t)j*inter, (const uint8_t*)(uintptr_t)p3[3*j+2], n_out, inter, yd + (size_t)j*n_out, 0, n_out);
        }
        #pragma omp single
        { double _t = _now(); m6_prof_t[20] += _t - _m0; m6_prof_n[20]++; _m0 = _t; }   // down 相位
        #pragma omp for schedule(static)
        for (int i = 0; i < n_out; i++) {
            float a = 0.f;
            for (int j = 0; j < n_used; j++) a += wtop[j] * yd[(size_t)j*n_out + i];
            out[i] = a;
        }
        #pragma omp single
        { double _t = _now(); m6_prof_t[21] += _t - _m0; m6_prof_n[21]++; }             // 加权求和
    }
}

// ===================== granite-hybrid MoE: softmax 路由 + 同一个合并段 =====================
// 与 Bailing 的差别只有路由：**softmax 过全部 n_exp**（不是 sigmoid）→ top-n_used
// → 权重 = probs[选中] 后归一（和 clamp 到 f16 最小正数，与 ggml build_moe_ffn 的 norm_w 一致）
// → **没有 w_scale**（granite 的 GGUF 没有 expert_weights_scale 键 ⇒ w_scale=0 ⇒ 不缩放）。
// 语义来源：llama-graph.cpp build_moe_ffn + granite-hybrid.cpp 的调用参数（SOFTMAX/norm_w=true）。
void m6_granite_moe(const float* x0, const uint64_t* ptrs, const int* codes,
                    const float* gate_inp, int n_exp, int n_used,
                    int inter, int n_in, int n_out, float* scratch, float* out) {
    float lg[512];
    double _r0 = _now();
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < n_exp; e++) {
        const float* w = gate_inp + (size_t)e * n_in;
        float acc = 0.f;
        for (int i = 0; i < n_in; i++) acc += x0[i] * w[i];
        lg[e] = acc;
    }
    // softmax（减最大值，与 ggml_soft_max 一致）
    float mx = -1e30f;
    for (int e = 0; e < n_exp; e++) if (lg[e] > mx) mx = lg[e];
    float s = 0.f;
    for (int e = 0; e < n_exp; e++) { lg[e] = expf(lg[e] - mx); s += lg[e]; }
    const float inv = 1.f / s;
    for (int e = 0; e < n_exp; e++) lg[e] *= inv;
    { double _t = _now(); m6_prof_t[22] += _t - _r0; m6_prof_n[22]++; _r0 = _t; }  // 路由点积+softmax
    int idx[512];
    topk_idx(lg, n_exp, n_used, idx);
    float wtop[512], ws = 0.f;
    for (int j = 0; j < n_used; j++) { wtop[j] = lg[idx[j]]; ws += wtop[j]; }
    if (ws < 6.103515625e-5f) ws = 6.103515625e-5f;
    for (int j = 0; j < n_used; j++) wtop[j] /= ws;
    uint64_t p3[192]; int c3[192];
    for (int j = 0; j < n_used; j++)
        for (int t = 0; t < 3; t++) { p3[3*j+t] = ptrs[3*idx[j]+t]; c3[3*j+t] = codes[3*idx[j]+t]; }
    moe_experts(x0, p3, c3, wtop, n_used, inter, n_in, n_out, scratch, out);
    { double _t = _now(); m6_prof_t[23] += _t - _r0; m6_prof_n[23]++; }           // 专家合并
}

// ===================== Ling (bailingmoe3) MoE: 路由 + 单区域三相位 =====================
// 路由 = sigmoid(+exp_probs_b 偏置) → noaux_tc 分组 top-k → 权重归一×scale
// 三相位 (gate+up / act / down) 放在同一个 parallel 区内, 区域数从 3 降到 1。
// gate_inp: [NEXP][H] (已转置好); scratch >= (3*inter + n_out)*n_exp
static inline int topk_idx(const float* v, int n, int k, int* out) {
    // 稳定的 top-k (并列取小下标, 与 np.argsort(-v) 一致); k 很小 → 选择排序足够
    char used[512];
    for (int i = 0; i < n; i++) used[i] = 0;
    for (int t = 0; t < k; t++) {
        int best = -1; float bv = -1e30f;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            if (v[i] > bv) { bv = v[i]; best = i; }
        }
        out[t] = best; used[best] = 1;
    }
    return k;
}

void m6_bailing_moe(const float* x0, const uint64_t* ptrs, const int* codes,
                    const float* gate_inp, const float* probs_b,
                    int n_exp, int n_used, int n_group, int n_group_used,
                    int norm_w, float w_scale,
                    int inter, int n_in, int n_out,
                    float* scratch, float* out) {
    const int nexp_grp = n_exp / n_group;
    // ---- 路由 (并行: 128 个专家的点积互相独立) ----
    float sel[512], probs[512];
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < n_exp; e++) {
        const float* w = gate_inp + (size_t)e * n_in;
        float acc = 0.f;
        for (int i = 0; i < n_in; i++) acc += x0[i] * w[i];
        const float p = 1.f / (1.f + expf(-acc));
        probs[e] = p;
        sel[e] = p + (probs_b ? probs_b[e] : 0.f);
    }
    if (n_group > 1) {
        // 组得分 = 组内 top2 之和
        float gscore[64];
        for (int g = 0; g < n_group; g++) {
            float m1 = -1e30f, m2 = -1e30f;
            for (int i = 0; i < nexp_grp; i++) {
                const float v = sel[g*nexp_grp + i];
                if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
            }
            gscore[g] = m1 + m2;
        }
        int gt[64];
        topk_idx(gscore, n_group, n_group_used, gt);
        char keep[64] = {0};
        for (int i = 0; i < n_group_used; i++) keep[gt[i]] = 1;
        for (int g = 0; g < n_group; g++)
            if (!keep[g])
                for (int i = 0; i < nexp_grp; i++) sel[g*nexp_grp + i] = -1e30f;
    }
    int idx[64];
    topk_idx(sel, n_exp, n_used, idx);
    float wtop[64];
    float s = 0.f;
    for (int j = 0; j < n_used; j++) { wtop[j] = probs[idx[j]]; s += wtop[j]; }
    if (norm_w) { if (s < 6.103515625e-5f) s = 6.103515625e-5f; for (int j = 0; j < n_used; j++) wtop[j] /= s; }
    if (w_scale != 0.f && w_scale != 1.f) for (int j = 0; j < n_used; j++) wtop[j] *= w_scale;
    // 把选中的专家指针/码紧凑化到局部数组 (并行区内按 j 索引)
    uint64_t p3[192]; int c3[192];
    for (int j = 0; j < n_used; j++)
        for (int t = 0; t < 3; t++) { p3[3*j+t] = ptrs[3*idx[j]+t]; c3[3*j+t] = codes[3*idx[j]+t]; }
    // ---- 三相位: 单区域 ----
    moe_experts(x0, p3, c3, wtop, n_used, inter, n_in, n_out, scratch, out);
}

// ===================== Ling (bailingmoe3) KDA 层: 融合整层 =====================
// 顺序严格对齐 ling_proto.kda_step / ggml GATED_DELTA_NET 的 kda 分支:
//   3 分支 causal conv1d(silu) → l2n(q,k) → 逐通道门控 delta-net AR 递推
//   → 逐头 rms(o_norm) → ×sigmoid(out_gate) → wo
// scratch 需要 >= 6*DI + NH floats (q,k,v,f_a,og,beta 各 DI/NH)
void m6_bailing_kda(const float* x,
                    const uint8_t* wq, int cq, const uint8_t* wk, int ck, const uint8_t* wv, int cv,
                    const float* conv_w,                       // [3][DI][4]
                    const uint8_t* f_a, int cfa,               // [DI][H]
                    const float* dt_b,                         // [DI]
                    const float* ssm_a,                        // [NH]
                    const float* beta_w,                       // [NH][H]  (行=头)
                    const uint8_t* g_a, int cga,               // [DI][H]
                    const float* o_norm,                       // [HD]
                    const uint8_t* wo, int cwo,                // [H][DI]
                    float gate_lb, int NH, int HD, int H,
                    float* conv_state,                         // [3][DI][3]
                    float* S,                                  // [NH][HD][HD]
                    float* scratch, float* out) {
    const int DI = NH * HD;
    float* q  = scratch;
    float* k  = q + DI;
    float* v  = k + DI;
    float* fa = v + DI;         // f_a 输出 = 逐通道门值
    float* og = fa + DI;        // out_gate
    float* beta = og + DI;      // [NH]
    const int nseg = 4;

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int s = 0; s < nseg; s++) gemv_range_any(cq,  x, wq, DI, H, q,  s*DI/nseg, (s+1)*DI/nseg);
        #pragma omp for schedule(static) nowait
        for (int s = 0; s < nseg; s++) gemv_range_any(ck,  x, wk, DI, H, k,  s*DI/nseg, (s+1)*DI/nseg);
        #pragma omp for schedule(static) nowait
        for (int s = 0; s < nseg; s++) gemv_range_any(cv,  x, wv, DI, H, v,  s*DI/nseg, (s+1)*DI/nseg);
        #pragma omp for schedule(static) nowait
        for (int s = 0; s < nseg; s++) gemv_range_any(cfa, x, f_a, DI, H, fa, s*DI/nseg, (s+1)*DI/nseg);
        #pragma omp for schedule(static) nowait
        for (int s = 0; s < nseg; s++) gemv_range_any(cga, x, g_a, DI, H, og, s*DI/nseg, (s+1)*DI/nseg);
        #pragma omp for schedule(static)
        for (int h = 0; h < NH; h++) {          // beta: [NH] 输出, 每头一次 1536 长点积
            const float* bw = beta_w + (size_t)h * H;
            float acc = 0.f;
            for (int i = 0; i < H; i++) acc += x[i] * bw[i];
            beta[h] = acc;
        }
        // conv1d + silu  (三分支一次搞定; 每个 c 自身独立)
        #pragma omp for schedule(static)
        for (int t = 0; t < 3*DI; t++) {
            const int br = t / DI, c = t % DI;
            float* st = conv_state + ((size_t)br*DI + c) * 3;
            float* dst = (br == 0 ? q : br == 1 ? k : v) + c;
            const float* xs = (br == 0 ? q : br == 1 ? k : v) + c;
            const float proj = xs[0];            // 注意: 就地写会污染 proj, 先取
            const float* w4 = conv_w + ((size_t)br*DI + c) * 4;
            float acc = w4[0]*st[0] + w4[1]*st[1] + w4[2]*st[2] + w4[3]*proj;
            st[0] = st[1]; st[1] = st[2]; st[2] = proj;
            dst[0] = acc / (1.f + expf(-acc));   // silu
        }
        // l2n (每头)
        #pragma omp for schedule(static)
        for (int h = 0; h < NH; h++) {
            float nq = 0.f, nk = 0.f;
            for (int i = 0; i < HD; i++) { nq += q[h*HD+i]*q[h*HD+i]; nk += k[h*HD+i]*k[h*HD+i]; }
            nq = 1.f / fmaxf(sqrtf(nq), EPS); nk = 1.f / fmaxf(sqrtf(nk), EPS);
            for (int i = 0; i < HD; i++) { q[h*HD+i] *= nq; k[h*HD+i] *= nk; }
        }
        // delta-net AR 递推 (每头一个 128×128 状态)
        #pragma omp for schedule(static)
        for (int h = 0; h < NH; h++) {
            float* Sh = S + (size_t)h*HD*HD;
            float eg[256];
            for (int i = 0; i < HD; i++) {
                // 门 = sigmoid((f_a·x + dt_b) * ssm_a) * lower_bound  (dt_b 漏用过, 见 README 教训)
                const float gt = (fa[h*HD+i] + dt_b[h*HD+i]) * ssm_a[h];
                const float gi = 1.f / (1.f + expf(-gt)) * gate_lb;
                eg[i] = expf(gi);
            }
            for (int i = 0; i < HD; i++) {          // S[i][:] *= exp(g[i])
                float* row = Sh + (size_t)i*HD;
                const float e = eg[i];
                for (int j = 0; j < HD; j++) row[j] *= e;
            }
            const float bh = 1.f / (1.f + expf(-beta[h]));
            float kv[256], dl[256];
            for (int j = 0; j < HD; j++) {          // kv[j] = Σ_i S[i][j] k[i]
                float s = 0.f;
                for (int i = 0; i < HD; i++) s += Sh[(size_t)i*HD + j] * k[h*HD+i];
                kv[j] = s;
            }
            for (int j = 0; j < HD; j++) dl[j] = (v[h*HD+j] - kv[j]) * bh;
            for (int i = 0; i < HD; i++) {          // S[i][j] += k[i]*delta[j]
                float* row = Sh + (size_t)i*HD;
                const float ki = k[h*HD+i];
                for (int j = 0; j < HD; j++) row[j] += ki * dl[j];
            }
            float on = 0.f, oacc[256];
            for (int j = 0; j < HD; j++) {          // o[j] = Σ_i S[i][j] q[i]
                float s = 0.f;
                for (int i = 0; i < HD; i++) s += Sh[(size_t)i*HD + j] * q[h*HD+i];
                oacc[j] = s * (1.0f / sqrtf((float)HD));
                on += oacc[j]*oacc[j];
            }
            on = 1.f / sqrtf(on/HD + EPS);
            // 写回 q 缓冲 (o_norm 后 × sigmoid(out_gate))
            for (int j = 0; j < HD; j++) {
                const float ogt = 1.f / (1.f + expf(-og[h*HD+j]));
                q[h*HD+j] = oacc[j] * on * o_norm[j] * ogt;
            }
        }
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++) gemv_range_any(cwo, q, wo, H, DI, out, s*H/nseg, (s+1)*H/nseg);
    }
}

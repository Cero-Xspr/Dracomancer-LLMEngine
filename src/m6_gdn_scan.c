// m6_gdn_scan.c — GDN 两段式 prefill 的第二段（扫描）
// 前提：qkv/gate 已由 m5_gemm 对全部 T 个位置批量投影完。本算子逐 token 走
// conv→silu→q/k rms→delta rule→门控 rms，**只碰激活与 S/tail 状态，不读任何大权重**
// （这正是 llama.cpp ssm_scan 的设计）。数学与 m6_layer.c m6_gdn_attn 逐式对应。
// 编译: gcc -O3 -march=native -fopenmp -shared -fPIC m6_gdn_scan.c -o m6_gdn_scan.so
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>

#ifndef GDN_EPS
#define GDN_EPS 1e-6f   // 与 m6_layer.c #define EPS 一致
#endif

// QKV[T*n_qkv] Z[T*n_vh*128] B[T*n_vh](beta) G[T*n_vh](=exp(alpha))
// conv[n_qkv*4] ssm_norm[128] tail[n_qkv*3] S[n_vh*128*128]
// ON[T*n_vh*128] 门控 rms 输出
void m6_gdn_scan(const float* QKV, const float* Z, const float* B, const float* G,
                 const float* conv, const float* ssm_norm, float* tail,
                 float* S, int T, int n_qkv, int n_vh, float* ON, float* work) {
    const int SV = 128;
    const int NHQ = (n_qkv - n_vh * SV) / (2 * SV);   // q/k 头数（v = n_vh*SV，q+k 平分余下）
    const float scale = 1.f / sqrtf((float)SV);
    #pragma omp parallel
    {
        float* cout = work;   // ★ 共享暂存 [n_qkv]：相位间有栅栏，跨 t 复用安全
        for (int t = 0; t < T; t++) {
            const float* qkv = QKV + (size_t)t * n_qkv;
            // ① depthwise conv(4 tap) + silu + tail 滑动（逐通道独立）
            #pragma omp for schedule(static)
            for (int i = 0; i < n_qkv; i++) {
                const float* wk = conv + i * 4;
                float acc = wk[0] * tail[i*3] + wk[1] * tail[i*3+1] + wk[2] * tail[i*3+2] + wk[3] * qkv[i];
                cout[i] = acc / (1.f + expf(-acc));
                tail[i*3] = tail[i*3+1];
                tail[i*3+1] = tail[i*3+2];
                tail[i*3+2] = qkv[i];
            }
            // ② q/k 每头 RMS（SUM 版，fmax(sqrt, EPS)——与 m6_gdn_attn 同式）
            #pragma omp for schedule(static)
            for (int j = 0; j < 2*NHQ; j++) {
                float* h = cout + j * SV;
                float n2 = 0.f;
                for (int u = 0; u < SV; u++) n2 += h[u] * h[u];
                const float inv = 1.f / (sqrtf(n2) > GDN_EPS ? sqrtf(n2) : GDN_EPS);
                for (int u = 0; u < SV; u++) h[u] *= inv;
            }
            // ③ delta rule（32 v 头并行；k/q 头 = h%16）+ 门控 rms
            const float* z = Z + (size_t)t * n_vh * SV;
            #pragma omp for schedule(static)
            for (int h = 0; h < n_vh; h++) {
                const int j = h % NHQ;
                const float g = G[t * n_vh + h], b = B[t * n_vh + h];
                float* S_h = S + (size_t)h * SV * SV;
                const float* kj = cout + NHQ * SV + j * SV;       // k 头（q 在 0，k 在 +NHQ*SV——与 m6_gdn_attn 同布局）
                const float* qj = cout + j * SV;                  // q 头（已归一化）
                const float* vh = cout + 2 * NHQ * SV + h * SV;   // v
                float sk[128], oh[128];
                for (int kk = 0; kk < SV; kk++) {
                    float* row = S_h + (size_t)kk * SV;
                    for (int vv = 0; vv < SV; vv++) row[vv] *= g;
                }
                for (int vv = 0; vv < SV; vv++) sk[vv] = 0.f;
                for (int kk = 0; kk < SV; kk++) {
                    const float* row = S_h + (size_t)kk * SV;
                    const float kv = kj[kk];
                    for (int vv = 0; vv < SV; vv++) sk[vv] += row[vv] * kv;
                }
                for (int vv = 0; vv < SV; vv++) oh[vv] = (vh[vv] - sk[vv]) * b;
                for (int kk = 0; kk < SV; kk++) {
                    float* row = S_h + (size_t)kk * SV;
                    const float kv = kj[kk];
                    for (int vv = 0; vv < SV; vv++) row[vv] += kv * oh[vv];
                }
                for (int vv = 0; vv < SV; vv++) sk[vv] = 0.f;
                for (int kk = 0; kk < SV; kk++) {
                    const float* row = S_h + (size_t)kk * SV;
                    const float qv = qj[kk];
                    for (int vv = 0; vv < SV; vv++) sk[vv] += row[vv] * qv;
                }
                // ④ 门控 rms：on = o·inv·ssm_norm·silu(z)
                float ms2 = 0.f;
                for (int vv = 0; vv < SV; vv++) { oh[vv] = sk[vv] * scale; ms2 += oh[vv] * oh[vv]; }
                const float inv2 = 1.f / sqrtf(ms2 / SV + GDN_EPS);
                float* on = ON + (size_t)t * n_vh * SV + (size_t)h * SV;
                for (int vv = 0; vv < SV; vv++) {
                    const float zv = z[h * SV + vv];
                    on[vv] = oh[vv] * inv2 * ssm_norm[vv] * (zv / (1.f + expf(-zv)));
                }
            }
        }   // t —— 区内 for t：隐式栅栏保证相位次序，整段只 fork/join 一次
    }
}

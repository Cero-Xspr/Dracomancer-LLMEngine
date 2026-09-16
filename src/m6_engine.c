// m6_engine.c: 通用层引擎 —— 架构无关的 token 前向循环 + 算子接口
//
// 适配新架构只需两步:
//   ① C 侧实现该架构的 attn / ffn 算子 (签名固定):
//        void attn_op(const float* xn, void* p, int pos, float* out);
//        void ffn_op (const float* xn, void* p, float* out);
//      xn = 引擎已做过 rms_norm(attn_norm) 的输入; out = 子层输出 (残差由引擎加)
//   ② Python 侧按 GGUF 元数据填 M6LayerDesc 数组 (权重指针 / 算子 / 状态指针)
//
// 引擎固化的管线 (与 llama.cpp 多数 decoder 层同构):
//   x → rms(attn_norm) → attn → x+=attn → rms(ffn_norm) → ffn → x+=ffn
// 管线不同的架构 (如 granite 的残差/嵌入 scale) 把系数折进算子参数, 或另写 forward 循环。
//
// 编译: 与 m6_layer.c 合成一个编译单元 (unity build), 避免跨 .so 取静态符号。
#include "m6_layer.c"

// ===================== 通用层描述符 =====================
typedef void (*m6_attn_op)(const float* xn, void* p, int pos, float* out);
typedef void (*m6_ffn_op)(const float* xn, void* p, float* out);

typedef struct {
    const float* attn_norm;   // [H]
    const float* ffn_norm;    // [H]
    m6_attn_op   attn;
    void*        attn_p;
    m6_ffn_op    ffn;
    void*        ffn_p;
} M6LayerDesc;

// ggml 语义的 rms_norm: x / sqrt(mean(x²)+eps) * w
void m6_rms_norm(const float* x, const float* w, int h, float eps, float* out) {
    float ms = 0.f;
    for (int i = 0; i < h; i++) ms += x[i] * x[i];
    const float inv = 1.f / sqrtf(ms / (float)h + eps);
    if (w) for (int i = 0; i < h; i++) out[i] = x[i] * inv * w[i];
    else   for (int i = 0; i < h; i++) out[i] = x[i] * inv;
}

// 量化格式 → 每行字节 (n_in 个值占多少字节)。新增格式只改这一张表。
size_t m6_rowbytes(int code, int n_in) {
    switch (code) {
        case 0: return (size_t)(n_in / 32) * 34;    // Q8_0
        case 1: return (size_t)(n_in / 32) * 18;    // IQ4_NL
        case 2: return (size_t)(n_in / 256) * 110;  // IQ3_S
        case 3: return (size_t)(n_in / 256) * 176;  // Q5_K
        case 4: return (size_t)(n_in / 256) * 210;  // Q6_K
        case 5: return (size_t)(n_in / 256) * 144;  // Q4_K
        case 6: return (size_t)(n_in / 256) * 136;  // IQ4_XS
        case 7: return (size_t)n_in * 2;            // F16
        case 8: return (size_t)(n_in / 32) * 22;    // Q5_0
        case 9: return (size_t)n_in * 4;            // F32 (ZAYA 的 conv/路由小张量)
        default: return 0;
    }
}

// ---- 内建剖面：定义已移到 m6_layer.c 顶部（m6_granite_moe 在那里也要用）----

// ===================== 通用 token 前向 =====================
// x: [H] 输入 embedding, 就地覆盖为末层输出
// scratch: >= 3*(H+64) floats
int m6_forward_token(float* x, int h, float eps, int pos,
                     const M6LayerDesc* layers, int n_layer, float* scratch) {
    const int span = h + 64;
    float* xn  = scratch;
    float* xn2 = scratch + span;
    float* sub = scratch + 2 * span;
    for (int il = 0; il < n_layer; il++) {
        const M6LayerDesc* L = layers + il;
        double t0 = _now();
        m6_rms_norm(x, L->attn_norm, h, eps, xn);
        m6_rms_norm(x, L->ffn_norm, h, eps, xn2);   // 与 attn 无依赖, 一起计时
        double t1 = _now(); m6_prof_t[0] += t1 - t0;
        L->attn(xn, L->attn_p, pos, sub);
        double t2 = _now(); m6_prof_t[1] += t2 - t1; m6_prof_n[1]++;
        for (int i = 0; i < h; i++) x[i] += sub[i];
        m6_rms_norm(x, L->ffn_norm, h, eps, xn2);
        double t3 = _now();
        L->ffn(xn2, L->ffn_p, sub);
        double t4 = _now(); m6_prof_t[2] += t4 - t3; m6_prof_n[2]++;
        for (int i = 0; i < h; i++) x[i] += sub[i];
    }
    return 0;
}

const char* m6_engine_version(void) { return "m6_engine/1"; }

// MLA work 里 ctx (注意力输出) 的偏移, 供调试读取中间值
int m6_mla_off_ctx(int nh, int k_mla, int kv_lora, int rot) {
    const int KM = k_mla, KL = kv_lora, R = rot;
    return 256 + nh*KM + (KL+R) + nh*KL + nh*(KL+R);
}
// MLA work 里 q_full 的偏移
int m6_mla_off_qfull(int nh, int k_mla, int kv_lora, int rot) {
    const int KM = k_mla, KL = kv_lora, R = rot;
    return 256 + nh*KM + (KL+R) + nh*KL;
}

// MLA K 缓存的行步长 (调用方按 max_t * 此值 分配)
int m6_mla_kstride(int k_mla, int kv_lora, int rot) { (void)k_mla; return kv_lora + rot; }

// MLA work 缓冲需要多少 float (调用方按此申请)
int m6_mla_work_size(int nh, int k_mla, int v_mla, int kv_lora, int rot, int max_t) {
    (void)k_mla; (void)v_mla;
    const int KM = k_mla, KL = kv_lora, R = rot;
    // ★ + nh*max_t（不是 + max_t）：注意力改逐头并行后每头一行 score
    return 256 + nh * KM + (KL + R) + nh * KL + nh * (KL + R) + nh * KL + nh + nh * max_t;
}

// ===================== bailingmoe3 (Ling) 算子 =====================

// ---- MLA 注意力 ----
// work 布局 (需 >= 28672 - max_t + NH*max_t floats；调用方一律用 m6_mla_work_size()):
//   qa[256] qb[NH*KM] kva[1024] qnope[8192] qfull[4096] ctx[8192] ov[2048] gt[512] sc[NH*max_t]
typedef struct {
    const uint8_t* q_a;   int cqa;    // [q_lora][H]
    const float*   q_a_norm;          // [q_lora]
    const uint8_t* q_b;   int cqb;    // [NH*K_MLA][q_lora]
    const uint8_t* kv_a;  int ckva;   // [KV_LORA+ROT][H]
    const float*   kv_a_norm;         // [KV_LORA]
    const uint8_t* k_b;   int ckb;    // [NH][KV_LORA][NOPE]
    const uint8_t* v_b;   int cvb;    // [NH][V_MLA][KV_LORA]
    const uint8_t* gate;  int cg;     // [NH][H]  作用在 attn_norm 后的输入上
    const uint8_t* wo;    int cwo;    // [H][NH*V_MLA]
    int nh, k_mla, v_mla, kv_lora, rot, h, q_lora;
    float rope_base, eps;
    float* kcache;        // [max_t][K_MLA]
    float* vcache;        // [max_t][KV_LORA]
    int*   tlen;
    int    max_t;
    float* work;
} M6MlaP;

// NORM 风格 rope (相邻对), 作用于前 rot 维
static void rope_norm_rot(float* x, int rot, int pos, float base) {
    const int half = rot >> 1;
    for (int i = 0; i < half; i++) {
        const float theta = (float)pos * powf(base, -2.0f * (float)i / (float)rot);
        const float c = cosf(theta), s = sinf(theta);
        const float x1 = x[2*i], x2 = x[2*i + 1];
        x[2*i]     = x1 * c - x2 * s;
        x[2*i + 1] = x1 * s + x2 * c;
    }
}

void m6_mla_op(const float* xn, void* pv, int pos, float* out) {
    M6MlaP* p = (M6MlaP*)pv;
    const int H = p->h, NH = p->nh, KM = p->k_mla, VM = p->v_mla, KL = p->kv_lora;
    const int ROT = p->rot, NOPE = KM - ROT, QL = p->q_lora, T = *p->tlen;
    const int KS = KL + ROT;   // ★ K 缓存每行是 kv_lora+rot (=576), 不是 k_mla (=192)
                               //   曾误用 KM 当步长 → 行互相覆盖 + 越界
    const int Tn = T + 1;
    // work 布局: 每头 q_full 是 KL+ROT (=K_MLA) 个值, 不能按 KM 步长排 —— 曾因此每头互相覆盖
    const int o_qa    = 0;
    const int o_qb    = o_qa + 256;
    const int o_kva   = o_qb + NH * KM;
    const int o_qnope = o_kva + (KL + ROT);
    const int o_qfull = o_qnope + NH * KL;
    const int o_ctx   = o_qfull + NH * (KL + ROT);
    const int o_gt    = o_ctx + NH * KL;
    const int o_sc    = o_gt + NH;
    if (o_sc > 32768) return;   // 固定段越界兜底 (调用方用 m6_mla_work_size 申请总量)
    float* w      = p->work;
    float* qa     = w + o_qa;
    float* qb     = w + o_qb;
    float* kva    = w + o_kva;
    float* qnope  = w + o_qnope;
    float* qfull  = w + o_qfull;
    float* ctx    = w + o_ctx;
    float* gt     = w + o_gt;
    float* sc     = w + o_sc;

    // q: q_a → rms(q_a_norm) → q_b
    gemv_any(p->cqa, xn, p->q_a, QL, H, qa);
    m6_rms_norm(qa, p->q_a_norm, QL, p->eps, qa);
    gemv_any(p->cqb, qa, p->q_b, NH * KM, QL, qb);
    // kv: kv_a → 前 KL 做 rms, 后 ROT 做 rope
    gemv_any(p->ckva, xn, p->kv_a, KL + ROT, H, kva);
    m6_rms_norm(kva, p->kv_a_norm, KL, p->eps, kva);
    rope_norm_rot(kva + KL, ROT, pos, p->rope_base);
    // ★★ 2026-09-15：下面三个"每头一次 gemv_any"的循环（每层 NH=16 头）原来各自开一次 OMP 区
    //   —— 每层 48 次 fork/join，而单头工作量很小（KL×NOPE = 512×128），固定开销占大头；
    //   注意力循环还完全是串行的，且对 Tn 是 O(上下文长度)（注释里原来写着"Tn 一般远小于 KL,
    //   单线程足够"，实测在长上下文下不成立）。现在合并成**一个并行区**（改用无内嵌 OMP 的
    //   + 注意力逐头并行 + **每头一行 score**。
    //   ★ 这两个调用保持 gemv_any。区内调它会让内层并行区被 GOMP 退化成单线程（本仓库多处
    //     依赖这条，结果正确、头与头之间仍并行）。
    //   ⚠️⚠️ 2026-09-15 这里出过一个"看似数值不等价"的悬案，**真相是我自己的行区间传错**：
    //     当时把这两个调用换成 gemv_range_any 并传 `0, NOPE`(=128) 当行区间，而 n_out 是
    //     KL(=512) ⇒ 只算了前 128 行、其余 384 行留着**上一轮的陈旧数据**（不崩、不 NaN，
    //     只是微微偏：6 token 后 logits cos 0.9938）。改成 `0, KL` 后**逐位相同**。
    //     ⇒ 结论：两个入口**是等价的**（gemv_entry_audit.py 逐 (格式,形状) 对账过，
    //        含这两处的真实组合 Q8_0 512×128 / Q6_K 128×512）；换成 range 也**没有实测收益**
    //        （交替 A/B：Tn=32 1.08×、256 0.94×、768 0.97×，全在噪声内）⇒ 保留 gemv_any。
    //     ⇒ 教训（连续两次归因错误后总结）：**先查参数与单位，再怀疑引擎**；
    //        "换回原来的写法就好了"不是根因证据，只是掩盖了越界区间。
    //   逐头独立 ⇒ 数值逐位不变（已用改前/改后 .so 的同 token logits 逐位比对，max|Δ|=0）。
    const size_t rbk = m6_rowbytes(p->ckb, NOPE);
    const size_t rbv = m6_rowbytes(p->cvb, KL);
    const float scale = 1.0f / sqrtf((float)KM);
    // 注：rope_norm_rot 无静态状态（每对就地算 powf/cosf/sinf），多线程调用本来就安全；
    //     ZAYA 的 NEOX 版有静态角度表，那条路的表调用在并行区**之外**（见 m6_zaya_cca_op）。
    #pragma omp parallel
    {
        // q_nope 吸收 wk_b (每头 [KL][NOPE])
        #pragma omp for schedule(static)
        for (int hh = 0; hh < NH; hh++) {
            const uint8_t* kb = p->k_b + (size_t)hh * KL * rbk;
            gemv_any(p->ckb, qb + hh * KM, kb, KL, NOPE, qnope + hh * KL);
        }
        // q_pe rope + 拼 q_full = [qnope_absorbed | q_pe]
        #pragma omp for schedule(static)
        for (int hh = 0; hh < NH; hh++) {
            float* qph = qb + hh * KM + NOPE;
            rope_norm_rot(qph, ROT, pos, p->rope_base);
            memcpy(qfull + hh * (KL + ROT), qnope + hh * KL, sizeof(float) * KL);
            memcpy(qfull + hh * (KL + ROT) + KL, qph, sizeof(float) * ROT);
        }
        // 写 KV cache: k = [kv_norm | k_pe], v = kv_norm （必须在注意力之前）
        #pragma omp single
        {
            float* kc = p->kcache + (size_t)T * KS;
            memcpy(kc, kva, sizeof(float) * KL);
            memcpy(kc + KL, kva + KL, sizeof(float) * ROT);
            memcpy(p->vcache + (size_t)T * KL, kva, sizeof(float) * KL);
        }
        // attention (逐头并行; 点积长度 = KS(576), 不是 KM(192) —— 曾误用 KM 当步长)
        #pragma omp for schedule(static)
        for (int hh = 0; hh < NH; hh++) {
            const float* qh = qfull + hh * (KL + ROT);
            float* sch = sc + (size_t)hh * p->max_t;      // ★ 每头一行
            float mx = -1e30f;
            for (int t = 0; t < Tn; t++) {
                const float* kt = p->kcache + (size_t)t * KS;
                float s = 0.f;
                for (int i = 0; i < KS; i++) s += qh[i] * kt[i];
                s *= scale;
                sch[t] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.f;
            for (int t = 0; t < Tn; t++) { sch[t] = expf(sch[t] - mx); sum += sch[t]; }
            const float inv = 1.f / sum;
            float* ch = ctx + hh * KL;            // 注意力输出 [KL] (V 缓存本身就是 KL 维)
            for (int i = 0; i < KL; i++) ch[i] = 0.f;
            for (int t = 0; t < Tn; t++) {
                const float pw = sch[t] * inv;
                const float* vt = p->vcache + (size_t)t * KL;
                for (int i = 0; i < KL; i++) ch[i] += pw * vt[i];
            }
        }
        // v 吸收 wv_b (每头 [VM][KL]) → 结果放 qnope 复用
        #pragma omp for schedule(static)
        for (int hh = 0; hh < NH; hh++) {
            const uint8_t* vb = p->v_b + (size_t)hh * VM * rbv;
            gemv_any(p->cvb, ctx + hh * KL, vb, VM, KL, qnope + hh * VM);
        }
    }   // 并行区结束
    // attn_gate × sigmoid, 然后 wo
    gemv_any(p->cg, xn, p->gate, NH, H, gt);
    for (int hh = 0; hh < NH; hh++) {
        const float g = 1.f / (1.f + expf(-gt[hh]));
        float* src = qnope + hh * VM;
        for (int i = 0; i < VM; i++) src[i] *= g;
    }
    gemv_any(p->cwo, qnope, p->wo, H, NH * VM, out);
    *p->tlen = Tn;
}

// ---- dense FFN (首层) ----
typedef struct {
    const uint8_t *g, *u, *d;
    int cg, cu, cd, n_ff, h;
} M6DenseP;

void m6_dense_op(const float* xn, void* pv, float* out) {
    // ★ 原来 3 个 gemv 都是单线程调用 (gemv_range_any 不含 OMP) → 小模型上成为主瓶颈。
    // 改成一个 parallel 区, 每个矩阵按 nseg 段并行 (段内仍是 range 入口)。
    M6DenseP* p = (M6DenseP*)pv;
    float g[8192], u[8192];
    const int FF = p->n_ff, H = p->h;
    const int nseg = (FF >= 4 * 128) ? 8 : 1;   // 太小的层不值得开区域
    if (nseg == 1) {
        gemv_range_any(p->cg, xn, p->g, FF, H, g, 0, FF);
        gemv_range_any(p->cu, xn, p->u, FF, H, u, 0, FF);
        for (int i = 0; i < FF; i++) { const float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
        gemv_range_any(p->cd, g, p->d, H, FF, out, 0, H);
        return;
    }
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++)
            gemv_range_any(p->cg, xn, p->g, FF, H, g, s*FF/nseg, (s+1)*FF/nseg);
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++)
            gemv_range_any(p->cu, xn, p->u, FF, H, u, s*FF/nseg, (s+1)*FF/nseg);
        #pragma omp for schedule(static)
        for (int i = 0; i < FF; i++) { const float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++)
            gemv_range_any(p->cd, g, p->d, H, FF, out, s*H/nseg, (s+1)*H/nseg);
    }
}

// ---- MoE (融合路由 + 共享专家) ----
typedef struct {
    const uint64_t* exp_ptrs;   // [3*NEXP] 全专家指针 (mmap 下地址固定)
    const int*      exp_codes;  // [3*NEXP]
    const float *gate_inp, *probs_b;
    int n_exp, n_used, n_group, n_group_used, norm_w;
    float w_scale;
    int inter, n_in, n_out;
    const uint8_t *sh_g, *sh_u, *sh_d;   // 共享专家 (dense)
    int csg, csu, csd, sh_ff;
    float* scratch;             // >= (3*inter + n_out)*n_used
} M6MoeP;

void m6_moe_op(const float* xn, void* pv, float* out) {
    M6MoeP* p = (M6MoeP*)pv;
    m6_bailing_moe(xn, p->exp_ptrs, p->exp_codes, p->gate_inp, p->probs_b,
                   p->n_exp, p->n_used, p->n_group, p->n_group_used,
                   p->norm_w, p->w_scale, p->inter, p->n_in, p->n_out,
                   p->scratch, out);
    // 共享专家 (无门): out += down(silu(gate)·up)
    // 原来 3 个 gemv 全串行 → 放进一个 parallel 区, 各矩阵按 4 段用 range 并行
    // 相位必须分开 (omp for 之间有隐式屏障): gate/up → act → down
    float g[8192], u[8192];
    const int nseg = 4, FF = p->sh_ff;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++)
            gemv_range_any(p->csg, xn, p->sh_g, FF, p->n_in, g, s*FF/nseg, (s+1)*FF/nseg);
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++)
            gemv_range_any(p->csu, xn, p->sh_u, FF, p->n_in, u, s*FF/nseg, (s+1)*FF/nseg);
        #pragma omp for schedule(static)
        for (int i = 0; i < FF; i++) {
            const float gv = g[i];
            g[i] = (gv / (1.f + expf(-gv))) * u[i];
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < p->n_out; i++) u[i] = 0.f;
        #pragma omp for schedule(static)
        for (int s = 0; s < nseg; s++)
            gemv_range_any(p->csd, g, p->sh_d, p->n_out, FF, u, s*p->n_out/nseg, (s+1)*p->n_out/nseg);
    }
    for (int i = 0; i < p->n_out; i++) out[i] += u[i];
}

// ---- KDA 线性注意力 (薄包装: 复用已验证的 m6_bailing_kda) ----
typedef struct {
    const uint8_t *wq, *wk, *wv, *f_a, *g_a, *wo;
    int cq, ck, cv, cfa, cga, cwo;
    const float *conv_w, *dt_b, *ssm_a, *beta_w, *o_norm;
    float gate_lb, eps;
    int nh, hd, h;
    float *conv_state, *S, *scratch;
} M6KdaP;

void m6_kda_op(const float* xn, void* pv, int pos, float* out) {
    (void)pos;
    M6KdaP* p = (M6KdaP*)pv;
    m6_bailing_kda(xn, p->wq, p->cq, p->wk, p->ck, p->wv, p->cv,
                   p->conv_w, p->f_a, p->cfa, p->dt_b, p->ssm_a, p->beta_w,
                   p->g_a, p->cga, p->o_norm, p->wo, p->cwo,
                   p->gate_lb, p->nh, p->hd, p->h,
                   p->conv_state, p->S, p->scratch, out);
}


// ---- 归一化 + 词表投影 (末层收尾) ----
int m6_head_op(const float* x, const float* final_norm, int h, float eps,
               const uint8_t* head_w, int code, int vocab, float* logits, float* scratch) {
    m6_rms_norm(x, final_norm, h, eps, scratch);
    gemv_any(code, scratch, head_w, vocab, h, logits);
    return 0;
}

// ===================== 标准 Llama 系 (GQA + NEOX 全头 rope + SwiGLU) =====================
// 适用: llama / SmolLM2 / Qwen2 等"每层=注意力+dense FFN"的标准 decoder。
// 与引擎固化管线完全同构 (rms(attn_norm) → attn → += → rms(ffn_norm) → FFN → +=), 无需改 forward。
typedef struct {
    const uint8_t *wq, *wk, *wv, *wo;
    int cq, ck, cv, co;
    int n_head, n_kv, head_dim, hidden, rot;
    float rope_base;
    float* kcache;      // [max_t][n_kv*head_dim]
    float* vcache;      // [max_t][n_kv*head_dim]
    int*   tlen;
    int    max_t;
    float* work;        // >= n_head*head_dim + 2*n_kv*head_dim + n_head*max_t
} M6LlamaAttnP;

// NEOX rope: 半区旋转, 作用于前 rot 维 (theta_i = pos * base^(-2i/rot))
static void rope_neox(float* x, int rot, int pos, float base) {
    const int half = rot >> 1;
    for (int i = 0; i < half; i++) {
        const float theta = (float)pos * powf(base, -2.0f * (float)i / (float)rot);
        const float c = cosf(theta), s = sinf(theta);
        const float x1 = x[i], x2 = x[i + half];
        x[i]        = x1 * c - x2 * s;
        x[i + half] = x1 * s + x2 * c;
    }
}

// ★ 抽成带 scale 参数的内核：granite 要复用同一套「一个并行区 + 逐头并行」的注意力，
//   但它有两处不同 ① **NoPE**（把 rot 设 0 即彻底跳过 rope，见 m6_granite_attn_op）
//   ② kq_scale = 1/head_dim（llama 家族是 1/sqrt(head_dim)，差 sqrt(128)≈11.3 倍）。
//   为什么抽函数而不是给 M6LlamaAttnP 加字段：Python 侧用 ctypes.Structure 自己分配这个结构，
//   加字段会让老调用方（smol/ling）分配得比 C 期望的小 ⇒ C 读到堆上垃圾值，**静默改掉 scale**。
static void llama_attn_core(M6LlamaAttnP* p, const float* xn, int pos, float* out, float scale, int use_neox) {
    const int NH = p->n_head, NKV = p->n_kv, HD = p->head_dim, H = p->hidden, ROT = p->rot;
    const int QD = NH * HD, KD = NKV * HD, T = *p->tlen, Tn = T + 1;
    float* w = p->work;
    float* q = w;
    float* k = q + QD;
    float* v = k + KD;
    float* sc = v + KD;                       // [NH][max_t] —— ★ 每头一行，头并行时不打架
    const int rep = NH / NKV;
    // ★★ 2026-09-15 一个并行区 + 逐头并行（原来 4 次 gemv 各开一次 OMP 区、且**每头的
    //   softmax/加权求和是串行的**）。实测动因（smol-360M，T=8，DRACO_ENG_PROF 分解）：
    //     fwd 7.21ms/token，其中 attn 段 2.2~2.5ms —— 而 attn 的权重只有 45MB，
    //     按同机 Q5_0/Q6_K 内核实测的 43GB/s 只应花 1.05ms ⇒ ~1.2ms 是"非流式"开销：
    //     4 次 fork/join（每次约 4~6µs×32 层=0.6ms）+ 串行的头循环（0.3~0.7ms）。
    //   分段用既有的 gemv_range_any（无内嵌 OMP 的行区间入口，就是为"胖区域"准备的），
    //   范式与 m6_dense_op 一致。**逐行/逐头独立 ⇒ 与并行度无关，数值逐位不变**（已用
    //   before/after .so 同 token 比对：max|Δ|=0）。
    int nq = QD / 32; if (nq < 1) nq = 1; if (nq > 32) nq = 32;
    int nk = KD / 32; if (nk < 1) nk = 1; if (nk > 32) nk = 32;
    const int no = H / 32 > 0 ? (H / 32 > 32 ? 32 : H / 32) : 1;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int s = 0; s < nq; s++)
            gemv_range_any(p->cq, xn, p->wq, QD, H, q, s * QD / nq, (s + 1) * QD / nq);
        #pragma omp for schedule(static)
        for (int s = 0; s < nk; s++)
            gemv_range_any(p->ck, xn, p->wk, KD, H, k, s * KD / nk, (s + 1) * KD / nk);
        #pragma omp for schedule(static)
        for (int s = 0; s < nk; s++)
            gemv_range_any(p->cv, xn, p->wv, KD, H, v, s * KD / nk, (s + 1) * KD / nk);
        // ★ rope：llama 家族用**连续成对** rope（ggml 的 GGML_ROPE_TYPE_NORMAL），
        //   当年误用 rope_neox（半分裂，ZAYA/NEOX 系的变体）—— SmolLM2 只测过 tok/s 没对数值
        //   所以漏网，2026-09-13 修。见 m6_op_isolate.py 的逐算子对账。
        //   逐头展开成 omp for：头之间完全独立（角度表按 (pos,base,rot) 算，与头无关）。
        // ★ rope 风格两派：llama/SmolLM2/granite 是「连续成对」（rope_norm_rot），
        //   Falcon-H1/ZAYA 系是 NEOX「半区旋转」（rope_neox）。llama.cpp 把 FALCON_H1 归在
        //   "the pairs of head values are offset by n_rot/2" 那一段 ⇒ NEOX。
        //   用错风格不会崩：softmax 过 T 个 token 后误差被摊平，只剩"长上下文变笨"这类软症状
        //   ⇒ 必须靠对账抓，不能靠"跑起来正常"。
        if (use_neox) {
            #pragma omp for schedule(static)
            for (int h = 0; h < NH; h++)  rope_neox(q + h * HD, ROT, pos, p->rope_base);
            #pragma omp for schedule(static)
            for (int h = 0; h < NKV; h++) rope_neox(k + h * HD, ROT, pos, p->rope_base);
        } else {
            #pragma omp for schedule(static)
            for (int h = 0; h < NH; h++)  rope_norm_rot(q + h * HD, ROT, pos, p->rope_base);
            #pragma omp for schedule(static)
            for (int h = 0; h < NKV; h++) rope_norm_rot(k + h * HD, ROT, pos, p->rope_base);
        }
        // KV 写回必须在 rope 之后、注意力之前（omp for 末尾的隐式栅栏已保证）
        #pragma omp single
        {
            memcpy(p->kcache + (size_t)T * KD, k, sizeof(float) * KD);
            memcpy(p->vcache + (size_t)T * KD, v, sizeof(float) * KD);
        }
        #pragma omp for schedule(static)
        for (int h = 0; h < NH; h++) {
            const int kv = h / rep;               // GQA: 连续的 rep 个 q 头共享一个 kv 头
            const float* qh = q + h * HD;
            float* sch = sc + (size_t)h * p->max_t;
            float mx = -1e30f;
            for (int t = 0; t < Tn; t++) {
                const float* kt = p->kcache + (size_t)t * KD + kv * HD;
                float s = 0.f;
                for (int i = 0; i < HD; i++) s += qh[i] * kt[i];
                s *= scale; sch[t] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.f;
            for (int t = 0; t < Tn; t++) { sch[t] = expf(sch[t] - mx); sum += sch[t]; }
            const float inv = 1.f / sum;
            float* oh = q + h * HD;               // 复用 q 缓冲存注意力输出
            for (int i = 0; i < HD; i++) oh[i] = 0.f;
            for (int t = 0; t < Tn; t++) {
                const float pw = sch[t] * inv;
                const float* vt = p->vcache + (size_t)t * KD + kv * HD;
                for (int i = 0; i < HD; i++) oh[i] += pw * vt[i];
            }
        }
        #pragma omp for schedule(static)
        for (int s = 0; s < no; s++)
            gemv_range_any(p->co, q, p->wo, H, QD, out, s * H / no, (s + 1) * H / no);
    }
    *p->tlen = Tn;                            // ★ 忘写这行 = 永远只看 1 个 token (无上下文)
}

// llama 家族：kq_scale = 1/sqrt(head_dim)（llama-graph.cpp build_attn 的默认）
void m6_llama_attn_op(const float* xn, void* pv, int pos, float* out) {
    M6LlamaAttnP* p = (M6LlamaAttnP*)pv;
    llama_attn_core(p, xn, pos, out, 1.0f / sqrtf((float)p->head_dim), 0);
}

// Falcon-H1 的注意力：kq_scale 与 llama 同为 1/sqrt(head_dim)（它没读 attention.scale 键），
// 唯一区别是 **NEOX rope** ⇒ 与 llama 共用同一个 M6LlamaAttnP 结构（不加字段，免得破坏
// smol/ling 的 ctypes 结构布局），风格由算子本身决定。
void m6_falcon_attn_op(const float* xn, void* pv, int pos, float* out) {
    M6LlamaAttnP* p = (M6LlamaAttnP*)pv;
    llama_attn_core(p, xn, pos, out, 1.0f / sqrtf((float)p->head_dim), 1);
}

// granite-hybrid 的 4 个注意力层：
//   · **NoPE** —— GGUF 里 rope.scaling.finetuned=0 ⇒ llama.cpp 的 granite-hybrid 把
//     hparams.rope_pattern 全填 false ⇒ has_rope(il)==false ⇒ 图里 inp_pos=nullptr、
//     build_attention_layer 里整段 `if (hparams.has_rope(il))` 被跳过。**不加 rope 才是对的**。
//     实现上把 p->rot 传 0：rope_norm_rot 的循环次数 half=rot/2=0 ⇒ 天然空操作。
//   · kq_scale = attention.scale = 1/head_dim（granite 的 GGUF 明确给了 0.0078125=1/128）
//   头数/kv 布局与 llama 系相同 ⇒ 复用同一个 M6LlamaAttnP 结构。
void m6_granite_attn_op(const float* xn, void* pv, int pos, float* out) {
    M6LlamaAttnP* p = (M6LlamaAttnP*)pv;
    llama_attn_core(p, xn, pos, out, 1.0f / (float)p->head_dim, 0);
}

// ===================== ZAYA1 (CCA 注意力 + MLP 路由 MoE + 带偏置残差标定) =====================
// 与固化管线不同: 残差合并是 (sub+hs_b)*hs_w + (res+r_b)*r_w, 所以另写 forward。
// 数学已与 transformers 参考实现逐段对账 (zaya_check2.py, L0/L1 全 token cos=1.0)。
typedef struct {
    const uint8_t *wq, *wk, *wv1, *wv2, *wo;
    int cq, ck, cv1, cv2, co;
    const float *dw_w, *dw_b;      // depthwise [1280][2] / [1280]
    const float *grp_w, *grp_b;    // grouped  [1280][128][2] / [1280]
    const float *temp;             // [nkv]
    int nh, nkv, hd, rot, hidden;
    float rope_base;
    // 状态 (每层独立; pos==0 时按参考语义重置)
    float* conv_state;             // [2][1280] → qk_prev, dw_prev
    float* kbuf;                   // [max_t][nkv*hd]
    float* vbuf;                   // [max_t][nkv*hd]
    float* vdel;                   // [nkv*hd/2] 上一 token 的 delayed 投影
    int*   tlen;                   // 已缓存 token 数
    int    max_t;
    float* work;                   // >= nh*hd + 2*nkv*hd + max_t + 1280*3
    float* dbg;                    // [CQK] 或 NULL: 每次调用后把卷积输出留一份 (探针用)
} M6ZayaCcaP;

static inline void gelu_erf(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.70710678118654752f));
}

// NEOX 半旋转, 作用前 rot 维。
// ★ 角度表缓存: th/c/s 只依赖 (pos, base, rot), 与层/头无关 —— 原来每层×每头重算
//   powf+cosf+sinf (10 次调用 × 32 对 × ~300 周期 ≈ 30-50µs/层, 占 attn 的 ~25%)。
//   缓存后每 token 只算一次表, rope 本体退化成 32 对 fma。
// 线程安全: cca_op 每 token 串行进入 (OMP 只在内部 gemv 里), 静态缓存无竞争。
static float g_rope_c[32], g_rope_s[32];
static int   g_rope_pos = -1;
static float g_rope_base = 0.f;
static int   g_rope_rot = 0;
static void zaya_rope_table(int rot, int pos, float base) {
    if (pos == g_rope_pos && base == g_rope_base && rot == g_rope_rot) return;
    const int half = rot >> 1;
    for (int i = 0; i < half; i++) {
        const float th = (float)pos * powf(base, -2.0f * (float)i / (float)rot);
        g_rope_c[i] = cosf(th); g_rope_s[i] = sinf(th);
    }
    g_rope_pos = pos; g_rope_base = base; g_rope_rot = rot;
}
static inline void zaya_rope(float* v, int rot) {
    const int half = rot >> 1;
    for (int i = 0; i < half; i++) {
        const float c = g_rope_c[i], s = g_rope_s[i];
        const float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

void m6_zaya_cca_op(const float* xn, void* pv, int pos, float* out) {
    M6ZayaCcaP* p = (M6ZayaCcaP*)pv;
    const int NH = p->nh, NKV = p->nkv, HD = p->hd, H = p->hidden, ROT = p->rot;
    const int QD = NH * HD, KD = NKV * HD, VH = KD / 2, CQK = QD + KD;
    const int rep = NH / NKV;
    float* w = p->work;
    // 布局: [q QD][k KD][v KD][sc NH*max_t][qk CQK][dw_t CQK]
    //   ★ sc 是 **NH*max_t**（每头一行）：注意力改成"逐头并行"后不能再共用一个 max_t 缓冲。
    //     调用方（zaya_gguf.py 的 work 分配）必须同步改，否则是静默越界。
    float *q = w, *k = q + QD, *v = k + KD, *sc = v + KD;
    float* qk  = sc + (size_t)NH * p->max_t;
    float* dwt = qk + CQK;
    float* qkp = p->conv_state;             // [CQK] 上一帧 qk
    float* dwp = p->conv_state + CQK;       // [CQK] 上一帧 depthwise 输出

    if (pos == 0) {                         // ★ t=0: 输入两拍全 0 → depthwise 输出 = 偏置, 不是 0
        *p->tlen = 0;                       //   新序列: 自己复位, 不依赖调用方清 tlen
        memset(qkp, 0, sizeof(float) * CQK);
        memcpy(dwp, p->dw_b, sizeof(float) * CQK);
        memset(p->vdel, 0, sizeof(float) * VH);
    }
    // ★ T 必须在 pos==0 复位**之后**再读, 否则复用序列时会把旧 tlen 带进来 (会多看几个陈旧 token)
    const int T = *p->tlen, Tn = T + 1;
    gemv_any(p->cq, xn, p->wq, QD, H, q);                        // 2M MAC → 开并行区
    gemv_range_any(p->ck, xn, p->wk, KD, H, k, 0, KD);           // 524k MAC → 串行更划算
    memcpy(qk, q, sizeof(float) * QD);
    memcpy(qk + QD, k, sizeof(float) * KD);
    for (int j = 0; j < CQK; j++)
        dwt[j] = p->dw_w[2*j] * qkp[j] + p->dw_w[2*j+1] * qk[j] + p->dw_b[j];
    memcpy(qkp, qk, sizeof(float) * CQK);
    // grouped conv: 10 组, 每组 out[128] = W[128][128][2] · (dwp, dwt)
    const int NG = CQK / 128;
    // ★ 原来这里是**嵌套标量循环** (每输出 128 次迭代、单累加器依赖链, NG 组全串行)。
    //   微基准 (m5/m5_convbench.c, 真实 grp_w): 标量串行 70.9us/层 = 实测 attn 耗时的 32%;
    //   AVX512 版 17.3us/层 → 4.1x, rel 误差 1.7e-07 (求和次序变化)。
    //   做法: 每组先把 (dwp,dwt) 交错成 256 长激活向量 (每组一次, 摊到 128 个输出),
    //   再让每个输出做干净的 256 长点积 (4 个独立累加器, 无依赖链)。
    if (NG > 0 && NG <= 16) {
        float vact[16 * 128 * 2];
        for (int g = 0; g < NG; g++) {
            const int base = g * 128;
            for (int i = 0; i < 128; i++) {
                vact[g * 256 + 2 * i]     = dwp[base + i];
                vact[g * 256 + 2 * i + 1] = dwt[base + i];
            }
        }
        #pragma omp parallel for schedule(static) collapse(2) if(!omp_in_parallel())
        for (int g = 0; g < NG; g++) {
            for (int o = 0; o < 128; o++) {
                const int base = g * 128;
                const float* wr = p->grp_w + ((size_t)base * 128 + (size_t)o * 128) * 2;
                const float* va = vact + g * 256;
                __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
                __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
                for (int j = 0; j < 256; j += 64) {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j),      _mm512_loadu_ps(va + j),      a0);
                    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j + 16), _mm512_loadu_ps(va + j + 16), a1);
                    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j + 32), _mm512_loadu_ps(va + j + 32), a2);
                    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + j + 48), _mm512_loadu_ps(va + j + 48), a3);
                }
                out[base + o] = p->grp_b[base + o] +
                    _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
            }
        }
    } else {                                 // 兜底: 形状意外时保持原标量路径
        for (int g = 0; g < NG; g++) {
            const int base = g * 128;
            const float* Wg = p->grp_w + (size_t)base * 128 * 2;
            for (int o = 0; o < 128; o++) {
                const float* wr = Wg + (size_t)o * 128 * 2;
                float s = p->grp_b[base + o];
                for (int i = 0; i < 128; i++)
                    s += wr[2*i] * dwp[base + i] + wr[2*i+1] * dwt[base + i];
                out[base + o] = s;
            }
        }
    }
    if (p->dbg) memcpy(p->dbg, out, sizeof(float) * CQK);   // 卷积输出留档
    // ★ dwp 的更新必须在 grouped conv **之后**: 本次卷积要用 (d_t, d_{t+1}) 即 (旧 dwp, 新 dwt)。
    //   放在前面会把旧帧覆盖掉 → 卷积变成 (d_{t+1}, d_{t+1})。tok0 看不出 (T=1 的注意力输出恒等于 v),
    //   tok>=1 立刻崩。
    memcpy(dwp, dwt, sizeof(float) * CQK);
    // q/k 残差来自 **线性投影** (不是卷积输出!): q_res=(q+repeat_k(k))*0.5, k_res=rep 个 q 头均值
    // qres 复用 qk 缓冲 (它已 memcpy 到 qkp, 之后就没用了)
    float* qres = qk;
    for (int h = 0; h < NH; h++) {
        const int kv = h / rep;
        const float* ql = q + h * HD;
        const float* kl = k + kv * HD;
        float* qr = qres + h * HD;
        for (int i = 0; i < HD; i++) qr[i] = (ql[i] + kl[i]) * 0.5f;
    }
    for (int h = 0; h < NH; h++) {
        float* qh = out + h * HD;
        const float* qr = qres + h * HD;
        for (int i = 0; i < HD; i++) qh[i] += qr[i];
    }
    for (int j = 0; j < NKV; j++) {
        float* kr = out + QD + j * HD;
        for (int i = 0; i < HD; i++) {
            float s = 0.f;
            for (int r = 0; r < rep; r++) s += qres[(j * rep + r) * HD + i];
            kr[i] += s / (float)rep;
        }
    }
    // l2 归一到 sqrt(hd), k 再乘 temp, 然后 rope
    // v: cat([v_cur(h_t), v_proj_delayed(h_{t-1})])
    // ★ 必须在并行区**之前**完成：注意力要读 p->vbuf，而 `omp for` 入口没有栅栏 ——
    //   把这段留在区内会让某个线程在别人还没写完 vbuf 时就开始注意力（第一次改就踩了这个坑，
    //   逐位对账立刻炸出来 cos≈−0.01）。
    gemv_range_any(p->cv1, xn, p->wv1, VH, H, v, 0, VH);        // 小矩阵走 range 版 (串行无 OMP)
    memcpy(v + VH, p->vdel, sizeof(float) * VH);                 // 延迟项 = 上一 token 的 v_proj2 输出
    gemv_range_any(p->cv2, xn, p->wv2, VH, H, p->vdel, 0, VH);
    memcpy(p->vbuf + (size_t)T * KD, v, sizeof(float) * KD);
    zaya_rope_table(ROT, pos, p->rope_base);          // ★ 每 token 一次 (角度与层/头无关)
    const float sq = sqrtf((float)HD);
    // ★★ 2026-09-15：把"L2norm+rope(逐头) → kbuf 写回 → 注意力(逐头)"收进**一个并行区**。
    //   原来这四段全是串行：头内是 AVX512 矢量的，但**头与头之间没有并行**，而注意力对 Tn 是
    //   O(上下文长度) 的（本函数自己的注释早就写着"真实 decode 在 T=数百 时这里是 attn 的主要
    //   标量开销"）。smol 的同类修复实测：Tn=768 时整 token 2.55×、attn 段 3.53×。
    //   数值不变：逐头/逐行互相独立，只是换了执行线程（已用改前/改后 .so 的 8 个 token 隐状态
    //   逐位比对，max|Δ|=0）。
    const float asc = 1.0f / sqrtf((float)HD);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int h = 0; h < NH; h++) {
            float* qh = out + h * HD;
            __m512 s = _mm512_set1_ps(1e-12f);
            for (int i = 0; i + 16 <= HD; i += 16)
                s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_loadu_ps(qh + i), _mm512_loadu_ps(qh + i)));
            const float g = sq / sqrtf(_mm512_reduce_add_ps(s));
            const __m512 gv = _mm512_set1_ps(g);
            for (int i = 0; i < HD; i += 16) _mm512_storeu_ps(qh + i, _mm512_mul_ps(_mm512_loadu_ps(qh + i), gv));
            zaya_rope(qh, ROT);
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < NKV; j++) {
            float* kh = out + QD + j * HD;
            __m512 s = _mm512_set1_ps(1e-12f);
            for (int i = 0; i + 16 <= HD; i += 16)
                s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_loadu_ps(kh + i), _mm512_loadu_ps(kh + i)));
            const float g = sq / sqrtf(_mm512_reduce_add_ps(s)) * p->temp[j];
            const __m512 gv = _mm512_set1_ps(g);
            for (int i = 0; i < HD; i += 16) _mm512_storeu_ps(kh + i, _mm512_mul_ps(_mm512_loadu_ps(kh + i), gv));
            zaya_rope(kh, ROT);
        }
        #pragma omp single
        { memcpy(p->kbuf + (size_t)T * KD, out + QD, sizeof(float) * KD); }
        // 注意力 (GQA, scale=1/sqrt(hd)) —— QK 点积/加权 V 求和向量化
        // (基准 T=5 时占比小, 但真实 decode 在 T=数百 时这里是 attn 的主要标量开销)
        #pragma omp for schedule(static)
        for (int h = 0; h < NH; h++) {
            const int kv = h / rep;
            const float* qh = out + h * HD;
            float* sch = sc + (size_t)h * p->max_t;      // ★ 每头一行
            float mx = -1e30f;
            for (int t = 0; t < Tn; t++) {
                const float* kt = p->kbuf + (size_t)t * KD + kv * HD;
                __m512 sv = _mm512_setzero_ps();
                for (int i = 0; i + 16 <= HD; i += 16)
                    sv = _mm512_fmadd_ps(_mm512_loadu_ps(qh + i), _mm512_loadu_ps(kt + i), sv);
                float s = _mm512_reduce_add_ps(sv);
                for (int i = (HD / 16) * 16; i < HD; i++) s += qh[i] * kt[i];
                s *= asc; sch[t] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.f;
            for (int t = 0; t < Tn; t++) { sch[t] = expf(sch[t] - mx); sum += sch[t]; }
            const float inv = 1.f / sum;
            float* oh = q + h * HD;
            for (int i = 0; i < HD; i++) oh[i] = 0.f;
            for (int t = 0; t < Tn; t++) {
                const float pw = sch[t] * inv;
                const float* vt = p->vbuf + (size_t)t * KD + kv * HD;
                const __m512 pv = _mm512_set1_ps(pw);
                for (int i = 0; i + 16 <= HD; i += 16)
                    _mm512_storeu_ps(oh + i, _mm512_fmadd_ps(pv, _mm512_loadu_ps(vt + i),
                                                             _mm512_loadu_ps(oh + i)));
                for (int i = (HD / 16) * 16; i < HD; i++) oh[i] += pw * vt[i];
            }
        }
    }   // 并行区结束
    gemv_any(p->co, q, p->wo, H, QD, out);      // 注意: 此时 out[0..QD) 已是 q 缓冲的别名, 覆盖安全
    *p->tlen = Tn;                              // ★ 别忘了
}

// ---- MoE: 路由 MLP (2048→256→256→256→17) + top-1 专家 + EDA 递归状态 ----
typedef struct {
    const uint8_t *wdown, *wfc1, *wfc2, *wout;
    int cdown, cfc1, cfc2, cout;
    const float *bdown, *bfc1, *bfc2;
    const float *rnorm;            // [256] router MLP 的 norm gain (base 里叫 rmsnorm_eda)
    const float *eda_scale;        // [256] 或 NULL (layer 0)
    const float *bal;              // [17] balancing_biases
    const uint64_t *exp_gp, *exp_dp;
    const int *exp_cg, *exp_cd;
    int nexp, nclass, inter, hidden, hidden2, iskip;
    float* rh_state;               // [256] 跨层递归 (含 EDA 项)
    float* scratch;
} M6ZayaMoeP;

void m6_zaya_moe_op(const float* xn, void* pv, float* out) {
    M6ZayaMoeP* p = (M6ZayaMoeP*)pv;
    const int H = p->hidden, RH = p->hidden2;
    float* s = p->scratch;
    float* rh = s;                 // [RH]
    float* t1 = rh + RH;           // [RH]
    float* lg = t1 + RH;           // [nclass]
    float* gu = lg + p->nclass;    // [2*inter] 专家 gate/up 输出 (必须独立缓冲, 别蹭 t1)
    gemv_range_any(p->cdown, xn, p->wdown, RH, H, rh, 0, RH);      // 524k MAC → 串行
    for (int i = 0; i < RH; i++) rh[i] += p->bdown[i];
    if (p->eda_scale) for (int i = 0; i < RH; i++) rh[i] += p->rh_state[i] * p->eda_scale[i];
    memcpy(p->rh_state, rh, sizeof(float) * RH);          // 递归传出 (含 EDA)
    m6_rms_norm(rh, p->rnorm, RH, 1e-5f, t1);
    gemv_range_any(p->cfc1, t1, p->wfc1, RH, RH, rh, 0, RH);
    for (int i = 0; i < RH; i++) rh[i] += p->bfc1[i];
    gelu_erf(rh, RH);
    gemv_range_any(p->cfc2, rh, p->wfc2, RH, RH, t1, 0, RH);
    for (int i = 0; i < RH; i++) t1[i] += p->bfc2[i];
    gelu_erf(t1, RH);
    gemv_range_any(p->cout, t1, p->wout, p->nclass, RH, lg, 0, p->nclass);
    float mx = -1e30f;
    for (int i = 0; i < p->nclass; i++) if (lg[i] > mx) mx = lg[i];
    float sum = 0.f;
    for (int i = 0; i < p->nclass; i++) { lg[i] = expf(lg[i] - mx); sum += lg[i]; }
    const float inv = 1.f / sum;
    int best = 0; float bv = -1e30f;
    for (int i = 0; i < p->nclass; i++) {
        const float v = lg[i] * inv + p->bal[i];
        if (v > bv) { bv = v; best = i; }
    }
    memset(out, 0, sizeof(float) * H);
    if (best >= p->nexp || best == p->iskip) return;      // skip 专家 → 输出 0
    const float wgt = lg[best] * inv;
    gemv_any(p->exp_cg[best], xn, (const uint8_t*)p->exp_gp[best], 2 * p->inter, H, gu);
    for (int i = 0; i < p->inter; i++) {
        const float gv = gu[i], uv = gu[p->inter + i];
        gu[i] = (gv / (1.f + expf(-gv))) * uv;
    }
    gemv_any(p->exp_cd[best], gu, (const uint8_t*)p->exp_dp[best], H, p->inter, out);
    for (int i = 0; i < H; i++) out[i] *= wgt;
}

// ---- ZAYA 专用 forward (带偏置的残差标定) ----
typedef struct {
    const float *input_norm, *post_norm;
    const float *a_hsw, *a_hsb, *a_rsw, *a_rsb;   // post_attention_residual_scale [H]
    const float *m_hsw, *m_hsb, *m_rsw, *m_rsb;   // post_mlp_residual_scale [H]
    m6_attn_op attn; void* attn_p;
    m6_ffn_op  ffn;  void*  ffn_p;
} M6ZayaLayer;

int m6_zaya_forward_token(float* x, int h, float eps, int pos,
                          const M6ZayaLayer* layers, int n_layer, float* scratch) {
    float* xn  = scratch;
    float* sub = scratch + h;
    float* res = sub + h;
    for (int il = 0; il < n_layer; il++) {
        const M6ZayaLayer* L = layers + il;
        memcpy(res, x, sizeof(float) * h);
        m6_rms_norm(x, L->input_norm, h, eps, xn);
        double _t0 = _now();
        L->attn(xn, L->attn_p, pos, sub);
        double _t1 = _now(); m6_prof_t[1] += _t1 - _t0; m6_prof_n[1]++;
        for (int i = 0; i < h; i++)
            x[i] = (sub[i] + L->a_hsb[i]) * L->a_hsw[i] + (res[i] + L->a_rsb[i]) * L->a_rsw[i];
        memcpy(res, x, sizeof(float) * h);
        m6_rms_norm(x, L->post_norm, h, eps, xn);
        _t0 = _now();
        L->ffn(xn, L->ffn_p, sub);
        _t1 = _now(); m6_prof_t[2] += _t1 - _t0; m6_prof_n[2]++;
        for (int i = 0; i < h; i++)
            x[i] = (sub[i] + L->m_hsb[i]) * L->m_hsw[i] + (res[i] + L->m_rsb[i]) * L->m_rsw[i];
    }
    return 0;
}
const char* m6_zaya_version(void) { return "m6_zaya/1"; }

// ═══════════════════════════════════════════════════════════════════════════════
// granite-hybrid（Mamba-1/S4D 标量衰减 × 36 + GQA 注意力 × 4 + MoE 64选6 + 共享专家）
// ═══════════════════════════════════════════════════════════════════════════════
// 语义全部来自 llama.cpp 逐行核对，且**已逐层对账通过**（hybrid/s4d_reconcile2.py：
// pos 0..3 × 全部 36 个 SSM 层，链头 cos 中位 0.99999 / 最小 0.99915，链尾 0.9995）。
//   ① in_proj(6448) 切 [z(3072) | xBC(3328) | dt(48)]（xBC 内部再切 [x(3072)|B(128)|C(128)])
//   ② conv：state(3 帧，最老在前) + 当前帧 → 4 tap 点积 + bias → silu；**只作用在 xBC 上**
//   ③ scan：dt = softplus(dt_pre + dt_bias)（**无 clamp**，与 ggml_compute_softplus_f32 同式）；
//      A 形状 [1,dt_rank] ⇒ 每头一个标量 dA = exp(dt·A)；h = dA·h + (x·dt)⊗B；y = Σ_s h·C
//      （读出的是**更新后**的状态 —— 与内核 t0 = s0*dA; t0 += B*xdt; s = t0; sum += t0*C 一致）
//   ④ y += x·ssm_d（按头，D 跳连）→ gate = silu(z)·y（swiglu_split(z, y_add_d)）
//   ⑤ grouped RMSNorm(d_inner/n_group)（ssm_norm，eps = attention.layer_norm_rms_epsilon）
//   ⑥ ssm_out 投影回 n_embd
// ★ y 的展开顺序必须是 **h 主序**（flat[h*head_dim + k]，dtype 的 ne0=head_dim 最快）。
//   写成转置（flat[k*dt_rank + h]）实测链尾 cos 从 0.9998 掉到 −0.044 —— 这种错不会崩、
//   只会静默算错，只能靠对账抓。
typedef struct {
    const uint8_t *win, *wout;     // in_proj [d_in_proj, H] / ssm_out [H, d_inner]（量化）
    int cin, cout;                 // 两者的量化类型码
    const float* conv_w;           // [XBC][d_conv]（tap 最快）
    const float* conv_b;           // [XBC]
    const float* a;                // [dt_rank]  （A={1,dt_rank} ⇒ 每头标量衰减）
    const float* d;                // [dt_rank]  （D 跳连，按头）
    const float* dt_b;             // [dt_rank]
    const float* norm;             // [d_inner/n_group]
    float* hist;                   // [(d_conv-1) * XBC]  conv 状态
    float* hst;                    // [dt_rank * head_dim * d_state]  ssm 状态
    float* work;                   // >= d_in_proj + XBC + dt_rank + d_inner
    int d_inner, d_state, dt_rank, n_group, d_conv, hidden;
    float eps;
} M6GraniteS4dP;

// work 内部布局（调用方只给一块够大的缓冲）
#define M6_SOFTPLUS(x) ((x) > 20.0f ? (x) : logf(1.0f + expf(x)))

void m6_granite_s4d_op(const float* xn, void* pv, int pos, float* out) {
    (void)pos;
    M6GraniteS4dP* p = (M6GraniteS4dP*)pv;
    const int DI = p->d_inner, DS = p->d_state, NH = p->dt_rank, CG = p->n_group, DC = p->d_conv;
    const int HD = DI / NH, XBC = DI + 2 * CG * DS, DIP = 2 * DI + 2 * CG * DS + NH;
    const int GW = DI / CG;                   // 每组的通道数（norm 的分组宽度）
    float* zall = p->work;                    // [DIP]
    float* cv   = zall + DIP;                 // [XBC] conv 输出（silu 之后）
    float* sdt  = cv + XBC;                   // [NH]
    float* gy   = sdt + NH;                   // [DI] 门控后的 y，就地做 RMSNorm

    // ① in_proj（gemv_any 内含 OMP；此处处于串行上下文）
    double _p0 = _now();
    gemv_any(p->cin, xn, p->win, DIP, p->hidden, zall);
    double _p1 = _now(); m6_prof_t[13] += _p1 - _p0; m6_prof_n[13]++;

    // ② conv：out[c] = silu(Σ_tap conv_w[c*DC+tap]·s[tap][c] + bias)，s = [hist | 当前帧]
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < XBC; c++) {
        const float* cw = p->conv_w + (size_t)c * DC;
        float acc = p->conv_b[c];
        for (int t = 0; t < DC - 1; t++) acc += cw[t] * p->hist[(size_t)t * XBC + c];
        acc += cw[DC - 1] * zall[DI + c];
        cv[c] = acc / (1.0f + expf(-acc));
    }
    // 状态移位：丢掉最老一帧，把当前帧放到最后（DC==1 时无历史，跳过）
    if (DC >= 2) {
        for (int t = 0; t < DC - 2; t++)
            memcpy(p->hist + (size_t)t * XBC, p->hist + (size_t)(t + 1) * XBC, sizeof(float) * XBC);
        memcpy(p->hist + (size_t)(DC - 2) * XBC, zall + DI, sizeof(float) * XBC);
    }

    { double _t = _now(); m6_prof_t[14] += _t - _p1; m6_prof_n[14]++; _p1 = _t; }   // conv+移位
    // ③ dt = softplus(dt_pre + dt_bias)（逐头标量）
    for (int h = 0; h < NH; h++) sdt[h] = M6_SOFTPLUS(zall[DI + XBC + h] + p->dt_b[h]);

    // ④ scan（逐头并行；每头 64×128 的状态递推）+ ⑤ D 跳连 + 门控
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) {
        const float sp = sdt[h];
        const float dA = expf(sp * p->a[h]);
        const int g = h / (NH / CG);          // 组内 repeat：B/C 每组共享
        const float* B = cv + DI + (size_t)g * DS;
        const float* C = cv + DI + (size_t)CG * DS + (size_t)g * DS;
        const float* xh = cv + (size_t)h * HD;
        const float* zh = zall + (size_t)h * HD;
        float* hs = p->hst + (size_t)h * HD * DS;
        for (int k = 0; k < HD; k++) {
            const float xdt = xh[k] * sp;
            float* hh = hs + (size_t)k * DS;
            float acc = 0.f;
            for (int s = 0; s < DS; s++) {
                const float v = hh[s] * dA + xdt * B[s];
                hh[s] = v;
                acc += v * C[s];
            }
            const float yv = acc + xh[k] * p->d[h];        // D 跳连（在 swiglu 之前）
            const float zv = zh[k];
            gy[h * HD + k] = (zv / (1.0f + expf(-zv))) * yv;
        }
    }

    { double _t = _now(); m6_prof_t[15] += _t - _p1; m6_prof_n[15]++; _p1 = _t; }   // scan+门控
    // ⑥ grouped RMSNorm（分 CG 组，每组宽度 GW，权重按组内位置共享）。
    //    ★ falcon-h1 没有 ssm_norm 张量（loader 里 TENSOR_NOT_REQUIRED，图里 if (ssm_norm) 才做）
    //    ⇒ norm==NULL 时整段跳过。缺这个 falcon 会静默算错（多乘一个无权重归一化）。
    if (!p->norm) return gemv_any(p->cout, gy, p->wout, p->hidden, DI, out);
    for (int g = 0; g < CG; g++) {
        float* seg = gy + (size_t)g * GW;
        double ss = 0.0;
        for (int i = 0; i < GW; i++) ss += (double)seg[i] * seg[i];
        const float r = 1.0f / sqrtf((float)(ss / GW) + p->eps);
        for (int i = 0; i < GW; i++) seg[i] = seg[i] * r * p->norm[i];
    }

    { double _t = _now(); m6_prof_t[16] += _t - _p1; m6_prof_n[16]++; _p1 = _t; }   // norm
    // ⑦ ssm_out
    gemv_any(p->cout, gy, p->wout, p->hidden, DI, out);
    { double _t = _now(); m6_prof_t[17] += _t - _p1; m6_prof_n[17]++; }             // ssm_out
}

// ── 一层 granite-hybrid（分支 + MoE + 共享专家 + 两处残差缩放）──
typedef struct {
    int is_attn;                        // 0 = SSM 层, 1 = 注意力层
    void* branch_p;                     // M6GraniteS4dP* 或 M6LlamaAttnP*
    const float* attn_norm;             // [H]
    const float* ffn_norm;              // [H]
    const uint64_t* exp_ptrs;           // [3*n_exp] (gate,up,down) × n_exp
    const int*      exp_codes;
    const float*    gate_inp;           // [n_exp][H]（已按行连续）
    const uint64_t* sh_ptrs;            // [3] 共享专家的 (gate,up,down)
    const int*      sh_codes;
    int n_exp, n_used, inter, sh_inter;
} M6GraniteLayer;

// 管线（granite-hybrid.cpp 的 graph + build_layer_ffn）：
//   xn = rms(x, attn_norm); br = branch(xn); x += res_scale*br;
//   xn = rms(x, ffn_norm); br = moe(xn) + shexp(xn); x += res_scale*br
// ★ 分支输出**先乘 res_scale 再加**（两处都是），这是 granite 与标准 decoder 的区别，
//   所以不能套 m6_forward_token 的固化管线（当初 ZAYA 也是因为残差标定不同才另写 forward）。
// work 需求：tmp >= 2*H+2*sh_inter；moe_scratch >= 3*n_used*inter + n_used*H
//   ★ 这个式子必须按 n_out 算，不能想当然写 4*n_used*inter：yd 那一块是 n_used*n_out（每专家一份
//   完整的 n_out 输出）。granite 的 n_out=H=1536 > inter=512，按 inter 估会少 24KB ⇒ 堆越界。
int m6_granite_forward_token(float* x, int H, float eps, int pos, float res_scale,
                             const M6GraniteLayer* layers, int n_layer,
                             float* tmp, float* moe_scratch, float* probe) {
    for (int il = 0; il < n_layer; il++) {
        const M6GraniteLayer* L = &layers[il];
        float* xn = tmp;                     // [H] 归一化后的层输入
        float* br = tmp + H;                 // [H] 分支输出
        float* sh = tmp + 2 * H;             // [2*sh_inter] 共享专家中间量
        double _a0 = _now();
        m6_rms_norm(x, L->attn_norm, H, eps, xn);
        double _a1 = _now(); m6_prof_t[8] += _a1 - _a0; m6_prof_n[8]++;
        const m6_attn_op branch = L->is_attn ? m6_granite_attn_op : m6_granite_s4d_op;
        branch(xn, L->branch_p, pos, br);
        double _a2 = _now(); m6_prof_t[9] += _a2 - _a1; m6_prof_n[9]++;
        for (int i = 0; i < H; i++) x[i] += res_scale * br[i];
        double _a3 = _now(); m6_prof_t[12] += _a3 - _a2; m6_prof_n[12]++;

        m6_rms_norm(x, L->ffn_norm, H, eps, xn);
        m6_granite_moe(xn, L->exp_ptrs, L->exp_codes, L->gate_inp,
                       L->n_exp, L->n_used, L->inter, H, H, moe_scratch, br);
        double _a4 = _now(); m6_prof_t[10] += _a4 - _a3; m6_prof_n[10]++;
        if (L->sh_inter > 0) {               // 共享专家：SwiGLU 密集 FFN，然后与 MoE 输出相加
            // ★ 三个 gemv 原来各用一次 gemv_any（整调用路径）：实测那一路对"512 行"这种小形状
            //   极慢（微基准 512x1536 Q4_K 整调用 7.1ms vs 行区间 0.47ms，15 倍），
            //   而共享专家每层 3 次 × 40 层 ⇒ 端到端差出 ~120ms/token。
            //   这里合成**一个并行区 + 行区间入口**（范式与 m6_llama_attn_op / moe_experts 一致）。
            const int NS = 8;
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for (int sp = 0; sp < NS; sp++) {
                    const int r0 = sp * L->sh_inter / NS, r1 = (sp + 1) * L->sh_inter / NS;
                    gemv_range_any(L->sh_codes[0], xn, (const uint8_t*)(uintptr_t)L->sh_ptrs[0],
                                   L->sh_inter, H, sh, r0, r1);
                    gemv_range_any(L->sh_codes[1], xn, (const uint8_t*)(uintptr_t)L->sh_ptrs[1],
                                   L->sh_inter, H, sh + L->sh_inter, r0, r1);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < L->sh_inter; i++) {
                    const float gv = sh[i];
                    sh[i] = (gv / (1.0f + expf(-gv))) * sh[L->sh_inter + i];
                }
                #pragma omp for schedule(static)
                for (int sp = 0; sp < NS; sp++)
                    gemv_range_any(L->sh_codes[2], sh, (const uint8_t*)(uintptr_t)L->sh_ptrs[2],
                                   H, L->sh_inter, xn, sp * H / NS, (sp + 1) * H / NS);
                #pragma omp for schedule(static)
                for (int i = 0; i < H; i++) br[i] += xn[i];
            }
        }
        double _a5 = _now(); m6_prof_t[11] += _a5 - _a4; m6_prof_n[11]++;   // shexp
        for (int i = 0; i < H; i++) x[i] += res_scale * br[i];
        double _a6 = _now(); m6_prof_t[12] += _a6 - _a5; m6_prof_n[12]++;   // 第二处残差
        // 对账探针（probe==NULL 时零开销）：把每层的输出留一份，便于与 dump 的 l_out-{il} 逐层比对
        if (probe) memcpy(probe + (size_t)il * H, x, sizeof(float) * H);
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Falcon-H1（每层：注意力 ∥ Mamba-2 并行相加 + 残差，再接 dense FFN；无任何 scale）
// ═══════════════════════════════════════════════════════════════════════════════
// 与 granite 的差别（都从 falcon-h1.cpp 逐行核对）：
//   · **每层两条分支并行**：x→rms(attn_norm) 同时喂 attn 与 SSM，out = attn_out + ssm_out + x
//     （granite 是二选一）。SSM 与 granite 完全同构（A={1,n_heads} 标量衰减 ⇒ 直接复用
//     m6_granite_s4d_op；0.5B 的形状是 d_inner 1536 / heads 24 / state 128 / group 1 / conv 4）。
//   · 注意力：NEOX rope + 1/sqrt(hd)（m6_falcon_attn_op）。
//   · FFN：普通 dense SwiGLU，无 bias（复用 m6_dense_op）。
//   · **没有任何 residual/embedding/logit scale**（falcon-h1.cpp 里一处 ggml_scale 都没有）。
typedef struct {
    void* attn_p;               // M6LlamaAttnP
    void* ssm_p;                // M6GraniteS4dP（语义与 granite 相同 ⇒ 结构体共用）
    const float *attn_norm, *ffn_norm;
    const uint8_t *g, *u, *d;   // dense FFN（无 bias）
    int cg, cu, cd, n_ff;
} M6FalconLayer;

int m6_falcon_forward_token(float* x, int H, float eps, int pos,
                            const M6FalconLayer* layers, int n_layer, float* tmp,
                            float* probe) {
    M6DenseP dp;
    for (int il = 0; il < n_layer; il++) {
        const M6FalconLayer* L = &layers[il];
        float* xn = tmp;          // [H]   归一化输入（attn 与 ssm 共用同一份）
        float* ao = tmp + H;      // [H]   注意力输出
        float* so = tmp + 2 * H;  // [H]   SSM 输出
        m6_rms_norm(x, L->attn_norm, H, eps, xn);
        m6_falcon_attn_op(xn, L->attn_p, pos, ao);
        m6_granite_s4d_op(xn, L->ssm_p, pos, so);
        for (int i = 0; i < H; i++) x[i] += ao[i] + so[i];

        dp.g = L->g; dp.u = L->u; dp.d = L->d;
        dp.cg = L->cg; dp.cu = L->cu; dp.cd = L->cd; dp.n_ff = L->n_ff; dp.h = H;
        m6_rms_norm(x, L->ffn_norm, H, eps, xn);
        m6_dense_op(xn, &dp, ao);
        for (int i = 0; i < H; i++) x[i] += ao[i];
        if (probe) memcpy(probe + (size_t)il * H, x, sizeof(float) * H);   // 对账探针（l_out-{il}）
    }
    return 0;
}

// qwen35moe 的共享专家：SwiGLU(shexp) × sigmoid(gate_inp_shexp·x)（标量门，granite 没有的语义）
// shp/shc = [gate, up, down, gate_inp] × 4
void m6_granite_shexp(const float* x, const uint64_t* shp, const int* shc,
                      float* sg, float* sh, float* shg, float* out, int shi, int h) {
    gemv_any(shc[3], x, (const uint8_t*)(uintptr_t)shp[3], 1, h, sg);
    gemv_any(shc[0], x, (const uint8_t*)(uintptr_t)shp[0], shi, h, sh);
    gemv_any(shc[1], x, (const uint8_t*)(uintptr_t)shp[1], shi, h, sh + shi);
    for (int i = 0; i < shi; i++) {
        const float gv = sh[i];
        sh[i] = (gv / (1.0f + expf(-gv))) * sh[shi + i];
    }
    gemv_any(shc[2], sh, (const uint8_t*)(uintptr_t)shp[2], h, shi, shg);
    // ★ 是 sigmoid 不是 silu：共享专家门 = 1/(1+e^-x)。写成 silu 时负门变负增益
    //   （症状：shexp 输出与参考精确反相关，cos −0.9994——靠逐元素 out/shg 恒为 −0.1818 抓到）。
    const float g = 1.0f / (1.0f + expf(-sg[0]));
    for (int i = 0; i < h; i++) out[i] = shg[i] * g;
}

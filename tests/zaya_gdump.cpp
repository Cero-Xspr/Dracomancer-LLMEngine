// zaya_gdump.cpp: 用 llama_context_params.cb_eval 把 zaya 图的命名中间张量 dump 到磁盘，
// 用来和 hybrid/m6_engine.c 的 oracle 逐层对账。不需要重编 llama.cpp。
//
// 编译:
//   g++ -O2 -std=c++17 zaya_gdump.cpp -I<src>/include -I<src>/ggml/include \
//       -L<build>/bin -lllama -lggml-base -o zaya_gdump
// 运行:
//   LD_LIBRARY_PATH=<build>/bin ZDUMP_POS=7 ./zaya_gdump <zaya.gguf> "2,1234,..." /tmp/zgdump
//
// ★ 关键点：**逐 token decode**（n_tokens==1），这样才能和 oracle 的 pos 逐步推进一致。
//   一次 prefill 整段会让 conv 状态/EDA 的批内语义和 oracle 不同，对账就没意义了。
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <algorithm>

static std::string OUTDIR = "/tmp/zgdump";
static int  DUMP_POS = -1;       // -1 = 最后一个 token
static int  CUR_POS  = -1;
static FILE * g_manifest = nullptr;

// ★ 图里的张量名最终由 llama_context::graph_get_cb() 那个 lambda 决定：
//   il >= 0 → ggml_format_name(cur, "%s-%d", name, il)（**会覆盖** llm_graph_context::cb 里设的名字）
//   il == -1 → ggml_set_name(cur, name)
//   所以这里按 `名-层号` / 裸名 两种形式匹配，而不是我原先以为的 L{il}.{name}。
static const char * const CB_NAMES[] = {
    // ★ Mamba/S4D 族（granite-hybrid 走 mamba-base 的图，cb 名只有这两个 —— 够对账用：
    //   "mamba2_y_add_d" = scan 读出 + D 跳连之后（我们链的 y += x·ssm_d 那一步的输出），
    //   "mamba_out"       = ssm_out 投影回 n_embd 之后（层的最终输出）。
    "mamba2_y_add_d", "mamba_out",
    "dwt", "conv_grp", "k_res", "q_final", "k_final", "cca_out",
    "attn_norm", "q_proj", "k_proj", "l_out",
    "router_down", "router_down_eda", "router_logits", "expert_idx",
    "moe_gate_up", "moe_down", "moe_weight", "cca_v", "kqv_out", "conv_intl", "conv_act",
    // ★ llama.cpp 自己给 MoE 内部起的名字（build_moe_ffn 里的 cb）：拿来当**MoE 家族的对账抓手**。
    //   加这批名字是因为实测发现：本地所有 MoE 模型都是"SSM+MoE"混合体，MoE 子块要单独验，
    //   而这些中间量（路由 logits/probs/分组 top-k/最终权重）正是逐算子对账需要的。
    "ffn_moe_logits", "ffn_moe_logits_biased", "ffn_moe_probs", "ffn_moe_probs_biased",
    "ffn_moe_group_topk", "ffn_moe_probs_masked", "ffn_moe_argsort", "ffn_moe_weights",
    "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_out", "ffn_norm", "ffn_inp",
    "ffn_moe_weights_softmax", "ffn_moe_weights_sum", "ffn_moe_topk", "ffn_moe_weighted",
    "ffn_moe_silu", "ffn_moe_down_scaled", "ffn_moe_gate_scaled", "ffn_shexp",
    // ★ falcon-h1 的图名（每层 attn∥ssm 并行）：分支级对账锚点
    "Qcur-post-rope", "Kcur-post-rope", "Vcur-post-rope", "attn_out", "ssm_in",
    "layer_out", "ffn_out",
    // ★ qwen35 的图名（19 GDN + 6 全注意力；分支级锚点 + linear_attn 内部量）
    "attn_residual", "attn_post_norm", "post_ffn", "linear_attn_qkv_mixed", "z",
    "Qcur_full", "Qcur_reshaped", "Qcur_normed", "Kcur", "Vcur", "Kcur_normed",
    "gate_reshaped", "h_nextn",
};
static const char * const BARE_NAMES[] = {
    "model.input_embed", "zaya_inp_scaled", "result_norm", "result_output",
};

static bool want(const char * n) {
    if (n == nullptr || *n == '\0') return false;
    for (const char * b : BARE_NAMES) {
        if (strcmp(n, b) == 0) return true;
    }
    // KV 缓存本体（set_rows 的节点名就是张量名）
    if (strncmp(n, "cache_k_l", 9) == 0 || strncmp(n, "cache_v_l", 9) == 0) return true;
    // 形如 <base>-<digits>
    const char * dash = strrchr(n, '-');
    if (dash == nullptr || dash == n) return false;
    for (const char * p = dash + 1; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    const size_t blen = (size_t) (dash - n);
    for (const char * b : CB_NAMES) {
        if (strlen(b) == blen && strncmp(n, b, blen) == 0) return true;
    }
    return false;
}

static bool LIST_ALL = false;
// ── 逐节点计时（ZPROF=1）：cb_eval 是逐节点执行的，于是"本次回调 - 上次回调"就是该节点的墙钟时间。
//    ★ 注意：逐节点执行本身会放大调度开销，所以绝对值不代表性；但要找**热点在哪类算子**足够。
#include <chrono>
#include <map>
static bool PROF = false;
static std::map<std::string, std::pair<double,long>> g_prof;   // op -> (秒, 次数)
static std::chrono::steady_clock::time_point g_t0;

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    GGML_UNUSED(ud);
    if (ask) {
        if (PROF) g_t0 = std::chrono::steady_clock::now();
        return true;
    }
    if (PROF) {
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
        auto & e = g_prof[ggml_op_name(t->op)];
        e.first += dt; e.second += 1;
    }
    const char * n = t->name;
    if (LIST_ALL) {
        fprintf(g_manifest, "NODE %s %s %s %lld %lld %lld %lld\n", (n && *n) ? n : "(empty)", ggml_op_name(t->op), ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        fflush(g_manifest);
    }
    if (CUR_POS != DUMP_POS) return true;
    if (!want(n)) return true;

    const size_t nb = ggml_nbytes(t);
    std::vector<char> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    static int g_seq = 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.%05d.bin", OUTDIR.c_str(), n, g_seq++);
    FILE * f = fopen(path, "wb");
    if (f == nullptr) { printf("[DUMP] 打不开 %s\n", path); return true; }
    fwrite(buf.data(), 1, nb, f);
    fclose(f);
    if (g_manifest) {
        fprintf(g_manifest, "%s %s %lld %lld %lld %lld %zu\n", n, ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], nb);
        fflush(g_manifest);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("用法: zaya_gdump <zaya.gguf> <tok1,tok2,...> [outdir]\n");
        printf("      ZDUMP_POS=<n>  只 dump 第 n 个 token 的中间量 (默认最后一个)\n");
        return 1;
    }
    if (argc > 3) OUTDIR = argv[3];
    const char * ep = getenv("ZDUMP_POS");
    if (ep) DUMP_POS = atoi(ep);
    if (getenv("ZDUMP_LIST")) LIST_ALL = true;
    if (getenv("ZPROF"))      PROF = true;

    mkdir(OUTDIR.c_str(), 0755);

    std::vector<llama_token> toks;
    {
        // ★ 容错解析：曾把 "[17, 1243, ...]"（带方括号的 python 打印格式）直接传进来，
        //   atoi("[17") 返回 0 ⇒ 首token悄悄变成 0 ⇒ 参考输出整体作废，对账全错还以为模型分歧。
        //   先删掉所有非数字非逗号字符。
        std::string clean;
        for (const char * q = argv[2]; *q; ++q)
            if ((*q >= '0' && *q <= '9') || *q == ',' || *q == '-') clean += *q;
        char * s = strdup(clean.c_str());
        for (char * p = strtok(s, ","); p != nullptr; p = strtok(nullptr, ",")) {
            toks.push_back((llama_token) atoi(p));
        }
        free(s);
    }
    if (toks.empty()) { printf("token 列表为空\n"); return 1; }
    if (DUMP_POS < 0) DUMP_POS = (int) toks.size() - 1;

    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(getenv("ZGL") ? getenv("ZGL") : "0");   // 0=CPU, 99=全部上 GPU(需 Vulkan 构建)
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) { printf("模型加载失败\n"); return 1; }

    auto cp = llama_context_default_params();
    cp.n_seq_max = (uint32_t) atoi(getenv("ZSEQ") ? getenv("ZSEQ") : "1");
    cp.n_ctx    = 512;
    cp.n_batch  = 64;
    cp.n_ubatch = 64;
    cp.flash_attn_type = (llama_flash_attn_type) atoi(getenv("ZFA") ? getenv("ZFA") : "1");
    // ★ 把 KV 缓存从默认的 f16 换成 f32：用来判定"批 vs 逐 token 的差异"是不是被
    //   **f16 缓存的四舍五入阶跃**放大的（1e-6 的差异要么不翻转、要么翻到相邻 f16 ≈ 5e-4）。
    if (getenv("ZKVTYPE")) {
        const std::string t = getenv("ZKVTYPE");
        if (t == "f32") { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    }
    cp.cb_eval  = cb_eval;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { printf("ctx 创建失败\n"); return 1; }

    {
        std::string mf = OUTDIR + "/manifest.txt";
        g_manifest = fopen(mf.c_str(), "w");
    }
    printf("[DUMP] tokens=%zu dump_pos=%d outdir=%s\n", toks.size(), DUMP_POS, OUTDIR.c_str());

    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_token one = 0;

    // ── 不变式测试模式 ──────────────────────────────────────────────
    //   ZBATCH=1 : 整段 prompt **一次** decode（n_st>1，测批内位移）
    //   ZSEQN=n  : n 个**独立序列**各 1 个 token、都在 pos 0（测按序列的状态与广播）
    //   两者都把 logits 写成 logits.bin，和"逐 token decode"的结果比。
    {
        const char * zb = getenv("ZBATCH");
        const char * zn = getenv("ZSEQN");
        if (zb || zn) {
            if (DUMP_POS >= 0) CUR_POS = DUMP_POS;   // 让 cb_eval 也把层间张量 dump 出来
            llama_batch b;
            if (zb) {
                b = llama_batch_get_one(toks.data(), (int) toks.size());
            } else {
                const int n = atoi(zn);
                b = llama_batch_init(n, 0, 1);
                for (int i = 0; i < n; ++i) {
                    b.token[i] = toks[i % (int) toks.size()];
                    b.pos[i]   = 0;
                    b.n_seq_id[i] = 1;
                    b.seq_id[i][0] = i;
                    b.logits[i] = 1;
                }
                b.n_tokens = n;   // ★ llama_batch_init 之后必须自己设 n_tokens（否则 decode 报 n_tokens == 0）
            }
            if (llama_decode(ctx, b) != 0) { printf("[TEST] decode 失败\n"); return 1; }
            const float * lg = llama_get_logits_ith(ctx, -1);
            FILE * f = fopen((OUTDIR + "/logits.bin").c_str(), "wb");
            if (f) { fwrite(lg, sizeof(float), (size_t) nv, f); fclose(f); }
            int best = 0; for (int j = 1; j < nv; ++j) { if (lg[j] > lg[best]) best = j; }
            printf("[TEST] %s -> argmax=%d\n", zb ? "ZBATCH" : "ZSEQN", best);
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 0;
        }
    }
    for (int i = 0; i < (int) toks.size(); ++i) {
        CUR_POS = i;
        one = toks[i];
        llama_batch b = llama_batch_get_one(&one, 1);
        if (llama_decode(ctx, b) != 0) { printf("decode 失败 @%d\n", i); return 1; }
        if (i == DUMP_POS) {
            const float * lg = llama_get_logits(ctx);
            FILE * f = fopen((OUTDIR + "/logits.bin").c_str(), "wb");
            if (f) { fwrite(lg, sizeof(float), (size_t) nv, f); fclose(f); }
            int best = 0;
            for (int j = 1; j < nv; ++j) { if (lg[j] > lg[best]) best = j; }
            printf("[DUMP] pos=%d argmax=%d (%s)\n", i, best,
                   llama_vocab_get_text(llama_model_get_vocab(model), best));
            // 顺手把该 token 的输入嵌入也存一份成 f32，方便和 oracle 的 deq_row 对
        }
    }
    // ZGREEDY=n: 不套模板，直接从给定 token 贪心续写 n 个 token（与 oracle 的 GEN 同口径）
    const char * ng = getenv("ZGREEDY");
    if (ng) {
        const int n_gen = atoi(ng);
        CUR_POS = -1;                       // 后续 token 不再 dump
        printf("[GREEDY] 续写 %d 个 token:\n", n_gen);
        for (int i = 0; i < n_gen; ++i) {
            const float * lg = llama_get_logits(ctx);
            int best = 0;
            for (int j = 1; j < nv; ++j) { if (lg[j] > lg[best]) best = j; }
            printf("%d ", best);
            fflush(stdout);
            one = best;
            llama_batch b2 = llama_batch_get_one(&one, 1);
            if (llama_decode(ctx, b2) != 0) { printf("\n续写 decode 失败\n"); break; }
        }
        printf("\n");
    }
    if (PROF) {
        double tot = 0;
        for (auto & kv : g_prof) tot += kv.second.first;
        std::vector<std::pair<double, std::string>> v;
        for (auto & kv : g_prof) v.push_back({kv.second.first, kv.first + " x" + std::to_string(kv.second.second)});
        std::sort(v.begin(), v.end());
        printf("[PROF] 逐节点计时（合计 %.1f ms/token，注意逐节点执行放大了调度开销）\n", tot * 1000);
        for (auto it = v.rbegin(); it != v.rend() && it != v.rbegin() + 12; ++it) {
            printf("  %-26s %8.2f ms  %5.1f%%\n", it->second.c_str(), it->first * 1000, 100.0 * it->first / tot);
        }
    }
    if (g_manifest) fclose(g_manifest);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("[DUMP] 完成\n");
    return 0;
}

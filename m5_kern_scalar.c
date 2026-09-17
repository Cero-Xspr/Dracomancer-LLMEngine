// m5_kern_scalar.c: 标量回退内核（无 AVX-512 机器的发行回退路径，E 组硬需求）
// 导出 m5_gemv / m5_gemv_range / m5_scalar_supported，签名与其它 kern 一致。
// 纯 C 逐块解量化 + 点积；性能只有 AVX-512 内核的 1/4~1/8，但让引擎在任何 x86-64 上跑对。
// 支持：Q8_0(0) Q5_K(3) Q6_K(4) Q4_K(5) F16(7) F32(9)；IQ 系返回 -1（fail loudly）。
// 块布局全部对照本仓 gguf-py/gguf/quants.py 的 dequantize_blocks（2026-09-16）。
#include <stdint.h>
#include <string.h>

static inline float f16tof32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t o;
    if (exp == 0)  o = (man == 0) ? sign : sign | (0x03800000u + (man << 13));
    else if (exp == 31) o = sign | 0x7F800000u | (man << 13);
    else o = sign | ((exp + 112u) << 23) | (man << 13);
    float f; memcpy(&f, &o, 4); return f;
}
static inline float hload(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return f16tof32(h); }
static inline float i8tof(int8_t v) { return (float)v; }

// 每行字节数
static size_t row_stride(int code, int n_in) {
    switch (code) {
        case 0: return (size_t)(n_in / 32) * 34;      // Q8_0
        case 3: return (size_t)(n_in / 256) * 176;    // Q5_K
        case 4: return (size_t)(n_in / 256) * 210;    // Q6_K
        case 5: return (size_t)(n_in / 256) * 144;    // Q4_K
        case 7: return (size_t)n_in * 2;              // F16
        case 9: return (size_t)n_in * 4;              // F32
        default: return 0;
    }
}

// Q4_K/Q5_K 共用：12 字节 scales → sc[8], mn[8]
static void k_scales(const uint8_t* s, float* sc8, float* mn8) {
    for (int i = 0; i < 4; i++) {
        sc8[i]     = (float)(s[i] & 0x3F);
        sc8[4 + i] = (float)((s[8 + i] & 0x0F) | ((s[i] >> 2) & 0x30));
        mn8[i]     = (float)(s[4 + i] & 0x3F);
        mn8[4 + i] = (float)((s[8 + i] >> 4) | ((s[4 + i] >> 2) & 0x30));
    }
}

static float dot_row(int code, const uint8_t* row, const float* x, int n_in) {
    float dot = 0.f;
    if (code == 7) {
        for (int i = 0; i < n_in; i++) dot += hload(row + i * 2) * x[i];
        return dot;
    }
    if (code == 9) {
        for (int i = 0; i < n_in; i++) { float w; memcpy(&w, row + i * 4, 4); dot += w * x[i]; }
        return dot;
    }
    if (code == 0) {                       // Q8_0: [d f16][32×i8] × n_in/32
        const int nb = n_in / 32;
        for (int b = 0; b < nb; b++) {
            const uint8_t* p = row + b * 34;
            const float d = hload(p);
            float acc = 0.f;
            for (int i = 0; i < 32; i++) acc += i8tof((int8_t)p[2 + i]) * x[b * 32 + i];
            dot += d * acc;
        }
        return dot;
    }
    if (code == 5) {                       // Q4_K: [d][dmin][sc12][qs128] × n_in/256
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* p = row + b * 144;
            const float d = hload(p), dmin = hload(p + 2);
            float sc8[8], mn8[8]; k_scales(p + 4, sc8, mn8);
            const uint8_t* q = p + 16;
            for (int j = 0; j < 8; j++) {
                const float dsc = d * sc8[j], dm = dmin * mn8[j];
                const int sel = j % 2;                     // 偶行低 nibble、奇行高 nibble
                const uint8_t* qb = q + (j / 2) * 32;      // ★ 同一组 32 字节服务 (j/2*2, +1) 两行
                float acc = 0.f;
                for (int k = 0; k < 32; k++) {
                    const float qv = (float)((sel == 0) ? (qb[k] & 0xF) : (qb[k] >> 4));
                    const float w = dsc * qv - dm;
                    acc += w * x[b * 256 + j * 32 + k];
                }
                dot += acc;
            }
        }
        return dot;
    }
    if (code == 3) {                       // Q5_K: [d][dmin][sc12][qh32][qs128] × n_in/256
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* p = row + b * 176;
            const float d = hload(p), dmin = hload(p + 2);
            float sc8[8], mn8[8]; k_scales(p + 4, sc8, mn8);
            const uint8_t* qh = p + 16;
            const uint8_t* qs = p + 48;
            for (int j = 0; j < 8; j++) {
                const float dsc = d * sc8[j], dm = dmin * mn8[j];
                const int sel = j % 2;
                const uint8_t* qb = qs + (j / 2) * 32;     // ★ 同 Q4_K：两行共享一组 32 字节
                float acc = 0.f;
                for (int k = 0; k < 32; k++) {
                    const int nib = (sel == 0) ? (qb[k] & 0xF) : (qb[k] >> 4);
                    const int hi = (qh[k] >> j) & 1;
                    const float w = dsc * (float)(nib | (hi << 4)) - dm;
                    acc += w * x[b * 256 + j * 32 + k];
                }
                dot += acc;
            }
        }
        return dot;
    }
    if (code == 4) {                       // Q6_K: [ql128][qh64][sc16 i8][d f16] × n_in/256
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* p = row + b * 210;
            const uint8_t* qlB = p;
            const uint8_t* qhB = p + 128;
            const int8_t* sc = (const int8_t*)(p + 192);
            const float d = hload(p + 208);
            for (int r = 0; r < 8; r++) {
                // ★ (n,4,2,64)→(n,16,32) 的行分解：r = g2*4 + sel*2 + half
                const int g2 = r / 4, sel = (r / 2) % 2, half = r % 2;
                const int g3 = r / 4, shl = (r % 4) * 2; // qh：(n,2,4,32)→(n,8,32)
                for (int halfg = 0; halfg < 2; halfg++) {
                    const float dsc = d * i8tof(sc[r * 2 + halfg]);
                    float acc = 0.f;
                    for (int k = 0; k < 16; k++) {
                        const int kk = halfg * 16 + k;
                        const int nib = (qlB[g2 * 64 + half * 32 + k] >> (sel * 4)) & 0xF;
                        const int b2 = (qhB[g3 * 32 + k] >> shl) & 0x3;
                        acc += (float)((nib | (b2 << 4)) - 32) * x[b * 256 + r * 32 + kk];
                    }
                    dot += dsc * acc;
                }
            }
        }
        return dot;
    }
    return 0.f;
}

int m5_scalar_supported(int code) {
    return (code == 0 || code == 3 || code == 4 || code == 5 || code == 7 || code == 9) ? 1 : 0;
}

int m5_gemv(int code, const float* x, const uint8_t* buf, int n_out, int n_in, float* y) {
    if (!m5_scalar_supported(code)) return -1;
    const size_t rs = row_stride(code, n_in);
    for (int r = 0; r < n_out; r++)
        y[r] = dot_row(code, buf + rs * (size_t)r, x, n_in);
    return 0;
}

int m5_gemv_range(int code, const float* x, const uint8_t* buf, int n_out, int n_in,
                  float* y, int o0, int o1) {
    if (!m5_scalar_supported(code)) return -1;
    const size_t rs = row_stride(code, n_in);
    for (int r = o0; r < o1; r++)
        y[r] = dot_row(code, buf + rs * (size_t)r, x, n_in);
    return 0;
}

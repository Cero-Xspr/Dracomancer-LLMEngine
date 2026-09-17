#!/usr/bin/env python3
"""SmolLM2 (标准 llama 架构) 的 Dracomancer 引擎适配 —— 通用引擎 m6_engine.so 的第二个架构。

这个文件是 README「适配新架构五步」的一次实做：
  ① 已确认 llama.cpp 支持该 arch (llama) → 有 ground truth
  ② 元数据 → 超参 (本文件顶部)
  ③ 算子: attn = m6_llama_attn_op (GQA + NEOX rope), ffn = m6_dense_op (SwiGLU) —— 都是 C 侧现成的
  ④ 描述符: 每层填 (attn_norm, ffn_norm, 两个算子指针)
  ⑤ 一次 C 调用/token
格式: 需要 Q5_0 (code 8) → m5_kern11.so (本次新增)

用法:
  python3 smol_engine.py                 # 跑 + 基准
  python3 smol_engine.py --cmp "文本"    # 与 llama.cpp 对照 token id (需先跑 llama.cpp 拿参考)
"""
import ctypes as ct, numpy as np, sys, time, os as _os
MODEL = _os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/smol/SmolLM2-360M-Instruct-Q4_K_M.gguf")

# ===================== ② 元数据 → 超参 (llama 家族) =====================
import m5sel
import gguf_fast as _gf
import gguf
# ★ 元数据解析换 gguf_fast（gguf.GGUFReader 在 llama32-1B 上 6.4s / ZAYA 上 13.6s，
#   全花在每字符串元素一次 numpy 封装；gguf_fast 0.15s / 0.29s，逐字节对账一致）。
R = _gf.FastGGUF(MODEL)
T = {t.name: t for t in R.tensors}
A = "llama"
def kv(name, default=None):
    f = R.fields.get(f"{A}.{name}")
    if f is None: return default
    v = f.contents() if callable(f.contents) else f.contents
    return v
NL    = int(kv("block_count"))
H     = int(kv("embedding_length"))
NH    = int(kv("attention.head_count"))
NKV   = int(kv("attention.head_count_kv"))
EPS   = float(kv("attention.layer_norm_rms_epsilon"))
ROT   = int(kv("rope.dimension_count"))
BASE  = float(kv("rope.freq_base"))
NFF   = int(kv("feed_forward_length"))
VOCAB = int(kv("vocab_size"))
HD    = H // NH
print(f"[CFG] llama: 层={NL} H={H} 头={NH}/{NKV} head_dim={HD} rope={ROT}@{BASE:.0f} "
      f"ffn={NFF} vocab={VOCAB} eps={EPS:.1e}", flush=True)

CODE = {'Q8_0': 0, 'IQ4_NL': 1, 'IQ3_S': 2, 'Q5_K': 3, 'Q6_K': 4, 'Q4_K': 5, 'IQ4_XS': 6,
        'F16': 7, 'Q5_0': 8}
_M5 = {0: "m5_kern6.so", 1: "m5_kern6.so", 2: "m5_kern6.so", 3: "m5_kern9.so", 4: "m5_kern8.so",
       5: "m5_kern7.so", 6: "m5_kern7.so", 7: "m5_kernF.so", 8: "m5_kern11.so"}
BASEDIR = "/media/xiao_/OverSys1/npu-direct/hybrid/"

# ★★ 必须在首次 dlopen 之前设 OMP 环境变量（libgomp 初始化时读走）。granite/falcon 都踩过
#    「默认 20 逻辑核 ⇒ 灾难性退化」的坑（llama32-1B 实测：默认 18 → OMP=8 33 t/s，1.9×）。
#    smol 引擎此前一直没接 autotune，纯靠运气（smol 模型小、每层算子大，退化不明显）。
import mcfg as _mcfg      # noqa: E402
import autotune as _at    # noqa: E402

_CFG = _mcfg.load_cfg(R, T)
_plan = _at.plan(_CFG, sum(t.n_bytes for t in R.tensors))
for _k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
    _os.environ.setdefault(_k, str(_plan[_k]))
print(f"[autotune] OMP={_os.environ['OMP_NUM_THREADS']} ({_plan.get('OMP_src', '?')}) "
      f"WAIT={_os.environ['OMP_WAIT_POLICY']} | {_plan.get('machine', '')}", flush=True)

M6E = ct.CDLL(_os.environ.get("M6_SO", BASEDIR + "m6_engine.so"))
M6E.m6_init_dl.argtypes = [ct.c_char_p] * 5; M6E.m6_init_dl.restype = ct.c_int
M6E.m6_init_extra_dl.argtypes = [ct.c_char_p]; M6E.m6_init_extra_dl.restype = ct.c_int
_rc = M6E.m6_init_dl(*( (BASEDIR + "m5/" + n).encode() for n in
                      m5sel.paths()))
assert _rc == 0, f"m6_init_dl rc={_rc}"
_rc2 = M6E.m6_init_extra_dl((BASEDIR + "m5/m5_kern11.so").encode())
assert _rc2 == 0, f"Q5_0 内核加载失败 rc={_rc2} (m5_kern11.so 存在?)"
M6E.m6_rms_norm.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int,
                            ct.c_float, ct.POINTER(ct.c_float)]
M6E.m6_engine_version.restype = ct.c_char_p
M6E.m6_forward_token.argtypes = [ct.POINTER(ct.c_float), ct.c_int, ct.c_float, ct.c_int,
                                 ct.c_void_p, ct.c_int, ct.POINTER(ct.c_float)]
M6E.m6_head_op.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int, ct.c_float,
                           ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(ct.c_float),
                           ct.POINTER(ct.c_float)]
M6E.m6_moe_batch4.argtypes = [ct.c_void_p]*5 + [ct.c_int]*4 + [ct.c_void_p]
print(f"[ENGINE] {M6E.m6_engine_version().decode()}  (Q5_0 内核 rc={_rc2})", flush=True)
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
# code → 内核库 (供基准/调试直接调用)
_M5LIB = {}
for _c, _f in _M5.items():
    if _f not in _M5LIB.values().__str__():
        pass
for _c, _f in _M5.items():
    if _c not in _M5LIB:
        _lib = ct.CDLL(BASEDIR + "m5/" + _f)
        _lib.m5_gemv.restype = ct.c_int
        _lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                                 ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
        _M5LIB[_c] = _lib

class M6LayerDesc(ct.Structure):
    _fields_ = [("attn_norm", ct.POINTER(ct.c_float)), ("ffn_norm", ct.POINTER(ct.c_float)),
                ("attn", ct.c_void_p), ("attn_p", ct.c_void_p),
                ("ffn", ct.c_void_p), ("ffn_p", ct.c_void_p)]

class M6LlamaAttnP(ct.Structure):
    _fields_ = [("wq", ct.c_void_p), ("wk", ct.c_void_p), ("wv", ct.c_void_p), ("wo", ct.c_void_p),
                ("cq", ct.c_int), ("ck", ct.c_int), ("cv", ct.c_int), ("co", ct.c_int),
                ("n_head", ct.c_int), ("n_kv", ct.c_int), ("head_dim", ct.c_int),
                ("hidden", ct.c_int), ("rot", ct.c_int),
                ("rope_base", ct.c_float),
                ("kcache", ct.POINTER(ct.c_float)), ("vcache", ct.POINTER(ct.c_float)),
                ("tlen", ct.POINTER(ct.c_int)), ("max_t", ct.c_int),
                ("work", ct.POINTER(ct.c_float))]

class M6DenseP(ct.Structure):
    _fields_ = [("g", ct.c_void_p), ("u", ct.c_void_p), ("d", ct.c_void_p),
                ("cg", ct.c_int), ("cu", ct.c_int), ("cd", ct.c_int),
                ("n_ff", ct.c_int), ("h", ct.c_int)]

ATTN = ct.cast(M6E.m6_llama_attn_op, ct.c_void_p).value
FFND = ct.cast(M6E.m6_dense_op, ct.c_void_p).value

MAXT = int(_os.environ.get("MAXT", "1024"))
KEEP = []
STATES = []     # 跨 token 状态（会话槽快照用）：每层 kc/vc/tlen
_QB = {}
def qb(name):
    if name not in _QB:
        _QB[name] = np.frombuffer(bytes(T[name].data), np.uint8)
    return _QB[name], CODE[T[name].tensor_type.name]

def fa(name):
    t = T[name]
    return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(-1)

# ===================== ④ 描述符 =====================
descs = (M6LayerDesc * NL)()
for il in range(NL):
    p = f"blk.{il}."
    # ★ 必须持有数组本体: pf(fa(...)) 只取地址, 匿名数组随即被 GC → 悬空指针 → NaN
    _an, _fn = fa(p + "attn_norm.weight"), fa(p + "ffn_norm.weight")
    KEEP.extend([_an, _fn])
    descs[il].attn_norm = pf(_an)
    descs[il].ffn_norm = pf(_fn)
    a = M6LlamaAttnP()
    a.wq, a.cq = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_q.weight"))
    a.wk, a.ck = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_k.weight"))
    a.wv, a.cv = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_v.weight"))
    a.wo, a.co = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "attn_output.weight"))
    a.n_head, a.n_kv, a.head_dim, a.hidden, a.rot = NH, NKV, HD, H, ROT
    a.rope_base = BASE
    kc = np.zeros(MAXT * NKV * HD, np.float32); vc = np.zeros(MAXT * NKV * HD, np.float32)
    tl = np.zeros(1, np.int32); # work = q[NH*HD] + k[NKV*HD] + v[NKV*HD] + sc[NH*MAXT]（★ sc 每头一行：头并行时各写各的）
    wk = np.zeros(NH * HD + 2 * NKV * HD + NH * MAXT + 64, np.float32)
    a.kcache, a.vcache = pf(kc), pf(vc)
    a.tlen = tl.ctypes.data_as(ct.POINTER(ct.c_int)); a.max_t = MAXT
    a.work = pf(wk)
    KEEP.extend([kc, vc, tl, wk, a])
    STATES.extend([kc, vc])
    descs[il].attn, descs[il].attn_p = ATTN, ct.cast(ct.pointer(a), ct.c_void_p)
    d = M6DenseP()
    d.g, d.cg = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ffn_gate.weight"))
    d.u, d.cu = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ffn_up.weight"))
    d.d, d.cd = (lambda e: (e[0].ctypes.data, e[1]))(qb(p + "ffn_down.weight"))
    d.n_ff, d.h = NFF, H
    KEEP.append(d)
    descs[il].ffn, descs[il].ffn_p = FFND, ct.cast(ct.pointer(d), ct.c_void_p)

_ONORM = fa("output_norm.weight")
_HEAD = qb("token_embd.weight") if "output.weight" not in T else qb("output.weight")
_EMB = T["token_embd.weight"]
_EMB_ROW = _EMB.n_bytes // VOCAB
_FD = open(MODEL, "rb", buffering=0)
SCRATCH = np.empty(3 * (H + 64), np.float32)
X = np.empty(H, np.float32)
LOGITS = np.empty(VOCAB, np.float32)
_HRMS = np.empty(H, np.float32)
print(f"[DESC] {NL} 层描述符就绪; head={'output' if 'output.weight' in T else '绑定词嵌入'} "
      f"({_HEAD[1]}); emb_row={_EMB_ROW}B", flush=True)


def forward(tok_id, pos):
    buf = np.frombuffer(_os.pread(_FD.fileno(), _EMB_ROW, _EMB.data_offset + tok_id * _EMB_ROW), np.uint8)
    if _EMB.tensor_type.name == "Q8_0":
        blk = buf.reshape(-1, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].copy().view(np.int8).astype(np.float32)
        X[:] = (q * d).reshape(-1)
    else:
        X[:] = np.asarray(gguf.quants.dequantize(buf, _EMB.tensor_type), np.float32).reshape(-1)
    M6E.m6_forward_token(pf(X), H, np.float32(EPS), pos, ct.cast(descs, ct.c_void_p), NL, pf(SCRATCH))
    return X


def logits_of_x():
    M6E.m6_head_op(pf(X), pf(_ONORM), H, np.float32(EPS), _HEAD[0].ctypes.data_as(ct.c_void_p),
                   _HEAD[1], VOCAB, pf(LOGITS), pf(_HRMS))
    return LOGITS


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASEDIR + "tok-smol")
    ids = tok.encode(_os.environ.get("PROMPT", "中国的首都是"))
    t0 = time.time()
    for pos, tid in enumerate(ids):
        forward(int(tid), pos)
    lg = logits_of_x()
    top5 = np.argsort(-lg)[:5].tolist()
    print(f"[PREFILL] {len(ids)} tok {time.time()-t0:.2f}s ids={ids}")
    print(f"  top5={top5} tokens={[tok.decode([i]) for i in top5]}", flush=True)
    gen = [top5[0]]; NST = int(_os.environ.get("NSTEPS", "40")); t0 = time.time(); n = 0
    for s in range(NST):
        forward(int(gen[-1]), len(ids) + s)
        nid = int(np.argmax(logits_of_x()))
        gen.append(nid); n += 1
        if s < 6 or s % 10 == 9:
            print(f"  [{s:2d}] {tok.decode([nid])!r} cum={tok.decode(gen)!r}", flush=True)
    dt = time.time() - t0
    print(f"[BENCH] {n} 步 {dt:.2f}s → {n/dt:.2f} tok/s  ({dt/n*1000:.2f} ms/步)", flush=True)
    print(f"  生成: {tok.decode(gen)!r}")

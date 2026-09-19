#!/usr/bin/env python3
"""autotune.py: 引擎参数自动调优 — 用真实内核微基准 + 机器状态自动选配置
- OMP 线程数: 跑真实 gemv 测各候选线程数, 取最优 (含超线程/大小核拓扑感知)
- PREAD 线程数: MoE 流式模型按 NVMe 队列与核数推导
- ECACHE_MB: 按 MemAvailable 与模型规模推导
- 内核选择: 按格式探测最佳 kernel (避免老旧内核拖累)
用法: import autotune; plan = autotune.plan(cfg, gguf_bytes); print(plan)
"""
import ctypes as ct, numpy as np, os, time

_SYS = "/sys/devices/system/cpu"

def topology():
    """返回 (大核数, 小核数, 物理核数) — 按 max_freq 分档, 只数物理核 (去 SMT 重复)"""
    freqs = {}
    for d in os.listdir(_SYS):
        if not d.startswith("cpu") or not d[3:].isdigit(): continue
        cid = int(d[3:])
        # 物理核: 取 core_id+socket 唯一的代表 (thread_siblings 的第一个)
        try:
            sib = open(f"{_SYS}/{d}/topology/thread_siblings_list").read().strip()
            if int(sib.split(",")[0].split("-")[0]) != cid: continue   # 只留每个物理核的首个逻辑核
        except Exception:
            pass
        p = f"{_SYS}/{d}/cpufreq/cpuinfo_max_freq"
        if os.path.exists(p):
            try: freqs[cid] = int(open(p).read().strip())
            except Exception: pass
    if not freqs: return 0, 0, os.cpu_count() or 1
    fmax = max(freqs.values())
    fast = sum(1 for f in freqs.values() if f == fmax)
    slow = len(freqs) - fast
    return fast, slow, fast + slow

def avail_mem():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) * 1024
    return 0

def load_avg():
    try: return os.getloadavg()[0]
    except Exception: return 0.0

def probe_omp(lib_path, code, n_out=6144, n_in=2048, candidates=(4, 6, 8, 10, 12), reps=40):
    """用真实 gemv (F16/Q8_0 取决于内核) 测各 OMP 线程数, 返回 (最优线程数, {t: ms})"""
    lib = ct.CDLL(lib_path)
    lib.m5_gemv.restype = ct.c_int
    lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                            ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
    rng = np.random.default_rng(0)
    # 非零随机权重 (全零会被页去重/L3 影响) + 32MB 级张量
    W = rng.integers(0, 256, size=n_out * n_in * 2, dtype=np.uint8)
    x = rng.standard_normal(n_in).astype(np.float32); y = np.zeros(n_out, np.float32)
    pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
    p8 = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_uint8))
    res = {}
    old = os.environ.get("OMP_NUM_THREADS")
    for t in candidates:
        os.environ["OMP_NUM_THREADS"] = str(t)    # ★须在任何内核初始化前调用 (libgomp 只读一次)
        for _ in range(5): lib.m5_gemv(code, pf(x), p8(W), n_out, n_in, pf(y))
        best_ms = 1e9
        for _ in range(3):                        # 3 轮取最优 (抗噪声)
            t0 = time.perf_counter()
            for _ in range(reps): lib.m5_gemv(code, pf(x), p8(W), n_out, n_in, pf(y))
            best_ms = min(best_ms, (time.perf_counter() - t0) / reps * 1e3)
        res[t] = best_ms
    if old: os.environ["OMP_NUM_THREADS"] = old
    best = min(res, key=res.get)
    return best, res

_PROBE_SRC = r"""
import os, sys, ctypes as ct, numpy as np, time, json
os.environ["OMP_NUM_THREADS"] = sys.argv[1]          # ★子进程内, 内核初始化前生效
os.environ.setdefault("OMP_WAIT_POLICY", "active")
os.environ.setdefault("GOMP_SPINCOUNT", "5000")
lib = ct.CDLL(sys.argv[2]); code = int(sys.argv[3]); n_out = int(sys.argv[4]); n_in = int(sys.argv[5])
lib.m5_gemv.restype = ct.c_int
lib.m5_gemv.argtypes = [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8),
                        ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
rng = np.random.default_rng(0)
W = rng.integers(0, 256, size=n_out*n_in*2, dtype=np.uint8)
x = rng.standard_normal(n_in).astype(np.float32); y = np.zeros(n_out, np.float32)
pf = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float)); p8 = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_uint8))
for _ in range(5): lib.m5_gemv(code, pf(x), p8(W), n_out, n_in, pf(y))
best = 1e9
for _ in range(3):
    t0 = time.perf_counter()
    for _ in range(30): lib.m5_gemv(code, pf(x), p8(W), n_out, n_in, pf(y))
    best = min(best, (time.perf_counter()-t0)/30*1e3)
print(json.dumps(best))
"""

def probe_omp_subprocess(lib_path, code=7, candidates=(4, 6, 8), n_out=6144, n_in=2048, quiet=True):
    """子进程探测各 OMP 线程数 (libgomp 只读一次 env, 必须独立进程), 返回 (最优, {t: ms})"""
    import subprocess, json, sys
    res = {}
    for t in candidates:
        try:
            r = subprocess.run([sys.executable, "-c", _PROBE_SRC, str(t), lib_path, str(code),
                                str(n_out), str(n_in)], capture_output=True, text=True, timeout=90)
            res[t] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:
            res[t] = float("nan")
    ok = {k: v for k, v in res.items() if v == v}
    return (min(ok, key=ok.get) if ok else 6), res

def plan(cfg, gguf_bytes=None, probe=True):
    """返回推荐引擎参数 dict"""
    fast, slow, ncores = topology()
    mem = avail_mem()
    la = load_avg()
    p = {'machine': f'{fast} 大核 + {slow} 小核, 可用内存 {mem/1e9:.1f}G, load {la:.1f}'}
    # OMP: 默认 4 大核 + 2 小核 (实测最优点); 有负载时进一步收敛
    omp = min(6, max(2, fast + min(2, max(0, slow - 2))))
    if la > ncores * 0.5:      # 机器已被占用, 收敛线程数避免过度竞争
        omp = max(2, omp - 2)
    p['OMP_NUM_THREADS'] = omp
    p['OMP_WAIT_POLICY'] = os.environ.get('DRACO_WAIT_POLICY', 'active')   # 能效实验口子：passive 省自旋电
    p['GOMP_SPINCOUNT'] = '5000'
    # 模型规模判定
    gb = (gguf_bytes or 0) / 1e9
    # SSM/FFN 类型影响 I/O 模式
    moe = cfg.get('ffn_kind') == 'moe' and cfg.get('n_expert', 0) > 0
    if moe:
        # MoE 流式: pread 池 + 专家 LRU
        # 池要 ≥ 每层在途请求数 (缺专家 ~3-4 个 × 3 张量 ≈ 12) 才能把随机读摊满;
        # 池太小的实测代价: 每层等待 ×3 (3 次串行 pread 落到同一线程)
        pread = min(16, max(6, ncores + 2))
        if la > ncores * 0.5: pread = min(pread, 12)     # 桌面也在跑, 别再叠线程
        p['PREAD_THREADS'] = pread
        # ECACHE: 用户态 LRU 与页缓存抢同一块内存。模型装不下时大缓存反而害事
        # (匿名页被换出 → 交换 I/O 抢盘, 实测 ECACHE=6G 中位 2.86 vs 2.6G 的 3.71)
        reserve = 1.5e9                              # 桌面喘息
        budget = max(0.0, mem - reserve)
        if budget < gb * 1e9:                        # 模型放不下 → 页缓存才是主缓存, LRU 保持小
            p['ECACHE_MB'] = int(min(budget * 0.3, 3000e6) / 1e6)
        else:
            p['ECACHE_MB'] = int(min(budget * 0.6, 6000e6) / 1e6)
        p['SLAB'] = '建议 gguf_repack.py 生成 slab 边车 (每专家单次连续 pread)'
    else:
        p['PREAD_THREADS'] = 2                       # dense 只读 embedding 行
        p['ECACHE_MB'] = 0
    # GPU 卸载建议: dense 且常驻 (模型 < 可用内存 60%) → GPU 有利 (批量提交后实测反超)
    resident = gb * 1e9 < (mem - 1.5e9) * 0.9
    p['GPU_DENSE'] = '1 (dense 常驻; 批量提交 hd_gemv_multi)' if (not moe and resident) else '0'
    p['WARMUP'] = '1 (首步加载不计入基准)'
    if probe:
        lib = "/media/xiao_/OverSys1/npu-direct/hybrid/m5/m5_kernF.so"
        best, res = probe_omp_subprocess(lib, candidates=tuple(sorted({max(2, omp-2), omp, omp+2})))
        p['probe_omp_ms'] = {k: round(v, 3) for k, v in res.items()}
        # 仅在显著更优 (>20%) 时覆盖规则值 — 机器有桌面负载时探测噪声可达 ±15%
        if res.get(best, 1e9) < res.get(omp, 1e9) * 0.8:
            p['OMP_NUM_THREADS'] = best
            p['OMP_src'] = f'probe 覆盖 (快 {res[omp]/res[best]:.2f}×)'
        else:
            p['OMP_src'] = '规则值 (探测差异不显著)'
    return p

if __name__ == '__main__':
    import sys, gguf
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mcfg import load_cfg, describe
    print(f"拓扑: {topology()}")
    if len(sys.argv) > 1:
        r = gguf.GGUFReader(sys.argv[1]); tm = {t.name: t for t in r.tensors}
        cfg = load_cfg(r, tm); describe(cfg)
        gb = sum(t.n_bytes for t in r.tensors)
        print(f"模型 {gb/1e9:.1f}GB")
        import json
        print(json.dumps(plan(cfg, gb), ensure_ascii=False, indent=1))

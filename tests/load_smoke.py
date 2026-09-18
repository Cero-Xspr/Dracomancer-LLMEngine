#!/usr/bin/env python3
"""8 家族装载冒烟：每个家族经 draco_engine_server.load_engine 装载 + prefill + 4 个贪心 token。
目的：把"打不开"这类事故（snap_state 误插/NameError/路径错）固化进套件。
用法：load_smoke.py [--only eng1,eng2]；环境变量 MODEL/ENG 由 run.py 传。"""
import os, sys, time

CASES = [
    ("smol",      "/media/xiao_/OverSys1/gguf/smol/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    ("falcon",    "/media/xiao_/OverSys1/gguf/falcon-h1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf"),
    ("llama",     "/media/xiao_/OverSys1/gguf/llama32/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("granite",   "/media/xiao_/OverSys1/gguf/granite/granite-h-tiny-Q4_K_M.gguf"),
    ("qwen35",    "/media/xiao_/OverSys1/gguf/Qwen3.5-2B-f16.gguf"),
    ("zaya",      "/media/xiao_/OverSys1/gguf/zaya1/ZAYA1-8B-Q4_K_M.gguf"),
    ("ling",      "/media/xiao_/OverSys1/gguf/ling/Ling-3.0-tiny-Q4_K_M.gguf"),
    ("qwen35moe", "/media/xiao_/OverSys1/gguf/Qwen3.6-35B-A3B-REAP-48-v2.gguf"),
]

def run_one(eng, path):
    os.environ["OMP_NUM_THREADS"] = "4"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import argparse
    for m in [m for m in list(sys.modules) if m.startswith(("draco_engine", "qwen35_", "ling_", "granite_", "falcon_", "smol_", "zaya_", "ling_proto", "granite_proto"))]:
        del sys.modules[m]
    import draco_engine_server as S
    S.ARGS = argparse.Namespace(model=path, port=1, engine=eng, threads=4, ctx=0)
    S.load_engine()
    ids = S.AP["encode"]("你好")
    for p, t in enumerate(ids):
        S.AP["forward"](int(t), p)
    for i in range(4):
        nid = int(S.AP["logits"]().argmax())
        assert nid == nid and 0 <= nid < 10**9, f"{eng}: argmax 异常"
        S.AP["forward"](nid, len(ids) + i)
    return True

def main():
    only = os.environ.get("ONLY", "")
    bad = []
    for eng, path in CASES:
        if only and eng not in only.split(","):
            continue
        if not os.path.exists(path):
            print(f"  ⚠ {eng}: 模型缺失 {path}（跳过，不算失败）")
            continue
        t0 = time.time()
        try:
            run_one(eng, path)
            print(f"  ✅ {eng:10s} {time.time()-t0:5.1f}s")
        except Exception as e:
            bad.append(eng)
            print(f"  ❌ {eng:10s} {type(e).__name__}: {e}")
    if bad:
        return False, f"装载失败：{','.join(bad)}"
    return True, "全部通过"

if __name__ == "__main__":
    ok, msg = main()
    sys.exit(0 if ok else 1)

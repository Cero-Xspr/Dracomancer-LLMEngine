#!/usr/bin/env python3
"""Falcon-H1 的 T2 闸门（自包含）：确保夹具 dump 存在（缺则现场生成）→ 跑引擎带逐位置探针
→ 与 dump 的 l_out-{il} 逐 (pos,层) 比 cos，判据取自金标准阈值。

背景（2026-09-15 对账时踩过的坑，别删）：
  · 夹具的 ids 参数**不能带方括号**（atoi("[17")=0 ⇒ 参考整个跑在错误的 prompt 上，
    对账"全错"其实是 prompt 不同）。解析已加固，但生成 dump 的调用方也要注意。
  · falcon 图把 build_mamba2 的输出重 cb 成 "ssm_out"（覆盖内部名 mamba_out）。
  · 探针必须**逐位置**记录（只存最后一次 forward 会拿 pos3 的探针比 pos0 的 dump，全红假象）。
"""
import ctypes as ct
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/gguf-py")

MODEL = os.environ.get("MODEL", "/media/xiao_/OverSys1/gguf/falcon-h1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf")
LDP = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"
FIXTURE = os.environ.get("FALCON_FIXTURE", "/media/xiao_/OverSys1/npu-direct/hybrid/zaya_gdump")
PROMPT = [17, 1243, 7889, 813, 10513, 860]      # "<|begin_of_text|>The capital of France is"
P = int(os.environ.get("FP", "3"))
GOLDEN = os.path.join(HERE, "golden", "falcon_layers.json")


def ensure_dumps():
    for p in range(P + 1):
        d = f"/tmp/frec{p}"
        if glob.glob(d + "/l_out-0.*.bin"):
            continue
        os.makedirs(d, exist_ok=True)
        env = {**os.environ, "LD_LIBRARY_PATH": LDP, "ZDUMP_POS": str(p)}
        r = subprocess.run([FIXTURE, MODEL, ",".join(map(str, PROMPT)), d],
                           env=env, capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0, f"夹具失败 pos{p}: {r.stderr[-200:]}"


def dump(p, name, il):
    hits = glob.glob(f"/tmp/frec{p}/{name}-{il}.*.bin")
    assert len(hits) == 1, (name, il, len(hits))
    return np.frombuffer(open(hits[0], "rb").read(), np.float32)


def main():
    ensure_dumps()
    os.environ["GPROBE"] = "1"
    import falcon_engine as F
    for pos in range(P + 1):
        F.forward(PROMPT[pos], pos)
    gold = json.load(open(GOLDEN)) if os.path.isfile(GOLDEN) else {"min_cos_threshold": 0.996}
    thr = float(gold["min_cos_threshold"])
    worst = (1.0, None)
    bad = []
    for pos in range(P + 1):
        mine = F.PROBE["by_pos"][pos]
        for il in range(F.NL):
            c = float(np.dot(mine[il] / np.linalg.norm(mine[il]),
                             dump(pos, "l_out", il) / np.linalg.norm(dump(pos, "l_out", il))))
            if c < worst[0]:
                worst = (c, (pos, il))
            if c < thr:
                bad.append((pos, il, c))
    if bad:
        return False, (f"{len(bad)} 个 (pos,层) 低于阈值 {thr}；最差 {worst[1]} cos={worst[0]:.5f}；"
                       f"（若普遍略降：先查夹具 ids 是否带方括号、探针是否逐位置）")
    return True, (f"pos 0..{P} × {F.NL} 层全部 ≥ {thr}；最差 cos={worst[0]:.5f}（{worst[1]}）")


if __name__ == "__main__":
    ok, msg = main()
    print(("✅ " if ok else "❌ ") + msg)
    sys.exit(0 if ok else 1)

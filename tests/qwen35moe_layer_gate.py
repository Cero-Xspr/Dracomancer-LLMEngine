#!/usr/bin/env python3
"""qwen35moe 的 T2 闸门（自包含）：确保夹具 dump 存在（缺则现场生成）→ 引擎带逐位置探针
→ 与 dump 的 l_out-{il} 逐 (pos,层) 比 cos，判据取自金标准阈值（0.99，IQ3_S+Q8_K 预期差异）。"""
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

MODEL = os.environ.get("QWEN35MOE_MODEL",
                       "/media/xiao_/OverSys1/gguf/Qwen3.6-35B-A3B-REAP-48-v2.gguf")
LDP = "/media/xiao_/OverSys1/npu-direct/llama.cpp-b10819/build-dbg/bin"
FIXTURE = os.environ.get("QWEN_FIXTURE", os.path.join(HERE, "zaya_gdump"))
PROMPT = [760, 6511, 314, 9338, 369]
P = int(os.environ.get("QMP", "3"))
GOLDEN = os.path.join(HERE, "golden", "qwen35moe_layers.json")


def ensure_dumps():
    for p in range(P + 1):
        d = f"/tmp/qmrec{p}"
        if glob.glob(d + "/l_out-0.*.bin"):
            continue
        os.makedirs(d, exist_ok=True)
        env = {**os.environ, "LD_LIBRARY_PATH": LDP, "ZDUMP_POS": str(p)}
        r = subprocess.run([FIXTURE, MODEL, ",".join(map(str, PROMPT)), d],
                           env=env, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, f"夹具失败 pos{p}: {r.stderr[-200:]}"


def dump(pos, il):
    hits = sorted(glob.glob(f"/tmp/qmrec{pos}/l_out-{il}.*.bin"))
    assert len(hits) == 1, (il, len(hits))
    return np.frombuffer(open(hits[0], "rb").read(), np.float32)


def main():
    ensure_dumps()
    os.environ["GPROBE"] = "1"
    os.environ["MODEL"] = MODEL
    import qwen35_engine as Q
    for pos in range(P + 1):
        Q.forward(PROMPT[pos], pos)
    gold = json.load(open(GOLDEN)) if os.path.isfile(GOLDEN) else {"min_cos_threshold": 0.99}
    thr = float(gold["min_cos_threshold"])
    cos = lambda a, b: float(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)))
    worst = (1.0, None)
    bad = []
    for pos in range(P + 1):
        mine = Q.PROBE["by_pos"][pos]
        for il in range(Q.NL):
            c = cos(mine[il], dump(pos, il))
            if c < worst[0]:
                worst = (c, (pos, il))
            if c < thr:
                bad.append((pos, il, round(c, 5)))
    if bad:
        return False, (f"{len(bad)} 个 (pos,层) 低于阈值 {thr}；最差 {worst[1]} cos={worst[0]:.5f}")
    return True, f"pos 0..{P} × {Q.NL} 层全部 ≥ {thr}；最差 cos={worst[0]:.5f}（{worst[1]}）"


if __name__ == "__main__":
    ok, msg = main()
    print(("✅ " if ok else "❌ ") + msg)
    sys.exit(0 if ok else 1)

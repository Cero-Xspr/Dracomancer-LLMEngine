#!/usr/bin/env python3
"""CPU 批量 prefill 分相解剖：定位 T=170 时 ~500ms/tok 的病灶。"""
import os, sys, time
os.environ.setdefault("K2_VK", "1")   # 引擎照常装载（gemm 走 CPU，K2_PF_GPU 默认关）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_engine as KE
import k2_prefill as KP
import falcon_tok

pf = KP.Prefiller(KE)
SCOPE = ["pf"]
T = {"gemm": 0.0, "mova": 0.0, "moe": 0.0}
_gemm_calls = []   # (dt, n_out, T)

_gw = pf.gemm_w
def gw(*a, **k):
    t0 = time.perf_counter()
    r = _gw(*a, **k)
    dt = time.perf_counter() - t0
    T["gemm"] += dt
    if SCOPE[0] == "mova": T["mova_gemm"] = T.get("mova_gemm", 0) + dt
    if SCOPE[0] == "moe": T["moe_gemm"] = T.get("moe_gemm", 0) + dt
    _gemm_calls.append((dt, a[3] if len(a) > 3 else -1, a[5] if len(a) > 5 else -1))
    return r
pf.gemm_w = gw

def wrap(name, orig):
    def w(*a, **k):
        old = SCOPE[0]; SCOPE[0] = name
        t0 = time.perf_counter()
        r = orig(*a, **k)
        T[name] = T.get(name, 0.0) + (time.perf_counter() - t0)
        SCOPE[0] = old
        return r
    return w
pf.mova_v = wrap("mova", pf.mova_v)
pf.moe_chunk = wrap("moe", pf.moe_chunk)

# 真实长 prompt（模板+长材料 ≈ 170 tok）
tk, _ = falcon_tok.build(KE.MODEL, add_bos=False, pre="llama3")
import re as _re
from jinja2.sandbox import ImmutableSandboxedEnvironment
src = open(os.path.join(KE.BASE, "tests/k2ref/chat_template.jinja")).read()
src = _re.sub(r"\{%-?\s*endgeneration\s*-?%\}", "", _re.sub(r"\{%-?\s*generation\s*-?%\}", "", src))
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
P = ("请阅读：\n" + "大语言模型推理分预填充与解码两阶段。预填充是算力受限的矩阵乘，"
     "解码是访存受限的矩阵向量乘。量化降低权重体积，让同样的带宽支撑更大的模型。" * 8) + "\n问题：一句话总结。"
ids = tk.encode(env.from_string(src).render(messages=[{"role": "user", "content": P}],
                  bos_token="<|ifm|begin_of_text|>", eos_token="<|ifm|endoftext|>",
                  add_generation_prompt=True)).ids
print(f"T = {len(ids)}", flush=True)

t0 = time.perf_counter()
lg, _x = pf.prefill(ids, pos0=0)
tot = time.perf_counter() - t0
print(f"\nprefill total {tot:.2f}s  ({tot/len(ids)*1000:.0f} ms/tok)", flush=True)
print(f"  gemm_w 累计   {T.get('gemm',0):.2f}s ({T.get('gemm',0)/tot*100:.0f}%)", flush=True)
print(f"  mova_v 毛时   {T.get('mova',0):.2f}s  (其内 gemm {T.get('mova_gemm',0):.2f}s)", flush=True)
print(f"  moe_chunk 毛时 {T.get('moe',0):.2f}s (其内 gemm {T.get('moe_gemm',0):.2f}s)", flush=True)
nongemm = tot - T.get("mova",0) - (T.get("moe",0) - 0)  # mova/moe 毛时里 gemm 已计一次
# 直接报：attention+rms+embed+head+glue ≈ total - mova_gross - moe_gross + (mova_gemm+moe_gemm 双计回补) - prefill级gemm
pf_gemm = T.get("gemm",0) - T.get("mova_gemm",0) - T.get("moe_gemm",0)
glue = tot - (T.get("mova",0) - 0) - T.get("moe",0) - pf_gemm  # mova/moe 毛-内含
print(f"  attention/rms/embed/head 等 ≈ {glue:.2f}s", flush=True)
_gemm_calls.sort(reverse=True)
print("\n最慢 8 次 gemm_w (ms, n_out, T):", flush=True)
for dt, no, tt in _gemm_calls[:8]:
    print(f"   {dt*1e3:7.1f}  n_out={no:5d} T={tt}", flush=True)
print(f"gemm_w 调用数 {len(_gemm_calls)}", flush=True)

import os
"""m5sel —— 内核库选择：有 AVX-512 用手写内核，否则全部指向标量回退。
DRACO_FORCE_SCALAR=1 可在本机强制标量（验证用）。"""

def scalar():
    try:
        return os.environ.get("DRACO_FORCE_SCALAR") == "1" or \
            "avx512f" not in open("/proc/cpuinfo").read()
    except OSError:
        return False

def paths():
    if scalar():
        return ("m5_kern_scalar.so",) * 5
    return ("m5_kern6.so", "m5_kern9.so", "m5_kern8.so",
            "m5_kern7.so", "m5_kernF.so")

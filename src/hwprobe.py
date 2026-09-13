#!/usr/bin/env python3
"""hwprobe —— 按**名字+label**找传感器，绝不按 hwmon 编号。

★ 为什么单独一个模块：我（和旧脚本 `measure_jtok.py`）一开始把 APU 封装功率写成
  `/sys/class/hwmon/hwmon12/power1_average`。2026-09-13 复查发现它已经跑到 **hwmon11** ——
  hwmon 的编号由内核枚举顺序决定，**会随驱动/启动变化**（本机既出现过 hwmon12 也出现过
  hwmon11）。硬编码索引的后果不是报错，而是**静默读错传感器或读空**，能耗结论会悄悄错。
  所以：一律按目录里的 `name`（如 amdgpu）+ 通道的 `*_label`（如 PPT）定位。

用法：
    from hwprobe import find_power, PowerSampler
    p = find_power("PPT")            # → Path 或 None
    with PowerSampler("PPT") as s:   # 采样线程
        ...跑负载...
    print(s.mean_w)                  # 窗口内平均瓦特
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

HWMON = Path("/sys/class/hwmon")


def _iter_channels():
    """遍历所有 hwmon 下的 power*/energy* 通道，产出 (chip_name, chan_name, label, path)。"""
    if not HWMON.is_dir():
        return
    for h in sorted(HWMON.glob("hwmon*")):
        try:
            chip = (h / "name").read_text().strip()
        except OSError:
            chip = "?"
        for chan in sorted(h.glob("power*_average")) + sorted(h.glob("power*_input")):
            lab = ""
            labf = chan.with_name(chan.name.rsplit("_", 1)[0] + "_label")
            try:
                lab = labf.read_text().strip()
            except OSError:
                pass
            yield chip, chan.name, lab, chan


def list_channels():
    """给排障用：列出本机所有功率通道（编号+芯片+label）。"""
    return [(str(p), c, n, l) for c, n, l, p in _iter_channels()]


def find_power(label: str, chip: str | None = None):
    """按 label（大小写不敏感）找功率通道；多个候选时优先 *_average。

    ★ 优先 average：`_input` 是瞬时值（噪声大），`_average` 是驱动给的平滑值。
    """
    cands = []
    for c, n, l, p in _iter_channels():
        if l.lower() != label.lower():
            continue
        if chip and c != chip:
            continue
        cands.append((n.endswith("_average"), c, str(p)))
    if not cands:
        return None
    cands.sort(key=lambda t: (not t[0], t[1]))
    return Path(cands[0][2])


def find_npu_power():
    """NPU IP 功率（label=NPU_power，芯片 amdxdna）。即使 NPU 空闲也返回路径。"""
    for c, n, l, p in _iter_channels():
        if l == "NPU_power":
            return Path(p)
    return None


class PowerSampler:
    """在后台线程里按固定间隔采功率，退出时给出窗口内的平均瓦特。

    用法：`with PowerSampler("PPT") as s: run_load()`，然后 `s.mean_w` / `s.samples`。
    找不到传感器时不抛异常，`mean_w` 为 None（让调用方决定是放弃还是继续测 tok/s）。
    """

    def __init__(self, label="PPT", interval=0.02, path=None):
        self.path = Path(path) if path else find_power(label)
        self.interval = interval
        self.vals: list[float] = []
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        if self.path is None or not self.path.exists():
            return self
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.vals.append(int(self.path.read_text()) / 1e6)   # µW → W
            except (OSError, ValueError):
                pass
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)
        return False

    @property
    def samples(self) -> int:
        return len(self.vals)

    @property
    def mean_w(self):
        return sum(self.vals) / len(self.vals) if self.vals else None

    @property
    def median_w(self):
        if not self.vals:
            return None
        v = sorted(self.vals)
        return v[len(v) // 2]

    def summary(self):
        if not self.vals:
            return "无样本（找不到传感器或采样失败）"
        return (f"mean={self.mean_w:.1f}W median={self.median_w:.1f}W "
                f"min={min(self.vals):.1f} max={max(self.vals):.1f} n={self.samples}")


if __name__ == "__main__":
    print("本机功率通道（编号会变，所以只用来排障）：")
    for p, chip, chan, lab in list_channels():
        print(f"  {p:52s} chip={chip:10s} {chan:18s} label={lab}")
    ppt = find_power("PPT")
    npu = find_npu_power()
    print(f"\n找到 APU 封装功率(PPT) → {ppt}")
    print(f"找到 NPU IP 功率       → {npu}")
    if ppt:
        with PowerSampler("PPT") as s:
            time.sleep(1.0)
        print("1 秒空闲采样：", s.summary())

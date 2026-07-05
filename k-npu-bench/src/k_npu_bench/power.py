from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Sample:
    ts: float
    power_w: float | None = None
    memory_mb: float | None = None


@dataclass
class DeviceStats:
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    peak_memory_mb: float = 0.0
    energy_j: float = 0.0
    samples: int = 0


@dataclass
class NvidiaSmiSampler:
    interval_s: float = 0.2
    index: int = 0
    samples: list[Sample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> "NvidiaSmiSampler":
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> DeviceStats:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.stats()

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._read_once()
            if sample:
                self.samples.append(sample)
            time.sleep(self.interval_s)

    def _read_once(self) -> Sample | None:
        cmd = [
            "nvidia-smi",
            f"--id={self.index}",
            "--query-gpu=power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2)
            first = proc.stdout.strip().splitlines()[0]
            power, memory = [x.strip() for x in first.split(",", maxsplit=1)]
            return Sample(ts=time.time(), power_w=float(power), memory_mb=float(memory))
        except Exception:
            return None

    def stats(self) -> DeviceStats:
        if not self.samples:
            return DeviceStats()
        powers = [s.power_w for s in self.samples if s.power_w is not None]
        memories = [s.memory_mb for s in self.samples if s.memory_mb is not None]
        avg_power = sum(powers) / len(powers) if powers else 0.0
        peak_power = max(powers) if powers else 0.0
        peak_memory = max(memories) if memories else 0.0
        duration = max(0.0, self.samples[-1].ts - self.samples[0].ts)
        return DeviceStats(
            avg_power_w=avg_power,
            peak_power_w=peak_power,
            peak_memory_mb=peak_memory,
            energy_j=avg_power * duration,
            samples=len(self.samples),
        )


class NullSampler:
    def start(self) -> "NullSampler":
        return self

    def stop(self) -> DeviceStats:
        return DeviceStats()


def make_sampler(kind: str, interval_s: float = 0.2, index: int = 0):
    if kind == "nvidia-smi":
        return NvidiaSmiSampler(interval_s=interval_s, index=index)
    if kind == "none":
        return NullSampler()
    raise ValueError(f"Unsupported sampler: {kind}")


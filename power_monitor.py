"""
power_monitor.py
================
GPU Power Monitoring utility using nvidia-smi.

Runs in a background thread and samples power usage at a fixed interval.
Provides statistics like mean, peak, and total energy (if integrated).
Used to benchmark the energy efficiency of LLM training and inference.
"""

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PowerStats:
    mean_w: float
    peak_w: float
    min_w: float
    samples: int
    duration_s: float

    @property
    def energy_joules(self) -> float:
        return self.mean_w * self.duration_s


class PowerMonitor:
    """
    Background thread that samples GPU power draw via nvidia-smi.
    """

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self._readings: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

    def _sample_power(self) -> float:
        """
        Query nvidia-smi for current power draw.
        Returns power in Watts.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            return 0.0

    def _run(self):
        """
        Sampling loop.
        """
        while not self._stop_event.is_set():
            reading = self._sample_power()
            if reading > 0:
                self._readings.append(reading)
            self._stop_event.wait(timeout=self.sample_interval_ms / 1000.0)

    def start(self):
        """
        Start the background monitoring thread.
        """
        self._readings = []
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> PowerStats:
        """
        Stop monitoring and return statistics.
        """
        if self._thread is None:
            return PowerStats(0, 0, 0, 0, 0)

        self._stop_event.set()
        self._thread.join()
        duration = time.perf_counter() - self._start_time

        if not self._readings:
            return PowerStats(
                mean_w=0.0, peak_w=0.0, min_w=0.0, samples=0, duration_s=duration
            )

        mean_w = sum(self._readings) / len(self._readings)
        peak_w = max(self._readings)
        min_w = min(self._readings)

        return PowerStats(
            mean_w=mean_w,
            peak_w=peak_w,
            min_w=min_w,
            samples=len(self._readings),
            duration_s=duration,
        )

    @staticmethod
    def is_available() -> bool:
        """
        Check if nvidia-smi is available and working.
        """
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False


if __name__ == "__main__":
    if not PowerMonitor.is_available():
        print("nvidia-smi not found. Power monitoring disabled.")
    else:
        monitor = PowerMonitor(sample_interval_ms=500)
        print("Monitoring power for 5 seconds...")
        monitor.start()
        time.sleep(5)
        stats = monitor.stop()
        print(f"Stats: {stats}")
        print(f"Energy: {stats.energy_joules:.2f} J")

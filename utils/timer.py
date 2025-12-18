"""
Simple timer utility for profiling and ETA estimation.
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict

class PipelineTimer:
    """Tracks elapsed time and estimates ETA."""

    def __init__(self, total_steps: int = 0, name: str = "Process"):
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.total_steps = total_steps
        self.current_step = 0
        self.name = name
        self.step_times = []

    def start(self):
        """Reset timer start."""
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.current_step = 0
        self.step_times = []

    def step(self) -> str:
        """
        Record step completion and return timing string.
        """
        now = time.time()
        duration = now - self.last_step_time
        self.step_times.append(duration)
        self.last_step_time = now
        self.current_step += 1

        elapsed = now - self.start_time
        avg_step_time = elapsed / self.current_step if self.current_step > 0 else 0
        
        if self.total_steps > 0:
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_step_time
            eta_str = self._format_time(eta_seconds)
            progress = (self.current_step / self.total_steps) * 100
            return f"[{progress:.1f}%] Step: {self._format_time(duration)} | Elapsed: {self._format_time(elapsed)} | ETA: {eta_str}"
        else:
            return f"Step: {self._format_time(duration)} | Elapsed: {self._format_time(elapsed)}"

    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def save_stats(self, output_path: str):
        """Save timing statistics to JSON."""
        stats = {
            "name": self.name,
            "total_time": time.time() - self.start_time,
            "total_steps": self.current_step,
            "avg_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            "step_times": self.step_times
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

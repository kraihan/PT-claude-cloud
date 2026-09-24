"""Opt-in training measurements and CUDA optimizer selection."""
from contextlib import contextmanager
import time

import torch


class StepTimer:
    """CUDA events measure queued GPU work; disabled timers never synchronize."""

    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.records = []
        self.previous = self._stamp() if self.enabled else None

    def _stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def mark(self, name):
        if self.enabled:
            end = self._stamp()
            self.records.append((name, self.previous, end))
            self.previous = end

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()
        with torch.profiler.record_function(name):
            yield
        if self.device.type == "cuda":
            end.record()
        else:
            end = time.perf_counter()
        self.records.append((name, start, end))

    def metrics(self):
        if not self.records:
            return {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = {}
        for name, start, end in self.records:
            elapsed = start.elapsed_time(end) / 1000 if self.device.type == "cuda" else end - start
            key = f"profile/{name}_s"
            result[key] = result.get(key, 0.0) + elapsed
        return result


def adamw(params, *, fused="auto", **kwargs):
    """Keep FP32 master weights; use fused CUDA AdamW when explicitly enabled."""
    params = list(params)
    if fused not in ("auto", True, False):
        raise ValueError("optimizer.fused must be auto, true or false")
    use_fused = bool(params) and all(p.device.type == "cuda" for p in params) and fused is not False
    return torch.optim.AdamW(params, fused=use_fused, **kwargs)

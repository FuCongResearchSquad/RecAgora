from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

import torch


class AsyncDeviceTimer:
    """A CPU wall timer or a non-blocking CUDA-event interval."""

    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self._cpu_started_at: float | None = None
        self._cpu_elapsed_ms: float | None = None
        self._start_event: Any | None = None
        self._end_event: Any | None = None
        if not self.enabled:
            return
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_started_at = time.perf_counter()

    def stop(self) -> "AsyncDeviceTimer":
        if not self.enabled:
            return self
        if self.device.type == "cuda":
            if self._end_event is not None:
                self._end_event.record()
        elif self._cpu_started_at is not None:
            self._cpu_elapsed_ms = (time.perf_counter() - self._cpu_started_at) * 1000.0
        return self

    def ready(self) -> bool:
        if not self.enabled:
            return True
        if self.device.type != "cuda":
            return self._cpu_elapsed_ms is not None
        return self._end_event is not None and bool(self._end_event.query())

    def elapsed_ms(self) -> float:
        assert self.ready()
        if not self.enabled:
            return 0.0
        if self.device.type == "cuda":
            assert self._start_event is not None and self._end_event is not None
            return float(self._start_event.elapsed_time(self._end_event))
        assert self._cpu_elapsed_ms is not None
        return float(self._cpu_elapsed_ms)


@dataclass
class PendingTimingTrace:
    step: int
    rank: int
    cpu_ms: Mapping[str, float]
    device_timers: Mapping[str, AsyncDeviceTimer]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def ready(self) -> bool:
        return all(timer.ready() for timer in self.device_timers.values())

    def resolve(self) -> dict[str, Any]:
        assert self.ready()
        row: dict[str, Any] = {
            "step": int(self.step),
            "rank": int(self.rank),
            "scope": f"rank{self.rank}_local",
            "distributed_sync": False,
        }
        row.update({name: float(value) for name, value in self.cpu_ms.items()})
        row.update(
            {
                name: timer.elapsed_ms()
                for name, timer in self.device_timers.items()
            }
        )
        forward_ms = float(row.get("forward_ms", 0.0))
        lookup_ms = float(row.get("embedding_lookup_ms", 0.0))
        row["model_core_ms"] = max(0.0, forward_ms - lookup_ms)
        device_total_ms = float(row.get("device_step_ms", 0.0))
        row["step_total_ms"] = (
            float(row.get("fetch_batch_ms", 0.0))
            + float(row.get("prepare_batch_ms", 0.0))
            + device_total_ms
            + float(row.get("global_metric_comm_ms", 0.0))
        )
        row.update(self.metadata)
        return row


class RankLocalTimingTracer:
    """Own delayed timing traces for exactly one rank without collectives."""

    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        target_rank: int,
        every_steps: int,
        resolve_delay_steps: int,
    ) -> None:
        self.rank = int(rank)
        self.target_rank = int(target_rank)
        self.every_steps = int(every_steps)
        self.resolve_delay_steps = int(resolve_delay_steps)
        self.active = bool(enabled) and self.rank == self.target_rank
        self._pending: list[PendingTimingTrace] = []
        self._noop_timer = AsyncDeviceTimer(torch.device("cpu"), enabled=False)

    def should_record(self, step: int) -> bool:
        return self.active and step > 0 and step % self.every_steps == 0

    def new_timer(self, device: torch.device, *, enabled: bool) -> AsyncDeviceTimer:
        if not self.active or not enabled:
            return self._noop_timer
        return AsyncDeviceTimer(device, enabled=True)

    def submit(self, trace: PendingTimingTrace) -> None:
        assert self.active
        assert trace.rank == self.rank
        self._pending.append(trace)

    def resolve_ready(self, current_step: int) -> list[dict[str, Any]]:
        if not self.active:
            return []
        resolved: list[dict[str, Any]] = []
        remaining: list[PendingTimingTrace] = []
        cutoff = int(current_step) - self.resolve_delay_steps
        for trace in self._pending:
            if trace.step <= cutoff and trace.ready():
                resolved.append(trace.resolve())
            else:
                remaining.append(trace)
        self._pending = remaining
        return resolved

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def format_timing_trace(trace: Mapping[str, Any]) -> str:
    ordered_fields = (
        "fetch_batch_ms",
        "prepare_batch_ms",
        "h2d_ms",
        "embedding_lookup_ms",
        "model_core_ms",
        "forward_ms",
        "loss_ms",
        "backward_ms",
        "optimizer_ms",
        "device_step_ms",
        "global_metric_comm_ms",
        "step_total_ms",
    )
    fields = [
        "timing_trace",
        f"step={int(trace['step'])}",
        f"scope={trace['scope']}",
        "distributed_sync=false",
    ]
    for name in ordered_fields:
        if name in trace:
            fields.append(f"{name}={float(trace[name]):.3f}")
    return " ".join(fields)

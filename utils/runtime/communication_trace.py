from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch


COMMUNICATION_TOKENS = (
    "nccl",
    "c10d",
    "allreduce",
    "all_reduce",
    "alltoall",
    "all_to_all",
    "reduce_scatter",
    "allgather",
    "all_gather",
)


@dataclass
class CommunicationProfileSession:
    step: int
    profiler: Any
    started_at: float


class RankLocalCommunicationProfiler:
    """Sample communication activity on one rank without adding collectives."""

    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        target_rank: int,
        every_steps: int,
        active_steps: int,
        export_chrome_trace: bool,
        output_dir: str | Path,
    ) -> None:
        self.rank = int(rank)
        self.target_rank = int(target_rank)
        self.every_steps = int(every_steps)
        self.active_steps = int(active_steps)
        self.export_chrome_trace = bool(export_chrome_trace)
        self.output_dir = Path(output_dir)
        self.active = bool(enabled) and self.rank == self.target_rank

    def should_profile(self, step: int) -> bool:
        if not self.active or step <= 0:
            return False
        offset = step % self.every_steps
        return offset == 0 or 0 < offset < self.active_steps

    def start_step(self, step: int) -> CommunicationProfileSession | None:
        if not self.should_profile(step):
            return None
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.__enter__()
        return CommunicationProfileSession(
            step=int(step),
            profiler=profiler,
            started_at=time.perf_counter(),
        )

    def finish_step(
        self,
        session: CommunicationProfileSession | None,
    ) -> dict[str, Any] | None:
        if session is None:
            return None
        sampled_step_wall_ms = (time.perf_counter() - session.started_at) * 1000.0
        session.profiler.__exit__(None, None, None)
        if self.export_chrome_trace:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            session.profiler.export_chrome_trace(
                str(
                    self.output_dir
                    / f"communication_trace_rank{self.rank}_step{session.step}.json"
                )
            )
        summary = summarize_communication_events(session.profiler.events())
        communication_union_ms = float(summary["communication_union_ms"])
        return {
            "event": "communication_trace",
            "step": session.step,
            "rank": self.rank,
            "scope": f"rank{self.rank}_local",
            "distributed_sync": False,
            "sample_wall_ms": sampled_step_wall_ms,
            "communication_ratio": (
                communication_union_ms / sampled_step_wall_ms
                if sampled_step_wall_ms > 0.0
                else 0.0
            ),
            **summary,
        }


def summarize_communication_events(events: Iterable[Any]) -> dict[str, Any]:
    candidates = [event for event in events if _is_communication_event(event)]
    device_candidates = [event for event in candidates if _is_device_event(event)]
    selected = device_candidates or candidates
    buckets: dict[str, list[tuple[float, float]]] = {
        "dmp_forward": [],
        "dmp_backward": [],
        "ddp_backward": [],
        "global_metric_comm": [],
        "rank_diagnostics_comm": [],
        "trace_comm": [],
        "unclassified_comm": [],
    }
    for event in selected:
        interval = _event_interval_us(event)
        if interval is None:
            continue
        buckets[_communication_category(event)].append(interval)

    out: dict[str, Any] = {}
    all_intervals: list[tuple[float, float]] = []
    for category, intervals in buckets.items():
        all_intervals.extend(intervals)
        out[f"{category}_count"] = len(intervals)
        out[f"{category}_sum_ms"] = sum(end - start for start, end in intervals) / 1000.0
        out[f"{category}_union_ms"] = _interval_union_us(intervals) / 1000.0
    out["communication_event_count"] = len(all_intervals)
    out["communication_sum_ms"] = (
        sum(end - start for start, end in all_intervals) / 1000.0
    )
    out["communication_union_ms"] = _interval_union_us(all_intervals) / 1000.0
    return out


def format_communication_trace(trace: Mapping[str, Any]) -> str:
    categories = (
        "dmp_forward",
        "dmp_backward",
        "ddp_backward",
        "global_metric_comm",
        "rank_diagnostics_comm",
        "trace_comm",
        "unclassified_comm",
    )
    parts = [
        "communication_trace",
        f"step={int(trace['step'])}",
        f"scope={trace['scope']}",
        "distributed_sync=false",
        f"event_count={int(trace['communication_event_count'])}",
        f"union_ms={float(trace['communication_union_ms']):.3f}",
        f"ratio={float(trace['communication_ratio']):.4f}",
    ]
    for category in categories:
        count = int(trace[f"{category}_count"])
        if count <= 0:
            continue
        parts.append(
            f"{category}=count:{count},sum_ms:"
            f"{float(trace[f'{category}_sum_ms']):.3f},union_ms:"
            f"{float(trace[f'{category}_union_ms']):.3f}"
        )
    return " ".join(parts)


def _is_communication_event(event: Any) -> bool:
    name = _event_name(event).lower()
    return any(token in name for token in COMMUNICATION_TOKENS)


def _is_device_event(event: Any) -> bool:
    device_type = str(getattr(event, "device_type", "")).lower()
    return "cuda" in device_type or "privateuse" in device_type


def _communication_category(event: Any) -> str:
    name = _event_name(event).lower()
    ancestors = " ".join(_ancestor_names(event)).lower()
    context = f"{ancestors} {name}"
    if "recagora.embedding_lookup" in context or "recagora.model_forward" in context:
        return "dmp_forward"
    if "recagora.backward" in context:
        if "alltoall" in name or "all_to_all" in name:
            return "dmp_backward"
        return "ddp_backward"
    if "recagora.global_metrics" in context:
        return "global_metric_comm"
    if "recagora.rank_diagnostics" in context:
        return "rank_diagnostics_comm"
    return "unclassified_comm"


def _event_name(event: Any) -> str:
    return str(getattr(event, "name", getattr(event, "key", "")))


def _ancestor_names(event: Any) -> list[str]:
    names: list[str] = []
    parent = getattr(event, "cpu_parent", None)
    seen: set[int] = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        names.append(_event_name(parent))
        parent = getattr(parent, "cpu_parent", None)
    return names


def _event_interval_us(event: Any) -> tuple[float, float] | None:
    time_range = getattr(event, "time_range", None)
    start = getattr(time_range, "start", None)
    end = getattr(time_range, "end", None)
    if start is None or end is None:
        return None
    start_value = float(start)
    end_value = float(end)
    if end_value < start_value:
        return None
    return start_value, end_value


def _interval_union_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start

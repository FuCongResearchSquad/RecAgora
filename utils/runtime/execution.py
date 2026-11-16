from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from utils.runtime.data import build_request_dataset
from utils.runtime.data.request_batch import RequestBatch

from utils.compiler.config import DataConfig, TrainConfig
from utils.runtime.data.prepack import CategoricalPrepackPlan
from utils.runtime.distributed import DistributedContext
from utils.runtime.checkpoint.dcp import load_model_checkpoint
from utils.compiler.feature_contract import AdapterPlan, EmbeddingInputPlan


# ===== DataLoader Construction =====


def build_data_loader(
    data_config: DataConfig,
    train_config: TrainConfig,
    *,
    torch_device: torch.device,
    distributed_context: DistributedContext | None = None,
    shuffle: bool | None = None,
    start_rank_batch_index: int = 0,
    start_global_batch_id: int | None = None,
    resume_global_batch_stop: int | None = None,
    resume_previous_world_size: int | None = None,
    read_plan: object | None = None,
    parse_plan: object | None = None,
    adapter_plan: AdapterPlan | None = None,
    embedding_input_plan: EmbeddingInputPlan | None = None,
    required_categorical_prepack_source_paths: Iterable[str] = (),
) -> DataLoader:
    rank = distributed_context.rank if distributed_context is not None else 0
    world_size = (
        distributed_context.world_size if distributed_context is not None else 1
    )
    has_adapter = adapter_plan is not None
    has_embedding = embedding_input_plan is not None
    if has_adapter != has_embedding:
        raise ValueError(
            "build_data_loader requires adapter_plan and embedding_input_plan together"
        )
    if not has_adapter:
        raise ValueError(
            "build_data_loader requires worker categorical prepack plans"
        )
    preserve_source_paths = tuple(
        dict.fromkeys(
            (
                *required_categorical_prepack_source_paths,
                *train_config.categorical_prepack_preserve_sources,
            )
        )
    )
    missing_mask_fields = set(train_config.negative_mask_fields).difference(
        data_config.candidate_metadata_fields
    )
    if missing_mask_fields:
        raise ValueError(
            "negative_mask_fields must be declared in data.candidate_metadata: "
            + ", ".join(sorted(missing_mask_fields))
        )
    categorical_prepack_plan = (
        None
        if adapter_plan is None or embedding_input_plan is None
        else CategoricalPrepackPlan(
            adapter_plan=adapter_plan,
            input_plan=embedding_input_plan,
            preserve_source_paths=preserve_source_paths,
            negative_mask_fields=train_config.negative_mask_fields,
        )
    )
    dataset = build_request_dataset(
        data_config,
        batch_size=train_config.batch_size,
        distributed_rank=rank,
        distributed_world_size=world_size,
        start_rank_batch_index=start_rank_batch_index,
        start_global_batch_id=start_global_batch_id,
        resume_global_batch_stop=resume_global_batch_stop,
        resume_previous_world_size=resume_previous_world_size,
        read_plan=read_plan,
        parse_plan=parse_plan,
        categorical_prepack_plan=categorical_prepack_plan,
    )
    is_iterable_dataset = isinstance(dataset, IterableDataset)
    if not bool(getattr(dataset, "yields_batches", False)):
        raise ValueError("RecAgora runtime only supports pre-batched RequestDataset")
    should_shuffle = (not is_iterable_dataset) if shuffle is None else shuffle
    loader_kwargs = {
        "batch_size": None,
        "shuffle": should_shuffle,
        "num_workers": train_config.num_workers,
        "pin_memory": train_config.pin_memory and torch_device.type == "cuda",
        "persistent_workers": train_config.persistent_workers
        and train_config.num_workers > 0,
        "collate_fn": _identity,
    }
    if train_config.num_workers > 0 and train_config.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = train_config.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


# ===== Checkpoint Loading =====


def load_model_weights(
    path: str | Path,
    *,
    model: torch.nn.Module,
    distributed_context: DistributedContext | None = None,
    allow_missing_acr: bool = False,
) -> dict[str, Any]:
    return load_model_checkpoint(
        path,
        model=model,
        distributed_context=distributed_context,
        allow_missing_acr=allow_missing_acr,
    )


# ===== Device And AMP Helpers =====


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled)


def grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def move_to_device(value: Any, device: torch.device, *, non_blocking: bool) -> Any:
    if isinstance(value, RequestBatch):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: move_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    return value


# ===== Batch Runtime Metadata =====


def strip_batch_source_path(batch: Any) -> tuple[Any, str | None]:
    return strip_batch_runtime_metadata(batch)


def strip_batch_runtime_metadata(batch: Any) -> tuple[Any, str | None]:
    if isinstance(batch, RequestBatch):
        source_path = batch.metadata.source_paths[0] if batch.metadata.source_paths else None
        return batch, source_path
    if not isinstance(batch, dict):
        return batch, None
    source_path = batch.get("_source_path")
    out = dict(batch)
    out.pop("_source_path", None)
    out.pop("_data_cursor", None)
    return out, str(source_path) if source_path is not None else None


# ===== Progress Iteration =====


class ProgressIterable:
    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        enabled: bool,
        desc: str,
        total: int | None = None,
    ) -> None:
        if total is not None and (
            isinstance(total, bool) or not isinstance(total, int)
        ):
            raise TypeError("progress total must be an integer")
        if total is not None and total < 0:
            raise ValueError("progress total must be a non-negative integer")
        self.iterable = iterable
        self.enabled = enabled
        self.desc = desc
        self._configured_total = total
        self._progress: Any | None = None
        self._started_at: float | None = None
        self._n = 0
        self._total: int | None = None
        self._excluded_n = 0
        self._excluded_duration_sec = 0.0

    def __iter__(self) -> Iterator[Any]:
        self._started_at = time.monotonic()
        self._n = 0
        self._excluded_n = 0
        self._excluded_duration_sec = 0.0
        self._total = self._resolve_total()
        if not self.enabled:
            for item in self.iterable:
                self._n += 1
                yield item
            return

        try:
            from tqdm.auto import tqdm
        except ImportError:
            for item in self.iterable:
                self._n += 1
                yield item
            return

        with tqdm(
            self.iterable,
            total=self._total,
            desc=self.desc,
            unit="batch",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        ) as progress:
            self._progress = progress
            try:
                for item in progress:
                    self._n += 1
                    yield item
            finally:
                self._progress = None

    def set_postfix(
        self,
        values: Mapping[str, Any],
        *,
        refresh: bool = True,
    ) -> None:
        if self._progress is not None:
            self._progress.set_postfix(values, refresh=refresh)

    def exclude_sampled_step(self, duration_sec: float) -> None:
        """Exclude an instrumentation-heavy step from steady-state rate fields."""
        self._excluded_n += 1
        self._excluded_duration_sec += max(0.0, float(duration_sec))

    def timing_log_fields(self) -> dict[str, str]:
        if self._started_at is None:
            return {}
        elapsed = max(
            0.0,
            time.monotonic() - self._started_at - self._excluded_duration_sec,
        )
        measured_n = max(0, self._n - self._excluded_n)
        rate = measured_n / elapsed if measured_n > 0 and elapsed > 0 else None
        eta = None
        if self._total is not None and rate is not None and rate > 0:
            eta = max(0.0, (self._total - self._n) / rate)
        return {
            "elapsed": _format_duration(elapsed),
            "eta": _format_duration(eta) if eta is not None else "unknown",
            "rate": f"{rate:.2f}batch/s" if rate is not None else "unknown",
        }

    def _resolve_total(self) -> int | None:
        if self._configured_total is not None:
            return self._configured_total
        try:
            return len(self.iterable)  # type: ignore[arg-type]
        except TypeError:
            return None


def iter_with_progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    total: int | None = None,
) -> ProgressIterable:
    return ProgressIterable(
        iterable,
        enabled=enabled,
        desc=desc,
        total=total,
    )


def _format_duration(seconds: float) -> str:
    total_seconds = int(max(0.0, seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ===== Output IO =====


def prepare_result_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "outputs": root / "outputs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_log(
    path: Path,
    message: str,
    *,
    mirror_stdout: bool = True,
) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    if mirror_stdout:
        print(message, flush=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _identity(value: Any) -> Any:
    return value

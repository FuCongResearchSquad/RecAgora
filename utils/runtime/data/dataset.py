from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import gc
import glob
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Iterator, NoReturn

import torch
from torch.utils.data import IterableDataset, get_worker_info

from utils.compiler.config import DataConfig
from utils.runtime.data.reader import ContractFeatureReader
from utils.runtime.data.errors import DataContractError
from utils.runtime.data.hotuser import build_hotuser_index, validate_request_ids
from utils.runtime.data.request_batch import (
    HistoryGroupLengths,
    RaggedFeature,
    RequestBatch,
    RequestBatchMetadata,
)

if TYPE_CHECKING:
    from utils.runtime.data.prepack import CategoricalPrepackPlan


# ===== Data Planning Records =====


@dataclass(frozen=True)
class RowGroupSpan:
    path: Path
    row_group_index: int
    row_count: int
    global_start: int
    global_stop: int


@dataclass(frozen=True)
class BatchReadSpan:
    path: Path
    row_group_index: int
    row_group_offset: int
    row_count: int
    global_start: int


def _resolve_epoch_global_batch_stop(
    *,
    global_full_batches: int,
    current_world_size: int,
    start_global_batch_id: int | None,
    resume_global_batch_stop: int | None,
    resume_previous_world_size: int | None,
) -> int:
    native_stop = (
        int(global_full_batches) // int(current_world_size)
    ) * int(current_world_size)
    if start_global_batch_id is None:
        return native_stop
    start = int(start_global_batch_id)
    requested_stop = _resolve_requested_epoch_global_batch_stop(
        global_full_batches=global_full_batches,
        resume_global_batch_stop=resume_global_batch_stop,
        resume_previous_world_size=resume_previous_world_size,
    )
    if start > requested_stop:
        raise ValueError(
            "elastic resume global cursor exceeds the saved epoch boundary: "
            f"start={start} stop={requested_stop}"
        )
    remaining_batches = requested_stop - start
    return start + (
        remaining_batches // int(current_world_size)
    ) * int(current_world_size)


def _resolve_requested_epoch_global_batch_stop(
    *,
    global_full_batches: int,
    resume_global_batch_stop: int | None,
    resume_previous_world_size: int | None,
) -> int:
    if resume_global_batch_stop is not None:
        return int(resume_global_batch_stop)
    if resume_previous_world_size is not None:
        previous_world_size = int(resume_previous_world_size)
        return (
            int(global_full_batches) // previous_world_size
        ) * previous_world_size
    return int(global_full_batches)


class RequestDataset(IterableDataset):
    """Pre-batched request-level Parquet dataset.

    The yielded batch schema is the request-level batch consumed by models.
    DDP rank sharding and DataLoader worker sharding are done by global raw
    sample batches. Row groups remain the physical Parquet read unit only.
    """

    yields_batches = True

    def __init__(
        self,
        config: DataConfig,
        batch_size: int = 128,
        *,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        start_rank_batch_index: int = 0,
        start_global_batch_id: int | None = None,
        resume_global_batch_stop: int | None = None,
        resume_previous_world_size: int | None = None,
        read_plan: Any | None = None,
        parse_plan: Any | None = None,
        categorical_prepack_plan: "CategoricalPrepackPlan | None" = None,
    ):
        """Prepare immutable Parquet row-group and rank-sharding metadata."""
        self.config = config
        self.batch_size = batch_size
        self.distributed_rank = distributed_rank
        self.distributed_world_size = distributed_world_size
        if start_global_batch_id is not None and int(start_global_batch_id) < 0:
            raise ValueError("start_global_batch_id must be non-negative")
        self.start_global_batch_id = (
            None if start_global_batch_id is None else int(start_global_batch_id)
        )
        self._resume_global_batch_stop = (
            None
            if resume_global_batch_stop is None
            else int(resume_global_batch_stop)
        )
        self._resume_previous_world_size = (
            None
            if resume_previous_world_size is None
            else int(resume_previous_world_size)
        )
        if resume_global_batch_stop is not None and int(resume_global_batch_stop) < 0:
            raise ValueError("resume_global_batch_stop must be non-negative")
        if resume_previous_world_size is not None and int(resume_previous_world_size) <= 0:
            raise ValueError("resume_previous_world_size must be positive")
        if self.start_global_batch_id is None and (
            resume_global_batch_stop is not None
            or resume_previous_world_size is not None
        ):
            raise ValueError(
                "elastic resume batch boundary requires start_global_batch_id"
            )
        self.start_rank_batch_index = (
            int(start_rank_batch_index)
            if self.start_global_batch_id is None
            else self.start_global_batch_id // self.distributed_world_size
        )
        self._categorical_prepack_plan = categorical_prepack_plan
        if read_plan is None or parse_plan is None:
            raise ValueError(
                "RequestDataset requires FeatureContract read_plan and parse_plan on the new train chain"
            )
        self._contract_reader = ContractFeatureReader(
            read_plan,
            parse_plan,
            list_len=config.list_len,
            padding_value=config.history_padding_value,
        )
        self.data_identity = None
        if config.root is None:
            self.paths = self._apply_start_file(
                self._expand_paths(config.paths),
                config.parquet_start_file,
            )
        else:
            hotuser_index = build_hotuser_index(config)
            self.data_identity = hotuser_index.identity
            self.paths = [record.path for record in hotuser_index.files]
        self._required_columns = self._contract_reader.required_columns()
        self._row_groups, self._source_num_rows = self._build_row_groups(self.paths)
        self._row_group_stops = [span.global_stop for span in self._row_groups]
        self._valid_global_rows = (
            self._build_valid_global_rows()
            if self._contract_reader.parse_plan.bad_length_policy == "drop_sample"
            else None
        )
        self._num_rows = (
            self._source_num_rows
            if self._valid_global_rows is None
            else len(self._valid_global_rows)
        )
        self._global_full_batches = self._num_rows // self.batch_size
        self._native_usable_global_batches = (
            self._global_full_batches // self.distributed_world_size
        ) * self.distributed_world_size
        requested_epoch_stop = _resolve_requested_epoch_global_batch_stop(
            global_full_batches=self._global_full_batches,
            resume_global_batch_stop=self._resume_global_batch_stop,
            resume_previous_world_size=self._resume_previous_world_size,
        )
        if requested_epoch_stop > self._global_full_batches:
            raise ValueError(
                "resume_global_batch_stop exceeds the available full batches: "
                f"stop={requested_epoch_stop} "
                f"full_batches={self._global_full_batches}"
            )
        self._epoch_global_batch_stop = _resolve_epoch_global_batch_stop(
            global_full_batches=self._global_full_batches,
            current_world_size=self.distributed_world_size,
            start_global_batch_id=self.start_global_batch_id,
            resume_global_batch_stop=resume_global_batch_stop,
            resume_previous_world_size=resume_previous_world_size,
        )
        self._usable_global_batches = self._epoch_global_batch_stop
        if self.start_global_batch_id is None:
            self._total_rank_batches = (
                self._epoch_global_batch_stop // self.distributed_world_size
            )
            self._num_batches = max(
                0,
                self._total_rank_batches - self.start_rank_batch_index,
            )
        else:
            self._num_batches = (
                self._epoch_global_batch_stop - self.start_global_batch_id
            ) // self.distributed_world_size
            self._total_rank_batches = (
                self.start_rank_batch_index + self._num_batches
            )
        self._rank_num_rows = self._num_batches * self.batch_size
        self._stats = {
            "rows_seen": 0,
            "rows_emitted": 0,
            "rows_dropped": self._source_num_rows - self._num_rows,
            "batches_emitted": 0,
        }

    # ===== Dataset Protocol =====

    def __iter__(self) -> Iterator[RequestBatch]:
        """Yield pre-batched request tensors for this rank and worker."""
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        parquet_files: OrderedDict[Path, Any] = OrderedDict()
        for remaining_batch_index, global_batch_id in enumerate(
            self._rank_global_batch_ids()
        ):
            if remaining_batch_index % num_workers != worker_id:
                continue
            rank_batch_index = self._rank_batch_index(global_batch_id)
            source_paths: list[Path] = []
            converted_batches: list[RequestBatch] = []
            read_row_group_sec = 0.0
            convert_record_batch_sec = 0.0
            span_infos: list[dict[str, Any]] = []
            for span in self._batch_read_spans(global_batch_id):
                read_started_at = time.perf_counter()
                record_batch = self._read_span_record_batch(span, parquet_files)
                read_row_group_sec += time.perf_counter() - read_started_at
                source_paths.append(span.path)
                span_infos.append(
                    {
                        "path": span.path.as_posix(),
                        "row_group_index": int(span.row_group_index),
                        "row_group_offset": int(span.row_group_offset),
                        "row_count": int(span.row_count),
                        "global_start": int(span.global_start),
                    }
                )
                convert_started_at = time.perf_counter()
                batch = self._convert_record_batch(
                    record_batch,
                    fallback_request_id=span.global_start,
                )
                convert_record_batch_sec += time.perf_counter() - convert_started_at
                del record_batch
                converted_batches.append(batch)
            out = (
                converted_batches[0]
                if len(converted_batches) == 1
                else _concat_request_batches(converted_batches)
            )
            cursor = {
                "global_batch_id": int(global_batch_id),
                "rank_batch_index": int(rank_batch_index),
                "next_rank_batch_index": int(rank_batch_index + 1),
                "worker_id": int(worker_id),
                "num_workers": int(num_workers),
                "source_path": source_paths[0].as_posix() if source_paths else None,
                "read_row_group_sec": float(read_row_group_sec),
                "convert_record_batch_sec": float(convert_record_batch_sec),
                "span_count": int(len(span_infos)),
                "spans": span_infos,
            }
            out = out.with_metadata(
                RequestBatchMetadata(
                    source_paths=tuple(path.as_posix() for path in source_paths),
                    cursor=cursor,
                )
            )
            if self._categorical_prepack_plan is not None:
                from utils.runtime.data.prepack import (
                    prepack_request_batch_categorical,
                )

                out = prepack_request_batch_categorical(
                    out,
                    plan=self._categorical_prepack_plan,
                )
            self._stats["batches_emitted"] += 1
            yield out
            del out
            del converted_batches
            del source_paths
            self._maybe_release_unused_worker_memory()

    def __len__(self) -> int:
        return self._num_batches

    @property
    def num_rows(self) -> int:
        return self._rank_num_rows

    @property
    def global_num_rows(self) -> int:
        return self._num_rows

    @property
    def global_num_row_groups(self) -> int:
        return len(self._row_groups)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def planned_batch_stats(self, num_workers: int) -> dict[str, Any]:
        """Return the exact batch plan used by train/eval logging and DDP sync."""
        effective_workers = max(1, int(num_workers))
        worker_stats: list[dict[str, Any]] = []
        planned_batches = 0
        planned_rows_emitted = 0
        planned_rows_dropped_tail = 0
        for worker_id in range(effective_workers):
            batches = self._worker_batch_count(worker_id, effective_workers)
            emitted_rows = batches * self.batch_size
            dropped_tail_rows = 0
            planned_batches += batches
            planned_rows_emitted += emitted_rows
            planned_rows_dropped_tail += dropped_tail_rows
            worker_stats.append(
                {
                    "worker_id": worker_id,
                    "row_groups": None,
                    "rows": emitted_rows,
                    "planned_batches": batches,
                    "planned_rows_emitted": emitted_rows,
                    "planned_rows_dropped_tail": dropped_tail_rows,
                }
            )
        global_rows_dropped_tail = self._num_rows - (
            self._global_full_batches * self.batch_size
        )
        if self.start_global_batch_id is None:
            balance_limit = self._global_full_batches
        elif self._resume_global_batch_stop is not None:
            balance_limit = self._resume_global_batch_stop
        elif self._resume_previous_world_size is not None:
            balance_limit = (
                self._global_full_batches // self._resume_previous_world_size
            ) * self._resume_previous_world_size
        else:
            balance_limit = self._global_full_batches
        global_batches_dropped_for_balance = (
            balance_limit - self._epoch_global_batch_stop
        )
        return {
            "rank": self.distributed_rank,
            "world_size": self.distributed_world_size,
            "batch_size": self.batch_size,
            "num_workers": effective_workers,
            "sharding": "global_sample",
            "advertised_len": len(self),
            "rank_rows": self._rank_num_rows,
            "rank_batch_start_offset": self.start_rank_batch_index,
            "start_global_batch_id": self.start_global_batch_id,
            "epoch_global_batch_stop": self._epoch_global_batch_stop,
            "native_usable_global_batches": self._native_usable_global_batches,
            "total_rank_batches": self._total_rank_batches,
            "global_rows": self._num_rows,
            "source_rows": self._source_num_rows,
            "global_rows_dropped_invalid": self._source_num_rows - self._num_rows,
            "global_row_groups": len(self._row_groups),
            "global_full_batches": self._global_full_batches,
            "usable_global_batches": self._epoch_global_batch_stop,
            "global_rows_dropped_tail": global_rows_dropped_tail,
            "global_batches_dropped_for_balance": global_batches_dropped_for_balance,
            "planned_batches": planned_batches,
            "planned_rows_emitted": planned_rows_emitted,
            "planned_rows_dropped_tail": planned_rows_dropped_tail,
            "worker_stats": worker_stats,
        }

    # ===== Rank And Worker Sharding =====

    def _rank_global_batch_ids(self) -> range:
        if self.start_global_batch_id is not None:
            return range(
                self.start_global_batch_id + self.distributed_rank,
                self._usable_global_batches,
                self.distributed_world_size,
            )
        return range(
            self.distributed_rank
            + self.start_rank_batch_index * self.distributed_world_size,
            self._usable_global_batches,
            self.distributed_world_size,
        )

    def _rank_batch_index(self, global_batch_id: int) -> int:
        if self.start_global_batch_id is not None:
            return self.start_rank_batch_index + (
                global_batch_id
                - self.start_global_batch_id
                - self.distributed_rank
            ) // self.distributed_world_size
        return (global_batch_id - self.distributed_rank) // self.distributed_world_size

    def _worker_batch_count(self, worker_id: int, num_workers: int) -> int:
        if worker_id >= self._num_batches:
            return 0
        return 1 + (self._num_batches - 1 - worker_id) // num_workers

    # ===== Path And Row-Group Planning =====

    @staticmethod
    def _apply_start_file(
        paths: list[Path],
        start_file: str | Path | None,
    ) -> list[Path]:
        if start_file is None:
            return paths

        start = Path(start_file)
        for index, path in enumerate(paths):
            if RequestDataset._same_path_or_name(path, start):
                return paths[index:]
        raise DataContractError(
            f"Configured Parquet start_file {start_file!r} was not found "
            "in expanded data.paths"
        )

    @staticmethod
    def _same_path_or_name(path: Path, target: Path) -> bool:
        if path == target or path.as_posix() == target.as_posix():
            return True
        if path.name == target.name:
            return True
        try:
            return path.resolve() == target.resolve()
        except OSError:
            return False

    @staticmethod
    def _expand_paths(paths: SequencePath) -> list[Path]:
        out: list[Path] = []
        for raw_path in paths:
            path_str = str(raw_path)
            matches = (
                [Path(p) for p in sorted(glob.glob(path_str))]
                if any(ch in path_str for ch in "*?[]")
                else [Path(path_str)]
            )
            for path in matches:
                if not path.exists():
                    raise DataContractError(f"Parquet path does not exist: {path}")
                if path.is_dir():
                    out.extend(sorted(path.glob("*.parquet")))
                else:
                    out.append(path)
        if not out:
            raise DataContractError("No Parquet files matched the configured paths")
        return out

    # ===== Parquet Reads =====

    def _read_span_record_batch(
        self,
        span: BatchReadSpan,
        parquet_files: OrderedDict[Path, Any],
    ) -> Any:
        parquet_file = self._parquet_file_for_span(span, parquet_files)
        try:
            columns = self._available_required_columns(parquet_file)
            table = parquet_file.read_row_group(span.row_group_index, columns=columns)
            table = table.slice(span.row_group_offset, span.row_count).combine_chunks()
            if self.data_identity is not None:
                validate_request_ids(
                    table,
                    field=self.config.request_id_field,
                    first_id=self.data_identity["first_id"] + span.global_start,
                )
            batches = table.to_batches(max_chunksize=span.row_count)
            return batches[0]
        finally:
            if self.config.parquet_file_cache_size == 0:
                self._close_parquet_file(parquet_file)

    def _parquet_file_for_span(
        self,
        span: BatchReadSpan,
        parquet_files: OrderedDict[Path, Any],
    ) -> Any:
        cache_size = self.config.parquet_file_cache_size
        if cache_size == 0:
            pq = _import_pyarrow_parquet()
            return pq.ParquetFile(span.path)

        parquet_file = parquet_files.get(span.path)
        if parquet_file is None:
            pq = _import_pyarrow_parquet()
            parquet_file = pq.ParquetFile(span.path)
            parquet_files[span.path] = parquet_file
        else:
            parquet_files.move_to_end(span.path)

        if cache_size is not None:
            while len(parquet_files) > cache_size:
                _, evicted = parquet_files.popitem(last=False)
                self._close_parquet_file(evicted)
        return parquet_file

    @staticmethod
    def _close_parquet_file(parquet_file: Any) -> None:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()

    def _maybe_release_unused_worker_memory(self) -> None:
        interval = self.config.release_unused_every_batches
        if interval is None:
            return
        if self._stats["batches_emitted"] % interval != 0:
            return
        gc.collect()
        pa = _import_pyarrow()
        release_unused = getattr(pa.default_memory_pool(), "release_unused", None)
        if callable(release_unused):
            release_unused()

    def _available_required_columns(self, parquet_file: Any) -> list[str]:
        available_columns = set(parquet_file.schema_arrow.names)
        return [
            column for column in self._required_columns if column in available_columns
        ]

    def _global_batch_read_spans(self, global_batch_id: int) -> list[BatchReadSpan]:
        batch_start = global_batch_id * self.batch_size
        batch_stop = batch_start + self.batch_size
        spans: list[BatchReadSpan] = []
        current = batch_start
        while current < batch_stop:
            row_group_index = bisect_right(self._row_group_stops, current)
            row_group = self._row_groups[row_group_index]
            stop = min(batch_stop, row_group.global_stop)
            row_count = stop - current
            spans.append(
                BatchReadSpan(
                    path=row_group.path,
                    row_group_index=row_group.row_group_index,
                    row_group_offset=current - row_group.global_start,
                    row_count=row_count,
                    global_start=current,
                )
            )
            current = stop
        return spans

    def _batch_read_spans(self, global_batch_id: int) -> list[BatchReadSpan]:
        if self._valid_global_rows is None:
            return self._global_batch_read_spans(global_batch_id)
        start = global_batch_id * self.batch_size
        positions = self._valid_global_rows[start : start + self.batch_size]
        spans: list[BatchReadSpan] = []
        index = 0
        while index < len(positions):
            position = positions[index]
            row_group_index = bisect_right(self._row_group_stops, position)
            row_group = self._row_groups[row_group_index]
            stop_index = index + 1
            previous = position
            while stop_index < len(positions):
                candidate = positions[stop_index]
                if candidate != previous + 1 or candidate >= row_group.global_stop:
                    break
                previous = candidate
                stop_index += 1
            spans.append(
                BatchReadSpan(
                    path=row_group.path,
                    row_group_index=row_group.row_group_index,
                    row_group_offset=position - row_group.global_start,
                    row_count=stop_index - index,
                    global_start=position,
                )
            )
            index = stop_index
        return spans

    def _build_valid_global_rows(self) -> tuple[int, ...]:
        """Pre-validate rows before DDP sharding when drop_sample is requested.

        The scan is opt-in because it reads the selected columns twice. It makes
        row dropping deterministic across ranks and prevents placeholder batches.
        """
        pq = _import_pyarrow_parquet()
        valid_rows: list[int] = []
        for row_group in self._row_groups:
            parquet_file = pq.ParquetFile(row_group.path)
            try:
                columns = self._available_required_columns(parquet_file)
                missing = set(self._required_columns) - set(columns)
                if missing and self._contract_reader.parse_plan.missing_slot_policy == "fail_fast":
                    raise DataContractError(
                        f"Parquet file {row_group.path} is missing required columns: "
                        f"{sorted(missing)!r}"
                    )
                table = parquet_file.read_row_group(row_group.row_group_index, columns=columns)
            finally:
                self._close_parquet_file(parquet_file)
            for offset in range(row_group.row_count):
                try:
                    self._contract_reader.parse_table(table.slice(offset, 1))
                except DataContractError:
                    continue
                valid_rows.append(row_group.global_start + offset)
        return tuple(valid_rows)

    # ===== Record Batch Conversion =====

    def _convert_record_batch(
        self,
        record_batch: Any,
        *,
        fallback_request_id: int,
    ) -> RequestBatch:
        """Parse one physical record batch into canonical request-level data."""
        pa = _import_pyarrow()
        table = pa.Table.from_batches([record_batch])
        self._stats["rows_seen"] += int(record_batch.num_rows)
        batch = self._contract_reader.parse_table(table)
        self._stats["rows_emitted"] += int(record_batch.num_rows)
        return batch

    # ===== Row-Group Metadata =====

    @staticmethod
    def _build_row_groups(paths: list[Path]) -> tuple[list[RowGroupSpan], int]:
        pq = _import_pyarrow_parquet()
        row_groups: list[RowGroupSpan] = []
        total = 0
        for path in paths:
            metadata = pq.ParquetFile(path).metadata
            for row_group_index in range(metadata.num_row_groups):
                row_count = int(metadata.row_group(row_group_index).num_rows)
                row_groups.append(
                    RowGroupSpan(
                        path=path,
                        row_group_index=row_group_index,
                        row_count=row_count,
                        global_start=total,
                        global_stop=total + row_count,
                    )
                )
                total += row_count
        return row_groups, total



# ===== Module Helpers =====


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        _raise_missing_pyarrow(exc)
    return pq


def _import_pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        _raise_missing_pyarrow(exc)
    return pa


def _raise_missing_pyarrow(exc: ImportError) -> NoReturn:
    raise ImportError(
        "Parquet RequestDataset requires pyarrow. "
        "Install project dependencies with `.venv/bin/python -m pip install -r requirements.txt`."
    ) from exc


def _concat_request_batches(batches: list[RequestBatch]) -> RequestBatch:
    first = batches[0]
    feature_paths = first.features.keys()
    label_names = first.labels.keys()
    candidate_metadata_names = first.candidate_metadata.keys()
    history_group_names = first.history_group_lengths.keys()
    user_ids = [batch.user_id for batch in batches]
    return RequestBatch(
        request_id=torch.cat([batch.request_id for batch in batches], dim=0),
        item_id=torch.cat([batch.item_id for batch in batches], dim=0),
        labels={
            name: torch.cat([batch.labels[name] for batch in batches], dim=0)
            for name in label_names
        },
        target=first.target,
        features={
            path: _concat_feature_values([batch.features[path] for batch in batches])
            for path in feature_paths
        },
        user_id=(
            None
            if user_ids[0] is None
            else torch.cat([value for value in user_ids if value is not None], dim=0)
        ),
        candidate_metadata={
            name: torch.cat(
                [batch.candidate_metadata[name] for batch in batches], dim=0
            )
            for name in candidate_metadata_names
        },
        history_group_lengths={
            name: HistoryGroupLengths(
                raw_lengths=torch.cat(
                    [
                        batch.history_group_lengths[name].raw_lengths
                        for batch in batches
                    ],
                    dim=0,
                ),
                truncated_lengths=torch.cat(
                    [
                        batch.history_group_lengths[name].truncated_lengths
                        for batch in batches
                    ],
                    dim=0,
                ),
            )
            for name in history_group_names
        },
        candidate_valid_mask=(
            None
            if all(batch.candidate_valid_mask is None for batch in batches)
            else torch.cat(
                [
                    torch.ones_like(batch.item_id, dtype=torch.bool)
                    if batch.candidate_valid_mask is None
                    else batch.candidate_valid_mask
                    for batch in batches
                ],
                dim=0,
            )
        ),
    )


def _concat_feature_values(
    values: list[torch.Tensor | RaggedFeature],
) -> torch.Tensor | RaggedFeature:
    if not values:
        raise ValueError("cannot concatenate an empty feature-value list")
    first = values[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("cannot concatenate mixed dense and ragged feature values")
        return torch.cat(values, dim=0)
    if not all(isinstance(value, RaggedFeature) for value in values):
        raise TypeError("cannot concatenate mixed ragged and dense feature values")
    return RaggedFeature(
        values=torch.cat([value.values for value in values], dim=0),
        lengths=torch.cat([value.lengths for value in values], dim=0),
        source_path=first.source_path,
        level=first.level,
    )


SequencePath = list[str | Path] | tuple[str | Path, ...]

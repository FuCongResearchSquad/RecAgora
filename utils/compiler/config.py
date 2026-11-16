"""Versioned YAML configuration loading and override compilation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import yaml


# ===== Type Aliases And Slot Presets =====


DTypeName = Literal["float32", "float64", "int64", "int32", "bool"]
DataFormat = Literal["parquet_stream"]
MissingPolicy = Literal["fail_fast", "fill_zero"]
BadLengthPolicy = Literal["fail_fast", "pad_or_truncate", "drop_sample"]
HistoryTruncationSide = Literal["head", "tail"]
HistorySequenceLayout = Literal["ragged", "padded"]
HistoryEventOrder = Literal["oldest_to_newest", "newest_to_oldest"]
HistoryPaddingSide = Literal["left", "right", "model_configured"]
SparseIDMappingType = Literal["modulo_bucket", "dense_map"]
ResumeWorldSizePolicy = Literal["strict", "elastic"]

NEGATIVE_MASK_FIELDS = frozenset(
    {
        "negative_impression",
        "negative_non_impression",
        "negative_from_rank",
        "negative_from_recall",
        "negative_from_random",
    }
)


def normalize_negative_mask_fields(
    value: Sequence[str], *, name: str = "negative_mask_fields"
) -> tuple[str, ...]:
    """Validate explicit negative-label selection and preserve first-seen order."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of negative-label field names")
    if any(not isinstance(item, str) or item not in NEGATIVE_MASK_FIELDS for item in value):
        raise ValueError(
            f"{name} only supports: {', '.join(sorted(NEGATIVE_MASK_FIELDS))}"
        )
    return tuple(dict.fromkeys(value))


# ===== Slot Normalization =====


def normalize_slot_id(slot_id: int | str) -> str:
    slot = str(slot_id)
    if slot.startswith("slot_"):
        slot = slot[len("slot_") :]
    if not slot or not slot.isdigit():
        raise ValueError(f"slot id must be numeric or slot_<id>, got {slot_id!r}")
    return slot


# ===== Config Dataclasses =====


@dataclass(frozen=True)
class HistoryGroupConfig:
    """Dataset-side contract for one aligned history feature group."""

    max_len: int
    truncate_side: HistoryTruncationSide = "head"
    padding_value: float = 0.0
    layout: HistorySequenceLayout = "ragged"
    anchor: str | None = None
    event_order: HistoryEventOrder | None = None
    padding_side: HistoryPaddingSide | None = None
    direct_padding_side: HistoryPaddingSide | None = None
    validate_aligned_lengths: bool = False

    def __post_init__(self) -> None:
        if self.max_len <= 0:
            raise ValueError("history group max_len must be positive")
        if self.truncate_side not in {"head", "tail"}:
            raise ValueError("history group truncate_side must be one of: head, tail")
        if self.layout not in {"ragged", "padded"}:
            raise ValueError("history group layout must be one of: ragged, padded")
        if self.event_order not in {
            None,
            "oldest_to_newest",
            "newest_to_oldest",
        }:
            raise ValueError(
                "history group event_order must be one of: "
                "oldest_to_newest, newest_to_oldest"
            )
        if self.padding_side not in {None, "left", "right", "model_configured"}:
            raise ValueError(
                "history group padding_side must be one of: "
                "left, right, model_configured"
            )
        if self.direct_padding_side not in {None, "left", "right"}:
            raise ValueError(
                "history group direct_padding_side must be one of: left, right"
            )


@dataclass(frozen=True)
class SparseIDContractConfig:
    """Dataset-version contract for sparse IDs consumed by embedding lookup."""

    mapping_type: SparseIDMappingType = "modulo_bucket"
    layout: str | Path | None = None

    def __post_init__(self) -> None:
        if self.mapping_type not in {"modulo_bucket", "dense_map"}:
            raise ValueError(
                "sparse id contract type must be one of: modulo_bucket, dense_map"
            )
        if self.mapping_type == "dense_map" and self.layout is None:
            raise ValueError("dense_map sparse id contract requires a layout")


@dataclass(frozen=True)
class DataConfig:
    paths: Sequence[str | Path]
    root: str | Path | None = None
    split: str = "train"
    manifest_path: str | Path | None = None
    data_format: DataFormat = "parquet_stream"
    slot_registry: str | Path | None = None
    sparse_id_contract: SparseIDContractConfig = field(
        default_factory=SparseIDContractConfig
    )
    list_len: int = 24
    label_names: Sequence[str] = ("click", "atc", "order")
    target: str = "click"
    label_required: bool = True
    request_id_field: str = "request_id"
    user_id_field: str | None = None
    item_id_field: str | None = "item_id"
    candidate_metadata_fields: Sequence[str] = field(default_factory=tuple)
    candidate_metadata_dtypes: Mapping[str, DTypeName] = field(default_factory=dict)
    missing_slot_policy: MissingPolicy = "fail_fast"
    bad_length_policy: BadLengthPolicy = "fail_fast"
    history_len: int = 200
    history_padding_value: float = 0.0
    history_groups: Mapping[str, HistoryGroupConfig] = field(default_factory=dict)
    label_dtype: DTypeName = "float32"
    id_dtype: DTypeName = "int64"
    index_progress: bool = False
    index_progress_interval: int = 1
    index_num_workers: int = 0
    parquet_start_file: str | Path | None = None
    parquet_file_cache_size: int | None = None
    release_unused_every_batches: int | None = None

    def __post_init__(self) -> None:
        if self.root is None:
            if not self.paths:
                raise ValueError("at least one data path is required")
            if self.manifest_path is not None:
                raise ValueError("manifest_path requires HotUser data.root")
        else:
            root = str(self.root)
            if self.paths or not root.strip() or "://" in root or any(c in root for c in "*?[]"):
                raise ValueError("HotUser data.root must be a local directory and data.paths must be empty")
            if self.split not in {"train", "valid", "test"}:
                raise ValueError("HotUser data.split must be train, valid or test")
            if self.parquet_start_file is not None:
                raise ValueError("HotUser data does not support parquet_start_file")
        if self.data_format != "parquet_stream":
            raise ValueError(f"unsupported data format: {self.data_format!r}")
        if self.list_len <= 0:
            raise ValueError("list_len must be positive")
        if self.missing_slot_policy not in {"fail_fast", "fill_zero"}:
            raise ValueError("missing_slot_policy must be one of: fail_fast, fill_zero")
        if self.bad_length_policy not in {
            "fail_fast",
            "pad_or_truncate",
            "drop_sample",
        }:
            raise ValueError(
                "bad_length_policy must be one of: fail_fast, pad_or_truncate, "
                "drop_sample"
            )
        if not self.label_names:
            raise ValueError("label_names cannot be empty")
        if self.label_required and self.target not in self.label_names:
            raise ValueError(f"target {self.target!r} must be included in label_names")
        if len(set(self.candidate_metadata_fields)) != len(
            self.candidate_metadata_fields
        ):
            raise ValueError("candidate_metadata_fields cannot contain duplicates")
        extra_candidate_dtypes = sorted(
            set(self.candidate_metadata_dtypes) - set(self.candidate_metadata_fields)
        )
        if extra_candidate_dtypes:
            raise ValueError(
                "candidate_metadata_dtypes reference undeclared fields: "
                f"{extra_candidate_dtypes}"
            )
        reserved_candidate_metadata_names = {
            self.request_id_field,
            *(str(name) for name in self.label_names),
        }
        if self.item_id_field is not None:
            reserved_candidate_metadata_names.add(self.item_id_field)
        if self.user_id_field is not None:
            reserved_candidate_metadata_names.add(self.user_id_field)
        candidate_metadata_collisions = sorted(
            set(self.candidate_metadata_fields).intersection(
                reserved_candidate_metadata_names
            )
        )
        if candidate_metadata_collisions:
            raise ValueError(
                "candidate metadata must be separate from IDs and labels: "
                f"{candidate_metadata_collisions}"
            )
        for name in self.candidate_metadata_fields:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "candidate_metadata_fields must contain non-empty strings"
                )
            dtype = self.candidate_metadata_dtypes.get(name, "int64")
            if dtype not in {"float32", "float64", "int64", "int32", "bool"}:
                raise ValueError(
                    f"candidate metadata {name!r} has unsupported dtype {dtype!r}"
                )
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        for name, group in self.history_groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("history group names must be non-empty strings")
            if not isinstance(group, HistoryGroupConfig):
                raise ValueError(f"history_groups[{name!r}] must be HistoryGroupConfig")
        if self.index_progress_interval <= 0:
            raise ValueError("index_progress_interval must be positive")
        if self.index_num_workers < 0:
            raise ValueError("index_num_workers must be non-negative")
        if (
            self.parquet_file_cache_size is not None
            and self.parquet_file_cache_size < 0
        ):
            raise ValueError("parquet_file_cache_size must be non-negative when provided")
        if (
            self.release_unused_every_batches is not None
            and self.release_unused_every_batches <= 0
        ):
            raise ValueError(
                "release_unused_every_batches must be positive when provided"
            )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    class_path: str
    params: Mapping[str, Any] = field(default_factory=dict)
    feature_declaration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeriodicOutputConfig:
    enabled: bool = False
    every_steps: int = 100

    def __post_init__(self) -> None:
        if self.every_steps <= 0:
            raise ValueError("every_steps must be positive")


@dataclass(frozen=True)
class TimingTraceConfig:
    enabled: bool = False
    rank: int = 0
    every_steps: int = 100
    distributed_sync: bool = False
    resolve_delay_steps: int = 2

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("timing_trace.rank must be non-negative")
        if self.every_steps <= 0:
            raise ValueError("timing_trace.every_steps must be positive")
        if self.resolve_delay_steps < 1:
            raise ValueError("timing_trace.resolve_delay_steps must be positive")
        if self.distributed_sync:
            raise ValueError(
                "timing_trace.distributed_sync must remain false; timing traces "
                "must not add distributed synchronization"
            )


@dataclass(frozen=True)
class CommunicationTraceConfig:
    enabled: bool = False
    rank: int = 0
    every_steps: int = 500
    active_steps: int = 1
    distributed_sync: bool = False
    export_chrome_trace: bool = False

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("communication_trace.rank must be non-negative")
        if self.every_steps <= 0:
            raise ValueError("communication_trace.every_steps must be positive")
        if self.active_steps <= 0:
            raise ValueError("communication_trace.active_steps must be positive")
        if self.distributed_sync:
            raise ValueError(
                "communication_trace.distributed_sync must remain false; profiler "
                "sampling must not add distributed synchronization"
            )


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool = False
    persistent_workers: bool = False
    lr: float = 1e-3
    optimizer_name: str = "adam"
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    epochs: int = 1
    max_steps: int | None = None
    seed: int = 2026
    device: str = "auto"
    amp: bool = False
    output_dir: str = "results/default"
    checkpoint_every_steps: int | None = 5000
    checkpoint_max_to_keep: int = 2
    progress_every_steps: int = 10
    global_metrics: PeriodicOutputConfig = field(
        default_factory=lambda: PeriodicOutputConfig(enabled=True, every_steps=100)
    )
    rank_diagnostics: PeriodicOutputConfig = field(
        default_factory=lambda: PeriodicOutputConfig(enabled=False, every_steps=500)
    )
    timing_trace: TimingTraceConfig = field(default_factory=TimingTraceConfig)
    communication_trace: CommunicationTraceConfig = field(
        default_factory=CommunicationTraceConfig
    )
    categorical_prepack_preserve_sources: tuple[str, ...] = ()
    resume_from: str | Path | None = None
    resume_world_size_policy: ResumeWorldSizePolicy = "strict"
    init_from: str | Path | None = None
    negative_mask_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "negative_mask_fields",
            normalize_negative_mask_fields(
                self.negative_mask_fields, name="train.negative_mask_fields"
            ),
        )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.optimizer_name not in {"adam", "adamw", "rmsprop"}:
            raise ValueError("optimizer.name must be one of: adam, adamw, rmsprop")
        if not isinstance(self.optimizer_params, Mapping):
            raise ValueError("optimizer.params must be a mapping")
        if self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive when provided")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be non-negative when provided")
        if self.max_steps == 0:
            object.__setattr__(self, "max_steps", None)
        if self.checkpoint_every_steps is not None and self.checkpoint_every_steps <= 0:
            raise ValueError(
                "checkpoint_every_steps must be positive when provided"
            )
        if self.checkpoint_max_to_keep <= 0:
            raise ValueError("checkpoint_max_to_keep must be positive")
        if self.progress_every_steps <= 0:
            raise ValueError("progress_every_steps must be positive")
        if not all(
            isinstance(path, str) and path
            for path in self.categorical_prepack_preserve_sources
        ):
            raise ValueError(
                "categorical_prepack_preserve_sources must contain non-empty strings"
            )
        if self.resume_world_size_policy not in {"strict", "elastic"}:
            raise ValueError(
                "resume.world_size_policy must be one of: strict, elastic"
            )
        if self.resume_from is not None and self.init_from is not None:
            raise ValueError("resume_from and init_from cannot be used together")


@dataclass(frozen=True)
class EvalConfig:
    checkpoint_path: str | Path | None = None
    output_dir: str | Path | None = None
    paths: Sequence[str | Path] | None = None
    split: str | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    device: str | None = None
    amp: bool | None = None
    max_batches: int | None = None
    global_auc_workers: int | None = None
    negative_mask_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "negative_mask_fields",
            normalize_negative_mask_fields(
                self.negative_mask_fields, name="eval.negative_mask_fields"
            ),
        )
        if self.paths is not None and len(self.paths) == 0:
            raise ValueError("eval.paths cannot be empty when provided")
        if self.split is not None and self.split not in {"train", "valid", "test"}:
            raise ValueError("eval.split must be train, valid or test")
        if self.paths is not None and self.split is not None:
            raise ValueError("eval.paths and eval.split cannot be combined")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("eval.batch_size must be positive when provided")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("eval.num_workers must be non-negative when provided")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("eval.max_batches must be positive when provided")
        if self.global_auc_workers is not None and self.global_auc_workers <= 0:
            raise ValueError(
                "eval.global_auc_workers must be positive when provided"
            )


@dataclass(frozen=True)
class RecAgoraConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    source_config_path: str | None = None


# ===== Config Loading =====


DEFAULT_REQUEST_DATA_CONFIG = "configs/data/request_data.yaml"
DEFAULT_TRAINING_CONFIG = "configs/train/training_config.yaml"


def load_recagora_config(
    path: str | Path,
    *,
    data_config: str | Path | None = None,
    data_paths: Sequence[str | Path] | None = None,
    data_root: str | Path | None = None,
    data_split: str | None = None,
    data_manifest: str | Path | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    epochs: int | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    grad_clip_norm: float | None = None,
    disable_grad_clip: bool = False,
    progress_every_steps: int | None = None,
    global_metrics_every_steps: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    amp: bool | None = None,
    checkpoint_every_steps: int | None = None,
    checkpoint_max_to_keep: int | None = None,
    timing_trace_enabled: bool | None = None,
    timing_trace_every_steps: int | None = None,
    timing_trace_rank: int | None = None,
    timing_trace_distributed_sync: bool | None = None,
    communication_trace_enabled: bool | None = None,
    communication_trace_every_steps: int | None = None,
    communication_trace_rank: int | None = None,
    communication_trace_active_steps: int | None = None,
    rank_diagnostics_enabled: bool | None = None,
    rank_diagnostics_every_steps: int | None = None,
    resume_from: str | Path | None = None,
    resume_world_size_policy: ResumeWorldSizePolicy | None = None,
    init_from: str | Path | None = None,
    parquet_start_file: str | Path | None = None,
) -> RecAgoraConfig:
    """Load YAML config and apply CLI/runtime overrides as typed dataclasses."""
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_mapping(config_path)
    if data_config is not None:
        data_config_path = Path(data_config).expanduser().resolve()
        raw = dict(raw)
        raw["data_config"] = str(data_config_path)
    base_dir = config_path.parent
    data_raw, data_base_dir = _load_data_config_section(raw, base_dir=base_dir)
    train_raw, eval_raw, training_base_dir = _load_training_config_sections(
        raw,
        base_dir=base_dir,
    )
    config = RecAgoraConfig(
        data=_parse_data_config(
            data_raw,
            base_dir=data_base_dir,
        ),
        model=_parse_model_config(_required_mapping(raw, "model")),
        train=_parse_train_config(train_raw),
        eval=_parse_eval_config(eval_raw, base_dir=training_base_dir),
        source_config_path=str(config_path),
    )
    config = replace(
        config,
        data=_apply_data_overrides(
            config.data,
            data_paths=data_paths,
            data_root=data_root,
            data_split=data_split,
            data_manifest=data_manifest,
            parquet_start_file=parquet_start_file,
        ),
        train=_apply_train_overrides(
            config.train,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            disable_grad_clip=disable_grad_clip,
            progress_every_steps=progress_every_steps,
            global_metrics_every_steps=global_metrics_every_steps,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            amp=amp,
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_max_to_keep=checkpoint_max_to_keep,
            timing_trace_enabled=timing_trace_enabled,
            timing_trace_every_steps=timing_trace_every_steps,
            timing_trace_rank=timing_trace_rank,
            timing_trace_distributed_sync=timing_trace_distributed_sync,
            communication_trace_enabled=communication_trace_enabled,
            communication_trace_every_steps=communication_trace_every_steps,
            communication_trace_rank=communication_trace_rank,
            communication_trace_active_steps=communication_trace_active_steps,
            rank_diagnostics_enabled=rank_diagnostics_enabled,
            rank_diagnostics_every_steps=rank_diagnostics_every_steps,
            resume_from=resume_from,
            resume_world_size_policy=resume_world_size_policy,
            init_from=init_from,
        ),
    )
    return config


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def _load_data_config_section(
    raw: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[Mapping[str, Any], Path]:
    if raw.get("data") is not None:
        raise ValueError(
            "top-level data is no longer supported; use request_data.yaml"
        )

    data_path = _resolve_config_include_path(
        raw.get("data_config"),
        default_filename=DEFAULT_REQUEST_DATA_CONFIG,
        base_dir=base_dir,
    )
    data_raw = load_yaml_mapping(data_path)
    if "feature_universe" in data_raw:
        raise ValueError(
            "request data feature_universe include was removed; use feature"
        )
    if data_raw.get("feature") is not None or data_raw.get("dataset") is not None:
        return _compose_request_data_v1(data_raw, request_path=data_path), data_path.parent
    return _required_mapping(data_raw, "data"), data_path.parent


def _compose_request_data_v1(
    request_raw: Mapping[str, Any],
    *,
    request_path: Path,
) -> Mapping[str, Any]:
    """Compile the public v1 RequestData includes into the private DataConfig schema."""

    universe_path = _resolve_required_config_include_path(
        request_raw.get("feature"),
        field="feature",
        base_dir=request_path.parent,
    )
    dataset_path = _resolve_required_config_include_path(
        request_raw.get("dataset"),
        field="dataset",
        base_dir=request_path.parent,
    )
    universe = load_yaml_mapping(universe_path)
    dataset = load_yaml_mapping(dataset_path)
    if universe.get("config_kind") != "feature":
        raise ValueError("feature include must have config_kind=feature")
    if dataset.get("config_kind") != "dataset":
        raise ValueError("dataset include must have config_kind=dataset")
    if str(universe.get("version", "")) != str(dataset.get("version", "")):
        raise ValueError(
            "RequestData feature and dataset versions must match: "
            f"feature={universe.get('version')!r} dataset={dataset.get('version')!r}"
        )

    data = dict(_required_mapping(request_raw, "data"))
    forbidden = sorted(
        key
        for key in ("slot_registry", "sparse_id_contract", "sequence")
        if key in data
    )
    if forbidden:
        raise ValueError(
            "v1 RequestData derives private data contracts from its includes; "
            f"do not duplicate {forbidden}"
        )

    bindings = dataset.get("feature_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise ValueError("dataset.feature_bindings must be a feature list")
    source_by_slot: dict[int, str] = {}
    for raw_binding in bindings:
        if not isinstance(raw_binding, Mapping):
            raise ValueError("dataset.feature_bindings entries must be mappings")
        slot_id = int(raw_binding["slot_id"])
        if slot_id in source_by_slot:
            raise ValueError(f"dataset.feature_bindings duplicates slot_id {slot_id}")
        source_by_slot[slot_id] = str(raw_binding["source_path"])

    physical = _required_mapping(dataset, "physical_contract")
    request_physical = _mapping_or_empty(data.get("physical_contract"))
    for field in ("candidate_count",):
        if field in request_physical and int(request_physical[field]) != int(
            physical[field]
        ):
            raise ValueError(
                f"RequestData physical_contract.{field} does not match dataset: "
                f"request={request_physical[field]!r} dataset={physical[field]!r}"
            )
    physical_columns = physical.get("columns")
    if not isinstance(physical_columns, Sequence) or isinstance(
        physical_columns, (str, bytes)
    ):
        raise ValueError("dataset.physical_contract.columns must be a column list")
    physical_columns_by_name = {
        str(column["name"]): column
        for column in physical_columns
        if isinstance(column, Mapping)
    }
    physical_column_names = set(physical_columns_by_name)
    if len(physical_column_names) != len(physical_columns):
        raise ValueError("dataset.physical_contract.columns contains invalid duplicates")
    unexpected_absent = sorted(
        set(str(name) for name in request_physical.get("absent_columns", ()))
        & physical_column_names
    )
    if unexpected_absent:
        raise ValueError(
            "RequestData absent columns are present in the dataset: "
            f"{unexpected_absent}"
        )
    candidate_metadata = _string_tuple(
        data.get("candidate_metadata"), "data.candidate_metadata"
    )
    missing_candidate_metadata = sorted(
        set(candidate_metadata) - physical_column_names
    )
    if missing_candidate_metadata:
        raise ValueError(
            "RequestData candidate_metadata columns are missing from the dataset: "
            f"{missing_candidate_metadata}"
        )
    reserved_fields = {
        str(data.get("request_id_field", "request_id")),
        str(data.get("user_id_field", "user_id")),
        str(data.get("item_id_field", "item_id")),
        *(
            str(name)
            for name in _mapping_or_empty(data.get("labels")).get("names", ())
        ),
    }
    candidate_metadata_collisions = sorted(
        set(candidate_metadata).intersection(reserved_fields)
    )
    if candidate_metadata_collisions:
        raise ValueError(
            "RequestData candidate_metadata must be separate from IDs and labels: "
            f"{candidate_metadata_collisions}"
        )
    for name in candidate_metadata:
        column = physical_columns_by_name[name]
        if int(column.get("candidate_count", physical["candidate_count"])) != int(
            physical["candidate_count"]
        ):
            raise ValueError(
                f"candidate metadata {name!r} candidate_count does not match dataset"
            )
    data["candidate_metadata_dtypes"] = {
        name: str(physical_columns_by_name[name].get("dtype", "int64"))
        for name in candidate_metadata
    }
    raw_history_groups = _mapping_or_empty(physical.get("history_groups"))
    universe_views = _mapping_or_empty(universe.get("views"))
    public_histories = _mapping_or_empty(universe_views.get("histories"))
    history_groups: dict[str, dict[str, Any]] = {}
    for raw_name, raw_group in raw_history_groups.items():
        name = str(raw_name)
        if not isinstance(raw_group, Mapping):
            raise ValueError(
                f"dataset.physical_contract.history_groups.{name} must be a mapping"
            )
        anchor_slot_id = int(raw_group["length_anchor_slot_id"])
        anchor = source_by_slot.get(anchor_slot_id)
        if anchor is None:
            raise ValueError(
                f"history group {name!r} references unknown anchor slot {anchor_slot_id}"
            )
        public_group = _mapping_or_empty(public_histories.get(name))
        dataset_event_order = str(raw_group.get("event_order", "unspecified"))
        public_event_order = str(
            public_group.get("event_order", dataset_event_order)
        )
        if public_event_order != dataset_event_order:
            raise ValueError(
                f"history group {name!r} event_order differs between Universe "
                f"and dataset: {public_event_order!r} vs {dataset_event_order!r}"
            )
        dataset_truncate_side = str(raw_group.get("truncate_side", "tail"))
        public_truncate_side = str(
            public_group.get("truncate_side", dataset_truncate_side)
        )
        if public_truncate_side != dataset_truncate_side:
            raise ValueError(
                f"history group {name!r} truncate_side differs between Universe "
                f"and dataset: {public_truncate_side!r} vs "
                f"{dataset_truncate_side!r}"
            )
        history_groups[name] = {
            "max_len": int(raw_group["max_length"]),
            "truncate_side": dataset_truncate_side,
            "padding_value": 0.0,
            "layout": "ragged",
            "anchor": anchor,
            "event_order": dataset_event_order,
            "padding_side": str(
                public_group.get("padding_side", "model_configured")
            ),
            "direct_padding_side": (
                None
                if raw_group.get("direct_padding_side") is None
                else str(raw_group["direct_padding_side"])
            ),
            "validate_aligned_lengths": True,
        }

    data["slot_registry"] = str(universe_path)
    dataset_sparse_contract = _required_mapping(dataset, "sparse_id_contract")
    data["sparse_id_contract"] = {
        "type": str(dataset_sparse_contract["type"]),
        "layout": str(dataset_path),
    }
    data["sequence"] = {
        "history_len": max(
            (int(group["max_len"]) for group in history_groups.values()),
            default=int(universe.get("defaults", {}).get("history_len", 200)),
        ),
        "history_groups": history_groups,
        "padding_value": 0.0,
    }
    data.setdefault(
        "dtype",
        {
            "label": "float32",
            "id": "int64",
        },
    )
    return data


def _resolve_required_config_include_path(
    value: Any,
    *,
    field: str,
    base_dir: Path,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} include path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"{field} config does not exist: {path}")
    return path.resolve()


def _load_training_config_sections(
    raw: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    if raw.get("train") is not None or raw.get("eval") is not None:
        raise ValueError(
            "top-level train/eval is no longer supported; use training_config.yaml"
        )

    training_path = _resolve_config_include_path(
        raw.get("training_config"),
        default_filename=DEFAULT_TRAINING_CONFIG,
        base_dir=base_dir,
    )
    training_raw = load_yaml_mapping(training_path)
    return (
        _mapping_or_empty(training_raw.get("train")),
        _mapping_or_empty(training_raw.get("eval")),
        training_path.parent,
    )


def _validated_dataset_feature_bindings(
    dataset: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_bindings = dataset.get("feature_bindings")
    if not isinstance(raw_bindings, Sequence) or isinstance(
        raw_bindings,
        (str, bytes),
    ):
        raise ValueError("dataset.feature_bindings must be a feature list")
    bindings: list[Mapping[str, Any]] = []
    bindings_by_slot: dict[int, Mapping[str, Any]] = {}
    for raw_binding in raw_bindings:
        if not isinstance(raw_binding, Mapping):
            raise ValueError("dataset.feature_bindings entry must be a mapping")
        slot_id = int(raw_binding["slot_id"])
        existing = bindings_by_slot.get(slot_id)
        if existing is not None:
            if dict(existing) != dict(raw_binding):
                raise ValueError(
                    f"dataset.feature_bindings conflicts on slot_id {slot_id}"
                )
            continue
        bindings_by_slot[slot_id] = raw_binding
        bindings.append(raw_binding)
    return tuple(bindings)


def _resolve_config_include_path(
    value: Any,
    *,
    default_filename: str,
    base_dir: Path,
) -> Path:
    if value is None:
        path = Path(default_filename)
    else:
        if not isinstance(value, str) or not value:
            raise ValueError(f"config include path must be a non-empty string: {value!r}")
        path = Path(value)
    path = path.expanduser()
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


# ===== Section Parsers =====


def _parse_data_config(
    data: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> DataConfig:
    labels = _mapping_or_empty(data.get("labels"))
    policies = _mapping_or_empty(data.get("policies"))
    sequence = _mapping_or_empty(data.get("sequence"))
    history_groups = _history_groups(sequence.get("history_groups"))
    dtype = _mapping_or_empty(data.get("dtype"))
    index = _mapping_or_empty(data.get("index"))
    streaming = _mapping_or_empty(data.get("streaming"))
    slot_registry = _optional_string(data.get("slot_registry"))
    if slot_registry is None:
        raise ValueError("data.slot_registry is required for FeatureContract planning")
    sparse_id_contract = _parse_sparse_id_contract(
        data.get("sparse_id_contract"),
        base_dir=base_dir,
    )

    return DataConfig(
        paths=_resolve_config_paths(data.get("paths", ()), base_dir=base_dir),
        root=(None if data.get("root") is None else _resolve_config_paths((data["root"],), base_dir=base_dir)[0]),
        split=str(data.get("split", "train")),
        manifest_path=(None if data.get("manifest_path") is None else _resolve_config_paths((data["manifest_path"],), base_dir=base_dir)[0]),
        data_format=_parse_data_format(data.get("format", "parquet_stream")),
        slot_registry=_resolve_slot_registry_path(slot_registry, base_dir),
        sparse_id_contract=sparse_id_contract,
        list_len=int(data.get("list_len", 24)),
        label_names=tuple(labels.get("names", ("click", "atc", "order"))),
        target=str(labels.get("target", "click")),
        label_required=_bool_value(
            labels.get("required", True), "data.labels.required"
        ),
        request_id_field=str(data.get("request_id_field", "request_id")),
        user_id_field=_optional_string(data.get("user_id_field")),
        item_id_field=_optional_string(data.get("item_id_field", "item_id")),
        candidate_metadata_fields=_string_tuple(
            data.get("candidate_metadata"), "data.candidate_metadata"
        ),
        candidate_metadata_dtypes={
            str(name): str(dtype)
            for name, dtype in _mapping_or_empty(
                data.get("candidate_metadata_dtypes")
            ).items()
        },
        missing_slot_policy=policies.get("missing_slot", "fail_fast"),
        bad_length_policy=policies.get("bad_length", "fail_fast"),
        history_len=int(sequence.get("history_len", 200)),
        history_padding_value=float(sequence.get("padding_value", 0.0)),
        history_groups=history_groups,
        label_dtype=dtype.get("label", "float32"),
        id_dtype=dtype.get("id", "int64"),
        index_progress=_bool_value(index.get("progress", False), "data.index.progress"),
        index_progress_interval=int(index.get("progress_interval", 1)),
        index_num_workers=int(index.get("num_workers", 0)),
        parquet_start_file=streaming.get("start_file"),
        parquet_file_cache_size=_optional_non_negative_int(
            streaming.get("parquet_file_cache_size"),
            "data.streaming.parquet_file_cache_size",
        ),
        release_unused_every_batches=_optional_positive_int(
            streaming.get("release_unused_every_batches"),
            "data.streaming.release_unused_every_batches",
        ),
    )


def _parse_sparse_id_contract(
    value: Any,
    *,
    base_dir: Path | None,
) -> SparseIDContractConfig:
    raw = _mapping_or_empty(value)
    mapping_type = str(raw.get("type", "modulo_bucket"))
    if mapping_type not in {"modulo_bucket", "dense_map"}:
        raise ValueError(
            "data.sparse_id_contract.type must be one of: "
            "modulo_bucket, dense_map"
        )
    layout = _optional_string(raw.get("layout"))
    resolved_layout: str | Path | None = None
    if layout is not None:
        resolved_layout = _resolve_slot_registry_path(layout, base_dir)
    return SparseIDContractConfig(
        mapping_type=cast(SparseIDMappingType, mapping_type),
        layout=resolved_layout,
    )


def _parse_model_config(model: Mapping[str, Any]) -> ModelConfig:
    params = model.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("model.params must be a mapping")
    name = model.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("model.name must be a non-empty string")
    class_path = model.get("class_path")
    if not isinstance(class_path, str) or not class_path.strip():
        raise ValueError("model.class_path must be a non-empty string")
    if "feature_universe" in model:
        raise ValueError("model.feature_universe was removed; use model.feature")
    feature_version = model.get("feature")
    if not isinstance(feature_version, str) or not feature_version:
        raise ValueError("model.feature must be a non-empty string")
    features = model.get("features", {})
    embedding = model.get("embedding")
    if not isinstance(features, Mapping):
        raise ValueError("model.features must be a mapping")
    if not isinstance(embedding, Mapping):
        raise ValueError("model.embedding must be a mapping")
    feature_declaration = {
        "feature": feature_version,
        "features": dict(features),
        "embedding": dict(embedding),
    }
    return ModelConfig(
        name=name,
        class_path=class_path.strip(),
        params=params,
        feature_declaration=feature_declaration,
    )


def _resolve_slot_registry_path(
    slot_registry: str,
    base_dir: Path | None,
) -> Path:
    path = Path(slot_registry).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
        candidate = base_dir.parent / path
        if candidate.exists():
            return candidate
    return path

def _parse_train_config(train: Mapping[str, Any]) -> TrainConfig:
    optimizer = _mapping_or_empty(train.get("optimizer"))
    optimizer_params = _mapping_or_empty(optimizer.get("params"))
    global_metrics = _mapping_or_empty(train.get("global_metrics"))
    rank_diagnostics = _mapping_or_empty(train.get("rank_diagnostics"))
    timing_trace = _mapping_or_empty(train.get("timing_trace"))
    communication_trace = _mapping_or_empty(train.get("communication_trace"))
    resume = _mapping_or_empty(train.get("resume"))
    return TrainConfig(
        batch_size=int(train.get("batch_size", 128)),
        num_workers=int(train.get("num_workers", 0)),
        prefetch_factor=_optional_positive_int(
            train.get("prefetch_factor"),
            "train.prefetch_factor",
        ),
        pin_memory=_bool_value(train.get("pin_memory", False), "train.pin_memory"),
        persistent_workers=_bool_value(
            train.get("persistent_workers", False), "train.persistent_workers"
        ),
        lr=float(train.get("lr", 1e-3)),
        optimizer_name=str(optimizer.get("name", train.get("optimizer_name", "adam"))),
        optimizer_params=dict(optimizer_params),
        weight_decay=float(optimizer.get("weight_decay", train.get("weight_decay", 0.0))),
        grad_clip_norm=_optional_positive_float(
            train.get("grad_clip_norm"),
            "grad_clip_norm",
        ),
        epochs=int(train.get("epochs", 1)),
        max_steps=_optional_nonnegative_limit(train.get("max_steps"), "max_steps"),
        seed=int(train.get("seed", 2026)),
        device=str(train.get("device", "auto")),
        amp=_bool_value(train.get("amp", False), "train.amp"),
        output_dir=str(train.get("output_dir", "results/default")),
        checkpoint_every_steps=_optional_positive_int(
            train.get("checkpoint_every_steps", 5000),
            "checkpoint_every_steps",
        ),
        checkpoint_max_to_keep=_optional_positive_int(
            train.get("checkpoint_max_to_keep", 2),
            "checkpoint_max_to_keep",
        )
        or 2,
        progress_every_steps=_optional_positive_int(
            train.get("progress_every_steps", train.get("metrics_every_steps")),
            "train.progress_every_steps",
        ) or 10,
        global_metrics=PeriodicOutputConfig(
            enabled=_bool_value(
                global_metrics.get("enabled", True),
                "train.global_metrics.enabled",
            ),
            every_steps=_optional_positive_int(
                global_metrics.get("every_steps"),
                "train.global_metrics.every_steps",
            ) or 100,
        ),
        rank_diagnostics=PeriodicOutputConfig(
            enabled=_bool_value(
                rank_diagnostics.get("enabled", False),
                "train.rank_diagnostics.enabled",
            ),
            every_steps=_optional_positive_int(
                rank_diagnostics.get("every_steps"),
                "train.rank_diagnostics.every_steps",
            ) or 500,
        ),
        timing_trace=TimingTraceConfig(
            enabled=_bool_value(
                timing_trace.get(
                    "enabled",
                    train.get("timing_trace_enabled", False),
                ),
                "train.timing_trace.enabled",
            ),
            rank=int(timing_trace.get("rank", 0)),
            every_steps=_optional_positive_int(
                timing_trace.get(
                    "every_steps",
                    train.get("timing_trace_every_steps"),
                ),
                "train.timing_trace.every_steps",
            ) or 100,
            distributed_sync=_bool_value(
                timing_trace.get("distributed_sync", False),
                "train.timing_trace.distributed_sync",
            ),
            resolve_delay_steps=int(timing_trace.get("resolve_delay_steps", 2)),
        ),
        communication_trace=CommunicationTraceConfig(
            enabled=_bool_value(
                communication_trace.get("enabled", False),
                "train.communication_trace.enabled",
            ),
            rank=int(communication_trace.get("rank", 0)),
            every_steps=_optional_positive_int(
                communication_trace.get("every_steps"),
                "train.communication_trace.every_steps",
            ) or 500,
            active_steps=_optional_positive_int(
                communication_trace.get("active_steps"),
                "train.communication_trace.active_steps",
            ) or 1,
            distributed_sync=_bool_value(
                communication_trace.get("distributed_sync", False),
                "train.communication_trace.distributed_sync",
            ),
            export_chrome_trace=_bool_value(
                communication_trace.get("export_chrome_trace", False),
                "train.communication_trace.export_chrome_trace",
            ),
        ),
        categorical_prepack_preserve_sources=_string_tuple(
            train.get("categorical_prepack_preserve_sources"),
            "train.categorical_prepack_preserve_sources",
        ),
        resume_from=train.get("resume_from"),
        resume_world_size_policy=cast(
            ResumeWorldSizePolicy,
            resume.get(
                "world_size_policy",
                train.get("resume_world_size_policy", "strict"),
            ),
        ),
        init_from=train.get("init_from"),
        negative_mask_fields=normalize_negative_mask_fields(
            train.get("negative_mask_fields", ()), name="train.negative_mask_fields"
        ),
    )


def _parse_eval_config(
    eval_config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> EvalConfig:
    return EvalConfig(
        checkpoint_path=eval_config.get("checkpoint_path"),
        output_dir=eval_config.get("output_dir"),
        paths=_optional_paths(
            eval_config.get("paths"),
            "eval.paths",
            base_dir=base_dir,
        ),
        split=_optional_string(eval_config.get("split")),
        batch_size=_optional_positive_int(eval_config.get("batch_size"), "eval.batch_size"),
        num_workers=_optional_non_negative_int(
            eval_config.get("num_workers"),
            "eval.num_workers",
        ),
        pin_memory=_optional_bool(eval_config.get("pin_memory")),
        persistent_workers=_optional_bool(eval_config.get("persistent_workers")),
        device=_optional_string(eval_config.get("device")),
        amp=_optional_bool(eval_config.get("amp")),
        max_batches=_optional_positive_int(
            eval_config.get("max_batches"),
            "eval.max_batches",
        ),
        global_auc_workers=_optional_positive_int(
            eval_config.get("global_auc_workers"),
            "eval.global_auc_workers",
        ),
        negative_mask_fields=normalize_negative_mask_fields(
            eval_config.get("negative_mask_fields", ()), name="eval.negative_mask_fields"
        ),
    )


def _optional_paths(
    value: Any,
    name: str,
    *,
    base_dir: Path | None = None,
) -> tuple[str | Path, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return _resolve_config_paths((value,), base_dir=base_dir)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        paths = tuple(cast(Sequence[str | Path], value))
        if not paths:
            raise ValueError(f"{name} cannot be empty when provided")
        return _resolve_config_paths(paths, base_dir=base_dir)
    raise ValueError(f"{name} must be a path string or a sequence of path strings")


def _resolve_config_paths(
    values: Any,
    *,
    base_dir: Path | None,
) -> tuple[str | Path, ...]:
    if isinstance(values, (str, Path)):
        raw_paths: Sequence[str | Path] = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_paths = cast(Sequence[str | Path], values)
    else:
        raise ValueError("data.paths must be a path string or a sequence of path strings")
    resolved: list[str | Path] = []
    for raw_path in raw_paths:
        if isinstance(raw_path, str) and "://" in raw_path:
            resolved.append(raw_path)
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        resolved.append(path.as_posix())
    return tuple(resolved)


def _parse_data_format(value: Any) -> DataFormat:
    data_format = str(value)
    if data_format == "parquet":
        data_format = "parquet_stream"
    if data_format != "parquet_stream":
        raise ValueError(f"Unsupported data.format: {data_format!r}")
    return cast(DataFormat, data_format)


# ===== Override Application =====


def _apply_data_overrides(
    config: DataConfig,
    *,
    data_paths: Sequence[str | Path] | None,
    data_root: str | Path | None,
    data_split: str | None,
    data_manifest: str | Path | None,
    parquet_start_file: str | Path | None,
) -> DataConfig:
    values = config.__dict__.copy()
    if data_paths is not None:
        if not data_paths:
            raise ValueError("data_paths override cannot be empty")
        if config.root is not None:
            raise ValueError("data_paths cannot override HotUser data.root")
        values["paths"] = tuple(data_paths)
    if data_root is not None:
        if config.root is None:
            raise ValueError("data_root override requires a HotUser data config")
        values["root"] = data_root
    if data_split is not None:
        if config.root is None:
            raise ValueError("data_split override requires a HotUser data config")
        values["split"] = data_split
    if data_manifest is not None:
        if config.root is None:
            raise ValueError("data_manifest override requires a HotUser data config")
        values["manifest_path"] = data_manifest
    if parquet_start_file is not None:
        values["parquet_start_file"] = parquet_start_file
    if values == config.__dict__:
        return config
    return DataConfig(**values)


def _apply_train_overrides(
    config: TrainConfig,
    *,
    output_dir: str | None,
    device: str | None,
    epochs: int | None,
    max_steps: int | None,
    batch_size: int | None,
    lr: float | None,
    grad_clip_norm: float | None,
    disable_grad_clip: bool,
    progress_every_steps: int | None,
    global_metrics_every_steps: int | None,
    num_workers: int | None,
    pin_memory: bool | None,
    persistent_workers: bool | None,
    amp: bool | None,
    checkpoint_every_steps: int | None,
    checkpoint_max_to_keep: int | None,
    timing_trace_enabled: bool | None,
    timing_trace_every_steps: int | None,
    timing_trace_rank: int | None,
    timing_trace_distributed_sync: bool | None,
    communication_trace_enabled: bool | None,
    communication_trace_every_steps: int | None,
    communication_trace_rank: int | None,
    communication_trace_active_steps: int | None,
    rank_diagnostics_enabled: bool | None,
    rank_diagnostics_every_steps: int | None,
    resume_from: str | Path | None,
    resume_world_size_policy: ResumeWorldSizePolicy | None,
    init_from: str | Path | None,
) -> TrainConfig:
    if grad_clip_norm is not None and disable_grad_clip:
        raise ValueError(
            "grad_clip_norm and disable_grad_clip cannot be set together"
        )
    values = config.__dict__.copy()
    if output_dir is not None:
        values["output_dir"] = output_dir
    if device is not None:
        values["device"] = device
    if epochs is not None:
        values["epochs"] = epochs
    if max_steps is not None:
        values["max_steps"] = _normalize_nonnegative_limit(
            max_steps,
            "max_steps",
        )
    if batch_size is not None:
        values["batch_size"] = batch_size
    if lr is not None:
        values["lr"] = lr
    if disable_grad_clip:
        values["grad_clip_norm"] = None
    elif grad_clip_norm is not None:
        values["grad_clip_norm"] = grad_clip_norm
    if progress_every_steps is not None:
        values["progress_every_steps"] = progress_every_steps
    if global_metrics_every_steps is not None:
        values["global_metrics"] = replace(
            config.global_metrics,
            every_steps=global_metrics_every_steps,
        )
    if num_workers is not None:
        values["num_workers"] = num_workers
    if pin_memory is not None:
        values["pin_memory"] = pin_memory
    if persistent_workers is not None:
        values["persistent_workers"] = persistent_workers
    if amp is not None:
        values["amp"] = amp
    if checkpoint_every_steps is not None:
        values["checkpoint_every_steps"] = checkpoint_every_steps
    if checkpoint_max_to_keep is not None:
        values["checkpoint_max_to_keep"] = checkpoint_max_to_keep
    timing_trace_values: dict[str, Any] = {}
    if timing_trace_enabled is not None:
        timing_trace_values["enabled"] = timing_trace_enabled
    if timing_trace_every_steps is not None:
        timing_trace_values["every_steps"] = timing_trace_every_steps
    if timing_trace_rank is not None:
        timing_trace_values["rank"] = timing_trace_rank
    if timing_trace_distributed_sync is not None:
        timing_trace_values["distributed_sync"] = timing_trace_distributed_sync
    if timing_trace_values:
        values["timing_trace"] = replace(config.timing_trace, **timing_trace_values)

    communication_trace_values: dict[str, Any] = {}
    if communication_trace_enabled is not None:
        communication_trace_values["enabled"] = communication_trace_enabled
    if communication_trace_every_steps is not None:
        communication_trace_values["every_steps"] = communication_trace_every_steps
    if communication_trace_rank is not None:
        communication_trace_values["rank"] = communication_trace_rank
    if communication_trace_active_steps is not None:
        communication_trace_values["active_steps"] = communication_trace_active_steps
    if communication_trace_values:
        values["communication_trace"] = replace(
            config.communication_trace,
            **communication_trace_values,
        )

    rank_diagnostics_values: dict[str, Any] = {}
    if rank_diagnostics_enabled is not None:
        rank_diagnostics_values["enabled"] = rank_diagnostics_enabled
    if rank_diagnostics_every_steps is not None:
        rank_diagnostics_values["every_steps"] = rank_diagnostics_every_steps
    if rank_diagnostics_values:
        values["rank_diagnostics"] = replace(
            config.rank_diagnostics,
            **rank_diagnostics_values,
        )
    if resume_from is not None:
        values["resume_from"] = resume_from
    if resume_world_size_policy is not None:
        values["resume_world_size_policy"] = resume_world_size_policy
    if init_from is not None:
        values["init_from"] = init_from
    return TrainConfig(**values)


# ===== Scalar Coercion Helpers =====


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return parsed


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return parsed


def _optional_nonnegative_limit(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _normalize_nonnegative_limit(int(value), name)


def _normalize_nonnegative_limit(value: int, name: str) -> int | None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative when provided")
    if value == 0:
        return None
    return value


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative when provided")
    return parsed


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _bool_value(value, "boolean config value")


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    parsed = tuple(str(item) for item in value)
    if any(not item for item in parsed):
        raise ValueError(f"{name} must contain non-empty strings")
    return parsed


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Top-level {key} config must be a mapping")
    return value


def _history_groups(value: Any) -> Mapping[str, HistoryGroupConfig]:
    raw_groups = _mapping_or_empty(value)
    groups: dict[str, HistoryGroupConfig] = {}
    for raw_name, raw_group in raw_groups.items():
        name = str(raw_name)
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"sequence.history_groups.{name} must be a mapping")
        if "max_len" not in raw_group:
            raise ValueError(f"sequence.history_groups.{name} requires max_len")
        groups[name] = HistoryGroupConfig(
            max_len=int(raw_group["max_len"]),
            truncate_side=str(raw_group.get("truncate_side", "head")),  # type: ignore[arg-type]
            padding_value=float(raw_group.get("padding_value", 0.0)),
            layout=str(raw_group.get("layout", "ragged")),  # type: ignore[arg-type]
            anchor=_optional_string(raw_group.get("anchor")),
            event_order=(
                None
                if raw_group.get("event_order") is None
                else str(raw_group["event_order"])
            ),  # type: ignore[arg-type]
            padding_side=(
                None
                if raw_group.get("padding_side") is None
                else str(raw_group["padding_side"])
            ),  # type: ignore[arg-type]
            direct_padding_side=(
                None
                if raw_group.get("direct_padding_side") is None
                else str(raw_group["direct_padding_side"])
            ),  # type: ignore[arg-type]
            validate_aligned_lengths=_bool_value(
                raw_group.get("validate_aligned_lengths", False),
                f"sequence.history_groups.{name}.validate_aligned_lengths",
            ),
        )
    return groups


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a mapping, got {type(value).__name__}")
    return value

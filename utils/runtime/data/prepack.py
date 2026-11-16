from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Mapping, Protocol, Sequence

import torch

from recagora import RecAgoraSparseBatch

from utils.compiler.config import normalize_negative_mask_fields
from utils.runtime.data.candidate_mask import mask_request_batch_candidates
from utils.runtime.data.request_batch import (
    HistoryGroupLengths,
    PackedCategoricalBatch,
    RaggedFeature,
    RequestBatch,
)
from utils.compiler.feature_contract import AdapterPlan, EmbeddingInputPlan


class SparseFeatureLike(Protocol):
    name: str
    vocab_size: int


class JaggedFeatureLike(Protocol):
    values: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True)
class SparseIDMapping:
    mapping_type: str = "modulo_bucket"


@dataclass(frozen=True)
class CategoricalPrepackPlan:
    """Model-neutral worker plan compiled from the feature contract."""

    adapter_plan: AdapterPlan
    input_plan: EmbeddingInputPlan
    preserve_source_paths: tuple[str, ...] = ()
    negative_mask_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "negative_mask_fields",
            normalize_negative_mask_fields(self.negative_mask_fields),
        )


@dataclass(frozen=True)
class _MappedIDSpan:
    start: int
    length: int
    history_group: str | None


@dataclass(frozen=True)
class _PrepackedSparseFeatures:
    """Private carrier between worker prepack and the injected lookup module."""

    batch_size: int
    list_len: int
    sparse_kjt: Any
    feature_shapes: Mapping[str, tuple[int, ...]]
    feature_segment_lengths: Mapping[str, torch.Tensor]
    sequence_group_lengths: Mapping[str, torch.Tensor]
    mapped_id_spans: Mapping[int, _MappedIDSpan]

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "_PrepackedSparseFeatures":
        return _PrepackedSparseFeatures(
            batch_size=self.batch_size,
            list_len=self.list_len,
            sparse_kjt=self.sparse_kjt.to(device, non_blocking=non_blocking),
            feature_shapes=self.feature_shapes,
            feature_segment_lengths={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.feature_segment_lengths.items()
            },
            sequence_group_lengths={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.sequence_group_lengths.items()
            },
            mapped_id_spans=self.mapped_id_spans,
        )


def prepack_request_batch_categorical(
    batch: RequestBatch,
    *,
    plan: CategoricalPrepackPlan,
) -> RequestBatch:
    """Map and concatenate categorical IDs inside the DataLoader worker."""
    timing: dict[str, float] = {}
    total_started_at = time.perf_counter()
    _require_cpu_request_batch(batch)
    if plan.negative_mask_fields:
        mask_started_at = time.perf_counter()
        batch = mask_request_batch_candidates(
            batch, fields=plan.negative_mask_fields, input_plan=plan.input_plan
        )
        timing["prepare_candidate_mask_sec"] = time.perf_counter() - mask_started_at
    adapter = _normalized_adapter(plan.adapter_plan.as_mapping())
    packed = _build_listwise_sparse_payload(
        batch,
        adapter,
        categorical_features=plan.input_plan.categorical_features,
        list_len=batch.list_len,
        canonical_history_group_lengths=batch.history_group_lengths,
        preparation_timing=timing,
    )
    worker_timing = {
        f"prepare_worker_{name.removeprefix('prepare_')}": float(value)
        for name, value in timing.items()
        if name.startswith("prepare_") and name.endswith("_sec")
    }
    worker_timing["prepare_worker_pack_total_sec"] = (
        time.perf_counter() - total_started_at
    )
    packed = replace(packed, preparation_timing=worker_timing)

    categorical_sources = {
        _field_source_and_level(raw_spec)[0]
        for raw_spec in _mapping_or_empty(
            adapter.get("categorical_features")
        ).values()
    }
    direct_sources = {
        _field_source_and_level(raw_spec)[0]
        for raw_spec in _mapping_or_empty(adapter.get("direct_features")).values()
    }
    retained_sources = direct_sources | set(plan.preserve_source_paths)
    pruned_features = {
        source_path: value
        for source_path, value in batch.features.items()
        if source_path not in categorical_sources or source_path in retained_sources
    }
    return batch.with_packed_categorical(
        packed,
        features=pruned_features,
    )


def prepare_embedding_model_batch(
    batch: RequestBatch,
    *,
    adapter_plan: AdapterPlan,
    input_plan: EmbeddingInputPlan,
    keyed_jagged_tensor_cls: type[Any],
    collect_timing: bool = False,
) -> RecAgoraSparseBatch:
    """Prepare one native listwise KJT regardless of the downstream model view."""
    timing: dict[str, float] | None = {} if collect_timing else None
    total_started_at = _timing_started_at(timing)
    validate_started_at = _timing_started_at(timing)
    _require_cpu_request_batch(batch)
    _timing_add(timing, "prepare_validate_cpu_sec", validate_started_at)
    adapter_started_at = _timing_started_at(timing)
    adapter = _normalized_adapter(adapter_plan.as_mapping())
    _timing_add(timing, "prepare_adapter_sec", adapter_started_at)
    torchrec_batch = request_batch_to_torchrec_listwise_ranking_batch(
        batch,
        categorical_features=input_plan.categorical_features,
        keyed_jagged_tensor_cls=keyed_jagged_tensor_cls,
        preparation_timing=timing,
    )
    dense_started_at = _timing_started_at(timing)
    dense = _prepared_dense_features(batch, adapter)
    labels = _prepared_labels(batch, batch)
    metadata = _prepared_metadata(batch, torchrec_batch)
    _timing_add(timing, "prepare_dense_metadata_sec", dense_started_at)
    _timing_add(timing, "prepare_instrumented_total_sec", total_started_at)
    return RecAgoraSparseBatch(
        batch_size=torchrec_batch.batch_size,
        list_len=torchrec_batch.list_len,
        _lookup_payload=torchrec_batch,
        dense=dense,
        labels=labels,
        metadata=metadata,
        candidate_metadata=dict(batch.candidate_metadata),
        history_group_lengths=dict(batch.history_group_lengths),
        preparation_timing={} if timing is None else timing,
        candidate_valid_mask=batch.candidate_valid_mask,
    )


def request_batch_to_torchrec_listwise_ranking_batch(
    batch: RequestBatch,
    *,
    categorical_features: Sequence[SparseFeatureLike],
    keyed_jagged_tensor_cls: type[Any],
    preparation_timing: dict[str, float] | None = None,
) -> _PrepackedSparseFeatures:
    list_len = batch.list_len
    batch_size = batch.batch_size
    specs = tuple(categorical_features)
    (
        sparse_kjt,
        feature_shapes,
        feature_segment_lengths,
        sequence_group_lengths,
        mapped_id_spans,
    ) = _build_listwise_sparse_kjt(
        batch,
        categorical_features=specs,
        keyed_jagged_tensor_cls=keyed_jagged_tensor_cls,
        list_len=list_len,
        preparation_timing=preparation_timing,
    )
    return _PrepackedSparseFeatures(
        batch_size=batch_size,
        list_len=list_len,
        sparse_kjt=sparse_kjt,
        feature_shapes=feature_shapes,
        feature_segment_lengths=feature_segment_lengths,
        sequence_group_lengths=sequence_group_lengths,
        mapped_id_spans=mapped_id_spans,
    )


def _build_listwise_sparse_kjt(
    batch: RequestBatch,
    *,
    categorical_features: Sequence[SparseFeatureLike],
    keyed_jagged_tensor_cls: type[Any],
    list_len: int,
    preparation_timing: dict[str, float] | None = None,
) -> tuple[
    Any,
    dict[str, tuple[int, ...]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[int, _MappedIDSpan],
]:
    packed = batch.packed_categorical
    if packed is None:
        raise ValueError(
            "prepare_embedding_model_batch requires worker-prepacked categorical "
            "data; construct the runtime loader with build_data_loader and the "
            "compiled AdapterPlan/EmbeddingInputPlan"
        )
    _validate_packed_categorical(
        packed,
        categorical_features=categorical_features,
        batch_size=batch.batch_size,
        list_len=list_len,
        history_group_lengths=batch.history_group_lengths,
    )
    if preparation_timing is not None:
        for name, value in packed.preparation_timing.items():
            if name.startswith("prepare_") and name.endswith("_sec"):
                preparation_timing[name] = float(value)

    kjt_started_at = _timing_started_at(preparation_timing)
    kjt = keyed_jagged_tensor_cls.from_lengths_sync(
        keys=list(packed.keys),
        values=packed.values,
        lengths=packed.lengths,
    )
    _timing_add(preparation_timing, "prepare_kjt_wrap_sec", kjt_started_at)
    feature_segment_lengths = dict(packed.feature_segment_lengths)
    sequence_group_lengths = dict(packed.sequence_group_lengths)
    mapped_id_spans = _mapped_id_spans(packed)
    pin_started_at = _timing_started_at(preparation_timing)
    if batch.request_id.is_pinned():
        if not packed.values.is_pinned() or not packed.lengths.is_pinned():
            pin_kjt = getattr(kjt, "pin_memory", None)
            if not callable(pin_kjt):
                raise TypeError(
                    "TorchRec KJT object does not support pinned CPU transfer"
                )
            kjt = pin_kjt()
        feature_segment_lengths = {
            name: value if value.is_pinned() else value.pin_memory()
            for name, value in feature_segment_lengths.items()
        }
        sequence_group_lengths = {
            name: value if value.is_pinned() else value.pin_memory()
            for name, value in sequence_group_lengths.items()
        }
    _timing_add(preparation_timing, "prepare_pin_sec", pin_started_at)
    return (
        kjt,
        dict(packed.feature_shapes),
        feature_segment_lengths,
        sequence_group_lengths,
        mapped_id_spans,
    )


def _mapped_id_spans(
    packed: PackedCategoricalBatch,
) -> dict[int, _MappedIDSpan]:
    return {
        int(slot_id): _MappedIDSpan(
            start=int(span[0]),
            length=int(span[1]),
            history_group=None if span[2] is None else str(span[2]),
        )
        for slot_id, span in packed.mapped_id_spans.items()
    }


def _build_listwise_sparse_payload(
    batch: RequestBatch,
    adapter: Mapping[str, Any],
    *,
    categorical_features: Sequence[SparseFeatureLike],
    list_len: int,
    canonical_history_group_lengths: Mapping[str, HistoryGroupLengths] | None = None,
    preparation_timing: dict[str, float] | None = None,
) -> PackedCategoricalBatch:
    values_by_key: list[torch.Tensor] = []
    lengths_by_key: list[torch.Tensor] = []
    feature_shapes: dict[str, tuple[int, ...]] = {}
    feature_segment_lengths: dict[str, torch.Tensor] = {}
    mapped_id_spans: dict[int, tuple[int, int, str | None]] = {}
    keys: list[str] = []
    value_offset = 0

    categorical_mapping = _mapping_or_empty(adapter.get("categorical_features"))

    for spec in categorical_features:
        extract_started_at = _timing_started_at(preparation_timing)
        value = _listwise_feature_value(
            batch,
            categorical_mapping,
            spec.name,
            list_len=list_len,
        )
        _timing_add(
            preparation_timing,
            "prepare_feature_extract_sec",
            extract_started_at,
        )
        if _is_jagged_feature(value):
            safe_values, semantic_lengths, feature_shape = _safe_jagged_embedding_ids(
                value,
                spec=spec,
                preparation_timing=preparation_timing,
            )
            lengths_started_at = _timing_started_at(preparation_timing)
            if value.level == "item":
                if semantic_lengths.numel() % list_len != 0:
                    raise ValueError(
                        f"item feature {spec.name!r} segment count must be divisible "
                        f"by list_len={list_len}"
                    )
                lengths = semantic_lengths.reshape(-1, list_len).sum(dim=1)
                feature_shape = (int(lengths.numel()), list_len)
            else:
                lengths = semantic_lengths
            _timing_add(
                preparation_timing,
                "prepare_lengths_sec",
                lengths_started_at,
            )
        else:
            lengths_started_at = _timing_started_at(preparation_timing)
            value = _squeeze_sequence_value(value)
            feature_level = str(
                getattr(spec, "source_level", getattr(spec, "role", "request"))
            )
            if feature_level == "item":
                if value.ndim < 2 or int(value.shape[1]) != list_len:
                    raise ValueError(
                        f"listwise item feature {spec.name!r} must have shape "
                        f"[B, {list_len}, ...]"
                    )
                batch_size = int(value.shape[0])
                value_width = int(value[0, 0].numel()) if batch_size else 1
                feature_shape = tuple(int(dim) for dim in value.shape)
                semantic_lengths = torch.full(
                    (batch_size * list_len,),
                    value_width,
                    dtype=torch.long,
                    device=value.device,
                )
                lengths = semantic_lengths.reshape(batch_size, list_len).sum(dim=1)
            elif value.ndim == 2:
                sample_count = int(value.shape[0])
                value_width = int(value.shape[1])
                feature_shape = (sample_count, value_width)
                lengths = torch.full(
                    (sample_count,),
                    value_width,
                    dtype=torch.long,
                    device=value.device,
                )
                semantic_lengths = lengths
            elif value.ndim >= 3:
                sample_count = int(value.shape[0])
                feature_shape = tuple(int(dim) for dim in value.shape)
                lengths = torch.full(
                    (sample_count,),
                    int(value[0].numel()) if sample_count else 0,
                    dtype=torch.long,
                    device=value.device,
                )
                semantic_lengths = lengths
            else:
                raise ValueError(
                    f"listwise categorical feature {spec.name!r} must have shape [B, K] or [B, L, K]"
                )
            _timing_add(
                preparation_timing,
                "prepare_lengths_sec",
                lengths_started_at,
            )
            mapping_started_at = _timing_started_at(preparation_timing)
            safe_values = _safe_embedding_ids(value, spec=spec).reshape(-1)
            _timing_add(
                preparation_timing,
                "prepare_id_mapping_sec",
                mapping_started_at,
            )
        if lengths.ndim != 1:
            raise ValueError(
                f"listwise categorical feature {spec.name!r} lengths must be 1-D"
            )
        keys.append(spec.name)
        values_by_key.append(safe_values)
        lengths_by_key.append(lengths)
        feature_shapes[spec.name] = feature_shape
        feature_segment_lengths[spec.name] = semantic_lengths
        value_count = int(safe_values.numel())
        if bool(getattr(spec, "mapped_ids_exposed", False)):
            slot_id = getattr(spec, "slot_id", None)
            history_group = getattr(spec, "history_group", None)
            if not isinstance(slot_id, int):
                raise ValueError(
                    f"mapped-ID feature {spec.name!r} has no compiled slot ID"
                )
            if history_group is not None and not isinstance(history_group, str):
                raise ValueError(f"mapped-ID slot {slot_id} has invalid history group")
            if slot_id in mapped_id_spans:
                raise ValueError(f"duplicate mapped-ID slot {slot_id}")
            mapped_id_spans[slot_id] = (
                value_offset,
                value_count,
                history_group,
            )
        value_offset += value_count

    if not keys:
        raise ValueError("TorchRec KJT requires at least one sparse feature")

    cat_values_started_at = _timing_started_at(preparation_timing)
    values = torch.cat(values_by_key)
    _timing_add(
        preparation_timing,
        "prepare_cat_values_sec",
        cat_values_started_at,
    )
    cat_lengths_started_at = _timing_started_at(preparation_timing)
    lengths = torch.cat(lengths_by_key)
    _timing_add(
        preparation_timing,
        "prepare_cat_lengths_sec",
        cat_lengths_started_at,
    )
    sequence_validate_started_at = _timing_started_at(preparation_timing)
    sequence_group_lengths = _validate_sequence_group_lengths(
        categorical_features,
        feature_segment_lengths,
    )
    canonical_history_group_lengths = canonical_history_group_lengths or {}
    for group_name, computed in tuple(sequence_group_lengths.items()):
        canonical = canonical_history_group_lengths.get(group_name)
        if canonical is None:
            continue
        expected = canonical.truncated_lengths
        if not torch.equal(computed.to(torch.long), expected.to(torch.long)):
            raise ValueError(
                f"worker categorical history lengths for group {group_name!r} "
                "do not match reader-validated canonical lengths"
            )
        sequence_group_lengths[group_name] = expected
    _timing_add(
        preparation_timing,
        "prepare_sequence_validate_sec",
        sequence_validate_started_at,
    )
    return PackedCategoricalBatch(
        keys=tuple(keys),
        values=values,
        lengths=lengths,
        feature_shapes=feature_shapes,
        feature_segment_lengths=feature_segment_lengths,
        sequence_group_lengths=sequence_group_lengths,
        batch_size=batch.batch_size,
        list_len=list_len,
        preparation_timing={},
        mapped_id_spans=mapped_id_spans,
    )


def _validate_packed_categorical(
    packed: PackedCategoricalBatch,
    *,
    categorical_features: Sequence[SparseFeatureLike],
    batch_size: int,
    list_len: int,
    history_group_lengths: Mapping[str, HistoryGroupLengths] | None = None,
) -> None:
    expected_keys = tuple(spec.name for spec in categorical_features)
    if packed.keys != expected_keys:
        raise ValueError(
            "packed categorical keys do not match the active embedding input plan"
        )
    if packed.batch_size != batch_size or packed.list_len != list_len:
        raise ValueError(
            "packed categorical batch shape does not match the RequestBatch"
        )
    expected_mapped_slots = {
        int(getattr(spec, "slot_id"))
        for spec in categorical_features
        if bool(getattr(spec, "mapped_ids_exposed", False))
    }
    if set(packed.mapped_id_spans) != expected_mapped_slots:
        raise ValueError(
            "packed mapped-ID slots do not match the active embedding input plan"
        )
    for group_name, packed_lengths in packed.sequence_group_lengths.items():
        canonical = (history_group_lengths or {}).get(group_name)
        if canonical is None:
            continue
        if not torch.equal(
            packed_lengths.to(torch.long),
            canonical.truncated_lengths.to(torch.long),
        ):
            raise ValueError(
                f"packed categorical history lengths for group {group_name!r} "
                "do not match reader-validated canonical lengths"
            )


def _listwise_feature_value(
    batch: RequestBatch,
    feature_mapping: Mapping[str, Any],
    feature_name: str,
    *,
    list_len: int,
) -> torch.Tensor | RaggedFeature:
    raw_spec = feature_mapping.get(feature_name)
    if raw_spec is None:
        value = batch.feature(feature_name)
    else:
        source, level = _field_source_and_level(raw_spec)
        value = batch.feature(source)
        if isinstance(value, RaggedFeature):
            if level is not None and value.level != level:
                raise ValueError(
                    f"feature {feature_name!r} level mismatch: {value.level!r} vs {level!r}"
                )
            return value
        if level == "request":
            if value.ndim == 1:
                value = value.unsqueeze(-1)
        elif level == "item" and value.ndim >= 2 and value.shape[1] != list_len:
            raise ValueError(
                f"item-level feature {feature_name!r} second dim must equal list_len={list_len}"
            )
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"feature {feature_name!r} must be a tensor or RaggedFeature")
    return value


def _field_source_and_level(raw_spec: Any) -> tuple[str, str | None]:
    if isinstance(raw_spec, str):
        return raw_spec, None
    if isinstance(raw_spec, Mapping):
        source = raw_spec.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("TorchRec field mapping requires a non-empty string source")
        level = raw_spec.get("level")
        if level is not None and not isinstance(level, str):
            raise ValueError("TorchRec field mapping level must be a string when provided")
        return source, level
    raise ValueError("TorchRec field mapping entries must be strings or mappings")


def _feature_value(
    batch: Mapping[str, Any],
    container_name: str,
    feature_name: str,
) -> Any:
    container = batch.get(container_name) if container_name else None
    if isinstance(container, Mapping) and feature_name in container:
        return container[feature_name]
    if feature_name in batch:
        return batch[feature_name]
    raise KeyError(f"Missing feature {feature_name!r} in {container_name!r}")


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("TorchRec adapter sections must be mappings when provided")
    return value


def _normalized_adapter(adapter: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(adapter)
    out["categorical_features"] = dict(
        _mapping_or_empty(adapter.get("categorical_features"))
    )
    out["direct_features"] = dict(_mapping_or_empty(adapter.get("direct_features")))
    return out


def _prepared_dense_features(
    features: Mapping[str, Any] | RequestBatch,
    adapter: Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(features, RequestBatch):
        dense_mapping = _mapping_or_empty(adapter.get("direct_features"))
        out: dict[str, torch.Tensor] = {}
        for feature_name, raw_spec in dense_mapping.items():
            source, _ = _field_source_and_level(raw_spec)
            value = features.feature(source)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"listwise dense feature {feature_name!r} must be a tensor"
                )
            out[str(feature_name)] = value
        return out
    dense_features = features.get("direct_features")
    if isinstance(dense_features, Mapping):
        return dict(dense_features)
    return {}


def _prepared_labels(
    batch: RequestBatch,
    features: Mapping[str, Any] | RequestBatch,
) -> Mapping[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, value in batch.labels.items():
        out[str(name)] = value.to(torch.float32)
    out.setdefault("target_label", batch.target_label.to(torch.float32))
    pointwise_target = (
        features.get("target_label") if isinstance(features, Mapping) else None
    )
    if isinstance(pointwise_target, torch.Tensor):
        out.setdefault("target_label", pointwise_target.to(torch.float32))
    return out


def _prepared_metadata(
    batch: RequestBatch,
    torchrec_batch: _PrepackedSparseFeatures,
) -> Mapping[str, Any]:
    out: dict[str, Any] = {
        "batch_size": torchrec_batch.batch_size,
        "list_len": torchrec_batch.list_len,
    }
    out["request_id"] = batch.request_id
    out["item_id"] = batch.item_id
    if batch.user_id is not None:
        out["user_id"] = batch.user_id
    out["request_batch_metadata"] = batch.metadata
    return out


def map_sparse_ids(
    value: torch.Tensor,
    *,
    vocab_size: int,
    mapping: SparseIDMapping | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    mapping_type = _mapping_type(mapping)
    dense_id = sparse_id_to_dense_id(value)
    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive for sparse id mapping")
    dense_id = dense_id.long()
    if mapping_type == "modulo_bucket":
        return dense_id.remainder(int(vocab_size))
    if mapping_type == "dense_map":
        invalid = (dense_id < 0) | (dense_id >= int(vocab_size))
        if bool(invalid.any().item()):
            invalid_values = dense_id[invalid]
            namespace = (
                mapping.get("namespace")
                if isinstance(mapping, Mapping)
                else None
            )
            mapping_version = (
                mapping.get("mapping_version")
                if isinstance(mapping, Mapping)
                else None
            )
            raise ValueError(
                "dense_map sparse id is out of range: "
                f"namespace={namespace!r} mapping_version={mapping_version!r} "
                f"vocab_size={int(vocab_size)} "
                f"invalid_min={int(invalid_values.min().item())} "
                f"invalid_max={int(invalid_values.max().item())} "
                f"invalid_count={int(invalid.sum().item())}"
            )
        return dense_id
    raise ValueError(
        f"Unsupported sparse id mapping {mapping_type!r}; expected one of: "
        "modulo_bucket, dense_map"
    )


def sparse_id_to_dense_id(value: torch.Tensor) -> torch.Tensor:
    """Return the dataset-provided namespace-local ID before policy handling."""

    return value


def _safe_embedding_ids(
    value: torch.Tensor,
    *,
    spec: SparseFeatureLike,
) -> torch.Tensor:
    return map_sparse_ids(
        value,
        vocab_size=spec.vocab_size,
        mapping=_feature_id_mapping(spec),
    )


def _safe_jagged_embedding_ids(
    value: Any,
    *,
    spec: SparseFeatureLike,
    preparation_timing: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    lengths_started_at = _timing_started_at(preparation_timing)
    values = value.values
    lengths = value.lengths.to(torch.long)
    if not isinstance(values, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise ValueError(f"jagged sparse feature {spec.name!r} values/lengths must be tensors")
    if lengths.ndim != 1:
        raise ValueError(f"jagged sparse feature {spec.name!r} lengths must be 1-D")
    metadata = getattr(value, "metadata", None)
    if metadata is None:
        if values.device.type != "cpu":
            raise ValueError(
                f"jagged sparse feature {spec.name!r} requires CPU-validated metadata"
            )
        value_count = int(lengths.sum().item())
        max_len = int(lengths.max().item()) if int(lengths.numel()) else 0
    else:
        value_count = int(metadata.value_count)
        max_len = int(metadata.max_length)
    if value_count != int(values.numel()):
        raise ValueError(f"jagged sparse feature {spec.name!r} values/lengths are inconsistent")
    max_sequence_length = getattr(spec, "max_sequence_length", None)
    if max_sequence_length is not None and max_len > int(max_sequence_length):
        raise ValueError(
            f"jagged sparse feature {spec.name!r} max length {max_len} exceeds "
            f"contract maximum {int(max_sequence_length)}"
        )
    _timing_add(preparation_timing, "prepare_lengths_sec", lengths_started_at)
    mapping_started_at = _timing_started_at(preparation_timing)
    safe_values = _safe_embedding_ids(values, spec=spec).reshape(-1)
    _timing_add(
        preparation_timing,
        "prepare_id_mapping_sec",
        mapping_started_at,
    )
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple):
        feature_shape = tuple(int(dim) for dim in shape)
    else:
        feature_shape = (int(lengths.numel()), max_len)
    return safe_values, lengths, feature_shape


def _validate_sequence_group_lengths(
    specs: Sequence[SparseFeatureLike],
    semantic_lengths: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    anchors: dict[str, tuple[str, torch.Tensor]] = {}
    for spec in specs:
        group = getattr(spec, "history_group", None)
        if group is None:
            continue
        lengths = semantic_lengths.get(spec.name)
        if not isinstance(lengths, torch.Tensor):
            raise KeyError(f"sequence feature {spec.name!r} has no semantic lengths")
        if lengths.device.type != "cpu":
            raise ValueError("sequence alignment validation must run on CPU")
        current = anchors.get(str(group))
        if current is None:
            anchors[str(group)] = (spec.name, lengths)
        elif not torch.equal(lengths, current[1]):
            raise ValueError(
                f"sequence feature {spec.name!r} lengths do not match "
                f"history-group anchor {current[0]!r}"
            )
    return {group: lengths for group, (_name, lengths) in anchors.items()}


def _require_cpu_request_batch(batch: RequestBatch) -> None:
    tensors = (
        batch.request_id,
        batch.item_id,
        *batch.labels.values(),
        *batch.candidate_metadata.values(),
    )
    if batch.user_id is not None:
        tensors = (*tensors, batch.user_id)
    if batch.candidate_valid_mask is not None:
        tensors = (*tensors, batch.candidate_valid_mask)
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError("prepare_model_batch requires a CPU RequestBatch")
    for lengths in batch.history_group_lengths.values():
        if (
            lengths.raw_lengths.device.type != "cpu"
            or lengths.truncated_lengths.device.type != "cpu"
        ):
            raise ValueError(
                "prepare_model_batch requires CPU history group lengths"
            )
    for value in batch.features.values():
        tensors = (
            (value.values, value.lengths)
            if isinstance(value, RaggedFeature)
            else (value,)
        )
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("prepare_model_batch requires CPU feature tensors")
    packed = batch.packed_categorical
    if packed is not None and (
        packed.values.device.type != "cpu"
        or packed.lengths.device.type != "cpu"
        or any(
            value.device.type != "cpu"
            for value in packed.feature_segment_lengths.values()
        )
        or any(
            value.device.type != "cpu"
            for value in packed.sequence_group_lengths.values()
        )
    ):
        raise ValueError("prepare_model_batch requires a CPU packed categorical payload")


def _feature_id_mapping(
    spec: SparseFeatureLike,
) -> SparseIDMapping | Mapping[str, Any] | None:
    return getattr(spec, "id_mapping", None)


def _mapping_type(mapping: SparseIDMapping | Mapping[str, Any] | None) -> str:
    if mapping is None:
        return "modulo_bucket"
    if isinstance(mapping, SparseIDMapping):
        return mapping.mapping_type
    if isinstance(mapping, Mapping):
        return str(mapping.get("type", "modulo_bucket"))
    raise ValueError("sparse id mapping must be a mapping when provided")


def _is_jagged_feature(value: Any) -> bool:
    return (
        hasattr(value, "values")
        and hasattr(value, "lengths")
        and isinstance(value.values, torch.Tensor)
        and isinstance(value.lengths, torch.Tensor)
    )


def _squeeze_sequence_value(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def _timing_started_at(timing: dict[str, float] | None) -> float | None:
    return time.perf_counter() if timing is not None else None


def _timing_add(
    timing: dict[str, float] | None,
    name: str,
    started_at: float | None,
) -> None:
    if timing is None or started_at is None:
        return
    timing[name] = timing.get(name, 0.0) + (time.perf_counter() - started_at)

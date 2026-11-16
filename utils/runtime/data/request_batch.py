from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, TypeAlias

import torch


@dataclass(frozen=True)
class RaggedFeatureMetadata:
    """CPU-validated ragged structure reused after device transfer."""

    value_count: int
    segment_count: int
    max_length: int


@dataclass(frozen=True)
class RaggedFeature:
    """Canonical variable-length feature storage for a request batch.

    ``lengths`` describes consecutive segments in ``values``. Request-level
    features have one segment per request; item-level features have one segment
    per candidate item, ordered by request then candidate position.
    """

    values: torch.Tensor
    lengths: torch.Tensor
    source_path: str
    level: str
    metadata: RaggedFeatureMetadata | None = None

    def __post_init__(self) -> None:
        if self.level not in {"request", "item"}:
            raise ValueError(f"RaggedFeature level must be request/item, got {self.level!r}")
        if self.values.ndim != 1:
            raise ValueError("RaggedFeature values must be one-dimensional")
        if self.lengths.ndim != 1:
            raise ValueError("RaggedFeature lengths must be one-dimensional")
        if self.lengths.dtype != torch.long:
            raise ValueError("RaggedFeature lengths must have dtype torch.long")
        if self.values.device != self.lengths.device:
            raise ValueError("RaggedFeature values and lengths must share a device")
        metadata = self.metadata
        if metadata is None:
            if self.values.device.type == "cuda":
                raise ValueError(
                    "CUDA RaggedFeature construction requires CPU-validated metadata"
                )
            if self.values.device.type == "meta":
                metadata = RaggedFeatureMetadata(
                    value_count=int(self.values.numel()),
                    segment_count=int(self.lengths.numel()),
                    max_length=0,
                )
            else:
                if bool(self.lengths.lt(0).any().item()):
                    raise ValueError("RaggedFeature lengths must be non-negative")
                value_count = int(self.lengths.sum().item())
                if value_count != int(self.values.numel()):
                    raise ValueError("RaggedFeature values/lengths are inconsistent")
                metadata = RaggedFeatureMetadata(
                    value_count=value_count,
                    segment_count=int(self.lengths.numel()),
                    max_length=(
                        int(self.lengths.max().item())
                        if int(self.lengths.numel())
                        else 0
                    ),
                )
            object.__setattr__(self, "metadata", metadata)
        if metadata.value_count != int(self.values.numel()):
            raise ValueError("RaggedFeature metadata value_count is inconsistent")
        if metadata.segment_count != int(self.lengths.numel()):
            raise ValueError("RaggedFeature metadata segment_count is inconsistent")
        if metadata.max_length < 0:
            raise ValueError("RaggedFeature metadata max_length must be non-negative")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "RaggedFeature":
        return replace(
            self,
            values=self.values.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "RaggedFeature":
        return replace(
            self,
            values=self.values.pin_memory(),
            lengths=self.lengths.pin_memory(),
        )


FeatureValue: TypeAlias = torch.Tensor | RaggedFeature


@dataclass(frozen=True)
class PackedCategoricalBatch:
    """Worker-built categorical payload awaiting a lightweight KJT wrap."""

    keys: tuple[str, ...]
    values: torch.Tensor
    lengths: torch.Tensor
    feature_shapes: Mapping[str, tuple[int, ...]]
    feature_segment_lengths: Mapping[str, torch.Tensor]
    sequence_group_lengths: Mapping[str, torch.Tensor]
    batch_size: int
    list_len: int
    preparation_timing: Mapping[str, float]
    mapped_id_spans: Mapping[int, tuple[int, int, str | None]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("packed categorical values must be one-dimensional")
        if self.lengths.ndim != 1 or self.lengths.dtype != torch.long:
            raise ValueError("packed categorical lengths must be 1-D torch.long")
        if self.values.device.type != "cpu" or self.lengths.device.type != "cpu":
            raise ValueError("packed categorical payload must remain on CPU")
        if set(self.keys) != set(self.feature_shapes):
            raise ValueError("packed categorical keys and feature shapes must match")
        if set(self.keys) != set(self.feature_segment_lengths):
            raise ValueError(
                "packed categorical keys and semantic segment lengths must match"
            )
        for slot_id, span in self.mapped_id_spans.items():
            if not isinstance(slot_id, int):
                raise ValueError("packed mapped-ID span keys must be integer slot IDs")
            if not isinstance(span, tuple) or len(span) != 3:
                raise ValueError(
                    f"packed mapped-ID span for slot {slot_id} must be "
                    "(start, length, history_group)"
                )
            start, length, history_group = span
            if (
                not isinstance(start, int)
                or not isinstance(length, int)
                or start < 0
                or length < 0
                or start + length > int(self.values.numel())
            ):
                raise ValueError(f"packed mapped-ID span for slot {slot_id} is invalid")
            if history_group is not None and (not isinstance(history_group, str) or not history_group):
                raise ValueError(
                    f"packed mapped-ID span for slot {slot_id} has an invalid history group"
                )

    def pin_memory(self) -> "PackedCategoricalBatch":
        pinned_by_identity: dict[int, torch.Tensor] = {}

        def pin(value: torch.Tensor) -> torch.Tensor:
            cached = pinned_by_identity.get(id(value))
            if cached is not None:
                return cached
            pinned = value if value.is_pinned() else value.pin_memory()
            pinned_by_identity[id(value)] = pinned
            return pinned

        return replace(
            self,
            values=pin(self.values),
            lengths=pin(self.lengths),
            feature_segment_lengths={
                name: pin(value)
                for name, value in self.feature_segment_lengths.items()
            },
            sequence_group_lengths={
                name: pin(value)
                for name, value in self.sequence_group_lengths.items()
            },
        )


@dataclass(frozen=True)
class RequestBatchMetadata:
    """Non-tensor execution diagnostics that deliberately remain on the host."""

    source_paths: tuple[str, ...] = ()
    cursor: Mapping[str, object] | None = None


@dataclass(frozen=True)
class HistoryGroupLengths:
    """Reader-validated raw and effective lengths for one aligned history group."""

    raw_lengths: torch.Tensor
    truncated_lengths: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("raw_lengths", self.raw_lengths),
            ("truncated_lengths", self.truncated_lengths),
        ):
            if value.ndim != 1 or value.dtype != torch.long:
                raise ValueError(f"history group {name} must be 1-D torch.long")
            if value.device.type == "cpu" and bool(value.lt(0).any().item()):
                raise ValueError(f"history group {name} must be non-negative")
        if self.raw_lengths.shape != self.truncated_lengths.shape:
            raise ValueError("history group raw/truncated lengths must share shape")
        if self.raw_lengths.device != self.truncated_lengths.device:
            raise ValueError("history group raw/truncated lengths must share device")
        if self.raw_lengths.device.type == "cpu" and bool(
            self.truncated_lengths.gt(self.raw_lengths).any().item()
        ):
            raise ValueError("history group truncated lengths cannot exceed raw lengths")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "HistoryGroupLengths":
        return replace(
            self,
            raw_lengths=self.raw_lengths.to(device, non_blocking=non_blocking),
            truncated_lengths=self.truncated_lengths.to(
                device, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> "HistoryGroupLengths":
        return replace(
            self,
            raw_lengths=self.raw_lengths.pin_memory(),
            truncated_lengths=self.truncated_lengths.pin_memory(),
        )


@dataclass(frozen=True)
class RequestBatch:
    """Model-view-independent request-level batch emitted by the DataLoader."""

    request_id: torch.Tensor
    item_id: torch.Tensor
    labels: Mapping[str, torch.Tensor]
    target: str
    features: Mapping[str, FeatureValue]
    metadata: RequestBatchMetadata = RequestBatchMetadata()
    user_id: torch.Tensor | None = None
    packed_categorical: PackedCategoricalBatch | None = None
    candidate_metadata: Mapping[str, torch.Tensor] = field(default_factory=dict)
    history_group_lengths: Mapping[str, HistoryGroupLengths] = field(
        default_factory=dict
    )
    candidate_valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.request_id.ndim != 1:
            raise ValueError("request_id must have shape [B]")
        if self.item_id.ndim != 2:
            raise ValueError("item_id must have shape [B, L]")
        if int(self.request_id.shape[0]) != int(self.item_id.shape[0]):
            raise ValueError("request_id and item_id batch dimensions must match")
        if self.candidate_valid_mask is not None:
            mask = self.candidate_valid_mask
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != self.item_id.shape
            ):
                raise ValueError("candidate_valid_mask must be bool with shape [B, L]")
            if mask.device != self.item_id.device:
                raise ValueError("candidate_valid_mask and item_id must share a device")
        if self.target not in self.labels:
            raise ValueError(f"target {self.target!r} is not present in labels")
        for name, label in self.labels.items():
            if not isinstance(label, torch.Tensor) or tuple(label.shape) != tuple(self.item_id.shape):
                raise ValueError(
                    f"label {name!r} must have shape {tuple(self.item_id.shape)}"
                )
        if self.user_id is not None and tuple(self.user_id.shape) != tuple(self.request_id.shape):
            raise ValueError("user_id must have shape [B]")
        for name, value in self.candidate_metadata.items():
            if not isinstance(name, str) or not name:
                raise ValueError("candidate metadata names must be non-empty strings")
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
                self.item_id.shape
            ):
                raise ValueError(
                    f"candidate metadata {name!r} must have shape "
                    f"{tuple(self.item_id.shape)}"
                )
        for name, lengths in self.history_group_lengths.items():
            if not isinstance(name, str) or not name:
                raise ValueError("history group names must be non-empty strings")
            if not isinstance(lengths, HistoryGroupLengths):
                raise TypeError(
                    f"history_group_lengths[{name!r}] must be HistoryGroupLengths"
                )
            if int(lengths.raw_lengths.numel()) != self.batch_size:
                raise ValueError(
                    f"history group {name!r} lengths must have shape "
                    f"[{self.batch_size}]"
                )
        for source_path, feature in self.features.items():
            if isinstance(feature, RaggedFeature):
                if feature.source_path != source_path:
                    raise ValueError(
                        f"ragged feature key {source_path!r} does not match source_path "
                        f"{feature.source_path!r}"
                    )
                expected_segments = self.batch_size if feature.level == "request" else self.batch_size * self.list_len
                if int(feature.lengths.numel()) != expected_segments:
                    raise ValueError(
                        f"ragged feature {source_path!r} has {feature.lengths.numel()} segments; "
                        f"expected {expected_segments}"
                    )
            elif not isinstance(feature, torch.Tensor):
                raise TypeError(f"feature {source_path!r} must be a Tensor or RaggedFeature")
        if self.packed_categorical is not None:
            if self.packed_categorical.batch_size != self.batch_size:
                raise ValueError("packed categorical batch size does not match RequestBatch")
            if self.packed_categorical.list_len != self.list_len:
                raise ValueError("packed categorical list length does not match RequestBatch")

    @property
    def batch_size(self) -> int:
        return int(self.request_id.shape[0])

    @property
    def list_len(self) -> int:
        return int(self.item_id.shape[1])

    @property
    def target_label(self) -> torch.Tensor:
        return self.labels[self.target]

    def feature(self, source_path: str) -> FeatureValue:
        try:
            return self.features[source_path]
        except KeyError as exc:
            raise KeyError(f"RequestBatch has no feature {source_path!r}") from exc

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "RequestBatch":
        return replace(
            self,
            request_id=self.request_id.to(device, non_blocking=non_blocking),
            item_id=self.item_id.to(device, non_blocking=non_blocking),
            labels={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.labels.items()
            },
            features={
                path: value.to(device, non_blocking=non_blocking)
                if isinstance(value, RaggedFeature)
                else value.to(device, non_blocking=non_blocking)
                for path, value in self.features.items()
            },
            user_id=(
                None
                if self.user_id is None
                else self.user_id.to(device, non_blocking=non_blocking)
            ),
            packed_categorical=self.packed_categorical,
            candidate_metadata={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.candidate_metadata.items()
            },
            history_group_lengths={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.history_group_lengths.items()
            },
            candidate_valid_mask=(
                None if self.candidate_valid_mask is None
                else self.candidate_valid_mask.to(device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> "RequestBatch":
        return replace(
            self,
            request_id=self.request_id.pin_memory(),
            item_id=self.item_id.pin_memory(),
            labels={name: value.pin_memory() for name, value in self.labels.items()},
            features={
                path: value.pin_memory()
                for path, value in self.features.items()
            },
            user_id=None if self.user_id is None else self.user_id.pin_memory(),
            packed_categorical=(
                None
                if self.packed_categorical is None
                else self.packed_categorical.pin_memory()
            ),
            candidate_metadata={
                name: value.pin_memory()
                for name, value in self.candidate_metadata.items()
            },
            history_group_lengths={
                name: value.pin_memory()
                for name, value in self.history_group_lengths.items()
            },
            candidate_valid_mask=(
                None if self.candidate_valid_mask is None
                else self.candidate_valid_mask.pin_memory()
            ),
        )

    def with_metadata(self, metadata: RequestBatchMetadata) -> "RequestBatch":
        return replace(self, metadata=metadata)

    def with_packed_categorical(
        self,
        packed: PackedCategoricalBatch,
        *,
        features: Mapping[str, FeatureValue] | None = None,
    ) -> "RequestBatch":
        return replace(
            self,
            features=self.features if features is None else features,
            packed_categorical=packed,
        )

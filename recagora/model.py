from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from .losses import CandidateLossContext


@dataclass(frozen=True)
class RecAgoraSlotSpec:
    slot_id: int
    view: str
    role: str
    representation: str
    encoding: str
    reduction: str
    output_dim: int
    mapped_ids_exposed: bool = False


@dataclass(frozen=True)
class FixedFeatureSpec:
    slot_ids: tuple[int, ...]
    total_dim: int


@dataclass(frozen=True)
class HistoryGroupSpec:
    name: str
    categorical_slot_ids: tuple[int, ...]
    direct_slot_ids: tuple[int, ...]
    event_total_dim: int
    max_length: int
    event_order: str
    truncate_side: str
    padding_side: str
    direct_padding_side: str
    padding_value: float
    length_anchor_slot_id: int


@dataclass(frozen=True)
class RecAgoraInputSpec:
    universe_version: str
    list_len: int
    task_names: tuple[str, ...]
    query: FixedFeatureSpec
    candidate: FixedFeatureSpec
    histories: Mapping[str, HistoryGroupSpec]
    _slots: Mapping[int, RecAgoraSlotSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "histories", MappingProxyType(dict(self.histories)))
        object.__setattr__(self, "_slots", MappingProxyType(dict(self._slots)))
        if self.list_len <= 0:
            raise ValueError("RecAgoraInputSpec list_len must be positive")
        if not self.task_names:
            raise ValueError("RecAgoraInputSpec requires at least one task")

    def slot(self, slot_id: int) -> RecAgoraSlotSpec:
        return self._slots[int(slot_id)]

    def total_dim(self, slot_ids: Sequence[int]) -> int:
        return sum(self.slot(slot_id).output_dim for slot_id in slot_ids)

    @property
    def mapped_id_slot_ids(self) -> tuple[int, ...]:
        return tuple(
            slot_id
            for slot_id, slot in self._slots.items()
            if slot.mapped_ids_exposed
        )


@dataclass(frozen=True)
class RecAgoraSparseBatch:
    """Public model input before categorical embedding lookup.

    Categorical values have already been mapped and packed in the DataLoader
    worker.  ``get_slots`` owns the only supported conversion from these IDs to
    embedding-dimensional model views.
    """

    batch_size: int
    list_len: int
    _lookup_payload: Any
    dense: Mapping[str, Any]
    labels: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]
    candidate_metadata: Mapping[str, torch.Tensor]
    history_group_lengths: Mapping[str, Any]
    preparation_timing: Mapping[str, float]
    candidate_valid_mask: torch.Tensor | None = None
    candidate_loss_context: CandidateLossContext | None = None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "RecAgoraSparseBatch":
        return RecAgoraSparseBatch(
            batch_size=self.batch_size,
            list_len=self.list_len,
            _lookup_payload=_to_device(
                self._lookup_payload,
                device,
                non_blocking=non_blocking,
            ),
            dense=_to_device(self.dense, device, non_blocking=non_blocking),
            labels={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.labels.items()
            },
            metadata=_to_device(
                self.metadata,
                device,
                non_blocking=non_blocking,
            ),
            candidate_metadata={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.candidate_metadata.items()
            },
            history_group_lengths=_to_device(
                self.history_group_lengths,
                device,
                non_blocking=non_blocking,
            ),
            preparation_timing=self.preparation_timing,
            candidate_valid_mask=(
                None
                if self.candidate_valid_mask is None
                else self.candidate_valid_mask.to(device, non_blocking=non_blocking)
            ),
            candidate_loss_context=self.candidate_loss_context,
        )

    def _runtime_lookup_payload(self) -> Any:
        """Return the opaque worker payload for framework runtime code only."""
        return self._lookup_payload


class RecAgoraSlotLookup(nn.Module):
    """Framework-injected lookup module registered under each ranking model."""

    keyed_jagged_tensor_cls: type[Any]

    def forward(self, batch: RecAgoraSparseBatch) -> "RecAgoraModelBatch":
        raise NotImplementedError

    def set_timing_enabled(self, enabled: bool) -> None:
        del enabled

    def consume_timing_span(self) -> Any | None:
        return None

    def set_communication_trace_enabled(self, enabled: bool) -> None:
        del enabled

    @property
    def last_lookup_sec(self) -> float:
        return 0.0


class FixedFeatureView(Mapping[int, torch.Tensor]):
    def __init__(
        self,
        values: Mapping[int, torch.Tensor],
        *,
        slot_ids: Sequence[int],
        batch_size: int,
        list_len: int | None,
    ) -> None:
        ordered = tuple(int(slot_id) for slot_id in slot_ids)
        # Slot membership and tensor layouts are compiled once and materialized
        # by the framework-owned lookup.  Rechecking every slot here made this
        # lightweight public view another per-forward validation boundary.
        frozen = {slot_id: values[slot_id] for slot_id in ordered}
        self._slot_ids = ordered
        self._values = MappingProxyType(frozen)
        self._batch_size = int(batch_size)
        self._list_len = list_len

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return self._slot_ids

    def __getitem__(self, slot_id: int) -> torch.Tensor:
        return self._values[int(slot_id)]

    def __iter__(self) -> Iterator[int]:
        return iter(self._slot_ids)

    def __len__(self) -> int:
        return len(self._slot_ids)

    def concat(self, slot_ids: Sequence[int] | None = None) -> torch.Tensor:
        selected = self._slot_ids if slot_ids is None else tuple(map(int, slot_ids))
        return torch.cat([self[slot_id] for slot_id in selected], dim=-1)

    def flatten_candidates(self, list_len: int) -> "FixedFeatureView":
        if self._list_len is None:
            values = {
                slot_id: value[:, None, :]
                .expand(-1, list_len, -1)
                .reshape(self._batch_size * list_len, value.shape[-1])
                for slot_id, value in self._values.items()
            }
        else:
            values = {
                slot_id: value.reshape(self._batch_size * list_len, value.shape[-1])
                for slot_id, value in self._values.items()
            }
        return FixedFeatureView(
            values,
            slot_ids=self._slot_ids,
            batch_size=self._batch_size * list_len,
            list_len=None,
        )


@dataclass(frozen=True)
class JaggedFeatureView:
    values: torch.Tensor
    lengths: torch.Tensor

    def to_padded(
        self,
        *,
        max_length: int,
        padding_side: str = "left",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        lengths = self.lengths.clamp_max(max_length)
        output = self.values.new_full(
            (int(lengths.numel()), max_length, int(self.values.shape[-1])),
            padding_value,
        )
        total = int(self.values.shape[0])
        if total:
            owners = torch.repeat_interleave(
                torch.arange(self.lengths.numel(), device=self.lengths.device),
                self.lengths,
                output_size=total,
            )
            starts = torch.cumsum(self.lengths, dim=0) - self.lengths
            positions = torch.arange(total, device=self.lengths.device) - torch.repeat_interleave(
                starts,
                self.lengths,
                output_size=total,
            )
            retained_starts = (self.lengths - max_length).clamp_min(0)
            keep = positions >= retained_starts[owners]
            destination = positions[keep] - retained_starts[owners[keep]]
            if padding_side == "left":
                destination = destination + max_length - lengths[owners[keep]]
            output[owners[keep], destination] = self.values[keep]
        return output


@dataclass(frozen=True)
class JaggedIDView:
    """Mapped categorical IDs aligned with one public jagged feature."""

    values: torch.Tensor
    lengths: torch.Tensor

    def to_padded(
        self,
        *,
        max_length: int,
        padding_side: str = "left",
        padding_value: int = 0,
    ) -> torch.Tensor:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        lengths = self.lengths.clamp_max(max_length)
        output = self.values.new_full(
            (int(lengths.numel()), max_length),
            int(padding_value),
        )
        total = int(self.values.shape[0])
        if total:
            owners = torch.repeat_interleave(
                torch.arange(self.lengths.numel(), device=self.lengths.device),
                self.lengths,
                output_size=total,
            )
            starts = torch.cumsum(self.lengths, dim=0) - self.lengths
            positions = torch.arange(
                total,
                device=self.lengths.device,
            ) - torch.repeat_interleave(
                starts,
                self.lengths,
                output_size=total,
            )
            retained_starts = (self.lengths - max_length).clamp_min(0)
            keep = positions >= retained_starts[owners]
            destination = positions[keep] - retained_starts[owners[keep]]
            if padding_side == "left":
                destination = destination + max_length - lengths[owners[keep]]
            output[owners[keep], destination] = self.values[keep]
        return output


@dataclass(frozen=True)
class HistoryView:
    lengths: torch.Tensor
    categorical: Mapping[int, JaggedFeatureView]
    direct: Mapping[int, torch.Tensor]
    max_length: int
    mapped_ids: Mapping[int, JaggedIDView] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reader validation and the compiled adapter already establish the
        # canonical lengths and direct widths.  This object only freezes the
        # lookup result for model consumption.
        object.__setattr__(
            self,
            "categorical",
            MappingProxyType(dict(self.categorical)),
        )
        object.__setattr__(self, "direct", MappingProxyType(dict(self.direct)))
        object.__setattr__(
            self,
            "mapped_ids",
            MappingProxyType(dict(self.mapped_ids)),
        )


class NamedTensorView(Mapping[str, torch.Tensor]):
    def __init__(self, values: Mapping[str, torch.Tensor]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def flatten_candidates(self) -> "NamedTensorView":
        return NamedTensorView(
            {name: value.reshape(-1) for name, value in self._values.items()}
        )


@dataclass(frozen=True)
class BatchIDs:
    request_id: torch.Tensor
    user_id: torch.Tensor | None
    item_id: torch.Tensor


@dataclass(frozen=True)
class RecAgoraModelBatch:
    ids: BatchIDs
    query: FixedFeatureView
    candidate: FixedFeatureView
    histories: Mapping[str, HistoryView]
    labels: NamedTensorView
    candidate_metadata: NamedTensorView
    candidate_valid_mask: torch.Tensor | None = None
    candidate_mapped_ids: Mapping[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "histories", MappingProxyType(dict(self.histories)))
        object.__setattr__(self, "candidate_mapped_ids", MappingProxyType(dict(self.candidate_mapped_ids)))

    @property
    def batch_size(self) -> int:
        return int(self.ids.request_id.shape[0])

    @property
    def list_len(self) -> int:
        return int(self.ids.item_id.shape[1])

    def flatten_candidates(self) -> "RecAgoraPointwiseBatch":
        list_len = self.list_len
        histories = self.histories
        return RecAgoraPointwiseBatch(
            ids=BatchIDs(
                request_id=self.ids.request_id.repeat_interleave(list_len),
                user_id=(
                    None
                    if self.ids.user_id is None
                    else self.ids.user_id.repeat_interleave(list_len)
                ),
                item_id=self.ids.item_id.reshape(-1),
            ),
            query=self.query.flatten_candidates(list_len),
            candidate=self.candidate.flatten_candidates(list_len),
            histories=histories,
            labels=self.labels.flatten_candidates(),
            candidate_metadata=self.candidate_metadata.flatten_candidates(),
            request_index=torch.arange(
                self.batch_size,
                device=self.ids.request_id.device,
            ).repeat_interleave(list_len),
            candidate_index=torch.arange(
                list_len,
                device=self.ids.request_id.device,
            ).repeat(self.batch_size),
            candidate_valid_mask=(
                None
                if self.candidate_valid_mask is None
                else self.candidate_valid_mask.reshape(-1)
            ),
        )


@dataclass(frozen=True)
class RecAgoraPointwiseBatch:
    ids: BatchIDs
    query: FixedFeatureView
    candidate: FixedFeatureView
    histories: Mapping[str, HistoryView]
    labels: NamedTensorView
    candidate_metadata: NamedTensorView
    request_index: torch.Tensor
    candidate_index: torch.Tensor
    candidate_valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "histories", MappingProxyType(dict(self.histories)))


class RecAgoraRankingModel(nn.Module):
    # Opt-in only after all candidate mixing, losses, and auxiliary statistics
    # exclude invalid positions.  A capability flag never replaces that work.
    supports_candidate_mask = False
    candidate_mask_all_empty_policy = "unsupported"
    candidate_mask_loss_reduction = "model_defined"

    def __init__(self, config: Any, input_spec: RecAgoraInputSpec) -> None:
        super().__init__()
        self.config = config
        self.input_spec = input_spec
        self.query_input_dim = input_spec.query.total_dim
        self.candidate_input_dim = input_spec.candidate.total_dim
        self.list_len = input_spec.list_len
        self.task_names = input_spec.task_names
        self.slot_lookup: RecAgoraSlotLookup | None = None

    def bind_slot_lookup(self, slot_lookup: RecAgoraSlotLookup) -> None:
        """Bind the framework-owned embedding module before DMP wrapping."""
        if self.slot_lookup is not None:
            raise RuntimeError("RecAgoraRankingModel slot lookup is already bound")
        if not isinstance(slot_lookup, RecAgoraSlotLookup):
            raise TypeError("slot_lookup must implement RecAgoraSlotLookup")
        self.slot_lookup = slot_lookup

    def get_slots(self, batch: RecAgoraSparseBatch) -> RecAgoraModelBatch:
        """Look up all declared slots once and return structured model views."""
        if self.slot_lookup is None:
            raise RuntimeError(
                "RecAgoraRankingModel has no slot lookup; construct it through the "
                "framework model factory"
            )
        return self.slot_lookup(batch)

    def slot_dim(self, slot_id: int) -> int:
        return self.input_spec.slot(slot_id).output_dim

    def history_spec(self, name: str = "history_group_1") -> HistoryGroupSpec:
        return self.input_spec.histories[name]


def _to_device(
    value: Any,
    device: torch.device | str,
    *,
    non_blocking: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            name: _to_device(item, device, non_blocking=non_blocking)
            for name, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _to_device(item, device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            _to_device(item, device, non_blocking=non_blocking)
            for item in value
        ]
    move = getattr(value, "to", None)
    if callable(move):
        return move(device, non_blocking=non_blocking)
    return value

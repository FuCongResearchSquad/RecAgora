from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import (
    require_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
)


@dataclass(frozen=True)
class HyFormerInputsOutput:
    query: torch.Tensor
    candidates: torch.Tensor
    non_sequence_tokens: torch.Tensor
    history_events: torch.Tensor
    history_valid: torch.Tensor


class HyFormerInputs(nn.Module):
    """Build HyFormer-private Q/C/H tensors from public post-lookup views."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        model_dim: int,
        non_sequence_token_order: Sequence[str],
        non_sequence_groups: Mapping[str, Sequence[int]],
        history: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.model_dim = int(model_dim)
        self.non_sequence_token_order = tuple(
            str(name) for name in non_sequence_token_order
        )
        self.non_sequence_groups = {
            str(name): tuple(int(slot) for slot in slots)
            for name, slots in non_sequence_groups.items()
        }
        self._validate_non_sequence_groups()

        history = dict(history)
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        self.runtime_validation = str(
            history.get("runtime_validation", "shapes_only")
        )
        validate_left_padding(self.padding_side, context="HyFormer")
        if self.runtime_validation not in {"off", "shapes_only", "strict"}:
            raise ValueError(
                "HyFormer history.runtime_validation must be off, shapes_only, or strict"
            )

        history_spec = require_history_group(input_spec, self.history_group_name)
        validate_history_order(
            history_spec,
            context="HyFormer",
            truncate_side="tail",
        )
        self.sequence_slots = tuple(
            int(slot) for slot in history.get("sequence_slots", ())
        )
        _require_unique_nonempty(
            self.sequence_slots,
            label="history.sequence_slots",
        )
        if set(self.sequence_slots) != set(history_spec.categorical_slot_ids):
            raise ValueError(
                "HyFormer history.sequence_slots must exactly cover public categorical "
                "history"
            )
        for slot_id in self.sequence_slots:
            slot = input_spec.slot(slot_id)
            if (
                slot.view != "history"
                or slot.role != "sequence"
                or slot.encoding != "embedding"
                or slot.reduction != "none"
            ):
                raise ValueError(
                    f"HyFormer history slot {slot_id} must be an unreduced history "
                    "embedding"
                )

        self.history_length = int(history_spec.max_length)
        self.timestamp_slot: int | None = None
        self.time_gap_slot: int | None = None
        self.time_gap_dim = int(history.get("time_gap_dim", 0))
        if history_spec.direct_slot_ids:
            self.timestamp_slot = int(history["timestamp_slot"])
            self.time_gap_slot = int(history["time_gap_slot"])
            if set(history_spec.direct_slot_ids) != {
                self.timestamp_slot,
                self.time_gap_slot,
            }:
                raise ValueError(
                    "HyFormer timestamp_slot and time_gap_slot must exactly cover direct "
                    "history"
                )
            for slot_id in history_spec.direct_slot_ids:
                slot = input_spec.slot(slot_id)
                if (
                    slot.view != "history"
                    or slot.role != "sequence"
                    or slot.encoding != "direct"
                    or slot.reduction != "none"
                ):
                    raise ValueError(
                        f"HyFormer direct slot {slot_id} has an incompatible public "
                        "contract"
                    )
            if history_spec.direct_padding_side != "right":
                raise ValueError(
                    "HyFormer requires physically right-padded direct history"
                )
            if self.time_gap_dim <= 0:
                raise ValueError(
                    "HyFormer Native160 history requires a positive time_gap_dim"
                )
            self.time_gap_projection: nn.Linear | None = nn.Linear(
                1,
                self.time_gap_dim,
            )
            nn.init.xavier_uniform_(self.time_gap_projection.weight)
            nn.init.zeros_(self.time_gap_projection.bias)
        else:
            unknown_temporal = {
                "timestamp_slot",
                "time_gap_slot",
                "time_gap_dim",
            }.intersection(history)
            if unknown_temporal:
                raise ValueError(
                    "HyFormer categorical-only history must not configure direct-time "
                    f"fields: {sorted(unknown_temporal)}"
                )
            self.time_gap_dim = 0
            self.time_gap_projection = None

        self.query_dim = int(input_spec.query.total_dim)
        self.candidate_dim = int(input_spec.candidate.total_dim)
        self.history_event_dim = (
            input_spec.total_dim(self.sequence_slots) + self.time_gap_dim
        )
        self.group_views: dict[str, str] = {}
        self.group_projections = nn.ModuleDict()
        query_slots = set(input_spec.query.slot_ids)
        for name in self.non_sequence_token_order:
            slot_ids = self.non_sequence_groups[name]
            view = "query" if set(slot_ids) <= query_slots else "candidate"
            self.group_views[name] = view
            projection = nn.Linear(input_spec.total_dim(slot_ids), self.model_dim)
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.group_projections[name] = projection

    @property
    def num_non_sequence_tokens(self) -> int:
        return len(self.non_sequence_token_order)

    def _validate_non_sequence_groups(self) -> None:
        if not self.non_sequence_token_order or len(
            set(self.non_sequence_token_order)
        ) != len(self.non_sequence_token_order):
            raise ValueError(
                "HyFormer non_sequence_token_order must be non-empty and unique"
            )
        if set(self.non_sequence_groups) != set(self.non_sequence_token_order):
            raise ValueError(
                "HyFormer non_sequence_groups must exactly match "
                "non_sequence_token_order"
            )
        groups = tuple(
            self.non_sequence_groups[name]
            for name in self.non_sequence_token_order
        )
        if any(not slots for slots in groups):
            raise ValueError("HyFormer non-sequence groups must not be empty")
        validate_disjoint_slot_groups(
            groups,
            context="HyFormer non-sequence groups",
        )
        grouped = {slot for slots in groups for slot in slots}
        expected = set(self.input_spec.query.slot_ids) | set(
            self.input_spec.candidate.slot_ids
        )
        if grouped != expected:
            raise ValueError(
                "HyFormer non-sequence groups must exactly cover public query and "
                f"candidate slots; missing={sorted(expected - grouped)}, "
                f"extra={sorted(grouped - expected)}"
            )
        query_slots = set(self.input_spec.query.slot_ids)
        candidate_slots = set(self.input_spec.candidate.slot_ids)
        saw_query = False
        saw_candidate = False
        for name, slots in zip(self.non_sequence_token_order, groups, strict=True):
            selected = set(slots)
            if selected <= query_slots:
                expected_view, expected_role = "query", "request"
                saw_query = True
            elif selected <= candidate_slots:
                expected_view, expected_role = "candidate", "item"
                saw_candidate = True
            else:
                raise ValueError(
                    f"HyFormer non-sequence group {name!r} must not mix query and "
                    "candidate slots"
                )
            for slot_id in slots:
                slot = self.input_spec.slot(slot_id)
                if (
                    slot.view != expected_view
                    or slot.role != expected_role
                    or slot.encoding != "embedding"
                    or slot.reduction != "mean"
                ):
                    raise ValueError(
                        f"HyFormer non-sequence slot {slot_id} has an incompatible "
                        "public contract"
                    )
        if not saw_query or not saw_candidate:
            raise ValueError(
                "HyFormer requires both query-side and candidate-side NS tokens"
            )

    def forward(self, batch: RecAgoraModelBatch) -> HyFormerInputsOutput:
        history = batch.histories[self.history_group_name]
        history_valid = _suffix_valid_mask(
            history.lengths,
            self.history_length,
        )
        history_parts = [
            history.categorical[slot_id].to_padded(
                max_length=self.history_length,
                padding_side=self.padding_side,
            )
            for slot_id in self.sequence_slots
        ]

        if self.time_gap_projection is not None:
            assert self.timestamp_slot is not None
            assert self.time_gap_slot is not None
            timestamps = _right_to_left(
                history.direct[self.timestamp_slot],
                history.lengths,
                self.history_length,
            )
            gaps = _right_to_left(
                history.direct[self.time_gap_slot],
                history.lengths,
                self.history_length,
            )
            if self.runtime_validation == "strict":
                _validate_direct_history(timestamps, gaps, history_valid)
            gaps = gaps.to(dtype=history_parts[0].dtype)
            gaps = gaps.masked_fill(~history_valid, 0.0).clamp_min(0.0)
            gap_embedding = self.time_gap_projection(
                torch.log1p(gaps).unsqueeze(-1)
            )
            history_parts.append(gap_embedding)

        history_events = torch.cat(history_parts, dim=-1).masked_fill(
            ~history_valid.unsqueeze(-1),
            0.0,
        )
        query = batch.query.concat()
        candidates = batch.candidate.concat()
        token_values: list[torch.Tensor] = []
        for name in self.non_sequence_token_order:
            slot_ids = self.non_sequence_groups[name]
            if self.group_views[name] == "query":
                value = self.group_projections[name](batch.query.concat(slot_ids))
                value = value[:, None, :].expand(-1, batch.list_len, -1)
            else:
                value = self.group_projections[name](
                    batch.candidate.concat(slot_ids)
                )
            token_values.append(value)
        non_sequence_tokens = torch.stack(token_values, dim=2)
        return HyFormerInputsOutput(
            query=query,
            candidates=candidates,
            non_sequence_tokens=non_sequence_tokens,
            history_events=history_events,
            history_valid=history_valid,
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long).clamp_max(width)
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - lengths).unsqueeze(1)


def _right_to_left(
    values: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    lengths = lengths.to(device=values.device, dtype=torch.long).clamp_max(width)
    positions = torch.arange(width, device=values.device)[None, :]
    valid = positions >= (width - lengths).unsqueeze(1)
    source = (positions - (width - lengths).unsqueeze(1)).clamp_min(0)
    return values[:, :width].gather(1, source).masked_fill(~valid, 0.0)


def _validate_direct_history(
    timestamps: torch.Tensor,
    gaps: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    valid_timestamps = timestamps[valid]
    if not bool(
        (
            torch.isfinite(valid_timestamps)
            & valid_timestamps.gt(0.0)
        ).all().item()
    ):
        raise ValueError(
            "HyFormer valid history timestamps must be finite and positive"
        )
    comparable = valid[:, 1:] & valid[:, :-1]
    if bool(
        (comparable & timestamps[:, 1:].lt(timestamps[:, :-1])).any().item()
    ):
        raise ValueError("HyFormer history timestamps must be oldest-to-newest")
    valid_gaps = gaps[valid]
    if not bool(torch.isfinite(valid_gaps).all().item()) or bool(
        valid_gaps.lt(0.0).any().item()
    ):
        raise ValueError(
            "HyFormer valid history time gaps must be finite and non-negative"
        )


def _require_unique_nonempty(slots: tuple[int, ...], *, label: str) -> None:
    if not slots:
        raise ValueError(f"HyFormer {label} must not be empty")
    if len(slots) != len(set(slots)):
        raise ValueError(f"HyFormer {label} must contain unique slots")

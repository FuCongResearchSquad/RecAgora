from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import (
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
)


@dataclass(frozen=True)
class OneTransInputsOutput:
    sequence: torch.Tensor
    non_sequence_groups: tuple[torch.Tensor, ...]
    sequence_valid_mask: torch.Tensor


class OneTransInputs(nn.Module):
    """Create model-owned sequence and non-sequence groups from public views."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, Any],
        tokenizer: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.sequence_slots = tuple(int(slot) for slot in history["sequence_slots"])
        self.length_slot = int(history.get("length_slot", 45041))
        self.action_slot = int(history.get("action_slot", 45040))
        self.timestamp_slot = int(history.get("timestamp_slot", 45042))
        self.time_gap_slot = int(history.get("time_gap_slot", 45073))
        self.padding_side = str(history.get("padding_side", "left"))
        self.time_gap_transform = str(history.get("time_gap_transform", "log1p"))
        self.time_gap_dim = int(history.get("time_gap_dim", 16))
        self.runtime_validation = str(history.get("runtime_validation", "shapes_only"))
        history_spec = input_spec.histories[self.history_group_name]
        self.max_events = int(history.get("max_events", history_spec.max_length))

        self.token_order = tuple(str(name) for name in tokenizer["token_order"])
        self.groups = {
            str(name): tuple(int(slot) for slot in slots)
            for name, slots in tokenizer["groups"].items()
        }
        self._validate_contract()

        self.time_gap_projection = nn.Linear(1, self.time_gap_dim)
        self.sequence_input_dim = (
            input_spec.total_dim(self.sequence_slots) + self.time_gap_dim
        )
        self.ns_group_input_dims = {
            name: input_spec.total_dim(self.groups[name]) for name in self.token_order
        }

    def _validate_contract(self) -> None:
        history = self.input_spec.histories[self.history_group_name]
        if not 1 <= self.max_events <= history.max_length:
            raise ValueError(
                "OneTrans history.max_events must be between 1 and "
                f"the public history length ({history.max_length})"
            )
        if len(self.sequence_slots) != len(set(self.sequence_slots)):
            raise ValueError("history.sequence_slots must be unique")
        if set(self.sequence_slots) != set(history.categorical_slot_ids):
            raise ValueError(
                "history.sequence_slots must exactly cover public categorical history"
            )
        if self.length_slot != history.length_anchor_slot_id:
            raise ValueError("history.length_slot must match the public length anchor")
        if self.action_slot not in self.sequence_slots:
            raise ValueError("history.action_slot must belong to sequence_slots")
        if set(history.direct_slot_ids) != {self.timestamp_slot, self.time_gap_slot}:
            raise ValueError("direct history must be timestamp_slot plus time_gap_slot")
        if history.direct_padding_side != "right":
            raise ValueError("public direct history must be physically right padded")
        validate_left_padding(self.padding_side, context="OneTrans")
        validate_history_order(
            history,
            context="OneTrans",
            truncate_side="tail",
        )
        if self.time_gap_transform != "log1p" or self.time_gap_dim <= 0:
            raise ValueError("OneTrans requires a positive log1p time-gap projection")
        if self.runtime_validation not in {"off", "shapes_only", "strict"}:
            raise ValueError("runtime_validation must be off, shapes_only, or strict")

        if not self.token_order or len(self.token_order) != len(set(self.token_order)):
            raise ValueError("tokenizer.token_order must contain unique group names")
        if set(self.groups) != set(self.token_order):
            raise ValueError("tokenizer.groups must exactly match token_order")
        flattened = [slot for name in self.token_order for slot in self.groups[name]]
        validate_disjoint_slot_groups(
            tuple(self.groups[name] for name in self.token_order),
            context="tokenizer groups",
        )
        fixed = set(self.input_spec.query.slot_ids) | set(
            self.input_spec.candidate.slot_ids
        )
        if set(flattened) != fixed:
            raise ValueError(
                "tokenizer groups must exactly cover fixed public slots; "
                f"missing={sorted(fixed - set(flattened))}, "
                f"extra={sorted(set(flattened) - fixed)}"
            )
        query_slots = set(self.input_spec.query.slot_ids)
        candidate_slots = set(self.input_spec.candidate.slot_ids)
        for name in self.token_order:
            slots = set(self.groups[name])
            if not (slots <= query_slots or slots <= candidate_slots):
                raise ValueError(
                    f"tokenizer group {name!r} must not mix query and candidate slots"
                )
            for slot_id in slots:
                slot = self.input_spec.slot(slot_id)
                if slot.encoding != "embedding" or slot.reduction != "mean":
                    raise ValueError(
                        f"fixed slot {slot_id} must be a mean-reduced embedding"
                    )

    def forward(self, batch: RecAgoraModelBatch) -> OneTransInputsOutput:
        history = batch.histories[self.history_group_name]
        lengths = history.lengths
        max_length = self.max_events
        sequence = torch.cat(
            [
                history.categorical[slot_id].to_padded(
                    max_length=max_length,
                    padding_side=self.padding_side,
                )
                for slot_id in self.sequence_slots
            ],
            dim=-1,
        )
        valid_mask = _suffix_valid_mask(lengths, max_length, sequence.device)

        if self.runtime_validation == "strict":
            timestamps = _right_to_left(
                history.direct[self.timestamp_slot],
                lengths,
                max_length,
            )
            valid_timestamps = timestamps[valid_mask]
            if not bool(
                (
                    torch.isfinite(valid_timestamps)
                    & valid_timestamps.gt(0)
                ).all().item()
            ):
                raise ValueError("valid history timestamps must be finite and positive")
            if _has_decreasing_timestamps(timestamps, valid_mask):
                raise ValueError("history timestamps must be oldest-to-newest")

        gaps = _right_to_left(
            history.direct[self.time_gap_slot],
            lengths,
            max_length,
        )
        if self.runtime_validation == "strict":
            valid_gaps = gaps[valid_mask]
            if not bool(torch.isfinite(valid_gaps).all().item()) or bool(
                valid_gaps.lt(0).any().item()
            ):
                raise ValueError("valid history time gaps must be finite and non-negative")
        gaps = gaps.masked_fill(~valid_mask, 0).clamp_min(0)
        gap_embedding = self.time_gap_projection(torch.log1p(gaps).unsqueeze(-1))
        sequence = torch.cat((sequence, gap_embedding), dim=-1)

        groups: list[torch.Tensor] = []
        query_slots = set(self.input_spec.query.slot_ids)
        for name in self.token_order:
            slot_ids = self.groups[name]
            if set(slot_ids) <= query_slots:
                value = batch.query.concat(slot_ids)
                value = value[:, None, :].expand(-1, batch.list_len, -1)
            else:
                value = batch.candidate.concat(slot_ids)
            groups.append(value)
        return OneTransInputsOutput(sequence, tuple(groups), valid_mask)


def _right_to_left(
    values: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    lengths = lengths.to(device=values.device, dtype=torch.long)
    retained_lengths = lengths.clamp_max(max_length)
    positions = torch.arange(max_length, device=values.device)[None, :]
    valid = positions >= (max_length - retained_lengths).unsqueeze(1)
    source = (
        positions
        - (max_length - retained_lengths).unsqueeze(1)
        + (lengths - retained_lengths).unsqueeze(1)
    ).clamp_min(0)
    return values.gather(1, source).masked_fill(~valid, 0)


def _suffix_valid_mask(
    lengths: torch.Tensor,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(max_length, device=device)[None, :]
    return positions >= (
        max_length - lengths.to(device=device, dtype=torch.long).clamp_max(max_length)
    ).unsqueeze(1)


def _has_decreasing_timestamps(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> bool:
    comparable = valid_mask[:, 1:] & valid_mask[:, :-1]
    return bool((comparable & values[:, 1:].lt(values[:, :-1])).any().item())

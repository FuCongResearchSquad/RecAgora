from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import (
    require_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
    validate_positive_int,
)


@dataclass(frozen=True)
class MixFormerInputsOutput:
    query: torch.Tensor
    candidates: torch.Tensor
    history_events: torch.Tensor
    history_valid: torch.Tensor


class MixFormerInputs(nn.Module):
    """Map public slot views to MixFormer-private dense tensors."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
    ) -> None:
        super().__init__()
        history = dict(history)
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="MixFormer")

        history_spec = require_history_group(
            input_spec,
            self.history_group_name,
        )
        validate_history_order(history_spec, context="MixFormer")

        self.query_slots = tuple(int(slot) for slot in query_slots)
        self.candidate_slots = tuple(int(slot) for slot in candidate_slots)
        self.sequence_slots = tuple(
            int(slot) for slot in history.get("sequence_slots", ())
        )
        _require_unique_nonempty(self.query_slots, label="query_slots")
        _require_unique_nonempty(self.candidate_slots, label="candidate_slots")
        _require_unique_nonempty(self.sequence_slots, label="history.sequence_slots")
        if self.query_slots != tuple(input_spec.query.slot_ids):
            raise ValueError(
                "MixFormer query_slots must preserve and exactly cover the public query view"
            )
        if self.candidate_slots != tuple(input_spec.candidate.slot_ids):
            raise ValueError(
                "MixFormer candidate_slots must preserve and exactly cover the public "
                "candidate view"
            )
        if self.sequence_slots != tuple(history_spec.categorical_slot_ids):
            raise ValueError(
                "MixFormer history.sequence_slots must preserve and exactly cover the "
                "public categorical history view"
            )
        groups = (
            self.query_slots,
            self.candidate_slots,
            self.sequence_slots,
            tuple(history_spec.direct_slot_ids),
        )
        validate_disjoint_slot_groups(
            groups,
            context="MixFormer query, candidate, and history slots",
        )

        self.timestamp_slot: int | None = None
        self.timestamp_dim = 0
        self.timestamp_transform: str | None = None
        self.timestamp_projection: nn.Linear | None = None
        if history_spec.direct_slot_ids:
            required_temporal = {
                "timestamp_slot",
                "timestamp_dim",
                "timestamp_transform",
            }
            missing_temporal = required_temporal.difference(history)
            if missing_temporal:
                raise ValueError(
                    "MixFormer direct history requires executable timestamp fields: "
                    f"{sorted(missing_temporal)}"
                )
            self.timestamp_slot = int(history["timestamp_slot"])
            self.timestamp_dim = int(history["timestamp_dim"])
            self.timestamp_transform = str(history["timestamp_transform"])
            if tuple(history_spec.direct_slot_ids) != (self.timestamp_slot,):
                raise ValueError(
                    "MixFormer direct history must contain exactly timestamp_slot"
                )
            timestamp_spec = input_spec.slot(self.timestamp_slot)
            if (
                timestamp_spec.view != "history"
                or timestamp_spec.role != "sequence"
                or timestamp_spec.representation != "time"
                or timestamp_spec.encoding != "direct"
                or timestamp_spec.reduction != "none"
                or timestamp_spec.output_dim != 1
            ):
                raise ValueError(
                    "MixFormer timestamp_slot has an incompatible public contract"
                )
            if history_spec.direct_padding_side != "right":
                raise ValueError(
                    "MixFormer requires physically right-padded direct timestamp history"
                )
            validate_positive_int(
                self.timestamp_dim,
                field_name="history.timestamp_dim",
            )
            if self.timestamp_transform != "relative_age_log1p":
                raise ValueError(
                    "MixFormer history.timestamp_transform must be "
                    "relative_age_log1p"
                )
            self.timestamp_projection = nn.Linear(1, self.timestamp_dim)
            nn.init.xavier_uniform_(self.timestamp_projection.weight)
            nn.init.zeros_(self.timestamp_projection.bias)
        else:
            unexpected_temporal = {
                "timestamp_slot",
                "timestamp_dim",
                "timestamp_transform",
            }.intersection(history)
            if unexpected_temporal:
                raise ValueError(
                    "MixFormer categorical-only history must not configure timestamp "
                    f"fields: {sorted(unexpected_temporal)}"
                )

        self.history_length = int(history_spec.max_length)
        self.query_dim = input_spec.total_dim(self.query_slots)
        self.candidate_dim = input_spec.total_dim(self.candidate_slots)
        self.history_event_dim = (
            input_spec.total_dim(self.sequence_slots) + self.timestamp_dim
        )

    def forward(self, batch: RecAgoraModelBatch) -> MixFormerInputsOutput:
        history = batch.histories[self.history_group_name]
        history_valid = _suffix_valid_mask(history.lengths, self.history_length)
        history_parts = [
            history.categorical[slot].to_padded(
                max_length=self.history_length,
                padding_side=self.padding_side,
            )
            for slot in self.sequence_slots
        ]
        if self.timestamp_projection is not None:
            assert self.timestamp_slot is not None
            timestamp = _right_to_left(
                history.direct[self.timestamp_slot],
                history.lengths,
                self.history_length,
            )
            if timestamp.device.type == "cpu":
                _validate_timestamp_values(timestamp, history_valid)
            relative_age = _relative_age_log1p(timestamp, history_valid)
            relative_age = relative_age.to(dtype=history_parts[0].dtype)
            timestamp_embedding = self.timestamp_projection(
                relative_age.unsqueeze(-1)
            ).masked_fill(~history_valid.unsqueeze(-1), 0.0)
            history_parts.append(timestamp_embedding)
        history_events = torch.cat(history_parts, dim=-1)
        history_events = history_events.masked_fill(
            ~history_valid.unsqueeze(-1),
            0.0,
        )
        return MixFormerInputsOutput(
            query=batch.query.concat(self.query_slots),
            candidates=batch.candidate.concat(self.candidate_slots),
            history_events=history_events,
            history_valid=history_valid,
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long).clamp(0, width)
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - lengths).unsqueeze(1)


def _right_to_left(
    values: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("MixFormer timestamp history must have shape [batch, history]")
    if tuple(values.shape) != (int(lengths.numel()), width):
        raise ValueError(
            "MixFormer timestamp history shape must match batch and configured "
            f"history width; got {tuple(values.shape)}, expected "
            f"{(int(lengths.numel()), width)}"
        )
    lengths = lengths.to(device=values.device, dtype=torch.long).clamp(0, width)
    positions = torch.arange(width, device=values.device)[None, :]
    valid = positions >= (width - lengths).unsqueeze(1)
    source = (positions - (width - lengths).unsqueeze(1)).clamp_min(0)
    return values.gather(1, source).masked_fill(~valid, 0.0)


def _relative_age_log1p(
    timestamps: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    newest = timestamps[:, -1:]
    relative_age = (newest - timestamps).clamp_min(0.0)
    return torch.log1p(relative_age).masked_fill(~valid, 0.0)


def _validate_timestamp_values(
    timestamps: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    valid_values = timestamps[valid]
    if not bool(torch.isfinite(valid_values).all()):
        raise ValueError("MixFormer valid history timestamps must be finite")
    comparable = valid[:, 1:] & valid[:, :-1]
    if bool(
        (comparable & timestamps[:, 1:].lt(timestamps[:, :-1])).any()
    ):
        raise ValueError("MixFormer history timestamps must be oldest-to-newest")


def _require_unique_nonempty(slots: tuple[int, ...], *, label: str) -> None:
    if not slots:
        raise ValueError(f"MixFormer {label} must not be empty")
    if len(slots) != len(set(slots)):
        raise ValueError(f"MixFormer {label} must contain unique slots")

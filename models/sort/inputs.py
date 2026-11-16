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
)


@dataclass(frozen=True)
class SORTInputsOutput:
    history_content: torch.Tensor
    history_action: torch.Tensor
    history_valid: torch.Tensor
    history_time: torch.Tensor
    query_tokens: tuple[torch.Tensor, ...]
    candidates: torch.Tensor


class SORTInputs(nn.Module):
    """Build SORT's public post-lookup request views.

    Categorical action is intentionally kept separate from history content so
    its semantic contribution remains explicit.  The two public direct time
    fields stay dense and never enter the embedding lookup path.
    """

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        query_groups: Mapping[str, Sequence[int]],
        candidate_slots: Sequence[int],
    ) -> None:
        super().__init__()
        history = dict(history)
        self.group_name = str(history.get("group", "history_group_1"))
        self.action_slot = int(history.get("action_slot", 45040))
        self.timestamp_slot = int(history.get("timestamp_slot", 45042))
        self.time_gap_slot = int(history.get("time_gap_slot", 45073))
        self.padding_side = str(history.get("padding_side", "left"))
        self.direct_validation = str(
            history.get("direct_validation", "shapes_only")
        )
        history_spec = require_history_group(input_spec, self.group_name)
        validate_history_order(
            history_spec,
            context="SORT",
            truncate_side="tail",
        )
        validate_left_padding(self.padding_side, context="SORT")
        if history_spec.direct_padding_side != "right":
            raise ValueError("SORT requires physically right-padded direct history")
        if self.direct_validation not in {"strict", "shapes_only", "off"}:
            raise ValueError("unsupported history.direct_validation")

        configured_content = history.get("content_slots")
        self.content_slots = (
            tuple(int(slot_id) for slot_id in configured_content)
            if isinstance(configured_content, Sequence)
            and not isinstance(configured_content, str)
            else tuple(
                slot_id
                for slot_id in history_spec.categorical_slot_ids
                if slot_id != self.action_slot
            )
        )
        if self.action_slot not in history_spec.categorical_slot_ids:
            raise ValueError("SORT action slot must belong to categorical history")
        if set(self.content_slots) | {self.action_slot} != set(
            history_spec.categorical_slot_ids
        ) or set(self.content_slots) & {self.action_slot}:
            raise ValueError(
                "SORT content and action slots must disjointly cover categorical history"
            )
        if len(self.content_slots) != len(set(self.content_slots)):
            raise ValueError("SORT history content slots must be unique")
        if set(history_spec.direct_slot_ids) != {
            self.timestamp_slot,
            self.time_gap_slot,
        }:
            raise ValueError(
                "SORT timestamp and time-gap slots must exactly cover direct history"
            )

        self.query_group_names = tuple(str(name) for name in query_groups)
        self.query_groups = {
            str(name): tuple(int(slot_id) for slot_id in slots)
            for name, slots in query_groups.items()
        }
        if len(self.query_group_names) != 2:
            raise ValueError("SORT requires exactly two query/profile token groups")
        if len(set(self.query_group_names)) != 2:
            raise ValueError("SORT query/profile group names must be unique")
        if any(not self.query_groups[name] for name in self.query_group_names):
            raise ValueError("SORT query/profile groups must be non-empty")
        validate_disjoint_slot_groups(
            tuple(self.query_groups[name] for name in self.query_group_names),
            context="SORT query/profile groups",
        )
        covered_query_slots = {
            slot_id
            for name in self.query_group_names
            for slot_id in self.query_groups[name]
        }
        if covered_query_slots != set(input_spec.query.slot_ids):
            raise ValueError("SORT query/profile groups must exactly cover query slots")

        self.candidate_slots = tuple(int(slot_id) for slot_id in candidate_slots)
        if self.candidate_slots != input_spec.candidate.slot_ids:
            raise ValueError(
                "SORT candidate_slots must preserve and exactly cover public candidates"
            )
        self.history_length = int(history_spec.max_length)
        self.history_content_dim = input_spec.total_dim(self.content_slots)
        self.history_action_dim = input_spec.slot(self.action_slot).output_dim
        self.query_group_dims = tuple(
            input_spec.total_dim(self.query_groups[name])
            for name in self.query_group_names
        )
        self.candidate_dim = input_spec.total_dim(self.candidate_slots)

    def forward(self, batch: RecAgoraModelBatch) -> SORTInputsOutput:
        history = batch.histories[self.group_name]
        lengths = history.lengths.to(dtype=torch.long).clamp_max(self.history_length)
        history_content = torch.cat(
            [
                history.categorical[slot_id].to_padded(
                    max_length=self.history_length,
                    padding_side=self.padding_side,
                )
                for slot_id in self.content_slots
            ],
            dim=-1,
        )
        history_action = history.categorical[self.action_slot].to_padded(
            max_length=self.history_length,
            padding_side=self.padding_side,
        )
        history_valid = _suffix_valid_mask(
            lengths,
            self.history_length,
            device=history_content.device,
        )
        timestamps = _right_to_left(
            history.direct[self.timestamp_slot],
            lengths,
            self.history_length,
        ).to(device=history_content.device, dtype=torch.float32)
        time_gaps = _right_to_left(
            history.direct[self.time_gap_slot],
            lengths,
            self.history_length,
        ).to(device=history_content.device, dtype=torch.float32)
        if self.direct_validation == "strict" and history_content.device.type == "cpu":
            valid_timestamps = timestamps[history_valid]
            valid_gaps = time_gaps[history_valid]
            if not bool(torch.isfinite(valid_timestamps).all().item()) or bool(
                valid_timestamps.le(0).any().item()
            ):
                raise ValueError("SORT valid timestamps must be finite and positive")
            if not bool(torch.isfinite(valid_gaps).all().item()) or bool(
                valid_gaps.lt(0).any().item()
            ):
                raise ValueError("SORT valid time gaps must be finite and non-negative")

        # The request contract has no exposure timestamp.  The newest valid
        # event is therefore the only honest request-relative time anchor.
        newest = timestamps[:, -1:]
        elapsed = (newest - timestamps).clamp_min(0.0)
        history_time = torch.stack(
            (
                torch.log1p(elapsed),
                torch.log1p(time_gaps.clamp_min(0.0)),
            ),
            dim=-1,
        )
        history_time = history_time.masked_fill(~history_valid[..., None], 0.0)
        return SORTInputsOutput(
            history_content=history_content,
            history_action=history_action,
            history_valid=history_valid,
            history_time=history_time,
            query_tokens=tuple(
                batch.query.concat(self.query_groups[name])
                for name in self.query_group_names
            ),
            candidates=batch.candidate.concat(self.candidate_slots),
        )


def _suffix_valid_mask(
    lengths: torch.Tensor,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(width, device=device)[None, :]
    return positions >= width - lengths.to(device=device)[:, None]


def _right_to_left(
    values: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    values = values.to(device=lengths.device)
    positions = torch.arange(width, device=values.device)[None, :]
    offsets = width - lengths.to(device=values.device)[:, None]
    source = (positions - offsets).clamp_min(0)
    valid = positions >= offsets
    return values[:, :width].gather(1, source).masked_fill(~valid, 0.0)

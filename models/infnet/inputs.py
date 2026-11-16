from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import (
    require_history_group,
    validate_history_order,
    validate_left_padding,
)


@dataclass(frozen=True)
class INFNetRequestInputs:
    query_fields: torch.Tensor
    sequence_originals: torch.Tensor
    sequence_hubs: torch.Tensor
    history_valid: torch.Tensor
    action_gates: torch.Tensor
    time_features: torch.Tensor
    position_indices: torch.Tensor


class ActionFunnelRouter(nn.Module):
    """Build exact nested task masks from public worker-mapped action IDs."""

    task_names = ("click", "atc", "order")

    def __init__(self, action_ids: Mapping[str, object]) -> None:
        super().__init__()
        values = {str(name): int(value) for name, value in action_ids.items()}
        if set(values) != set(self.task_names):
            raise ValueError(
                "INFNet history.action_ids must contain click, atc, and order"
            )
        if len(set(values.values())) != len(values) or any(
            value <= 1 for value in values.values()
        ):
            raise ValueError(
                "INFNet history.action_ids must be distinct mapped IDs above OOV"
            )
        self.click_id = values["click"]
        self.atc_id = values["atc"]
        self.order_id = values["order"]

    def forward(
        self,
        action_ids: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if action_ids.shape != valid.shape:
            raise ValueError("INFNet action IDs and valid mask must share shape")
        order = action_ids.eq(self.order_id)
        atc = action_ids.eq(self.atc_id) | order
        click = action_ids.eq(self.click_id) | atc
        gates = torch.stack((click, atc, order), dim=-1)
        return gates & valid.unsqueeze(-1)


class HistoryEventTokenizer(nn.Module):
    """Build bounded event tokens from categorical action/content and time."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        content_slots: Sequence[int],
        action_slot: int,
        token_dim: int,
        history_length: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.content_slots = tuple(int(slot) for slot in content_slots)
        self.action_slot = int(action_slot)
        self.token_dim = int(token_dim)
        self.history_length = int(history_length)
        self.content_projections = nn.ModuleList(
            nn.Linear(input_spec.slot(slot).output_dim, self.token_dim, bias=False)
            for slot in self.content_slots
        )
        self.action_projection = nn.Linear(
            input_spec.slot(self.action_slot).output_dim,
            self.token_dim,
            bias=False,
        )
        self.time_projection = nn.Linear(2, self.token_dim, bias=False)
        self.position_embedding = nn.Embedding(self.history_length, self.token_dim)
        self.bias = nn.Parameter(torch.zeros(self.token_dim))
        self.norm = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.content_projections:
            nn.init.xavier_uniform_(projection.weight)
        nn.init.xavier_uniform_(self.action_projection.weight)
        nn.init.xavier_uniform_(self.time_projection.weight)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(
        self,
        *,
        content: Sequence[torch.Tensor],
        action: torch.Tensor,
        time_features: torch.Tensor,
        position_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if len(content) != len(self.content_projections):
            raise ValueError("INFNet history tokenizer received the wrong field count")
        tokens = self.bias.view(1, 1, -1).expand(
            action.shape[0], action.shape[1], -1
        )
        for values, projection in zip(
            content,
            self.content_projections,
            strict=True,
        ):
            tokens = tokens + projection(values)
        tokens = tokens + self.action_projection(action)
        tokens = tokens + self.time_projection(time_features)
        tokens = tokens + self.position_embedding(position_indices)
        tokens = self.norm(self.activation(tokens))
        return tokens.masked_fill(~valid.unsqueeze(-1), 0.0)


class INFNetInputs(nn.Module):
    """Compile Native160 public views into INFNet-private token groups."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        categorical_input_dim: int,
        token_dim: int,
        task_names: Sequence[str],
        norm_eps: float,
    ) -> None:
        super().__init__()
        history = dict(history)
        unknown = sorted(
            set(history)
            - {
                "group",
                "action_slot",
                "timestamp_slot",
                "time_gap_slot",
                "padding_side",
                "action_ids",
            }
        )
        if unknown:
            raise ValueError(f"INFNet history contains unknown fields: {unknown}")
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.action_slot = int(history.get("action_slot", 45040))
        self.timestamp_slot = int(history.get("timestamp_slot", 45042))
        self.time_gap_slot = int(history.get("time_gap_slot", 45073))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="INFNet")
        history_spec = require_history_group(input_spec, self.history_group_name)
        validate_history_order(
            history_spec,
            context="INFNet history",
            truncate_side="tail",
        )
        if self.action_slot not in history_spec.categorical_slot_ids:
            raise ValueError("INFNet requires action_slot in categorical history")
        if set(history_spec.direct_slot_ids) != {
            self.timestamp_slot,
            self.time_gap_slot,
        }:
            raise ValueError(
                "INFNet direct history must contain timestamp_slot and time_gap_slot"
            )
        if history_spec.direct_padding_side != "right":
            raise ValueError("INFNet requires physically right-padded direct history")

        self.query_slots = tuple(int(slot) for slot in input_spec.query.slot_ids)
        self.candidate_slots = tuple(int(slot) for slot in input_spec.candidate.slot_ids)
        self.history_slots = tuple(
            int(slot) for slot in history_spec.categorical_slot_ids
        )
        self.content_slots = tuple(
            slot for slot in self.history_slots if slot != self.action_slot
        )
        if not self.query_slots or not self.candidate_slots or not self.content_slots:
            raise ValueError("INFNet requires query, candidate, and history content")
        self.categorical_input_dim = int(categorical_input_dim)
        for slot in self.query_slots + self.candidate_slots + self.content_slots:
            spec = input_spec.slot(slot)
            if spec.encoding != "embedding" or spec.output_dim != self.categorical_input_dim:
                raise ValueError(
                    f"INFNet slot {slot} must be an embedding with output_dim="
                    f"{self.categorical_input_dim}"
                )
        action_spec = input_spec.slot(self.action_slot)
        if action_spec.encoding != "embedding":
            raise ValueError("INFNet action_slot must use public embedding lookup")
        if not action_spec.mapped_ids_exposed:
            raise ValueError(
                "INFNet action_slot must be declared in features.mapped_id_slots"
            )
        action_ids = history.get("action_ids")
        if not isinstance(action_ids, Mapping):
            raise TypeError("INFNet history.action_ids must be a mapping")

        normalized_tasks = tuple(str(name) for name in task_names)
        if normalized_tasks != ActionFunnelRouter.task_names:
            raise ValueError("INFNet task order must be click, atc, order")
        self.history_length = int(history_spec.max_length)
        self.token_dim = int(token_dim)
        self.history_tokenizer = HistoryEventTokenizer(
            input_spec=input_spec,
            content_slots=self.content_slots,
            action_slot=self.action_slot,
            token_dim=self.token_dim,
            history_length=self.history_length,
            norm_eps=float(norm_eps),
        )
        self.action_router = ActionFunnelRouter(action_ids)
        self.sequence_hub_projections = nn.ModuleDict(
            {
                name: nn.Linear(self.token_dim, self.token_dim)
                for name in normalized_tasks
            }
        )
        for projection in self.sequence_hub_projections.values():
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    @property
    def num_categorical_fields(self) -> int:
        return len(self.query_slots) + len(self.candidate_slots)

    def request_inputs(self, batch: RecAgoraModelBatch) -> INFNetRequestInputs:
        history = batch.histories[self.history_group_name]
        valid = _suffix_valid_mask(history.lengths, self.history_length)
        content = tuple(
            history.categorical[slot].to_padded(
                max_length=self.history_length,
                padding_side=self.padding_side,
            )
            for slot in self.content_slots
        )
        action = history.categorical[self.action_slot].to_padded(
            max_length=self.history_length,
            padding_side=self.padding_side,
        )
        try:
            mapped_action = history.mapped_ids[self.action_slot].to_padded(
                max_length=self.history_length,
                padding_side=self.padding_side,
            )
        except KeyError as exc:
            raise KeyError(
                f"INFNet history is missing mapped IDs for action slot "
                f"{self.action_slot}"
            ) from exc
        timestamps = _right_align_direct(
            history.direct[self.timestamp_slot],
            history.lengths,
            self.history_length,
        )
        time_gaps = _right_align_direct(
            history.direct[self.time_gap_slot],
            history.lengths,
            self.history_length,
        )
        latest = timestamps[:, -1:]
        recency = latest - timestamps
        time_features = torch.stack(
            (_signed_log1p(recency), _signed_log1p(time_gaps)),
            dim=-1,
        ).masked_fill(~valid.unsqueeze(-1), 0.0)
        positions = torch.arange(
            self.history_length,
            device=history.lengths.device,
        )[None]
        position_indices = (
            positions - (self.history_length - history.lengths).unsqueeze(1)
        ).clamp(min=0, max=self.history_length - 1)
        originals = self.history_tokenizer(
            content=content,
            action=action,
            time_features=time_features,
            position_indices=position_indices,
            valid=valid,
        )
        action_gates = self.action_router(mapped_action, valid=valid)
        sequence_hubs = torch.stack(
            tuple(
                _weighted_mean(
                    self.sequence_hub_projections[name](originals),
                    action_gates[..., index],
                )
                for index, name in enumerate(ActionFunnelRouter.task_names)
            ),
            dim=1,
        )
        query_fields = torch.stack(
            tuple(batch.query[slot] for slot in self.query_slots),
            dim=1,
        )
        return INFNetRequestInputs(
            query_fields=query_fields,
            sequence_originals=originals,
            sequence_hubs=sequence_hubs,
            history_valid=valid,
            action_gates=action_gates,
            time_features=time_features,
            position_indices=position_indices,
        )

    def candidate_fields(
        self,
        batch: RecAgoraModelBatch,
        *,
        start: int,
        end: int,
    ) -> torch.Tensor:
        if start < 0 or end <= start or end > batch.list_len:
            raise ValueError("invalid INFNet candidate chunk bounds")
        return torch.stack(
            tuple(batch.candidate[slot][:, start:end] for slot in self.candidate_slots),
            dim=2,
        )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded = weights.to(values.dtype).unsqueeze(-1)
    numerator = (values * expanded).sum(dim=1)
    denominator = expanded.sum(dim=1).clamp_min(torch.finfo(values.dtype).eps)
    pooled = numerator / denominator
    has_value = weights.sum(dim=1, keepdim=True).gt(0)
    return torch.where(has_value, pooled, torch.zeros_like(pooled))


def _right_align_direct(
    values: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    values = values.to(device=lengths.device, dtype=torch.float32)
    positions = torch.arange(width, device=values.device)[None]
    bounded = lengths.clamp(min=0, max=width)
    source = (positions - (width - bounded).unsqueeze(1)).clamp_min(0)
    aligned = values.gather(1, source)
    return aligned.masked_fill(~_suffix_valid_mask(bounded, width), 0.0)


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    bounded = lengths.clamp(min=0, max=width)
    positions = torch.arange(width, device=lengths.device)[None]
    return positions >= (width - bounded).unsqueeze(1)


def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(values.abs())

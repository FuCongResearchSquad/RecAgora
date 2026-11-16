from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch, JaggedFeatureView
from models._validation import (
    require_categorical_history_group,
    validate_disjoint_slot_groups,
)

from .blocks import RMSNorm


class SemanticTokenProjection(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        token_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.projection = nn.Linear(self.input_dim, self.token_dim)
        self.activation = nn.SiLU()
        self.norm = RMSNorm(self.token_dim, eps=rms_norm_eps)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"token projection expected input dimension {self.input_dim}, "
                f"got {values.shape[-1]}"
            )
        return self.norm(self.activation(self.projection(values)))


class TokenMixerInputs(nn.Module):
    """Build global and semantic group tokens from one public lookup result."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        token_dim: int,
        token_order: Sequence[str],
        input_groups: Mapping[str, Sequence[int]],
        query_group_name: str,
        history_group_name: str,
        history_token_name: str,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.token_dim = int(token_dim)
        self.token_order = tuple(str(name) for name in token_order)
        self.input_groups = {
            str(name): tuple(int(slot_id) for slot_id in slot_ids)
            for name, slot_ids in input_groups.items()
        }
        self.query_group_name = str(query_group_name)
        self.history_group_name = str(history_group_name)
        self.history_token_name = str(history_token_name)
        self.request_shared_token_names = (
            self.query_group_name,
            self.history_token_name,
        )
        self.history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="TokenMixer",
        )
        self._validate_contract()

        self.group_input_dims = {
            name: input_spec.total_dim(self.input_groups[name])
            for name in self.token_order
        }
        self.group_tokenizers = nn.ModuleDict(
            {
                name: SemanticTokenProjection(
                    input_dim=self.group_input_dims[name],
                    token_dim=self.token_dim,
                    rms_norm_eps=rms_norm_eps,
                )
                for name in self.token_order
            }
        )
        self.global_input_dim = sum(self.group_input_dims.values())
        self.global_tokenizer = SemanticTokenProjection(
            input_dim=self.global_input_dim,
            token_dim=self.token_dim,
            rms_norm_eps=rms_norm_eps,
        )

    @property
    def total_token_count(self) -> int:
        return 1 + len(self.token_order)

    def _validate_contract(self) -> None:
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if not self.token_order or len(set(self.token_order)) != len(self.token_order):
            raise ValueError("token_order must be non-empty and unique")
        if set(self.input_groups) != set(self.token_order):
            raise ValueError("input_groups must exactly match token_order")
        if any(not self.input_groups[name] for name in self.token_order):
            raise ValueError("input_groups must not contain empty groups")
        if self.query_group_name not in self.input_groups:
            raise ValueError("query_group_name must name one token group")
        if self.history_token_name not in self.input_groups:
            raise ValueError("history_token_name must name one token group")
        if self.query_group_name == self.history_token_name:
            raise ValueError("query and history token groups must be different")
        validate_disjoint_slot_groups(
            tuple(self.input_groups[name] for name in self.token_order),
            context="TokenMixer input_groups",
        )

        selected = (
            set(self.input_spec.query.slot_ids)
            | set(self.input_spec.candidate.slot_ids)
            | set(self.history_spec.categorical_slot_ids)
        )
        grouped = {
            slot_id
            for name in self.token_order
            for slot_id in self.input_groups[name]
        }
        if grouped != selected:
            raise ValueError(
                "TokenMixer input_groups must exactly cover selected slots; "
                f"missing={sorted(selected - grouped)}, "
                f"extra={sorted(grouped - selected)}"
            )
        if set(self.input_groups[self.query_group_name]) != set(
            self.input_spec.query.slot_ids
        ):
            raise ValueError("query token group must exactly cover query slots")
        if set(self.input_groups[self.history_token_name]) != set(
            self.history_spec.categorical_slot_ids
        ):
            raise ValueError(
                "history token group must exactly cover categorical history slots"
            )

        for group_name in self.token_order:
            expected_view = (
                "query"
                if group_name == self.query_group_name
                else "history"
                if group_name == self.history_token_name
                else "candidate"
            )
            expected_role = (
                "request"
                if expected_view == "query"
                else "sequence"
                if expected_view == "history"
                else "item"
            )
            expected_reduction = "none" if expected_view == "history" else "mean"
            for slot_id in self.input_groups[group_name]:
                slot = self.input_spec.slot(slot_id)
                if (slot.view, slot.role) != (expected_view, expected_role):
                    raise ValueError(
                        f"TokenMixer group {group_name!r} slot {slot_id} has "
                        f"view/role={slot.view!r}/{slot.role!r}, expected "
                        f"{expected_view!r}/{expected_role!r}"
                    )
                if slot.encoding != "embedding":
                    raise ValueError(
                        f"TokenMixer only supports embedding slot {slot_id}"
                    )
                if slot.reduction != expected_reduction:
                    raise ValueError(
                        f"TokenMixer slot {slot_id} reduction must be "
                        f"{expected_reduction!r}"
                    )

    def group_values(self, batch: RecAgoraModelBatch) -> dict[str, torch.Tensor]:
        batch_size = batch.batch_size
        list_len = batch.list_len
        history = batch.histories[self.history_group_name]
        values: dict[str, torch.Tensor] = {}
        for group_name in self.token_order:
            slot_ids = self.input_groups[group_name]
            if group_name == self.query_group_name:
                group = batch.query.concat(slot_ids)[:, None, :].expand(
                    -1,
                    list_len,
                    -1,
                )
            elif group_name == self.history_token_name:
                group = torch.cat(
                    [
                        _segment_mean(history.categorical[slot_id])
                        for slot_id in slot_ids
                    ],
                    dim=-1,
                )[:, None, :].expand(-1, list_len, -1)
            else:
                group = batch.candidate.concat(slot_ids)
            if tuple(group.shape[:2]) != (batch_size, list_len):
                raise ValueError(
                    f"TokenMixer group {group_name!r} did not preserve [B, L]"
                )
            values[group_name] = group
        return values

    def forward(self, batch: RecAgoraModelBatch) -> torch.Tensor:
        values = self.group_values(batch)
        group_tokens = torch.stack(
            [
                self._tokenize_group(name, values[name], batch.list_len)
                for name in self.token_order
            ],
            dim=2,
        )
        global_values = torch.cat(
            [values[name] for name in self.token_order],
            dim=-1,
        )
        global_token = self.global_tokenizer(global_values).unsqueeze(2)
        tokens = torch.cat((global_token, group_tokens), dim=2)
        return tokens.reshape(
            batch.batch_size * batch.list_len,
            self.total_token_count,
            self.token_dim,
        )

    def _tokenize_group(
        self,
        name: str,
        values: torch.Tensor,
        list_len: int,
    ) -> torch.Tensor:
        if name not in self.request_shared_token_names:
            return self.group_tokenizers[name](values)
        shared = self.group_tokenizers[name](values[:, 0, :])
        return shared[:, None, :].expand(-1, list_len, -1)


def _segment_mean(feature: JaggedFeatureView) -> torch.Tensor:
    values = feature.values
    lengths = feature.lengths
    batch_size = int(lengths.numel())
    output = values.new_zeros((batch_size, int(values.shape[-1])))
    if values.shape[0]:
        owners = torch.repeat_interleave(
            torch.arange(batch_size, device=lengths.device),
            lengths,
            output_size=int(values.shape[0]),
        )
        output.index_add_(0, owners, values)
    return output / lengths.clamp_min(1).to(values.dtype).unsqueeze(-1)

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch, JaggedFeatureView
from models._validation import require_categorical_history_group


class FieldWiseLayerNorm(nn.Module):
    """Layer normalization with private affine parameters for every field."""

    def __init__(self, *, num_fields: int, field_dim: int, eps: float) -> None:
        super().__init__()
        self.num_fields = _positive_int(num_fields, "num_fields")
        self.field_dim = _positive_int(field_dim, "field_dim")
        self.eps = _positive_float(eps, "eps")
        self.weight = nn.Parameter(torch.ones(self.num_fields, self.field_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_fields, self.field_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _validate_field_tensor(
            values,
            num_fields=self.num_fields,
            field_dim=self.field_dim,
            context="FieldWiseLayerNorm",
        )
        normalized = values.to(torch.float32)
        mean = normalized.mean(dim=-1, keepdim=True)
        inverse_std = torch.rsqrt(
            (normalized - mean).square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = ((normalized - mean) * inverse_std).to(dtype=values.dtype)
        return normalized * self.weight.to(dtype=values.dtype) + self.bias.to(
            dtype=values.dtype
        )


class FieldWiseTokenizer(nn.Module):
    """Vectorized field-private projection used by the HiFormer input layer."""

    def __init__(
        self,
        *,
        num_fields: int,
        input_dim: int,
        token_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_fields = _positive_int(num_fields, "num_fields")
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.weight = nn.Parameter(
            torch.empty(self.num_fields, self.input_dim, self.token_dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_fields, self.token_dim))
        self.activation = nn.SiLU()
        self.norm = FieldWiseLayerNorm(
            num_fields=self.num_fields,
            field_dim=self.token_dim,
            eps=norm_eps,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for field_index in range(self.num_fields):
            nn.init.xavier_uniform_(self.weight[field_index])
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _validate_field_tensor(
            values,
            num_fields=self.num_fields,
            field_dim=self.input_dim,
            context="FieldWiseTokenizer",
        )
        projected = torch.einsum("...fi,fid->...fd", values, self.weight)
        projected = projected + self.bias
        return self.norm(self.activation(projected))


class HiFormerFieldInputs(nn.Module):
    """Build fieldwise tokens from public post-lookup RecAgora views."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str,
        input_dim: int,
        token_dim: int,
        norm_eps: float,
        action: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.history_group_name = str(history_group_name)
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="HiFormer",
        )
        self.query_slot_ids = tuple(input_spec.query.slot_ids)
        self.candidate_slot_ids = tuple(input_spec.candidate.slot_ids)
        self.history_slot_ids = tuple(self.history_spec.categorical_slot_ids)
        self.action_slot_id = (
            None if action is None else int(action["slot_id"])
        )
        self.action_input_dim = (
            None if action is None else int(action["input_dim"])
        )
        self._validate_contract()

        if action is None:
            self.action_adapter: nn.Module | None = None
        else:
            self.action_adapter = nn.Linear(
                int(action["input_dim"]),
                int(action["output_dim"]),
                bias=bool(action["bias"]),
            )
            nn.init.xavier_uniform_(self.action_adapter.weight)
            if self.action_adapter.bias is not None:
                nn.init.zeros_(self.action_adapter.bias)

        self.query_tokenizer = FieldWiseTokenizer(
            num_fields=len(self.query_slot_ids),
            input_dim=self.input_dim,
            token_dim=self.token_dim,
            norm_eps=norm_eps,
        )
        self.candidate_tokenizer = FieldWiseTokenizer(
            num_fields=len(self.candidate_slot_ids),
            input_dim=self.input_dim,
            token_dim=self.token_dim,
            norm_eps=norm_eps,
        )
        self.history_tokenizer = FieldWiseTokenizer(
            num_fields=len(self.history_slot_ids),
            input_dim=self.input_dim,
            token_dim=self.token_dim,
            norm_eps=norm_eps,
        )

    @property
    def num_feature_tokens(self) -> int:
        return (
            len(self.query_slot_ids)
            + len(self.candidate_slot_ids)
            + len(self.history_slot_ids)
        )

    def _validate_contract(self) -> None:
        if not self.query_slot_ids or not self.candidate_slot_ids:
            raise ValueError("HiFormer requires query and candidate fields")
        if not self.history_slot_ids:
            raise ValueError("HiFormer requires categorical history fields")
        if (
            self.action_slot_id is not None
            and self.action_slot_id not in self.history_slot_ids
        ):
            raise ValueError(
                "HiFormer action.slot_id must belong to the configured history group"
            )
        for slot_id in self.query_slot_ids:
            self._validate_slot(slot_id, view="query", role="request", reduction="mean")
        for slot_id in self.candidate_slot_ids:
            self._validate_slot(slot_id, view="candidate", role="item", reduction="mean")
        for slot_id in self.history_slot_ids:
            self._validate_slot(
                slot_id,
                view="history",
                role="sequence",
                reduction="none",
                expected_dim=(
                    self.action_input_dim
                    if slot_id == self.action_slot_id
                    else self.input_dim
                ),
            )

    def _validate_slot(
        self,
        slot_id: int,
        *,
        view: str,
        role: str,
        reduction: str,
        expected_dim: int | None = None,
    ) -> None:
        slot = self.input_spec.slot(slot_id)
        if (
            slot.view != view
            or slot.role != role
            or slot.encoding != "embedding"
            or slot.reduction != reduction
        ):
            raise ValueError(
                f"HiFormer slot {slot_id} has incompatible public semantics"
            )
        required_dim = self.input_dim if expected_dim is None else expected_dim
        if slot.output_dim != required_dim:
            raise ValueError(
                f"HiFormer slot {slot_id} output_dim={slot.output_dim} does not "
                f"match required input dimension {required_dim}"
            )

    def request_shared_tokens(
        self,
        batch: RecAgoraModelBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_values = torch.stack(
            [batch.query[slot_id] for slot_id in self.query_slot_ids],
            dim=1,
        )
        history = batch.histories[self.history_group_name]
        pooled_history: list[torch.Tensor] = []
        for slot_id in self.history_slot_ids:
            field = _segment_mean(history.categorical[slot_id])
            if slot_id == self.action_slot_id:
                if self.action_adapter is None:
                    raise RuntimeError("HiFormer action adapter is not initialized")
                field = self.action_adapter(field)
            pooled_history.append(field)
        history_values = torch.stack(pooled_history, dim=1)
        return (
            self.query_tokenizer(query_values),
            self.history_tokenizer(history_values),
        )

    def candidate_tokens(
        self,
        batch: RecAgoraModelBatch,
        *,
        start: int,
        end: int,
    ) -> torch.Tensor:
        if start < 0 or end <= start or end > batch.list_len:
            raise ValueError("invalid HiFormer candidate chunk bounds")
        values = torch.stack(
            [
                batch.candidate[slot_id][:, start:end]
                for slot_id in self.candidate_slot_ids
            ],
            dim=2,
        )
        return self.candidate_tokenizer(values)


def _segment_mean(feature: JaggedFeatureView) -> torch.Tensor:
    values = feature.values
    lengths = feature.lengths
    batch_size = int(lengths.numel())
    output = values.new_zeros((batch_size, int(values.shape[-1])))
    total = int(values.shape[0])
    if total:
        owners = torch.repeat_interleave(
            torch.arange(batch_size, device=lengths.device),
            lengths,
            output_size=total,
        )
        output.index_add_(0, owners, values)
    return output / lengths.clamp_min(1).to(values.dtype).unsqueeze(-1)


def _validate_field_tensor(
    values: torch.Tensor,
    *,
    num_fields: int,
    field_dim: int,
    context: str,
) -> None:
    if values.ndim < 3 or tuple(values.shape[-2:]) != (num_fields, field_dim):
        raise ValueError(
            f"{context} expected [..., {num_fields}, {field_dim}], got "
            f"{tuple(values.shape)}"
        )


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted

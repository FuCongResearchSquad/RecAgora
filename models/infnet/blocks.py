from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


class FieldWiseCategoricalTokenizer(nn.Module):
    """Field-private original tokens plus auto-split categorical hubs."""

    def __init__(
        self,
        *,
        num_fields: int,
        input_dim: int,
        token_dim: int,
        num_hubs: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if num_fields <= 0 or input_dim <= 0 or token_dim <= 0 or num_hubs <= 0:
            raise ValueError("INFNet tokenizer dimensions must be positive")
        if num_hubs > num_fields:
            raise ValueError("INFNet num categorical hubs cannot exceed field count")
        self.num_fields = int(num_fields)
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.num_hubs = int(num_hubs)
        self.original_weight = nn.Parameter(
            torch.empty(self.num_fields, self.input_dim, self.token_dim)
        )
        self.original_bias = nn.Parameter(torch.zeros(self.num_fields, self.token_dim))
        self.original_norm = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        self.hub_slices = _balanced_slices(self.num_fields, self.num_hubs)
        self.hub_projections = nn.ModuleList(
            nn.Linear((end - start) * self.input_dim, self.token_dim)
            for start, end in self.hub_slices
        )
        self.hub_norm = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for field in range(self.num_fields):
            nn.init.xavier_uniform_(self.original_weight[field])
        nn.init.zeros_(self.original_bias)
        for projection in self.hub_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.ones_(self.original_norm.weight)
        nn.init.zeros_(self.original_norm.bias)
        nn.init.ones_(self.hub_norm.weight)
        nn.init.zeros_(self.hub_norm.bias)

    def forward(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if fields.ndim != 3 or tuple(fields.shape[1:]) != (
            self.num_fields,
            self.input_dim,
        ):
            raise ValueError(
                "INFNet categorical tokenizer expected "
                f"[rows, {self.num_fields}, {self.input_dim}], got "
                f"{tuple(fields.shape)}"
            )
        originals = torch.einsum(
            "bfd,fdt->bft",
            fields,
            self.original_weight,
        ) + self.original_bias
        originals = self.original_norm(self.activation(originals))
        hubs = torch.stack(
            tuple(
                projection(fields[:, start:end].flatten(start_dim=1))
                for projection, (start, end) in zip(
                    self.hub_projections,
                    self.hub_slices,
                    strict=True,
                )
            ),
            dim=1,
        )
        return originals, self.hub_norm(self.activation(hubs))


class HubSpecificAggregator(nn.Module):
    """Aggregate categorical, sequence and task views into one hub group."""

    def __init__(self, token_dim: int, dropout: float, norm_eps: float) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.query_projection = nn.Linear(self.token_dim, self.token_dim)
        self.fusion = nn.Linear(3 * self.token_dim, self.token_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        for projection in (self.query_projection, self.fusion):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        hubs: torch.Tensor,
        *,
        categorical_keys: torch.Tensor,
        categorical_values: torch.Tensor,
        sequence_keys: torch.Tensor,
        sequence_values: torch.Tensor,
        task_keys: torch.Tensor,
        task_values: torch.Tensor,
        sequence_valid: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_projection(hubs)
        categorical = _masked_cross_attention(
            query,
            categorical_keys,
            categorical_values,
            valid=None,
        )
        sequence = _masked_cross_attention(
            query,
            sequence_keys,
            sequence_values,
            valid=sequence_valid,
        )
        task = _masked_cross_attention(
            query,
            task_keys,
            task_values,
            valid=None,
        )
        update = self.fusion(torch.cat((categorical, sequence, task), dim=-1))
        return self.norm(hubs + self.dropout(update))


class MultiViewAggregation(nn.Module):
    def __init__(self, *, token_dim: int, dropout: float, norm_eps: float) -> None:
        super().__init__()
        self.key_projections = nn.ModuleDict(
            {name: nn.Linear(token_dim, token_dim) for name in ("c", "s", "t")}
        )
        self.value_projections = nn.ModuleDict(
            {name: nn.Linear(token_dim, token_dim) for name in ("c", "s", "t")}
        )
        self.aggregators = nn.ModuleDict(
            {
                name: HubSpecificAggregator(token_dim, dropout, norm_eps)
                for name in ("c", "s", "t")
            }
        )
        for projection in tuple(self.key_projections.values()) + tuple(
            self.value_projections.values()
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        *,
        categorical_hubs: torch.Tensor,
        sequence_hubs: torch.Tensor,
        task_hubs: torch.Tensor,
        categorical_originals: torch.Tensor,
        sequence_originals: torch.Tensor,
        task_originals: torch.Tensor,
        sequence_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = {
            "c": self.key_projections["c"](categorical_originals),
            "s": self.key_projections["s"](sequence_originals),
            "t": self.key_projections["t"](task_originals),
        }
        values = {
            "c": self.value_projections["c"](categorical_originals),
            "s": self.value_projections["s"](sequence_originals),
            "t": self.value_projections["t"](task_originals),
        }
        common = {
            "categorical_keys": keys["c"],
            "categorical_values": values["c"],
            "sequence_keys": keys["s"],
            "sequence_values": values["s"],
            "task_keys": keys["t"],
            "task_values": values["t"],
            "sequence_valid": sequence_valid,
        }
        return (
            self.aggregators["c"](categorical_hubs, **common),
            self.aggregators["s"](sequence_hubs, **common),
            self.aggregators["t"](task_hubs, **common),
        )


class BroadcastGatedUnit(nn.Module):
    """Generate affine token updates from the corresponding global hubs."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_hubs: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(token_dim) * int(expansion_factor)
        self.affine = nn.Sequential(
            nn.Linear(int(num_hubs) * int(token_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2 * int(token_dim)),
        )
        self.norm = nn.LayerNorm(int(token_dim), eps=float(norm_eps))
        for module in self.affine:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        originals: torch.Tensor,
        hubs: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        alpha, beta = self.affine(hubs.flatten(start_dim=1)).chunk(2, dim=-1)
        affine = originals * torch.sigmoid(alpha.unsqueeze(1)) + beta.unsqueeze(1)
        updated = self.norm(originals + affine)
        if valid is not None:
            updated = updated.masked_fill(~valid.unsqueeze(-1), 0.0)
        return updated


class MultiViewBroadcast(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_categorical_hubs: int,
        num_sequence_hubs: int,
        num_task_hubs: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        common = {
            "token_dim": token_dim,
            "expansion_factor": expansion_factor,
            "dropout": dropout,
            "norm_eps": norm_eps,
        }
        self.categorical = BroadcastGatedUnit(
            num_hubs=num_categorical_hubs,
            **common,
        )
        self.sequence = BroadcastGatedUnit(num_hubs=num_sequence_hubs, **common)
        self.task = BroadcastGatedUnit(num_hubs=num_task_hubs, **common)

    def forward(
        self,
        *,
        categorical_originals: torch.Tensor,
        sequence_originals: torch.Tensor,
        task_originals: torch.Tensor,
        categorical_hubs: torch.Tensor,
        sequence_hubs: torch.Tensor,
        task_hubs: torch.Tensor,
        sequence_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.categorical(categorical_originals, categorical_hubs),
            self.sequence(
                sequence_originals,
                sequence_hubs,
                valid=sequence_valid,
            ),
            self.task(task_originals, task_hubs),
        )


class INFNetBlock(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_categorical_hubs: int,
        num_sequence_hubs: int,
        num_task_hubs: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.aggregation = MultiViewAggregation(
            token_dim=token_dim,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.broadcast = MultiViewBroadcast(
            token_dim=token_dim,
            num_categorical_hubs=num_categorical_hubs,
            num_sequence_hubs=num_sequence_hubs,
            num_task_hubs=num_task_hubs,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        categorical_originals: torch.Tensor,
        sequence_originals: torch.Tensor,
        task_originals: torch.Tensor,
        categorical_hubs: torch.Tensor,
        sequence_hubs: torch.Tensor,
        task_hubs: torch.Tensor,
        sequence_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        categorical_hubs, sequence_hubs, task_hubs = self.aggregation(
            categorical_hubs=categorical_hubs,
            sequence_hubs=sequence_hubs,
            task_hubs=task_hubs,
            categorical_originals=categorical_originals,
            sequence_originals=sequence_originals,
            task_originals=task_originals,
            sequence_valid=sequence_valid,
        )
        categorical_originals, sequence_originals, task_originals = self.broadcast(
            categorical_originals=categorical_originals,
            sequence_originals=sequence_originals,
            task_originals=task_originals,
            categorical_hubs=categorical_hubs,
            sequence_hubs=sequence_hubs,
            task_hubs=task_hubs,
            sequence_valid=sequence_valid,
        )
        return (
            categorical_originals,
            sequence_originals,
            task_originals,
            categorical_hubs,
            sequence_hubs,
            task_hubs,
        )


class INFNetTaskReadoutBlock(nn.Module):
    """Finish INFNet with the only view consumed by the prediction towers.

    A symmetric Aggregate-Broadcast block would update categorical and sequence
    views after the final task update even though no later layer consumes them.
    Omitting those dead terminal branches keeps every instantiated parameter on
    the ranking-loss graph while preserving the task-hub aggregation from all
    three original-token views.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_task_hubs: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.key_projections = nn.ModuleDict(
            {name: nn.Linear(token_dim, token_dim) for name in ("c", "s", "t")}
        )
        self.value_projections = nn.ModuleDict(
            {name: nn.Linear(token_dim, token_dim) for name in ("c", "s", "t")}
        )
        for projection in tuple(self.key_projections.values()) + tuple(
            self.value_projections.values()
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.task_aggregation = HubSpecificAggregator(
            token_dim,
            dropout,
            norm_eps,
        )
        self.task_broadcast = BroadcastGatedUnit(
            token_dim=token_dim,
            num_hubs=num_task_hubs,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        categorical_originals: torch.Tensor,
        sequence_originals: torch.Tensor,
        task_originals: torch.Tensor,
        categorical_hubs: torch.Tensor,
        sequence_hubs: torch.Tensor,
        task_hubs: torch.Tensor,
        sequence_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        task_hubs = self.task_aggregation(
            task_hubs,
            categorical_keys=self.key_projections["c"](categorical_originals),
            categorical_values=self.value_projections["c"](
                categorical_originals
            ),
            sequence_keys=self.key_projections["s"](sequence_originals),
            sequence_values=self.value_projections["s"](sequence_originals),
            task_keys=self.key_projections["t"](task_originals),
            task_values=self.value_projections["t"](task_originals),
            sequence_valid=sequence_valid,
        )
        task_originals = self.task_broadcast(task_originals, task_hubs)
        return (
            categorical_originals,
            sequence_originals,
            task_originals,
            categorical_hubs,
            sequence_hubs,
            task_hubs,
        )


@dataclass(frozen=True)
class INFNetBackboneOutput:
    categorical_originals: torch.Tensor
    sequence_originals: torch.Tensor
    task_originals: torch.Tensor
    categorical_hubs: torch.Tensor
    sequence_hubs: torch.Tensor
    task_hubs: torch.Tensor


class INFNetBackbone(nn.Module):
    """Layer-wise INFNet Aggregate then Broadcast interaction stack."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_layers: int,
        num_categorical_hubs: int,
        num_sequence_hubs: int,
        num_task_hubs: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("INFNet num_layers must be positive")
        full_layers: list[nn.Module] = [
            INFNetBlock(
                token_dim=token_dim,
                num_categorical_hubs=num_categorical_hubs,
                num_sequence_hubs=num_sequence_hubs,
                num_task_hubs=num_task_hubs,
                expansion_factor=expansion_factor,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(int(num_layers) - 1)
        ]
        full_layers.append(
            INFNetTaskReadoutBlock(
                token_dim=token_dim,
                num_task_hubs=num_task_hubs,
                expansion_factor=expansion_factor,
                dropout=dropout,
                norm_eps=norm_eps,
            )
        )
        self.layers = nn.ModuleList(full_layers)

    def forward(
        self,
        *,
        categorical_originals: torch.Tensor,
        sequence_originals: torch.Tensor,
        task_originals: torch.Tensor,
        categorical_hubs: torch.Tensor,
        sequence_hubs: torch.Tensor,
        task_hubs: torch.Tensor,
        sequence_valid: torch.Tensor,
    ) -> INFNetBackboneOutput:
        state = (
            categorical_originals,
            sequence_originals,
            task_originals,
            categorical_hubs,
            sequence_hubs,
            task_hubs,
        )
        for layer in self.layers:
            state = layer(*state, sequence_valid=sequence_valid)
        return INFNetBackboneOutput(*state)


def _masked_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if valid is None:
        weights = torch.softmax(scores, dim=-1)
    else:
        mask = valid[:, None, :]
        floor = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~mask, floor)
        maximum = masked.max(dim=-1, keepdim=True).values
        unnormalized = torch.exp(masked - maximum) * mask.to(scores.dtype)
        weights = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(scores.dtype).eps
        )
    return torch.matmul(weights, value)


def _balanced_slices(num_fields: int, num_hubs: int) -> tuple[tuple[int, int], ...]:
    base, extra = divmod(num_fields, num_hubs)
    slices: list[tuple[int, int]] = []
    start = 0
    for index in range(num_hubs):
        width = base + int(index < extra)
        slices.append((start, start + width))
        start += width
    return tuple(slices)

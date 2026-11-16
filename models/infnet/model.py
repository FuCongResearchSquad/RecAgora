from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from recagora import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    RecAgoraRankingModel,
    RecAgoraSparseBatch,
    RankingOutput,
    masked_binary_cross_entropy_with_logits,
    structured_ranking_output,
)
from models._validation import (
    finite_nonnegative_float,
    finite_positive_float,
    validate_dropout,
    validate_known_weights,
    validate_positive_int,
    validate_positive_int_sequence,
    validate_standard_multitask_contract,
)

from .blocks import FieldWiseCategoricalTokenizer, INFNetBackbone
from .inputs import ActionFunnelRouter, INFNetInputs, INFNetRequestInputs


class INFNetModel(RecAgoraRankingModel):
    """INFNet Aggregate-Broadcast core adapted to request-level ranking."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        categorical_input_dim: int,
        token_dim: int,
        num_categorical_hubs: int,
        num_shared_task_hubs: int,
        num_layers: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
        candidate_isolation: bool,
        candidate_chunk_size: int,
        task_names: Sequence[str],
        tower_hidden_units: Sequence[int],
        tower_dropout: float,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        for value, name in (
            (categorical_input_dim, "categorical_input_dim"),
            (token_dim, "token_dim"),
            (num_categorical_hubs, "num_categorical_hubs"),
            (num_shared_task_hubs, "num_shared_task_hubs"),
            (num_layers, "num_layers"),
            (expansion_factor, "expansion_factor"),
            (candidate_chunk_size, "candidate_chunk_size"),
        ):
            validate_positive_int(int(value), field_name=name)
        validate_dropout(dropout)
        validate_dropout(tower_dropout, field_name="tower_dropout")
        finite_positive_float(norm_eps, "INFNet norm_eps")
        normalized_tasks = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="INFNet",
        )
        if normalized_tasks != ActionFunnelRouter.task_names:
            raise ValueError("INFNet task order must be click, atc, order")
        hidden_units = tuple(int(value) for value in tower_hidden_units)
        validate_positive_int_sequence(hidden_units, field_name="tower_hidden_units")
        configured_weights = {
            str(name): finite_nonnegative_float(weight, f"task_weights.{name}")
            for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(
            configured_weights,
            normalized_tasks,
            field_name="task_weights",
        )
        effective_weights = {
            name: configured_weights.get(name, 1.0) for name in normalized_tasks
        }
        if not any(weight > 0 for weight in effective_weights.values()):
            raise ValueError("INFNet requires a positive task weight")

        params: dict[str, Any] = {
            "history": dict(history),
            "categorical_input_dim": int(categorical_input_dim),
            "token_dim": int(token_dim),
            "num_categorical_hubs": int(num_categorical_hubs),
            "num_shared_task_hubs": int(num_shared_task_hubs),
            "num_layers": int(num_layers),
            "expansion_factor": int(expansion_factor),
            "dropout": float(dropout),
            "norm_eps": float(norm_eps),
            "candidate_isolation": bool(candidate_isolation),
            "candidate_chunk_size": int(candidate_chunk_size),
            "task_names": normalized_tasks,
            "tower_hidden_units": hidden_units,
            "tower_dropout": float(tower_dropout),
            "task_weights": effective_weights,
        }
        super().__init__(params, input_spec)
        self.task_names = normalized_tasks
        self.task_weights = effective_weights
        self.token_dim = int(token_dim)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.inputs = INFNetInputs(
            input_spec=input_spec,
            history=history,
            categorical_input_dim=int(categorical_input_dim),
            token_dim=self.token_dim,
            task_names=self.task_names,
            norm_eps=float(norm_eps),
        )
        self.categorical_tokenizer = FieldWiseCategoricalTokenizer(
            num_fields=self.inputs.num_categorical_fields,
            input_dim=int(categorical_input_dim),
            token_dim=self.token_dim,
            num_hubs=int(num_categorical_hubs),
            norm_eps=float(norm_eps),
        )
        self.task_originals = nn.Parameter(
            torch.empty(len(self.task_names), self.token_dim)
        )
        self.shared_task_hubs = nn.Parameter(
            torch.empty(int(num_shared_task_hubs), self.token_dim)
        )
        nn.init.xavier_uniform_(self.task_originals)
        nn.init.xavier_uniform_(self.shared_task_hubs)
        self.backbone = INFNetBackbone(
            token_dim=self.token_dim,
            num_layers=int(num_layers),
            num_categorical_hubs=int(num_categorical_hubs),
            num_sequence_hubs=len(self.task_names),
            num_task_hubs=len(self.task_names) + int(num_shared_task_hubs),
            expansion_factor=int(expansion_factor),
            dropout=float(dropout),
            norm_eps=float(norm_eps),
        )
        self.heads = nn.ModuleDict(
            {
                name: _build_task_tower(
                    input_dim=self.token_dim,
                    hidden_units=hidden_units,
                    dropout=float(tower_dropout),
                )
                for name in self.task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        return structured_ranking_output(self._forward_slots(embedded))

    def _forward_slots(self, embedded: RecAgoraModelBatch) -> dict[str, torch.Tensor]:
        request = self.inputs.request_inputs(embedded)
        chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.task_names
        }
        for start in range(0, embedded.list_len, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, embedded.list_len)
            candidate_fields = self.inputs.candidate_fields(
                embedded,
                start=start,
                end=end,
            )
            chunk = self._score_candidate_chunk(request, candidate_fields)
            for name, logits in zip(self.task_names, chunk, strict=True):
                chunks[name].append(logits)
        return {
            name: torch.cat(chunks[name], dim=1)
            for name in self.task_names
        }

    def _score_candidate_chunk(
        self,
        request: INFNetRequestInputs,
        candidate_fields: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, candidate_count = candidate_fields.shape[:2]
        query = request.query_fields[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
        )
        categorical_fields = torch.cat((query, candidate_fields), dim=2).reshape(
            batch_size * candidate_count,
            self.inputs.num_categorical_fields,
            self.inputs.categorical_input_dim,
        )
        categorical_originals, categorical_hubs = self.categorical_tokenizer(
            categorical_fields
        )
        sequence_originals = _expand_candidates(
            request.sequence_originals,
            candidate_count,
        )
        sequence_hubs = _expand_candidates(request.sequence_hubs, candidate_count)
        sequence_valid = _expand_candidates(
            request.history_valid,
            candidate_count,
        )
        rows = batch_size * candidate_count
        task_originals = self.task_originals[None].expand(rows, -1, -1)
        shared_task_hubs = self.shared_task_hubs[None].expand(rows, -1, -1)
        task_hubs = torch.cat((task_originals, shared_task_hubs), dim=1)
        state = self.backbone(
            categorical_originals=categorical_originals,
            sequence_originals=sequence_originals,
            task_originals=task_originals,
            categorical_hubs=categorical_hubs,
            sequence_hubs=sequence_hubs,
            task_hubs=task_hubs,
            sequence_valid=sequence_valid,
        )
        task_states = state.task_originals.reshape(
            batch_size,
            candidate_count,
            len(self.task_names),
            self.token_dim,
        )
        return tuple(
            self.heads[name](task_states[:, :, index]).squeeze(-1)
            for index, name in enumerate(self.task_names)
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name in self.task_names:
            labels = batch.labels[name].to(
                device=logits[name].device,
                dtype=logits[name].dtype,
            )
            value = masked_binary_cross_entropy_with_logits(
                logits[name],
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = value
            weighted.append(value * float(self.task_weights[name]))
        total = torch.stack(weighted).sum()
        losses["total"] = total
        return total, losses


def _expand_candidates(value: torch.Tensor, candidate_count: int) -> torch.Tensor:
    return value[:, None].expand(-1, candidate_count, *value.shape[1:]).reshape(
        value.shape[0] * candidate_count,
        *value.shape[1:],
    )


def _build_task_tower(
    *,
    input_dim: int,
    hidden_units: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_units:
        linear = nn.Linear(previous, int(hidden))
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.SiLU(), nn.Dropout(float(dropout))))
        previous = int(hidden)
    output = nn.Linear(previous, 1)
    nn.init.xavier_uniform_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)

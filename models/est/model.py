from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

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
    validate_head_dimensions,
    validate_known_weights,
    validate_positive_int,
    validate_standard_multitask_contract,
)

from .blocks import (
    ESTLCABlock,
    FeatureTokenizer,
    FieldWiseTokenizer,
    PartitionedTaskHead,
    build_activation,
)
from .inputs import ESTLCAInputs


class ESTLCAModel(RecAgoraRankingModel):
    """Request-shared EST-LCA ranker over the public RecAgora slot interface."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "model_defined"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
        model_dim: int,
        num_heads: int,
        num_layers: int,
        expansion_factor: int,
        tokenizer_hidden_dim: int,
        tokenizer_activation: str = "silu",
        ffn_activation: str = "relu",
        dropout: float = 0.0,
        norm_eps: float = 1.0e-6,
        attention_backend: str = "math",
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_dim: int = 128,
        head_activation: str = "relu",
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
        point_loss_weight: float = 1.0,
        list_loss_weight: float = 0.0,
    ) -> None:
        validate_positive_int(model_dim, field_name="model_dim")
        validate_positive_int(num_heads, field_name="num_heads")
        if model_dim % num_heads:
            raise ValueError("EST-LCA model_dim must be divisible by num_heads")
        validate_positive_int(num_layers, field_name="num_layers")
        validate_positive_int(expansion_factor, field_name="expansion_factor")
        validate_positive_int(
            tokenizer_hidden_dim,
            field_name="tokenizer_hidden_dim",
        )
        build_activation(tokenizer_activation)
        build_activation(ffn_activation)
        build_activation(head_activation)
        validate_dropout(dropout)
        finite_positive_float(norm_eps, "EST-LCA norm_eps")
        if attention_backend not in {"math", "sdpa"}:
            raise ValueError("EST-LCA attention_backend must be math or sdpa")
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="EST-LCA",
        )
        validate_head_dimensions(head_hidden_dim, head_dropout)
        task_weights = {
            str(name): float(weight) for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")
        finite_nonnegative_float(point_loss_weight, "point_loss_weight")
        finite_nonnegative_float(list_loss_weight, "list_loss_weight")
        if point_loss_weight == 0.0 and list_loss_weight == 0.0:
            raise ValueError("EST-LCA requires at least one positive loss weight")

        params: dict[str, Any] = {
            "query_slots": tuple(int(slot) for slot in query_slots),
            "candidate_slots": tuple(int(slot) for slot in candidate_slots),
            "history": dict(history),
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "expansion_factor": int(expansion_factor),
            "tokenizer_hidden_dim": int(tokenizer_hidden_dim),
            "tokenizer_activation": str(tokenizer_activation),
            "ffn_activation": str(ffn_activation),
            "dropout": float(dropout),
            "norm_eps": float(norm_eps),
            "attention_backend": str(attention_backend),
            "candidate_isolation": bool(candidate_isolation),
            "task_names": task_names,
            "head_hidden_dim": int(head_hidden_dim),
            "head_activation": str(head_activation),
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
            "point_loss_weight": float(point_loss_weight),
            "list_loss_weight": float(list_loss_weight),
        }
        super().__init__(params, input_spec)
        self.inputs = ESTLCAInputs(
            input_spec=input_spec,
            query_slots=query_slots,
            candidate_slots=candidate_slots,
            history=history,
        )
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.task_names = task_names
        self.task_weights = task_weights
        self.point_loss_weight = float(point_loss_weight)
        self.list_loss_weight = float(list_loss_weight)
        self.candidate_mask_loss_reduction = (
            "candidate_mean" if self.list_loss_weight == 0.0 else "model_defined"
        )
        self.query_tokenizer = FieldWiseTokenizer(
            input_dims=self.inputs.query_field_dims,
            hidden_dim=int(tokenizer_hidden_dim),
            model_dim=self.model_dim,
            activation=tokenizer_activation,
        )
        self.candidate_tokenizer = FieldWiseTokenizer(
            input_dims=self.inputs.candidate_field_dims,
            hidden_dim=int(tokenizer_hidden_dim),
            model_dim=self.model_dim,
            activation=tokenizer_activation,
        )
        self.history_tokenizer = FeatureTokenizer(
            input_dim=self.inputs.history_event_dim,
            hidden_dim=int(tokenizer_hidden_dim),
            model_dim=self.model_dim,
            activation=tokenizer_activation,
        )
        self.blocks = nn.ModuleList(
            ESTLCABlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                query_token_count=len(self.inputs.query_slots),
                candidate_token_count=len(self.inputs.candidate_slots),
                expansion_factor=int(expansion_factor),
                ffn_activation=ffn_activation,
                dropout=float(dropout),
                norm_eps=float(norm_eps),
                attention_backend=attention_backend,
            )
            for _ in range(num_layers)
        )
        request_input_dim = (
            len(self.inputs.query_slots) * self.model_dim + self.model_dim
        )
        candidate_input_dim = len(self.inputs.candidate_slots) * self.model_dim
        self.heads = nn.ModuleDict(
            {
                name: PartitionedTaskHead(
                    request_input_dim=request_input_dim,
                    candidate_input_dim=candidate_input_dim,
                    hidden_dim=int(head_hidden_dim),
                    activation=head_activation,
                    dropout=float(head_dropout),
                )
                for name in self.task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        return structured_ranking_output(self._forward_slots(embedded))

    def _forward_slots(self, embedded: RecAgoraModelBatch) -> dict[str, torch.Tensor]:
        inputs = self.inputs(embedded)
        query_tokens = self.query_tokenizer(inputs.query_fields)
        candidate_tokens = self.candidate_tokenizer(inputs.candidate_fields)
        history = self.history_tokenizer(inputs.history_events)
        history = history.masked_fill(~inputs.history_valid.unsqueeze(-1), 0.0)
        for block in self.blocks:
            history, query_tokens, candidate_tokens = block(
                history,
                query_tokens,
                candidate_tokens,
                history_valid=inputs.history_valid,
            )
        history_pool = _masked_mean(history, inputs.history_valid)
        request_features = torch.cat(
            (query_tokens.flatten(start_dim=1), history_pool),
            dim=-1,
        )
        candidate_features = candidate_tokens.flatten(start_dim=2)
        return {
            name: head(request_features, candidate_features)
            for name, head in self.heads.items()
        }

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = output["logits"]
        metrics: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.task_names:
            logits = logits_by_task[name]
            labels = batch.labels[name].to(logits.dtype)
            point = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            value = self.point_loss_weight * point
            metrics[f"{name}_point"] = point
            if self.list_loss_weight:
                listwise = _multi_positive_listwise_loss(
                    logits,
                    labels,
                    valid_mask=batch.candidate_valid_mask,
                )
                metrics[f"{name}_list"] = listwise
                value = value + self.list_loss_weight * listwise
            value = value * self.task_weights.get(name, 1.0)
            metrics[f"{name}_total"] = value
            total = value if total is None else total + value
        metrics["total"] = total  # type: ignore[assignment]
        return total, metrics  # type: ignore[return-value]


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(value.dtype).unsqueeze(-1)
    numerator = (value * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _multi_positive_listwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != logits.shape:
            raise ValueError("valid_mask must be bool with the same shape as logits")
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        has_candidate = valid_mask.any(dim=1)
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
        masked_logits = torch.where(
            has_candidate[:, None], masked_logits, torch.zeros_like(masked_logits)
        )
        log_probabilities = F.log_softmax(masked_logits, dim=1)
        log_probabilities = torch.where(
            valid_mask, log_probabilities, torch.zeros_like(log_probabilities)
        )
        positive_count = safe_labels.gt(0).sum(dim=1)
        target = safe_labels / safe_labels.sum(dim=1, keepdim=True).clamp_min(1.0)
        per_request = -(target * log_probabilities).sum(dim=1)
        valid = (positive_count.gt(0) & has_candidate).to(per_request.dtype)
        return (per_request * valid).sum() / valid.sum().clamp_min(1.0)
    positive_count = labels.gt(0).sum(dim=1)
    target = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    per_request = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    valid = positive_count.gt(0).to(per_request.dtype)
    return (per_request * valid).sum() / valid.sum().clamp_min(1.0)

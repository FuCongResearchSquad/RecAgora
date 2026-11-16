from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
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

from .blocks import MixFormerBlock
from .inputs import MixFormerInputs


class ChunkTokenizer(nn.Module):
    """Contiguous feature split followed by an independent linear head projection."""

    def __init__(self, *, input_dim: int, num_tokens: int, model_dim: int) -> None:
        super().__init__()
        validate_positive_int(input_dim, field_name="tokenizer.input_dim")
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.model_dim = int(model_dim)
        self.chunk_dim = int(math.ceil(input_dim / num_tokens))
        self.pad_dim = self.chunk_dim * self.num_tokens - self.input_dim
        self.projections = nn.ModuleList(
            nn.Linear(self.chunk_dim, self.model_dim)
            for _ in range(self.num_tokens)
        )
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.pad_dim:
            features = F.pad(features, (0, self.pad_dim))
        chunks = features.reshape(
            *features.shape[:-1],
            self.num_tokens,
            self.chunk_dim,
        )
        return torch.stack(
            tuple(
                projection(chunks[..., index, :])
                for index, projection in enumerate(self.projections)
            ),
            dim=-2,
        )


class MixFormerModel(RecAgoraRankingModel):
    """Request-shared, user-item decoupled MixFormer-style ranker."""

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
        num_user_heads: int,
        num_item_heads: int,
        num_layers: int,
        expansion_factor: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1.0e-6,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
        point_loss_weight: float = 1.0,
        list_loss_weight: float = 0.0,
    ) -> None:
        num_heads = int(num_user_heads + num_item_heads)
        validate_positive_int(model_dim, field_name="model_dim")
        validate_positive_int(num_user_heads, field_name="num_user_heads")
        validate_positive_int(num_item_heads, field_name="num_item_heads")
        if model_dim % num_heads:
            raise ValueError("MixFormer model_dim must be divisible by total feature heads")
        validate_positive_int(num_layers, field_name="num_layers")
        validate_positive_int(expansion_factor, field_name="expansion_factor")
        validate_dropout(dropout)
        finite_positive_float(rms_norm_eps, "MixFormer RMSNorm epsilon")
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="MixFormer",
        )
        head_hidden_dim = model_dim if head_hidden_dim is None else int(head_hidden_dim)
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
            raise ValueError("MixFormer requires at least one positive loss weight")

        params: dict[str, Any] = {
            "query_slots": tuple(int(slot) for slot in query_slots),
            "candidate_slots": tuple(int(slot) for slot in candidate_slots),
            "history": dict(history),
            "model_dim": int(model_dim),
            "num_user_heads": int(num_user_heads),
            "num_item_heads": int(num_item_heads),
            "num_layers": int(num_layers),
            "expansion_factor": int(expansion_factor),
            "dropout": float(dropout),
            "rms_norm_eps": float(rms_norm_eps),
            "candidate_isolation": bool(candidate_isolation),
            "task_names": task_names,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
            "point_loss_weight": float(point_loss_weight),
            "list_loss_weight": float(list_loss_weight),
        }
        super().__init__(params, input_spec)
        self.inputs = MixFormerInputs(
            input_spec=input_spec,
            query_slots=query_slots,
            candidate_slots=candidate_slots,
            history=history,
        )
        self.model_dim = int(model_dim)
        self.num_user_heads = int(num_user_heads)
        self.num_item_heads = int(num_item_heads)
        self.num_heads = num_heads
        self.task_names = task_names
        self.task_weights = task_weights
        self.point_loss_weight = float(point_loss_weight)
        self.list_loss_weight = float(list_loss_weight)
        self.candidate_mask_loss_reduction = (
            "candidate_mean" if self.list_loss_weight == 0.0 else "model_defined"
        )
        self.user_tokenizer = ChunkTokenizer(
            input_dim=self.inputs.query_dim,
            num_tokens=self.num_user_heads,
            model_dim=self.model_dim,
        )
        self.item_tokenizer = ChunkTokenizer(
            input_dim=self.inputs.candidate_dim,
            num_tokens=self.num_item_heads,
            model_dim=self.model_dim,
        )
        self.sequence_projection = nn.Linear(
            self.inputs.history_event_dim,
            self.model_dim,
        )
        nn.init.xavier_uniform_(self.sequence_projection.weight)
        nn.init.zeros_(self.sequence_projection.bias)
        self.blocks = nn.ModuleList(
            MixFormerBlock(
                model_dim=self.model_dim,
                num_user_heads=self.num_user_heads,
                num_item_heads=self.num_item_heads,
                expansion_factor=int(expansion_factor),
                dropout=float(dropout),
                norm_eps=float(rms_norm_eps),
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.RMSNorm(self.model_dim, eps=float(rms_norm_eps))
        self.heads = nn.ModuleDict(
            {
                name: _build_task_head(
                    input_dim=self.model_dim,
                    hidden_dim=head_hidden_dim,
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
        user_tokens = self.user_tokenizer(inputs.query)
        item_tokens = self.item_tokenizer(inputs.candidates)
        sequence = self.sequence_projection(inputs.history_events)
        sequence = sequence.masked_fill(~inputs.history_valid.unsqueeze(-1), 0.0)
        for block in self.blocks:
            sequence, user_tokens, item_tokens = block(
                sequence,
                user_tokens,
                item_tokens,
                history_valid=inputs.history_valid,
            )
        user_for_candidates = user_tokens[:, None].expand(
            -1,
            embedded.list_len,
            -1,
            -1,
        )
        candidate_states = torch.cat((user_for_candidates, item_tokens), dim=2).mean(
            dim=2
        )
        candidate_states = self.output_norm(candidate_states)
        return {
            name: head(candidate_states).squeeze(-1)
            for name, head in self.heads.items()
        }

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = output["logits"]
        metrics: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.task_names:
            logits = output[name]
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


def _build_task_head(*, input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    for module in head:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return head

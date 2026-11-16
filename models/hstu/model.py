from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from recagora import (
    RecAgoraInputSpec,
    RecAgoraRankingModel,
    RecAgoraSparseBatch,
    RankingOutput,
    masked_binary_cross_entropy_with_logits,
    structured_ranking_output,
)
from models._validation import validate_hstu_attention, validate_mlp_head

from .blocks import HSTUBlock, HSTUTokenPositionTimeEncoder
from .inputs import HSTUInputs
from .mask import build_target_aware_mask


class HSTUStyleModel(RecAgoraRankingModel):
    """Padded DLRM-HSTU-style target-aware ranker over the public lookup boundary."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "model_defined"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        model_dim: int = 256,
        num_heads: int = 4,
        attention_dim: int = 64,
        value_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
        attention_normalization: str = "token_count",
        position_bucket_count: int = 202,
        time_bucket_count: int = 128,
        time_bucket_transform: str = "log1p",
        history_tokenization: str = "content_action_projected_sum",
        position_time_encoding: str = "additive_once",
        candidate_position_mode: str = "shared_target",
        candidate_isolation: bool = True,
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        head_type: str = "mlp",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
        point_loss_weight: float = 1.0,
        list_loss_weight: float = 0.0,
        task_loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        validate_hstu_attention(
            model_dim=model_dim,
            num_heads=num_heads,
            value_dim=value_dim,
            normalization=attention_normalization,
        )
        if history_tokenization != "content_action_projected_sum":
            raise ValueError("unsupported history_tokenization")
        if position_time_encoding != "additive_once":
            raise ValueError("unsupported position_time_encoding")
        if candidate_position_mode != "shared_target":
            raise ValueError("unsupported candidate_position_mode")
        validate_mlp_head(head_type, head_hidden_dim, head_dropout)
        output_label_names = tuple(str(name) for name in output_label_names)
        task_loss_weights = {
            str(name): float(value) for name, value in (task_loss_weights or {}).items()
        }
        params: dict[str, Any] = {
            "history": dict(history),
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "attention_dim": int(attention_dim),
            "value_dim": int(value_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "attention_normalization": attention_normalization,
            "position_bucket_count": int(position_bucket_count),
            "time_bucket_count": int(time_bucket_count),
            "time_bucket_transform": time_bucket_transform,
            "history_tokenization": history_tokenization,
            "position_time_encoding": position_time_encoding,
            "candidate_position_mode": candidate_position_mode,
            "candidate_isolation": bool(candidate_isolation),
            "output_label_names": output_label_names,
            "head_type": head_type,
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "point_loss_weight": float(point_loss_weight),
            "list_loss_weight": float(list_loss_weight),
            "task_loss_weights": task_loss_weights,
        }
        super().__init__(params, input_spec)
        self.inputs = HSTUInputs(input_spec=input_spec, history=history)
        if position_bucket_count < self.inputs.history_length + 2:
            raise ValueError("position buckets must cover context, history, and shared target")
        self.model_dim = int(model_dim)
        self.candidate_isolation = bool(candidate_isolation)
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.head_type = head_type
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.point_loss_weight = float(point_loss_weight)
        self.list_loss_weight = float(list_loss_weight)
        self.candidate_mask_loss_reduction = (
            "candidate_mean" if self.list_loss_weight == 0.0 else "model_defined"
        )
        self.task_loss_weights = task_loss_weights
        self.context_projection = nn.Linear(self.inputs.context_input_dim, model_dim)
        self.history_projection = nn.Linear(self.inputs.history_input_dim, model_dim)
        self.action_projection = nn.Linear(self.inputs.action_input_dim, model_dim)
        self.history_event_norm = nn.LayerNorm(model_dim, eps=1.0e-6)
        self.candidate_projection = nn.Linear(self.inputs.candidate_input_dim, model_dim)
        self.token_type = nn.Embedding(3, model_dim)
        self.position_time_encoder = HSTUTokenPositionTimeEncoder(
            model_dim=model_dim,
            position_bucket_count=position_bucket_count,
            time_bucket_count=time_bucket_count,
            time_bucket_transform=time_bucket_transform,
        )
        block_params = {
            "model_dim": model_dim,
            "num_heads": num_heads,
            "attention_dim": attention_dim,
            "value_dim": value_dim,
            "dropout": dropout,
            "attention_normalization": attention_normalization,
        }
        self.blocks = nn.ModuleList(HSTUBlock(**block_params) for _ in range(num_layers))
        self.output_heads = nn.ModuleDict(
            {
                name: _build_task_head(
                    input_dim=model_dim,
                    head_type=head_type,
                    head_hidden_dim=head_hidden_dim,
                    head_dropout=head_dropout,
                )
                for name in output_label_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        inputs = self.inputs(embedded)
        context = self.context_projection(inputs.context)[:, None, :]
        history = self.history_projection(inputs.history_content)
        action = self.action_projection(inputs.history_action)
        history_events = self.history_event_norm(history + action)
        candidates = self.candidate_projection(inputs.candidates)
        if embedded.candidate_valid_mask is not None:
            candidates = torch.where(
                embedded.candidate_valid_mask[..., None],
                candidates,
                torch.zeros_like(candidates),
            )
        batch_size, history_length, _ = history_events.shape
        tokens = torch.cat((context, history_events, candidates), dim=1)
        token_types = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.long, device=tokens.device),
                torch.ones(
                    (batch_size, history_length), dtype=torch.long, device=tokens.device
                ),
                torch.full(
                    (batch_size, embedded.list_len),
                    2,
                    dtype=torch.long,
                    device=tokens.device,
                ),
            ),
            dim=1,
        )
        tokens = tokens + self.token_type(token_types)
        tokens = self.position_time_encoder(
            tokens,
            history_valid=inputs.history_valid,
            timestamps=inputs.timestamps,
            time_gaps=inputs.time_gaps,
        )
        allowed, valid, candidate_start = build_target_aware_mask(
            inputs.history_valid,
            list_len=embedded.list_len,
            candidate_isolation=self.candidate_isolation,
            candidate_valid=embedded.candidate_valid_mask,
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                allowed=allowed,
                valid=valid,
            )
        candidate_states = tokens[:, candidate_start : candidate_start + embedded.list_len]
        return structured_ranking_output(
            {
                name: head(candidate_states).squeeze(-1)
                for name, head in self.output_heads.items()
            }
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = output["logits"]
        metrics: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.output_label_names:
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
            value = value * self.task_loss_weights.get(name, 1.0)
            metrics[name] = value
            total = value if total is None else total + value
        metrics["total"] = total  # type: ignore[assignment]
        return total, metrics  # type: ignore[return-value]


def _build_task_head(
    *,
    input_dim: int,
    head_type: str,
    head_hidden_dim: int,
    head_dropout: float,
) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(input_dim, 1)
    hidden = nn.Linear(input_dim, head_hidden_dim)
    output = nn.Linear(head_hidden_dim, 1)
    for layer in (hidden, output):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    return nn.Sequential(
        hidden,
        nn.SiLU(),
        nn.Dropout(head_dropout),
        output,
    )


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
        valid = positive_count.gt(0) & has_candidate
        valid_weight = valid.to(per_request.dtype)
        return (per_request * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    positive_count = labels.gt(0).sum(dim=1)
    target = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    per_request = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    valid = positive_count.gt(0)
    valid_weight = valid.to(per_request.dtype)
    return (per_request * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

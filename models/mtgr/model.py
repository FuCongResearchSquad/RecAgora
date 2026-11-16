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
from models._validation import (
    finite_positive_float,
    validate_dropout,
    validate_head_dimensions,
    validate_model_dim_heads,
    validate_positive_int,
    validate_standard_multitask_contract,
)

from .blocks import GatedAttentionBlock
from .inputs import MTGRInputs, MTGRInputsOutput


class MTGRModel(RecAgoraRankingModel):
    """MTGR ranker adapted to request-level search data with offline history."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "model_defined"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        candidate_groups: Mapping[str, Sequence[int]],
        model_dim: int,
        num_heads: int,
        value_dim: int,
        query_key_dim: int,
        num_blocks: int,
        output_hidden_dim: int | None = None,
        group_norm_eps: float = 1.0e-5,
        dropout: float = 0.0,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
        point_loss_weight: float = 1.0,
        list_loss_weight: float = 0.0,
    ) -> None:
        validate_model_dim_heads(model_dim, num_heads)
        validate_positive_int(num_blocks, field_name="num_blocks")
        validate_positive_int(value_dim, field_name="value_dim")
        validate_positive_int(query_key_dim, field_name="query_key_dim")
        if num_heads * value_dim != model_dim or num_heads * query_key_dim != model_dim:
            raise ValueError("MTGR requires num_heads * value/query-key dim == model_dim")
        output_hidden_dim = (
            int(model_dim) if output_hidden_dim is None else int(output_hidden_dim)
        )
        head_hidden_dim = (
            int(model_dim) if head_hidden_dim is None else int(head_hidden_dim)
        )
        validate_positive_int(output_hidden_dim, field_name="output_hidden_dim")
        finite_positive_float(group_norm_eps, "group_norm_eps")
        validate_dropout(dropout)
        validate_head_dimensions(head_hidden_dim, head_dropout)
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="MTGR",
        )
        task_weights = {
            str(name): float(value) for name, value in (task_weights or {}).items()
        }
        params: dict[str, Any] = {
            "history": dict(history),
            "candidate_groups": {
                name: tuple(int(slot) for slot in slots)
                for name, slots in candidate_groups.items()
            },
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "value_dim": int(value_dim),
            "query_key_dim": int(query_key_dim),
            "num_blocks": int(num_blocks),
            "output_hidden_dim": output_hidden_dim,
            "group_norm_eps": float(group_norm_eps),
            "dropout": float(dropout),
            "candidate_isolation": bool(candidate_isolation),
            "task_names": task_names,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
            "point_loss_weight": float(point_loss_weight),
            "list_loss_weight": float(list_loss_weight),
        }
        super().__init__(params, input_spec)
        self.inputs = MTGRInputs(
            input_spec=input_spec,
            history=history,
            candidate_groups=candidate_groups,
        )
        self.model_dim = int(model_dim)
        self.task_names = task_names
        self.task_weights = task_weights
        self.point_loss_weight = float(point_loss_weight)
        self.list_loss_weight = float(list_loss_weight)
        self.candidate_mask_loss_reduction = (
            "candidate_mean" if self.list_loss_weight == 0.0 else "model_defined"
        )
        self.user_projections = nn.ModuleList(
            nn.Linear(input_dim, model_dim)
            for input_dim in self.inputs.user_feature_dims
        )
        for projection in self.user_projections:
            _initialize_linear(projection)
        self.history_tokenizer = _build_token_mlp(
            input_dim=self.inputs.history_event_dim,
            model_dim=model_dim,
        )
        self.candidate_tokenizer = _build_token_mlp(
            input_dim=self.inputs.candidate_dim,
            model_dim=model_dim,
        )
        self.blocks = nn.ModuleList(
            GatedAttentionBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                value_dim=value_dim,
                query_key_dim=query_key_dim,
                num_token_domains=3,
                output_hidden_dim=output_hidden_dim,
                dropout=dropout,
                group_norm_eps=group_norm_eps,
            )
            for _ in range(num_blocks)
        )
        self.heads = nn.ModuleDict(
            {
                name: _build_task_head(
                    input_dim=model_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=float(head_dropout),
                )
                for name in task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        inputs = self.inputs(embedded)
        user_tokens = torch.stack(
            tuple(
                projection(value)
                for projection, value in zip(
                    self.user_projections,
                    inputs.user_features,
                    strict=True,
                )
            ),
            dim=1,
        )
        history_tokens = self.history_tokenizer(inputs.history_events)
        candidate_tokens = self.candidate_tokenizer(inputs.candidates)
        tokens = torch.cat((user_tokens, history_tokens, candidate_tokens), dim=1)
        allowed = self._build_mask(
            inputs,
            candidate_count=embedded.list_len,
            candidate_valid=embedded.candidate_valid_mask,
        )
        valid, domain_ids = self._build_token_metadata(
            inputs,
            candidate_count=embedded.list_len,
            candidate_valid=embedded.candidate_valid_mask,
        )
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        for block in self.blocks:
            tokens = block(
                tokens,
                allowed=allowed,
                domain_ids=domain_ids,
                valid=valid,
            )
        candidate_states = tokens[
            :, len(self.inputs.user_slots) + self.inputs.history_length :
        ]
        return structured_ranking_output(
            {
                name: head(candidate_states).squeeze(-1)
                for name, head in self.heads.items()
            }
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = output["logits"]
        components: dict[str, torch.Tensor] = {}
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
            components[f"{name}_point"] = point
            if self.list_loss_weight:
                listwise = _multi_positive_listwise_loss(
                    logits,
                    labels,
                    valid_mask=batch.candidate_valid_mask,
                )
                components[f"{name}_list"] = listwise
                value = value + self.list_loss_weight * listwise
            value = value * self.task_weights.get(name, 1.0)
            components[f"{name}_total"] = value
            total = value if total is None else total + value
        components["total"] = total  # type: ignore[assignment]
        return total, components  # type: ignore[return-value]

    def _build_mask(
        self,
        inputs: MTGRInputsOutput,
        *,
        candidate_count: int,
        candidate_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, history_length = inputs.history_valid.shape
        user_count = len(self.inputs.user_slots)
        static_count = user_count + history_length
        token_count = static_count + candidate_count
        allowed = torch.zeros(
            (batch_size, token_count, token_count),
            dtype=torch.bool,
            device=inputs.history_events.device,
        )
        static_valid = torch.cat(
            (
                torch.ones(
                    (batch_size, user_count),
                    dtype=torch.bool,
                    device=allowed.device,
                ),
                inputs.history_valid,
            ),
            dim=1,
        )
        allowed[:, :static_count, :static_count] = (
            static_valid[:, :, None] & static_valid[:, None, :]
        )
        if candidate_valid is None:
            candidate_valid = torch.ones(
                (batch_size, candidate_count), dtype=torch.bool, device=allowed.device
            )
        allowed[:, static_count:, :static_count] = (
            candidate_valid[:, :, None] & static_valid[:, None, :]
        )
        allowed[:, static_count:, static_count:] = (
            candidate_valid[:, :, None]
            & torch.eye(
                candidate_count, dtype=torch.bool, device=allowed.device
            )[None]
        )
        return allowed

    def _build_token_metadata(
        self,
        inputs: MTGRInputsOutput,
        *,
        candidate_count: int,
        candidate_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_length = inputs.history_valid.shape
        device = inputs.history_valid.device
        user_count = len(self.inputs.user_slots)
        valid = torch.cat(
            (
                torch.ones(
                    (batch_size, user_count),
                    dtype=torch.bool,
                    device=device,
                ),
                inputs.history_valid,
                candidate_valid
                if candidate_valid is not None
                else torch.ones(
                    (batch_size, candidate_count), dtype=torch.bool, device=device
                ),
            ),
            dim=1,
        )
        domain_ids = torch.cat(
            (
                torch.zeros(
                    (batch_size, user_count),
                    dtype=torch.long,
                    device=device,
                ),
                torch.ones(
                    (batch_size, history_length),
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (batch_size, candidate_count),
                    2,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=1,
        )
        return valid, domain_ids


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
        count = safe_labels.gt(0).sum(dim=1)
        target = safe_labels / safe_labels.sum(dim=1, keepdim=True).clamp_min(1.0)
        per_request = -(target * log_probabilities).sum(dim=1)
        valid = count.gt(0) & has_candidate
        return per_request[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
    count = labels.gt(0).sum(dim=1)
    target = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    per_request = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    valid = count.gt(0)
    return per_request[valid].mean() if bool(valid.any()) else logits.sum() * 0.0


def _initialize_linear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _build_token_mlp(*, input_dim: int, model_dim: int) -> nn.Sequential:
    """Non-linear tokenization chosen for the paper's unspecified token MLP."""
    tokenizer = nn.Sequential(
        nn.Linear(input_dim, model_dim),
        nn.SiLU(),
        nn.Linear(model_dim, model_dim),
    )
    for module in tokenizer:
        if isinstance(module, nn.Linear):
            _initialize_linear(module)
    return tokenizer


def _build_task_head(*, input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Build the paper-aligned per-task MLP readout over candidate states."""
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    for module in head:
        if isinstance(module, nn.Linear):
            _initialize_linear(module)
    return head


# Public compatibility for the existing built-in registry.  Formal MTGR configs
# use the explicit ``models.mtgr.model:MTGRModel`` class path and model.name=mtgr.
MTGRStyleModel = MTGRModel

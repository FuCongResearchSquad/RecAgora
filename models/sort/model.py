from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from recagora import (
    RecAgoraInputSpec,
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
    validate_model_dim_heads,
    validate_positive_int,
    validate_standard_multitask_contract,
)

from .blocks import SORTBlock
from .inputs import SORTInputs


class SORTModel(RecAgoraRankingModel):
    """SORT architecture-core adaptation for request-level ranking."""

    fixed_non_candidate_tokens = 5  # BOS, SEP, two profile tokens, SEP.
    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "preserve_objectives"
    candidate_mask_loss_reduction = "model_defined"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        query_groups: Mapping[str, Sequence[int]],
        candidate_slots: Sequence[int],
        model_dim: int,
        num_layers: int,
        num_heads: int,
        local_window: int,
        history_keep_per_layer: Sequence[int],
        expert_hidden_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        routed_top_k: int,
        dropout: float = 0.0,
        norm_eps: float = 1.0e-6,
        qk_norm_eps: float = 1.0e-6,
        rope_base: float = 10_000.0,
        router_aux_loss_weight: float = 0.01,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        validate_model_dim_heads(model_dim, num_heads)
        for value, name in (
            (num_layers, "num_layers"),
            (local_window, "local_window"),
            (expert_hidden_dim, "expert_hidden_dim"),
            (num_shared_experts, "num_shared_experts"),
            (num_routed_experts, "num_routed_experts"),
            (routed_top_k, "routed_top_k"),
        ):
            validate_positive_int(value, field_name=name)
        if routed_top_k > num_routed_experts:
            raise ValueError("routed_top_k must not exceed num_routed_experts")
        validate_dropout(dropout)
        validate_head_dimensions(head_hidden_dim, head_dropout)
        finite_positive_float(norm_eps, "norm_eps")
        finite_positive_float(qk_norm_eps, "qk_norm_eps")
        finite_positive_float(rope_base, "rope_base")
        finite_nonnegative_float(router_aux_loss_weight, "router_aux_loss_weight")
        keeps = tuple(int(value) for value in history_keep_per_layer)
        if len(keeps) != int(num_layers):
            raise ValueError("history_keep_per_layer must contain one value per layer")
        if any(value < 0 for value in keeps) or any(
            left < right for left, right in zip(keeps, keeps[1:])
        ):
            raise ValueError(
                "history_keep_per_layer must be non-negative and non-increasing"
            )
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="SORT",
        )
        task_weights = {
            str(name): float(weight) for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")

        params: dict[str, Any] = {
            "history": dict(history),
            "query_groups": {
                str(name): tuple(int(slot_id) for slot_id in slots)
                for name, slots in query_groups.items()
            },
            "candidate_slots": tuple(int(slot_id) for slot_id in candidate_slots),
            "model_dim": int(model_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "local_window": int(local_window),
            "history_keep_per_layer": keeps,
            "expert_hidden_dim": int(expert_hidden_dim),
            "num_shared_experts": int(num_shared_experts),
            "num_routed_experts": int(num_routed_experts),
            "routed_top_k": int(routed_top_k),
            "dropout": float(dropout),
            "norm_eps": float(norm_eps),
            "qk_norm_eps": float(qk_norm_eps),
            "rope_base": float(rope_base),
            "router_aux_loss_weight": float(router_aux_loss_weight),
            "candidate_isolation": bool(candidate_isolation),
            "task_names": task_names,
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
        }
        super().__init__(params, input_spec)
        self.model_dim = int(model_dim)
        self.history_keep_per_layer = keeps
        self.router_aux_loss_weight = float(router_aux_loss_weight)
        self.task_names = task_names
        self.task_weights = task_weights
        self.inputs = SORTInputs(
            input_spec=input_spec,
            history=history,
            query_groups=query_groups,
            candidate_slots=candidate_slots,
        )
        if keeps[0] > self.inputs.history_length:
            raise ValueError("history_keep_per_layer exceeds configured history length")
        self.retained_non_candidate_counts = tuple(
            keep + self.fixed_non_candidate_tokens for keep in keeps
        )

        self.history_content_projection = nn.Linear(
            self.inputs.history_content_dim,
            self.model_dim,
        )
        self.history_action_projection = nn.Linear(
            self.inputs.history_action_dim,
            self.model_dim,
        )
        self.history_time_projection = nn.Linear(2, self.model_dim)
        self.history_norm = nn.RMSNorm(self.model_dim, eps=norm_eps)
        self.query_projections = nn.ModuleList(
            nn.Linear(input_dim, self.model_dim)
            for input_dim in self.inputs.query_group_dims
        )
        self.candidate_projection = nn.Linear(
            self.inputs.candidate_dim,
            self.model_dim,
        )
        self.bos_token = nn.Parameter(torch.empty(self.model_dim))
        self.separator_token = nn.Parameter(torch.empty(self.model_dim))
        self.token_type = nn.Embedding(5, self.model_dim)
        self.blocks = nn.ModuleList(
            SORTBlock(
                model_dim=self.model_dim,
                num_heads=int(num_heads),
                local_window=int(local_window),
                expert_hidden_dim=int(expert_hidden_dim),
                num_shared_experts=int(num_shared_experts),
                num_routed_experts=int(num_routed_experts),
                routed_top_k=int(routed_top_k),
                dropout=float(dropout),
                norm_eps=float(norm_eps),
                qk_norm_eps=float(qk_norm_eps),
                rope_base=float(rope_base),
            )
            for _ in range(int(num_layers))
        )
        self.final_norm = nn.RMSNorm(self.model_dim, eps=norm_eps)
        self.heads = nn.ModuleDict(
            {
                name: _TaskTower(
                    input_dim=self.model_dim,
                    hidden_dim=int(head_hidden_dim),
                    dropout=float(head_dropout),
                )
                for name in self.task_names
            }
        )
        self._reset_input_parameters()

    def _reset_input_parameters(self) -> None:
        projections = (
            self.history_content_projection,
            self.history_action_projection,
            self.history_time_projection,
            self.candidate_projection,
            *self.query_projections,
        )
        for layer in projections:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.bos_token, std=self.model_dim**-0.5)
        nn.init.normal_(self.separator_token, std=self.model_dim**-0.5)
        nn.init.normal_(self.token_type.weight, std=self.model_dim**-0.5)

    def _tokenize(
        self,
        embedded,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.inputs(embedded)
        batch_size = embedded.batch_size
        history = self.history_norm(
            self.history_content_projection(inputs.history_content)
            + self.history_action_projection(inputs.history_action)
            + self.history_time_projection(inputs.history_time)
        )
        history = history + self.token_type.weight[1]
        query = torch.stack(
            [
                projection(values)
                for projection, values in zip(
                    self.query_projections,
                    inputs.query_tokens,
                    strict=True,
                )
            ],
            dim=1,
        ) + self.token_type.weight[3]
        candidates = self.candidate_projection(inputs.candidates) + self.token_type.weight[4]
        bos = (self.bos_token + self.token_type.weight[0]).reshape(1, 1, -1).expand(
            batch_size, -1, -1
        )
        separator = (
            self.separator_token + self.token_type.weight[2]
        ).reshape(1, 1, -1).expand(batch_size, -1, -1)
        tokens = torch.cat(
            (bos, history, separator, query, separator, candidates),
            dim=1,
        )
        fixed_valid = torch.ones(
            (batch_size, self.fixed_non_candidate_tokens),
            dtype=torch.bool,
            device=tokens.device,
        )
        candidate_valid = (
            torch.ones(
                (batch_size, embedded.list_len),
                dtype=torch.bool,
                device=tokens.device,
            )
            if embedded.candidate_valid_mask is None
            else embedded.candidate_valid_mask
        )
        valid = torch.cat(
            (
                fixed_valid[:, :1],
                inputs.history_valid,
                fixed_valid[:, 1:],
                candidate_valid,
            ),
            dim=1,
        )
        history_positions = torch.arange(
            1,
            self.inputs.history_length + 1,
            dtype=torch.long,
            device=tokens.device,
        )
        post_history_positions = torch.arange(
            self.inputs.history_length + 1,
            self.inputs.history_length + self.fixed_non_candidate_tokens,
            dtype=torch.long,
            device=tokens.device,
        )
        candidate_position = torch.full(
            (embedded.list_len,),
            self.inputs.history_length + self.fixed_non_candidate_tokens,
            dtype=torch.long,
            device=tokens.device,
        )
        position_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                history_positions,
                post_history_positions,
                candidate_position,
            )
        )
        if embedded.candidate_valid_mask is None:
            # Preserve the legacy arithmetic path exactly when masking is disabled.
            tokens = tokens * valid[..., None].to(tokens.dtype)
        else:
            tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        return tokens, valid, position_ids

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        tokens, valid, position_ids = self._tokenize(embedded)
        auxiliary_losses: list[torch.Tensor] = []
        expert_loads: list[torch.Tensor] = []
        mean_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        history_count = self.inputs.history_length
        for keep, block in zip(
            self.history_keep_per_layer,
            self.blocks,
            strict=True,
        ):
            tokens, valid, position_ids = _retain_recent_history(
                tokens,
                valid,
                position_ids,
                history_count=history_count,
                retained_history_count=keep,
            )
            history_count = keep
            routed = block(
                tokens,
                valid=valid,
                position_ids=position_ids,
                candidate_count=embedded.list_len,
            )
            tokens = routed.values
            auxiliary_losses.append(routed.auxiliary_loss)
            expert_loads.append(routed.expert_load)
            mean_probabilities.append(routed.mean_probability)
            entropies.append(routed.entropy)
        candidate_states = self.final_norm(tokens[:, -embedded.list_len :])
        logits = {
            name: head(candidate_states).squeeze(-1)
            for name, head in self.heads.items()
        }
        return structured_ranking_output(
            logits,
            aux={
                "router_aux_loss": torch.stack(auxiliary_losses).mean(),
                "routing_metrics": {
                    "expert_load_by_layer": torch.stack(expert_loads),
                    "mean_probability_by_layer": torch.stack(mean_probabilities),
                    "entropy_by_layer": torch.stack(entropies),
                },
            },
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name in self.task_names:
            logits = output["logits"][name]
            labels = batch.labels[name].to(device=logits.device, dtype=logits.dtype)
            value = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = value
            weighted.append(value * float(self.task_weights.get(name, 1.0)))
        router_auxiliary = output["aux"]["router_aux_loss"]
        weighted_router_auxiliary = router_auxiliary * self.router_aux_loss_weight
        losses["moe_load_balance"] = weighted_router_auxiliary
        weighted.append(weighted_router_auxiliary)
        total = torch.stack(weighted).sum()
        losses["total"] = total
        return total, losses


class _TaskTower(nn.Sequential):
    def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for layer in (self[0], self[3]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def _retain_recent_history(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    history_count: int,
    retained_history_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if retained_history_count > history_count:
        raise ValueError("SORT retained history count must not increase")
    if retained_history_count == history_count:
        return tokens, valid, position_ids
    first_history = 1
    retained_start = first_history + history_count - retained_history_count
    after_history = first_history + history_count
    indices = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=tokens.device),
            torch.arange(
                retained_start,
                after_history,
                dtype=torch.long,
                device=tokens.device,
            ),
            torch.arange(
                after_history,
                tokens.shape[1],
                dtype=torch.long,
                device=tokens.device,
            ),
        )
    )
    return (
        tokens.index_select(1, indices),
        valid.index_select(1, indices),
        position_ids.index_select(0, indices),
    )

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
    validate_known_weights,
    validate_positive_int_sequence,
)

from .blocks import SSRDynamicBlock
from .inputs import SSRDInputAdapter


IMPLEMENTATION_SCOPE = "ssr_d_faithful_core_gr_adapted"
SSR_VARIANT = "dynamic"


class SSRDModel(RecAgoraRankingModel):
    """SSR-D faithful core adapted to RecAgora's public Common157 contract.

    This is not an official full-experiment reproduction. The embedding lookup,
    request-list data path, candidate-conditioned history adapter and multitask
    heads are RecAgora adaptations; the stacked dynamic ICS/view core follows the
    SSR-D paper structure with independent filter and view dimensions.
    """

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str = "history_group_1",
        history_attention_dim: int = 64,
        history_padding_side: str = "right",
        num_views: int = 4,
        filter_dim: int = 192,
        view_dim: int = 96,
        num_layers: int = 2,
        ics_iterations: int = 5,
        alpha_init: float = 0.0,
        scale_init: float = 1.0,
        layer_norm_eps: float = 1.0e-6,
        head_hidden_dim: int = 96,
        head_dropout: float = 0.0,
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        validate_positive_int_sequence(
            (
                int(history_attention_dim),
                int(num_views),
                int(filter_dim),
                int(view_dim),
                int(num_layers),
                int(ics_iterations),
                int(head_hidden_dim),
            ),
            field_name="SSR-D dimensions, layers and iterations",
        )
        validate_dropout(head_dropout, field_name="head_dropout")
        finite_positive_float(layer_norm_eps, "layer_norm_eps")

        output_label_names = tuple(str(name) for name in output_label_names)
        if output_label_names != tuple(input_spec.task_names):
            raise ValueError("SSR-D output heads must follow the compiled task order")
        configured_weights = {
            str(name): float(value) for name, value in (loss_weights or {}).items()
        }
        validate_known_weights(configured_weights, output_label_names)
        for name, weight in configured_weights.items():
            finite_nonnegative_float(weight, f"loss_weights.{name}")

        params: dict[str, Any] = {
            "history_group_name": history_group_name,
            "history_attention_dim": int(history_attention_dim),
            "history_padding_side": history_padding_side,
            "num_views": int(num_views),
            "filter_dim": int(filter_dim),
            "view_dim": int(view_dim),
            "num_layers": int(num_layers),
            "ics_iterations": int(ics_iterations),
            "alpha_init": float(alpha_init),
            "scale_init": float(scale_init),
            "layer_norm_eps": float(layer_norm_eps),
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "output_label_names": output_label_names,
            "loss_weights": configured_weights,
        }
        super().__init__(params, input_spec)
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = configured_weights
        self.input_adapter = SSRDInputAdapter(
            input_spec=input_spec,
            history_group_name=history_group_name,
            history_attention_dim=int(history_attention_dim),
            history_padding_side=history_padding_side,
        )

        blocks = []
        block_input_dim = self.input_adapter.output_dim
        for _ in range(int(num_layers)):
            blocks.append(
                SSRDynamicBlock(
                    input_dim=block_input_dim,
                    num_views=int(num_views),
                    filter_dim=int(filter_dim),
                    view_dim=int(view_dim),
                    iterations=int(ics_iterations),
                    alpha_init=float(alpha_init),
                    scale_init=float(scale_init),
                    layer_norm_eps=float(layer_norm_eps),
                )
            )
            block_input_dim = int(num_views) * int(view_dim)
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleDict(
            {
                name: _TaskHead(
                    input_dim=int(view_dim),
                    hidden_dim=int(head_hidden_dim),
                    dropout=float(head_dropout),
                )
                for name in output_label_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        pointwise = embedded.flatten_candidates()
        hidden = self.input_adapter(embedded, pointwise)
        for index, block in enumerate(self.blocks):
            views = block(hidden)
            hidden = views.mean(dim=1) if index == len(self.blocks) - 1 else views.flatten(1)
        return structured_ranking_output(
            {
                name: head(hidden).reshape(embedded.batch_size, embedded.list_len)
                for name, head in self.heads.items()
            }
        )

    def compute_loss(
        self,
        model_output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model_output = model_output["logits"]
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.output_label_names:
            logits = model_output[name]
            labels = batch.labels[name].to(logits.dtype)
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted = loss * self.loss_weights.get(name, 1.0)
            total = weighted if total is None else total + weighted
        losses["total"] = total  # type: ignore[assignment]
        return total, losses  # type: ignore[return-value]


class _TaskHead(nn.Sequential):
    def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float) -> None:
        hidden = nn.Linear(input_dim, hidden_dim)
        output = nn.Linear(hidden_dim, 1)
        for layer in (hidden, output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        super().__init__(
            hidden,
            nn.SiLU(),
            nn.Dropout(dropout),
            output,
        )

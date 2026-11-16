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
    normalize_output_label_names,
    validate_dropout,
    validate_known_weights,
    validate_standard_multitask_contract,
)

from .blocks import TokenMixerBackbone, parse_pffn_config
from .inputs import TokenMixerInputs


class TokenMixerModel(RecAgoraRankingModel):
    """Dense RecAgora TokenMixer baseline over request-level ranking inputs."""

    # Every candidate is an independent row through token mixing and optional
    # S-P MoE routing; the model has no candidate-level auxiliary objective.
    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        token_dim: int,
        num_layers: int,
        expansion_ratio: int,
        token_order: Sequence[str],
        input_groups: Mapping[str, Sequence[int]],
        query_group_name: str,
        history_group_name: str,
        history_token_name: str,
        dropout: float,
        rms_norm_eps: float,
        activation_checkpointing: bool,
        candidate_isolation: bool,
        head_hidden_dim: int,
        head_dropout: float,
        output_label_names: Sequence[str],
        loss_weights: Mapping[str, float] | None = None,
        pffn: Mapping[str, Any] | None = None,
    ) -> None:
        token_dim = _positive_int(token_dim, "token_dim")
        num_layers = _positive_int(num_layers, "num_layers")
        expansion_ratio = _positive_int(expansion_ratio, "expansion_ratio")
        head_hidden_dim = _positive_int(head_hidden_dim, "head_hidden_dim")
        validate_dropout(dropout)
        validate_dropout(head_dropout, field_name="head_dropout")
        rms_norm_eps = finite_positive_float(rms_norm_eps, "rms_norm_eps")
        activation_checkpointing = _boolean(
            activation_checkpointing,
            "activation_checkpointing",
        )
        candidate_isolation = _boolean(
            candidate_isolation,
            "candidate_isolation",
        )
        output_label_names = normalize_output_label_names(output_label_names)
        output_label_names = validate_standard_multitask_contract(
            input_spec,
            output_label_names,
            candidate_isolation=candidate_isolation,
            context="TokenMixer",
        )
        configured_weights = {
            str(name): finite_nonnegative_float(
                weight,
                f"loss_weights.{name}",
            )
            for name, weight in (loss_weights or {}).items()
        }
        validate_known_weights(configured_weights, output_label_names)
        if configured_weights and not any(
            configured_weights.get(name, 1.0) > 0.0
            for name in output_label_names
        ):
            raise ValueError("at least one TokenMixer task weight must be positive")
        pffn_settings = parse_pffn_config(
            pffn,
            dense_hidden_dim=token_dim * expansion_ratio,
        )

        params: dict[str, Any] = {
            "token_dim": token_dim,
            "num_layers": num_layers,
            "expansion_ratio": expansion_ratio,
            "token_order": tuple(str(name) for name in token_order),
            "input_groups": {
                str(name): tuple(int(slot_id) for slot_id in slot_ids)
                for name, slot_ids in input_groups.items()
            },
            "query_group_name": str(query_group_name),
            "history_group_name": str(history_group_name),
            "history_token_name": str(history_token_name),
            "dropout": float(dropout),
            "rms_norm_eps": rms_norm_eps,
            "activation_checkpointing": activation_checkpointing,
            "candidate_isolation": candidate_isolation,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": float(head_dropout),
            "output_label_names": output_label_names,
            "loss_weights": configured_weights,
        }
        if pffn is not None:
            params["pffn"] = dict(pffn)
        super().__init__(params, input_spec)
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = configured_weights
        self.pffn_settings = pffn_settings
        self.candidate_isolation = candidate_isolation
        self.inputs = TokenMixerInputs(
            input_spec=input_spec,
            token_dim=token_dim,
            token_order=token_order,
            input_groups=input_groups,
            query_group_name=query_group_name,
            history_group_name=history_group_name,
            history_token_name=history_token_name,
            rms_norm_eps=rms_norm_eps,
        )
        self.backbone = TokenMixerBackbone(
            input_dim=token_dim,
            num_tokens=self.inputs.total_token_count,
            num_layers=num_layers,
            expansion_ratio=expansion_ratio,
            dropout=float(dropout),
            rms_norm_eps=rms_norm_eps,
            activation_checkpointing=activation_checkpointing,
            pffn_settings=pffn_settings,
        )
        self.output_heads = nn.ModuleDict(
            {
                name: TokenMixerTaskHead(
                    input_dim=self.backbone.output_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=float(head_dropout),
                )
                for name in output_label_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        tokens = self.inputs(embedded)
        tokens = self.backbone(tokens)
        representation = tokens.mean(dim=1)
        return structured_ranking_output(
            {
                name: head(representation).squeeze(-1).reshape(
                    embedded.batch_size,
                    embedded.list_len,
                )
                for name, head in self.output_heads.items()
            }
        )

    def compute_loss(
        self,
        model_output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = model_output["logits"]
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name in self.output_label_names:
            logits = logits_by_task[name]
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                batch.labels[name].to(device=logits.device, dtype=logits.dtype),
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted.append(loss * float(self.loss_weights.get(name, 1.0)))
        total = torch.stack(weighted).sum()
        losses["total"] = total
        return total, losses


class TokenMixerTaskHead(nn.Sequential):
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


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value

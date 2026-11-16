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

from .blocks import HiFormerCompositeBackbone
from .inputs import HiFormerFieldInputs


class HiFormerModel(RecAgoraRankingModel):
    """HiFormer composite feature-interaction core over selected RecAgora fields."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str,
        history_reduction: str,
        action: Mapping[str, object] | None = None,
        tokenization: Mapping[str, object],
        interaction: Mapping[str, object],
        candidate_isolation: bool,
        candidate_chunk_size: int,
        task_names: Sequence[str],
        tower_hidden_units: Sequence[int],
        tower_activation: str,
        tower_dropout: float,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        history_group_name = str(history_group_name)
        if history_reduction != "segment_mean":
            raise ValueError("HiFormer history_reduction must be segment_mean")
        if type(candidate_isolation) is not bool or not candidate_isolation:
            raise ValueError("HiFormer requires candidate isolation")
        validate_positive_int(
            candidate_chunk_size,
            field_name="candidate_chunk_size",
        )

        token_config = _parse_tokenization(tokenization)
        action_config = _parse_action(action)
        if (
            action_config is not None
            and int(action_config["output_dim"]) != int(token_config["input_dim"])
        ):
            raise ValueError(
                "HiFormer action.output_dim must match tokenization.input_dim"
            )
        interaction_config = _parse_interaction(interaction)
        token_dim = int(token_config["token_dim"])
        if token_dim % int(interaction_config["num_heads"]):
            raise ValueError("token_dim must be divisible by interaction.num_heads")

        normalized_tasks = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="HiFormer",
        )
        hidden_units = tuple(int(value) for value in tower_hidden_units)
        validate_positive_int_sequence(
            hidden_units,
            field_name="tower_hidden_units",
        )
        if tower_activation != "silu":
            raise ValueError("HiFormer tower_activation must be silu")
        validate_dropout(tower_dropout, field_name="tower_dropout")
        configured_weights = {
            str(name): finite_nonnegative_float(
                weight,
                f"task_weights.{name}",
            )
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
        if not any(weight > 0.0 for weight in effective_weights.values()):
            raise ValueError("HiFormer requires at least one positive task weight")

        params: dict[str, Any] = {
            "history_group_name": history_group_name,
            "history_reduction": history_reduction,
            "action": action_config,
            "tokenization": token_config,
            "interaction": interaction_config,
            "candidate_isolation": candidate_isolation,
            "candidate_chunk_size": int(candidate_chunk_size),
            "task_names": normalized_tasks,
            "tower_hidden_units": hidden_units,
            "tower_activation": tower_activation,
            "tower_dropout": float(tower_dropout),
            "task_weights": effective_weights,
        }
        super().__init__(params, input_spec)
        self.task_names = normalized_tasks
        self.task_weights = effective_weights
        self.candidate_isolation = candidate_isolation
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.token_dim = token_dim
        self.inputs = HiFormerFieldInputs(
            input_spec=input_spec,
            history_group_name=history_group_name,
            input_dim=int(token_config["input_dim"]),
            token_dim=self.token_dim,
            norm_eps=float(token_config["norm_eps"]),
            action=action_config,
        )
        self.num_task_tokens = len(self.task_names)
        self.num_tokens = self.inputs.num_feature_tokens + self.num_task_tokens
        self.task_tokens = nn.Parameter(
            torch.empty(self.num_task_tokens, self.token_dim)
        )
        nn.init.xavier_uniform_(self.task_tokens)
        self.backbone = HiFormerCompositeBackbone(
            num_tokens=self.num_tokens,
            token_dim=self.token_dim,
            num_layers=int(interaction_config["num_layers"]),
            num_heads=int(interaction_config["num_heads"]),
            qkv_rank=int(interaction_config["qkv_rank"]),
            expansion_factor=int(interaction_config["expansion_factor"]),
            attention_dropout=float(interaction_config["attention_dropout"]),
            ffn_dropout=float(interaction_config["ffn_dropout"]),
            norm_eps=float(interaction_config["norm_eps"]),
            activation_checkpointing=bool(
                interaction_config["activation_checkpointing"]
            ),
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
        query_tokens, history_tokens = self.inputs.request_shared_tokens(embedded)
        chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.task_names
        }
        for start in range(0, embedded.list_len, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, embedded.list_len)
            candidate_tokens = self.inputs.candidate_tokens(
                embedded,
                start=start,
                end=end,
            )
            chunk_logits = self._score_candidate_chunk(
                query_tokens,
                candidate_tokens,
                history_tokens,
            )
            for name, logits in zip(self.task_names, chunk_logits, strict=True):
                chunks[name].append(logits)
        return {
            name: torch.cat(chunks[name], dim=1)
            for name in self.task_names
        }

    def _score_candidate_chunk(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, candidate_count = candidate_tokens.shape[:2]
        query = query_tokens[:, None].expand(-1, candidate_count, -1, -1)
        history = history_tokens[:, None].expand(-1, candidate_count, -1, -1)
        tasks = self.task_tokens[None, None].expand(
            batch_size,
            candidate_count,
            -1,
            -1,
        )
        tokens = torch.cat((query, candidate_tokens, history, tasks), dim=2)
        tokens = tokens.reshape(
            batch_size * candidate_count,
            self.num_tokens,
            self.token_dim,
        )
        encoded = self.backbone(tokens)
        task_states = encoded[:, -self.num_task_tokens :].reshape(
            batch_size,
            candidate_count,
            self.num_task_tokens,
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
        logits_by_task = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name in self.task_names:
            logits = logits_by_task[name]
            labels = batch.labels[name].to(
                device=logits.device,
                dtype=logits.dtype,
            )
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted.append(loss * float(self.task_weights[name]))
        total = torch.stack(weighted).sum()
        losses["total"] = total
        return total, losses


def _parse_tokenization(config: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "mode",
        "input_dim",
        "token_dim",
        "activation",
        "norm",
        "norm_eps",
    }
    _require_exact_keys(config, expected, context="tokenization")
    if config["mode"] != "fieldwise":
        raise ValueError("HiFormer tokenization.mode must be fieldwise")
    if config["activation"] != "silu":
        raise ValueError("HiFormer tokenization.activation must be silu")
    if config["norm"] != "fieldwise_layer_norm":
        raise ValueError(
            "HiFormer tokenization.norm must be fieldwise_layer_norm"
        )
    input_dim = int(config["input_dim"])
    token_dim = int(config["token_dim"])
    validate_positive_int(input_dim, field_name="tokenization.input_dim")
    validate_positive_int(token_dim, field_name="tokenization.token_dim")
    return {
        "mode": "fieldwise",
        "input_dim": input_dim,
        "token_dim": token_dim,
        "activation": "silu",
        "norm": "fieldwise_layer_norm",
        "norm_eps": finite_positive_float(
            config["norm_eps"],
            "tokenization.norm_eps",
        ),
    }


def _parse_action(
    config: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if config is None:
        return None
    expected = {
        "slot_id",
        "adapter",
        "input_dim",
        "output_dim",
        "reduction",
        "bias",
    }
    _require_exact_keys(config, expected, context="action")
    for name in ("slot_id", "input_dim", "output_dim"):
        if type(config[name]) is not int:
            raise TypeError(f"HiFormer action.{name} must be an integer")
    slot_id = int(config["slot_id"])
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    validate_positive_int(slot_id, field_name="action.slot_id")
    validate_positive_int(input_dim, field_name="action.input_dim")
    validate_positive_int(output_dim, field_name="action.output_dim")
    if config["adapter"] != "linear":
        raise ValueError("HiFormer action.adapter must be linear")
    if config["reduction"] != "segment_mean":
        raise ValueError("HiFormer action.reduction must be segment_mean")
    if type(config["bias"]) is not bool:
        raise TypeError("HiFormer action.bias must be boolean")
    return {
        "slot_id": slot_id,
        "adapter": "linear",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "reduction": "segment_mean",
        "bias": config["bias"],
    }


def _parse_interaction(config: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "type",
        "num_layers",
        "num_heads",
        "qkv_rank",
        "expansion_factor",
        "attention_dropout",
        "ffn_dropout",
        "norm_eps",
        "attention_output_projection",
        "last_layer_task_pruning",
        "activation_checkpointing",
    }
    _require_exact_keys(config, expected, context="interaction")
    if config["type"] != "flattened_low_rank_composite_attention":
        raise ValueError("unsupported HiFormer interaction.type")
    if config["attention_output_projection"] is not False:
        raise ValueError("HiFormer does not implement attention output projection")
    if config["last_layer_task_pruning"] is not False:
        raise ValueError("HiFormer does not implement last-layer task pruning")
    if type(config["activation_checkpointing"]) is not bool:
        raise TypeError("interaction.activation_checkpointing must be boolean")
    normalized: dict[str, object] = {
        "type": "flattened_low_rank_composite_attention",
        "attention_output_projection": False,
        "last_layer_task_pruning": False,
        "activation_checkpointing": config["activation_checkpointing"],
    }
    for name in ("num_layers", "num_heads", "qkv_rank", "expansion_factor"):
        value = int(config[name])
        validate_positive_int(value, field_name=f"interaction.{name}")
        normalized[name] = value
    for name in ("attention_dropout", "ffn_dropout"):
        validate_dropout(float(config[name]), field_name=f"interaction.{name}")
        normalized[name] = float(config[name])
    normalized["norm_eps"] = finite_positive_float(
        config["norm_eps"],
        "interaction.norm_eps",
    )
    return normalized


def _require_exact_keys(
    config: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = {str(name) for name in config}
    if actual != expected:
        raise ValueError(
            f"HiFormer {context} keys mismatch: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _build_task_tower(
    *,
    input_dim: int,
    hidden_units: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_units:
        linear = nn.Linear(previous, hidden)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.SiLU(), nn.Dropout(dropout)))
        previous = hidden
    output = nn.Linear(previous, 1)
    nn.init.xavier_uniform_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)

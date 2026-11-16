from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraRankingModel, RecAgoraSparseBatch
from recagora import RankingOutput, structured_ranking_output
from recagora import masked_binary_cross_entropy_with_logits
from models._validation import (
    finite_nonnegative_float,
    normalize_output_label_names,
    validate_head_dimensions,
    validate_known_weights,
    validate_model_dim_heads,
)

from .blocks import OneTransBlock
from .inputs import OneTransInputs, OneTransInputsOutput


class TaskTokenReadout(nn.Module):
    """Project the non-sequence token axis to one representation per task."""

    def __init__(self, num_tokens: int, task_count: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.projection = nn.Linear(num_tokens, task_count)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens.transpose(-1, -2)).transpose(-1, -2)


class TaskMLPTower(nn.Sequential):
    """Independent two-layer prediction tower for one ranking task."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for layer in (self[0], self[3]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class OneTransModel(RecAgoraRankingModel):
    """One implementation whose tokenizer/encoder combinations live in YAML."""

    # Candidate branches only read shared history; candidates never read one
    # another.  All training objectives are candidate BCE terms below.
    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, Any],
        tokenizer: Mapping[str, Any],
        encoder: Mapping[str, Any],
        readout: Mapping[str, Any] | None = None,
        head_hidden_dim: int,
        head_dropout: float,
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        history = dict(history)
        tokenizer = dict(tokenizer)
        if str(tokenizer.get("type", "")) == "groupwise" and "token_order" in tokenizer:
            tokenizer["token_count"] = len(tuple(tokenizer["token_order"]))
        encoder = dict(encoder)
        readout = dict(readout or {"type": "task_projection"})
        head_hidden_dim = int(head_hidden_dim)
        head_dropout = float(head_dropout)
        output_label_names = normalize_output_label_names(output_label_names)
        loss_weights = {
            str(name): float(weight)
            for name, weight in (loss_weights or {}).items()
        }
        self._validate_parameters(
            input_spec=input_spec,
            history=history,
            tokenizer=tokenizer,
            encoder=encoder,
            readout=readout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            output_label_names=output_label_names,
            loss_weights=loss_weights,
        )
        params = {
            "history": history,
            "tokenizer": tokenizer,
            "encoder": encoder,
            "readout": readout,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": head_dropout,
            "output_label_names": output_label_names,
            "loss_weights": loss_weights,
        }
        super().__init__(params, input_spec)

        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = loss_weights
        self.tokenizer_type = str(tokenizer["type"])
        self.model_dim = int(encoder["model_dim"])
        self.sequence_keep_per_layer = tuple(
            int(value) for value in encoder["sequence_keep_per_layer"]
        )
        self.readout_type = str(readout.get("type", "task_projection"))
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.inputs = OneTransInputs(
            input_spec=input_spec,
            history=history,
            tokenizer=tokenizer,
        )
        self.sequence_projection = nn.Linear(
            self.inputs.sequence_input_dim,
            self.model_dim,
        )
        self.num_ns_tokens = int(tokenizer["token_count"])
        if self.tokenizer_type == "groupwise":
            self.ns_projections = nn.ModuleList(
                nn.Linear(
                    self.inputs.ns_group_input_dims[name],
                    self.model_dim,
                )
                for name in self.inputs.token_order
            )
            self.ns_auto_projection = None
        else:
            self.ns_projections = nn.ModuleList()
            self.ns_auto_projection = nn.Linear(
                sum(self.inputs.ns_group_input_dims.values()),
                self.num_ns_tokens * self.model_dim,
            )

        self.blocks = nn.ModuleList(
            OneTransBlock(
                model_dim=self.model_dim,
                num_heads=int(encoder["num_heads"]),
                ffn_dim=int(encoder["ffn_dim"]),
                num_ns_tokens=self.num_ns_tokens,
                dropout=float(encoder.get("dropout", 0.0)),
                rms_norm_eps=float(encoder.get("rms_norm_eps", 1e-6)),
            )
            for _ in self.sequence_keep_per_layer
        )
        self.output_heads = nn.ModuleDict(
            {
                name: TaskMLPTower(
                    self.model_dim,
                    self.head_hidden_dim,
                    self.head_dropout,
                )
                for name in output_label_names
            }
        )
        self.task_readout = (
            TaskTokenReadout(self.num_ns_tokens, len(output_label_names))
            if self.readout_type == "task_projection"
            else None
        )

    @staticmethod
    def _validate_parameters(
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, Any],
        tokenizer: Mapping[str, Any],
        encoder: Mapping[str, Any],
        readout: Mapping[str, Any],
        head_hidden_dim: int,
        head_dropout: float,
        output_label_names: tuple[str, ...],
        loss_weights: Mapping[str, float],
    ) -> None:
        required_history = {"sequence_slots"}
        required_tokenizer = {"type", "token_count", "token_order", "groups"}
        required_encoder = {
            "model_dim",
            "num_heads",
            "ffn_dim",
            "sequence_keep_per_layer",
        }
        for name, value, required in (
            ("history", history, required_history),
            ("tokenizer", tokenizer, required_tokenizer),
            ("encoder", encoder, required_encoder),
        ):
            missing = sorted(required - set(value))
            if missing:
                raise ValueError(f"OneTrans {name} is missing fields: {missing}")
        tokenizer_type = str(tokenizer["type"])
        if tokenizer_type not in {"groupwise", "auto_split"}:
            raise ValueError("tokenizer.type must be groupwise or auto_split")
        token_count = int(tokenizer["token_count"])
        token_order = tuple(tokenizer["token_order"])
        if token_count <= 0:
            raise ValueError("tokenizer.token_count must be positive")
        model_dim = int(encoder["model_dim"])
        num_heads = int(encoder["num_heads"])
        validate_model_dim_heads(model_dim, num_heads)
        if int(encoder["ffn_dim"]) <= 0:
            raise ValueError("encoder.ffn_dim must be positive")
        keeps = tuple(int(value) for value in encoder["sequence_keep_per_layer"])
        if not keeps or any(value < 0 for value in keeps):
            raise ValueError("sequence_keep_per_layer must contain non-negative values")
        if any(left < right for left, right in zip(keeps, keeps[1:])):
            raise ValueError("sequence_keep_per_layer must be non-increasing")
        readout_type = str(readout.get("type", "task_projection"))
        if readout_type not in {"task_projection", "mean_ns", "last_ns"}:
            raise ValueError("readout.type must be task_projection, mean_ns, or last_ns")
        validate_head_dimensions(head_hidden_dim, head_dropout)
        if not set(output_label_names) <= set(input_spec.task_names):
            raise ValueError("OneTrans outputs must be declared dataset tasks")
        if readout_type == "task_projection" and token_count < len(output_label_names):
            raise ValueError("task projection requires at least one token per task")
        validate_known_weights(loss_weights, output_label_names)
        for name, weight in loss_weights.items():
            finite_nonnegative_float(weight, f"loss_weights.{name}")

    def _project_tokens(
        self,
        inputs: OneTransInputsOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer_type == "auto_split":
            concatenated = torch.cat(inputs.non_sequence_groups, dim=-1)
            ns_tokens = self.ns_auto_projection(concatenated).reshape(
                *concatenated.shape[:-1],
                self.num_ns_tokens,
                self.model_dim,
            )
        else:
            ns_tokens = torch.stack(
                [
                    projection(group)
                    for projection, group in zip(
                        self.ns_projections,
                        inputs.non_sequence_groups,
                        strict=True,
                    )
                ],
                dim=2,
            )
        return self.sequence_projection(inputs.sequence), ns_tokens

    def _readout(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.readout_type == "task_projection":
            return self.task_readout(tokens)
        if self.readout_type == "last_ns":
            return tokens.select(dim=-2, index=tokens.shape[-2] - 1)
        return tokens.mean(dim=-2)

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        inputs = self.inputs(embedded)
        sequence, ns_tokens = self._project_tokens(inputs)
        valid_mask = inputs.sequence_valid_mask
        for keep, block in zip(
            self.sequence_keep_per_layer,
            self.blocks,
            strict=True,
        ):
            sequence, ns_tokens, valid_mask = block.forward_shared_candidates(
                sequence,
                ns_tokens,
                retained_sequence_count=min(int(keep), sequence.shape[1]),
                sequence_valid_mask=valid_mask,
            )
        representation = self._readout(ns_tokens)
        output: dict[str, torch.Tensor] = {}
        for index, (name, head) in enumerate(self.output_heads.items()):
            task_representation = (
                representation[..., index, :]
                if self.readout_type == "task_projection"
                else representation
            )
            output[name] = head(task_representation).squeeze(-1)
        return structured_ranking_output(output)

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name, logits in output.items():
            label = batch.labels[name]
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                label.to(device=logits.device, dtype=logits.dtype),
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted.append(loss * float(self.loss_weights.get(name, 1.0)))
        return torch.stack(weighted).sum(), losses

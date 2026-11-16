from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
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
    validate_positive_int,
    validate_positive_int_sequence,
    validate_standard_multitask_contract,
)

from .blocks import ChunkTokenizer, UniMixingLiteStack
from .inputs import UniMixerInputs


class UniMixerModel(RecAgoraRankingModel):
    """Request-level adaptation of the complete UniMixing-Lite backbone."""

    # Tokenization and UniMixing preserve candidate isolation in the [B,L]
    # leading dimensions, and there is no candidate-level auxiliary objective.
    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
        token_dim: int,
        num_tokens: int,
        num_layers: int,
        expansion_factor: int,
        num_basis: int,
        global_rank: int,
        sinkhorn_iterations: int,
        sinkhorn_eps: float,
        tau_start: float,
        tau_end: float,
        tau_schedule: str,
        tau_anneal_steps: int,
        tau_warmup_steps: int = 0,
        symmetric: bool = True,
        local_coefficient_softmax: bool = False,
        output_projection: bool = False,
        dropout: float = 0.0,
        norm_eps: float = 1.0e-6,
        activation_checkpointing: bool = True,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_units: Sequence[int] = (128, 64),
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        for value, name in (
            (token_dim, "token_dim"),
            (num_tokens, "num_tokens"),
            (num_layers, "num_layers"),
            (expansion_factor, "expansion_factor"),
            (num_basis, "num_basis"),
            (global_rank, "global_rank"),
            (sinkhorn_iterations, "sinkhorn_iterations"),
            (tau_anneal_steps, "tau_anneal_steps"),
        ):
            validate_positive_int(value, field_name=name)
        if type(tau_warmup_steps) is not int or tau_warmup_steps < 0:
            raise ValueError("tau_warmup_steps must be non-negative")
        finite_positive_float(sinkhorn_eps, "sinkhorn_eps")
        finite_positive_float(tau_start, "tau_start")
        finite_positive_float(tau_end, "tau_end")
        validate_dropout(dropout)
        validate_dropout(head_dropout, field_name="head_dropout")
        finite_positive_float(norm_eps, "norm_eps")
        for value, name in (
            (symmetric, "symmetric"),
            (local_coefficient_softmax, "local_coefficient_softmax"),
            (output_projection, "output_projection"),
            (activation_checkpointing, "activation_checkpointing"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        tau_schedule = str(tau_schedule).lower()
        if tau_schedule not in {"linear", "cosine", "exponential"}:
            raise ValueError(
                "tau_schedule must be linear, cosine, or exponential"
            )
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="UniMixer",
        )
        validate_positive_int_sequence(
            head_hidden_units,
            field_name="head_hidden_units",
        )
        head_hidden_units = tuple(int(value) for value in head_hidden_units)
        task_weights = {
            str(name): float(weight) for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")

        params: dict[str, Any] = {
            "query_slots": tuple(int(slot_id) for slot_id in query_slots),
            "candidate_slots": tuple(int(slot_id) for slot_id in candidate_slots),
            "history": dict(history),
            "token_dim": int(token_dim),
            "num_tokens": int(num_tokens),
            "num_layers": int(num_layers),
            "expansion_factor": int(expansion_factor),
            "num_basis": int(num_basis),
            "global_rank": int(global_rank),
            "sinkhorn_iterations": int(sinkhorn_iterations),
            "sinkhorn_eps": float(sinkhorn_eps),
            "tau_start": float(tau_start),
            "tau_end": float(tau_end),
            "tau_schedule": tau_schedule,
            "tau_anneal_steps": int(tau_anneal_steps),
            "tau_warmup_steps": int(tau_warmup_steps),
            "symmetric": symmetric,
            "local_coefficient_softmax": local_coefficient_softmax,
            "output_projection": output_projection,
            "dropout": float(dropout),
            "norm_eps": float(norm_eps),
            "activation_checkpointing": activation_checkpointing,
            "candidate_isolation": candidate_isolation,
            "task_names": task_names,
            "head_hidden_units": head_hidden_units,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
        }
        super().__init__(params, input_spec)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.task_names = task_names
        self.task_weights = task_weights
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.tau_schedule = tau_schedule
        self.tau_anneal_steps = int(tau_anneal_steps)
        self.tau_warmup_steps = int(tau_warmup_steps)
        self.register_buffer("tau_step", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "current_tau",
            torch.tensor(self.tau_start, dtype=torch.float32),
        )

        self.inputs = UniMixerInputs(
            input_spec=input_spec,
            query_slots=query_slots,
            candidate_slots=candidate_slots,
            history=history,
        )
        self.tokenizer = ChunkTokenizer(
            input_dim=self.inputs.output_dim,
            num_tokens=self.num_tokens,
            token_dim=self.token_dim,
            norm_eps=float(norm_eps),
        )
        self.backbone = UniMixingLiteStack(
            token_dim=self.token_dim,
            num_tokens=self.num_tokens,
            num_layers=int(num_layers),
            expansion_factor=int(expansion_factor),
            num_basis=int(num_basis),
            global_rank=int(global_rank),
            sinkhorn_iterations=int(sinkhorn_iterations),
            sinkhorn_eps=float(sinkhorn_eps),
            tau=self.tau_start,
            symmetric=symmetric,
            local_coefficient_softmax=local_coefficient_softmax,
            output_projection=output_projection,
            dropout=float(dropout),
            norm_eps=float(norm_eps),
            activation_checkpointing=activation_checkpointing,
        )
        head_input_dim = self.num_tokens * self.token_dim
        self.heads = nn.ModuleDict(
            {
                name: _TaskTower(
                    input_dim=head_input_dim,
                    hidden_units=head_hidden_units,
                    dropout=float(head_dropout),
                )
                for name in self.task_names
            }
        )
        self.update_tau(step=0)

    def _tau_tensor_for_step(self, step: torch.Tensor) -> torch.Tensor:
        step = step.to(device=self.current_tau.device, dtype=torch.float32).clamp_min(0)
        progress = (
            (step - float(self.tau_warmup_steps)) / float(self.tau_anneal_steps)
        ).clamp(min=0.0, max=1.0)
        if self.tau_schedule == "linear":
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress
        elif self.tau_schedule == "cosine":
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
                1.0 + torch.cos(math.pi * progress)
            )
        else:
            tau = self.tau_start * ((self.tau_end / self.tau_start) ** progress)
        return tau.clamp_min(1.0e-6)

    @torch.no_grad()
    def _copy_tau_for_step(self, step: torch.Tensor) -> None:
        tau = self._tau_tensor_for_step(step)
        self.current_tau.copy_(tau)
        self.backbone.copy_tau(tau)

    @torch.no_grad()
    def update_tau(self, *, step: int | None = None) -> float:
        """Explicit diagnostic setter; the training path remains host-sync free."""

        if step is None:
            step_tensor = self.tau_step
        else:
            step_tensor = torch.as_tensor(
                step,
                device=self.tau_step.device,
                dtype=self.tau_step.dtype,
            )
        self._copy_tau_for_step(step_tensor)
        return float(self.current_tau.item())

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        inputs = self.inputs(embedded)
        if self.training:
            self._copy_tau_for_step(self.tau_step)
            self.tau_step.add_(1)
        tokens = self.tokenizer(inputs.values)
        representation = self.backbone(tokens).flatten(start_dim=2)
        return structured_ranking_output(
            {
                name: head(representation).squeeze(-1)
                for name, head in self.heads.items()
            }
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.task_names:
            logits = logits_by_task[name]
            labels = batch.labels[name].to(device=logits.device, dtype=logits.dtype)
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted = loss * self.task_weights.get(name, 1.0)
            total = weighted if total is None else total + weighted
        losses["total"] = total  # type: ignore[assignment]
        return total, losses  # type: ignore[return-value]


class _TaskTower(nn.Sequential):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_units: Sequence[int],
        dropout: float,
    ) -> None:
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_units:
            linear = nn.Linear(previous, int(hidden))
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.SiLU(), nn.Dropout(dropout)))
            previous = int(hidden)
        output = nn.Linear(previous, 1)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        super().__init__(*layers)

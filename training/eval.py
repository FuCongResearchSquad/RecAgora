from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import os
from time import perf_counter
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .factory import build_model
from utils.compiler.config import RecAgoraConfig, DataConfig, TrainConfig
from utils.runtime.data.request_batch import RequestBatch
from utils.runtime.model.runner import unwrap_recagora_runtime_output
from utils.runtime.candidate_mask import require_candidate_mask_support
from utils.runtime.distributed import (
    DistributedContext,
    all_gather_object,
    destroy_distributed,
    init_distributed,
    resolve_device_for_context,
    unwrap_model,
)
from utils.compiler.feature_contract import build_feature_contract
from utils.runtime.prediction import binary_auc, logits_by_name
from utils.runtime.execution import (
    autocast,
    build_data_loader,
    iter_with_progress,
    load_model_weights,
    prepare_result_dirs,
    strip_batch_runtime_metadata,
    write_log,
)
from utils.runtime.diagnostics import (
    environment_diagnostic_lines,
    run_start_line,
    torchrec_model_diagnostic_lines,
)
from utils.runtime.checkpoint.dcp import read_model_checkpoint_metadata
from utils.runtime.torchrec import (
    torchrec_mode,
    uses_torchrec_dmp,
    wrap_torchrec_dmp,
)


# ===== Public API =====


@dataclass
class EvalMetricAccumulator:
    """Device-resident additive eval metrics with no per-batch host sync."""

    device: torch.device
    sums: dict[str, torch.Tensor] = field(default_factory=dict)
    counts: dict[str, torch.Tensor] = field(default_factory=dict)
    candidate_counts: dict[str, torch.Tensor] = field(default_factory=dict)

    def record_candidates(
        self,
        *,
        labels_by_name: Mapping[str, torch.Tensor],
        candidate_valid_mask: torch.Tensor | None,
    ) -> None:
        """Count candidates once per batch, not once per output task."""
        if not labels_by_name:
            return
        labels = next(iter(labels_by_name.values()))
        _validate_metric_mask(candidate_valid_mask, labels)
        zero = torch.zeros((), device=self.device, dtype=torch.int64)
        if candidate_valid_mask is None:
            counts = {
                "valid_candidates": zero.new_tensor(labels.numel()),
                "masked_candidates": zero,
                "empty_requests": zero,
            }
        else:
            counts = {
                "valid_candidates": candidate_valid_mask.sum(),
                "masked_candidates": (~candidate_valid_mask).sum(),
                "empty_requests": (~candidate_valid_mask.any(dim=1)).sum(),
            }
        for name, task_labels in labels_by_name.items():
            _validate_metric_mask(candidate_valid_mask, task_labels)
            counts[f"masked_positive/{name}"] = (
                zero
                if candidate_valid_mask is None
                else (task_labels.gt(0) & ~candidate_valid_mask).sum()
            )
        for name, value in counts.items():
            if name not in self.candidate_counts:
                self.candidate_counts[name] = torch.zeros(
                    (), device=self.device, dtype=torch.float64,
                )
            self.candidate_counts[name].add_(value.to(self.device))

    def update(
        self,
        *,
        name: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        candidate_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_metric_mask(candidate_valid_mask, labels)
        # Select before BCE: NaN/Inf values in an excluded position must not
        # contaminate the sum through multiplication by zero.
        loss_logits = (
            logits if candidate_valid_mask is None else logits[candidate_valid_mask]
        )
        loss_labels = (
            labels if candidate_valid_mask is None else labels[candidate_valid_mask]
        )
        loss_sum = F.binary_cross_entropy_with_logits(
            loss_logits,
            loss_labels,
            reduction="sum",
        )
        self._add(
            f"loss/{name}",
            loss_sum,
            logits.new_tensor(loss_labels.numel(), dtype=torch.int64),
        )
        scores = torch.sigmoid(logits.detach())
        metric_sums, metric_counts = _vectorized_request_metric_tensors(
            labels,
            scores,
            name=name,
            candidate_valid_mask=candidate_valid_mask,
        )
        for key, value in metric_sums.items():
            self._add(key, value, metric_counts[key])
        return scores

    def _add(
        self,
        key: str,
        value: torch.Tensor,
        count: torch.Tensor,
    ) -> None:
        value = value.detach().to(device=self.device, dtype=torch.float64)
        count = count.detach().to(device=self.device, dtype=torch.float64)
        if key not in self.sums:
            self.sums[key] = torch.zeros((), dtype=torch.float64, device=self.device)
            self.counts[key] = torch.zeros((), dtype=torch.float64, device=self.device)
        self.sums[key].add_(value)
        self.counts[key].add_(count)


def evaluate(
    config: RecAgoraConfig,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_batches: int | None = None,
    global_eval: bool = False,
    global_auc_workers: int | None = None,
) -> dict[str, Any]:
    """Run checkpoint evaluation and write aggregate ranking metrics to disk."""
    started_at = datetime.now().astimezone()
    dist_context = init_distributed()
    try:
        # Runtime setup mirrors training, with eval-specific overrides for data,
        # batch size, device, and AMP.
        runtime_train = _eval_train_config(config)
        torch.manual_seed(runtime_train.seed)
        runtime_data = _eval_data_config(config)
        feature_plans = build_feature_contract(
            runtime_data,
            config.model,
        ).build_all_plans()
        torch_device = resolve_device_for_context(runtime_train.device, dist_context)
        use_amp = runtime_train.amp and torch_device.type == "cuda"
        checkpoint = _resolve_checkpoint_path(config, checkpoint_path)
        result_dirs = prepare_result_dirs(_resolve_output_dir(config, output_dir))
        log_path = result_dirs["logs"] / "eval.log"
        if dist_context.is_main:
            write_log(
                log_path,
                run_start_line("eval", started_at=started_at),
            )
        metrics_path = result_dirs["outputs"] / "eval_metrics.json"
        model = build_model(config.model, input_contract=feature_plans).to(torch_device)
        require_candidate_mask_support(model, runtime_train.negative_mask_fields)
        model = wrap_torchrec_dmp(
            model,
            config=config.model,
            context=dist_context,
            device=torch_device,
        )
        checkpoint_metadata = read_model_checkpoint_metadata(checkpoint)
        _validate_ranking_checkpoint_interface(checkpoint_metadata)
        load_model_weights(
            checkpoint,
            model=model,
            distributed_context=dist_context,
        )
        model.eval()

        loader = build_data_loader(
            runtime_data,
            runtime_train,
            torch_device=torch_device,
            distributed_context=dist_context,
            shuffle=False,
            read_plan=feature_plans.read_plan,
            parse_plan=feature_plans.parse_plan,
            adapter_plan=feature_plans.adapter_plan,
            embedding_input_plan=feature_plans.embedding_input_plan,
        )
        if len(loader) == 0 and not dist_context.enabled:
            raise ValueError("Eval DataLoader is empty")

        batch_limit = max_batches or config.eval.max_batches
        requested_global_auc_workers = (
            global_auc_workers
            if global_auc_workers is not None
            else config.eval.global_auc_workers
        )
        if (
            requested_global_auc_workers is not None
            and requested_global_auc_workers <= 0
        ):
            raise ValueError("global_auc_workers must be positive when provided")
        if dist_context.is_main:
            for line in _eval_start_lines(
                config=config,
                model=model,
                runtime_data=runtime_data,
                runtime_train=runtime_train,
                torch_device=torch_device,
                use_amp=use_amp,
                checkpoint=checkpoint,
                batch_limit=batch_limit,
                global_eval=global_eval,
                global_auc_workers=requested_global_auc_workers,
                distributed_context=dist_context,
            ):
                write_log(log_path, line)

        metric_accumulator = EvalMetricAccumulator(torch_device)
        auc_labels: dict[str, list[torch.Tensor]] = {}
        auc_scores: dict[str, list[torch.Tensor]] = {}
        batches = 0
        request_groups = 0
        examples = 0
        last_progress_batch = 0

        # Main eval loop: collect loss for every output head, request-level ranking
        # metrics for all runs, and exact/global metrics only when requested.
        progress = iter_with_progress(
            loader,
            enabled=runtime_data.index_progress and dist_context.is_main,
            desc="eval",
        )
        with torch.inference_mode():
            for batch in progress:
                if batch_limit is not None and batches >= batch_limit:
                    break
                batch, _ = strip_batch_runtime_metadata(batch)
                request_groups += int(batch.request_id.shape[0])
                model_owner = unwrap_model(model)
                model_batch = _prepare_model_batch(model_owner, batch)
                model_batch = model_batch.to(
                    torch_device,
                    non_blocking=runtime_train.pin_memory
                    and torch_device.type == "cuda",
                )
                with autocast(torch_device, use_amp):
                    model_output = model(model_batch)
                    model_output, _ = unwrap_recagora_runtime_output(model_output)
                    named_logits = logits_by_name(
                        model_output,
                        default_name=config.data.target,
                    )

                batches += 1
                examples += int(model_batch.metadata["request_id"].shape[0])
                labels_by_name = {
                    name: _label_for_prepared_output(
                        model_batch.labels,
                        name=name,
                        default_name=config.data.target,
                    )
                    for name in named_logits
                }
                candidate_valid_mask = model_batch.candidate_valid_mask
                metric_accumulator.record_candidates(
                    labels_by_name=labels_by_name,
                    candidate_valid_mask=candidate_valid_mask,
                )
                for name, logits in named_logits.items():
                    labels = labels_by_name[name]
                    scores = metric_accumulator.update(
                        name=name,
                        logits=logits,
                        labels=labels,
                        candidate_valid_mask=candidate_valid_mask,
                    )
                    if global_eval:
                        valid_labels, valid_scores = _global_auc_values(
                            labels, scores, candidate_valid_mask,
                        )
                        auc_labels.setdefault(name, []).append(
                            valid_labels.to(torch.float32).cpu()
                        )
                        auc_scores.setdefault(name, []).append(
                            valid_scores.to(torch.float32).cpu()
                        )
                should_update_progress = _should_report_eval_metrics(
                    batches,
                    runtime_train.progress_every_steps,
                )
                if should_update_progress:
                    reduced_metrics = _reduce_eval_metric_accumulator(
                        metric_accumulator,
                        dist_context,
                    )
                    loss_sums, counts = _loss_state(reduced_metrics)
                    if dist_context.is_main:
                        progress.set_postfix(
                            {"loss": f"{_running_eval_loss(loss_sums, counts):.6f}"}
                        )
                        last_progress_batch = batches

        reduced_metrics = _reduce_eval_metric_accumulator(
            metric_accumulator,
            dist_context,
            counters={
                "batches": batches,
                "request_groups": request_groups,
                "examples": examples,
            },
        )
        loss_sums, counts = _loss_state(reduced_metrics)
        request_auc_sums, request_auc_counts = _request_metric_state(
            reduced_metrics,
            prefix="request_auc/",
        )
        request_hr_sums, request_hr_counts = _request_metric_state(
            reduced_metrics,
            prefix="request_hr/",
        )
        request_ndcg_sums, request_ndcg_counts = _request_metric_state(
            reduced_metrics,
            prefix="request_ndcg/",
        )
        if dist_context.is_main:
            if batches != last_progress_batch:
                progress.set_postfix(
                    {"loss": f"{_running_eval_loss(loss_sums, counts):.6f}"}
                )

        merged_global_state = {
            "auc_labels": {},
            "auc_scores": {},
        }
        if global_eval:
            gathered_states = _all_gather_dicts(
                {
                    "auc_labels": auc_labels,
                    "auc_scores": auc_scores,
                },
                dist_context,
            )
            merged_global_state = _merge_eval_states(
                gathered_states,
                global_eval=True,
            )
        merged_state = {
            "loss_sums": loss_sums,
            "counts": counts,
            "auc_labels": merged_global_state["auc_labels"],
            "auc_scores": merged_global_state["auc_scores"],
            "request_auc_sums": request_auc_sums,
            "request_auc_counts": request_auc_counts,
            "request_hr_sums": request_hr_sums,
            "request_hr_counts": request_hr_counts,
            "request_ndcg_sums": request_ndcg_sums,
            "request_ndcg_counts": request_ndcg_counts,
            **reduced_metrics["counters"],
        }

        if not merged_state["counts"]:
            raise ValueError("Eval produced no labeled predictions")

        if not dist_context.is_main:
            return {
                "status": "done_non_main",
                "rank": dist_context.rank,
                "world_size": dist_context.world_size,
                "distributed": dist_context.enabled,
            }

        result = _build_eval_result(
            checkpoint=checkpoint,
            loss_sums=merged_state["loss_sums"],
            counts=merged_state["counts"],
            auc_labels=merged_state["auc_labels"],
            auc_scores=merged_state["auc_scores"],
            request_auc_sums=merged_state["request_auc_sums"],
            request_auc_counts=merged_state["request_auc_counts"],
            request_hr_sums=merged_state["request_hr_sums"],
            request_hr_counts=merged_state["request_hr_counts"],
            request_ndcg_sums=merged_state["request_ndcg_sums"],
            request_ndcg_counts=merged_state["request_ndcg_counts"],
            batches=merged_state["batches"],
            request_groups=merged_state["request_groups"],
            examples=merged_state["examples"],
            torch_device=torch_device,
            use_amp=use_amp,
            metrics_path=metrics_path,
            log_path=log_path,
            data_paths=loader.dataset.paths,
            status="done",
            global_eval=global_eval,
            global_auc_workers=requested_global_auc_workers,
            distributed_context=dist_context,
            torchrec_mode_value=torchrec_mode(config.model),
            torchrec_dmp_enabled=uses_torchrec_dmp(config.model, dist_context),
            negative_mask_fields=runtime_train.negative_mask_fields,
            candidate_counts=reduced_metrics["counters"],
        )
        if loader.dataset.data_identity is not None:
            result["data_identity"] = dict(loader.dataset.data_identity)
        _write_eval_metrics(metrics_path, result)
        eval_loss_text = (
            "n/a" if result["eval_loss"] is None else f"{result['eval_loss']:.6f}"
        )
        done_line = (
            f"{result['status']} batches={merged_state['batches']} "
            f"requests={merged_state['request_groups']} "
            f"examples={merged_state['examples']} "
            f"eval_loss={eval_loss_text} "
            f"valid_candidates={result['valid_candidates']} "
            f"masked_candidates={result['masked_candidates']} "
            f"empty_requests={result['empty_requests']} "
            f"masked_positive_counts={result['masked_positive_counts']}"
        )
        if global_eval:
            done_line += (
                f" global_auc_workers={result['global_auc_workers']}"
                f" global_auc_compute_sec={result['global_auc_compute_sec']:.6f}"
            )
        write_log(log_path, done_line)
        return result
    finally:
        destroy_distributed(dist_context)


# ===== Loss Aggregation =====


def _running_eval_loss(
    loss_sums: dict[str, float],
    counts: dict[str, int],
) -> float:
    losses = _losses_from_sums(loss_sums, counts)
    if not losses:
        return 0.0
    return sum(losses.values()) / len(losses)


def _should_report_eval_metrics(batch_count: int, every_steps: int) -> bool:
    return batch_count > 0 and every_steps > 0 and batch_count % every_steps == 0


def _prepare_model_batch(model: torch.nn.Module, batch: Any) -> Any:
    prepare = getattr(model, "prepare_model_batch", None)
    if not callable(prepare):
        raise AttributeError(
            f"Model {type(model).__name__} must define prepare_model_batch"
        )
    return prepare(batch)


def _label_for_prepared_output(
    labels_by_name: Any,
    *,
    name: str,
    default_name: str,
) -> torch.Tensor:
    if isinstance(labels_by_name, dict) and isinstance(labels_by_name.get(name), torch.Tensor):
        return labels_by_name[name]
    if (
        name == default_name
        and isinstance(labels_by_name, dict)
        and isinstance(labels_by_name.get("target_label"), torch.Tensor)
    ):
        return labels_by_name["target_label"]
    raise KeyError(f"Prepared batch does not contain labels for model output {name!r}")


def _losses_from_sums(
    loss_sums: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    return {
        name: loss_sums[name] / counts[name]
        for name in sorted(counts)
        if counts[name] > 0
    }


def _validate_metric_mask(
    candidate_valid_mask: torch.Tensor | None,
    labels: torch.Tensor,
) -> None:
    if candidate_valid_mask is None:
        return
    if candidate_valid_mask.dtype != torch.bool:
        raise TypeError("candidate_valid_mask must have dtype bool")
    if labels.ndim != 2 or candidate_valid_mask.shape != labels.shape:
        raise ValueError("candidate_valid_mask must match the [B,L] labels shape")
    if candidate_valid_mask.device != labels.device:
        raise ValueError("candidate_valid_mask and labels must be on the same device")


def _global_auc_values(
    labels: torch.Tensor,
    scores: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only valid predictions enter the potentially large CPU AUC buffer."""
    _validate_metric_mask(candidate_valid_mask, labels)
    if candidate_valid_mask is None:
        return labels.detach().flatten(), scores.detach().flatten()
    return labels.detach()[candidate_valid_mask], scores.detach()[candidate_valid_mask]


def _vectorized_request_metric_tensors(
    labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    name: str,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return exact additive request metrics without leaving the device."""
    if labels.shape != scores.shape or labels.ndim != 2:
        return {}, {}
    _validate_metric_mask(candidate_valid_mask, labels)
    positives = labels.gt(0.5)
    if candidate_valid_mask is not None:
        positives = positives & candidate_valid_mask
    positive_counts = positives.sum(dim=1)
    negative_counts = (
        labels.shape[1] - positive_counts
        if candidate_valid_mask is None
        else candidate_valid_mask.sum(dim=1) - positive_counts
    )
    auc_valid = positive_counts.gt(0) & negative_counts.gt(0)

    positive_pairs = positives.unsqueeze(2)
    negatives = ~positives
    if candidate_valid_mask is not None:
        negatives = negatives & candidate_valid_mask
    negative_pairs = negatives.unsqueeze(1)
    pair_mask = positive_pairs & negative_pairs
    left = scores.unsqueeze(2)
    right = scores.unsqueeze(1)
    pair_wins = left.gt(right).to(torch.float64)
    pair_wins.add_(left.eq(right).to(torch.float64), alpha=0.5)
    auc_numerators = (pair_wins * pair_mask).sum(dim=(1, 2))
    auc_denominators = (positive_counts * negative_counts).clamp_min(1)
    request_auc = auc_numerators / auc_denominators.to(torch.float64)

    sums: dict[str, torch.Tensor] = {
        f"request_auc/{name}": request_auc[auc_valid].sum(),
    }
    counts: dict[str, torch.Tensor] = {
        f"request_auc/{name}": auc_valid.sum(),
    }

    ranking_valid = positive_counts.gt(0)
    # Preserve the existing tie-order convention and all-valid result. Scores
    # here are probabilities, so -inf sorts excluded slots strictly last,
    # including when their original logits were NaN or Inf.
    ranking_scores = (
        scores
        if candidate_valid_mask is None
        else scores.masked_fill(~candidate_valid_mask, float("-inf"))
    )
    order = torch.argsort(ranking_scores, dim=1, descending=True)
    ranked_positive = positives.gather(1, order)
    candidate_count = int(labels.shape[1])
    discounts = torch.arange(
        2,
        candidate_count + 2,
        dtype=torch.float64,
        device=labels.device,
    ).log2().reciprocal()
    cumulative_dcg = torch.cumsum(
        ranked_positive.to(torch.float64) * discounts,
        dim=1,
    )
    ideal_dcg = torch.cumsum(discounts, dim=0)
    for k in _request_ranking_ks(candidate_count):
        metric_name = f"{name}@{k}" if name else f"@{k}"
        topk = ranked_positive[:, :k]
        hr = topk.any(dim=1).to(torch.float64)
        ideal_hits = positive_counts.clamp_max(k)
        ideal = ideal_dcg[(ideal_hits - 1).clamp_min(0)]
        ndcg = cumulative_dcg[:, k - 1] / ideal.clamp_min(torch.finfo(torch.float64).tiny)
        sums[f"request_hr/{metric_name}"] = hr[ranking_valid].sum()
        counts[f"request_hr/{metric_name}"] = ranking_valid.sum()
        sums[f"request_ndcg/{metric_name}"] = ndcg[ranking_valid].sum()
        counts[f"request_ndcg/{metric_name}"] = ranking_valid.sum()
    return sums, counts


# ===== Distributed Aggregation =====


def _reduce_eval_metric_accumulator(
    accumulator: EvalMetricAccumulator,
    context: DistributedContext,
    *,
    counters: dict[str, int] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Reduce every additive eval metric with one collective and one D2H copy."""
    metric_names = sorted(accumulator.sums)
    counter_values = dict(accumulator.candidate_counts)
    counter_values.update({
        name: torch.tensor(value, dtype=torch.float64, device=accumulator.device)
        for name, value in (counters or {}).items()
    })
    counter_names = sorted(counter_values)
    rows = [
        torch.stack((accumulator.sums[name], accumulator.counts[name]))
        for name in metric_names
    ]
    rows.extend(
        torch.stack((counter_values[name], counter_values[name].new_zeros(())))
        for name in counter_names
    )
    if not rows:
        return {"sums": {}, "counts": {}, "counters": {}}
    packed = torch.stack(rows)
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    host_rows = packed.detach().cpu().tolist()
    metric_rows = host_rows[: len(metric_names)]
    counter_rows = host_rows[len(metric_names) :]
    return {
        "sums": {
            name: float(values[0])
            for name, values in zip(metric_names, metric_rows, strict=True)
        },
        "counts": {
            name: int(values[1])
            for name, values in zip(metric_names, metric_rows, strict=True)
        },
        "counters": {
            name: int(values[0])
            for name, values in zip(counter_names, counter_rows, strict=True)
        },
    }


def _loss_state(
    state: dict[str, dict[str, float | int]],
) -> tuple[dict[str, float], dict[str, int]]:
    return _request_metric_state(state, prefix="loss/")


def _request_metric_state(
    state: dict[str, dict[str, float | int]],
    *,
    prefix: str,
) -> tuple[dict[str, float], dict[str, int]]:
    sums = state.get("sums", {})
    counts = state.get("counts", {})
    return (
        {
            name.removeprefix(prefix): float(value)
            for name, value in sums.items()
            if name.startswith(prefix)
        },
        {
            name.removeprefix(prefix): int(value)
            for name, value in counts.items()
            if name.startswith(prefix)
        },
    )


def _all_gather_dicts(
    value: dict[str, Any],
    context: DistributedContext,
) -> list[dict[str, Any]]:
    gathered = all_gather_object(value, context)
    return [item for item in gathered if isinstance(item, dict)]


def _merge_eval_states(
    states: list[dict[str, Any]],
    *,
    global_eval: bool,
) -> dict[str, Any]:
    return {
        "loss_sums": _sum_float_dicts(
            _state_dicts(states, "loss_sums"),
        ),
        "counts": _sum_int_dicts(_state_dicts(states, "counts")),
        "auc_labels": (
            _merge_auc_tensor_lists(_state_dicts(states, "auc_labels"))
            if global_eval
            else {}
        ),
        "auc_scores": (
            _merge_auc_tensor_lists(_state_dicts(states, "auc_scores"))
            if global_eval
            else {}
        ),
        "request_auc_sums": _sum_float_dicts(
            _state_dicts(states, "request_auc_sums"),
        ),
        "request_auc_counts": _sum_int_dicts(
            _state_dicts(states, "request_auc_counts"),
        ),
        "request_hr_sums": _sum_float_dicts(
            _state_dicts(states, "request_hr_sums"),
        ),
        "request_hr_counts": _sum_int_dicts(
            _state_dicts(states, "request_hr_counts"),
        ),
        "request_ndcg_sums": _sum_float_dicts(
            _state_dicts(states, "request_ndcg_sums"),
        ),
        "request_ndcg_counts": _sum_int_dicts(
            _state_dicts(states, "request_ndcg_counts"),
        ),
        "batches": sum(_state_ints(states, "batches")),
        "request_groups": sum(_state_ints(states, "request_groups")),
        "examples": sum(_state_ints(states, "examples")),
    }


def _state_dicts(states: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for state in states:
        value = state.get(key)
        if isinstance(value, dict):
            out.append(value)
    return out


def _state_ints(states: list[dict[str, Any]], key: str) -> list[int]:
    return [int(state.get(key, 0)) for state in states]


def _sum_float_dicts(dicts: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for values in dicts:
        for key, value in values.items():
            out[str(key)] = out.get(str(key), 0.0) + float(value)
    return out


def _sum_int_dicts(dicts: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for values in dicts:
        for key, value in values.items():
            out[str(key)] = out.get(str(key), 0) + int(value)
    return out


def _merge_auc_tensor_lists(
    dicts: list[dict[str, Any]],
) -> dict[str, list[torch.Tensor]]:
    out: dict[str, list[torch.Tensor]] = {}
    for values in dicts:
        for key, tensor_list in values.items():
            if not isinstance(tensor_list, list):
                continue
            out.setdefault(str(key), []).extend(
                tensor for tensor in tensor_list if isinstance(tensor, torch.Tensor)
            )
    return out


# ===== Exact Global AUC =====


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, int(os.cpu_count() or 1))


def _compute_global_auc_task(
    task: tuple[str, list[torch.Tensor], list[torch.Tensor]],
) -> tuple[str, float | None]:
    name, label_tensors, score_tensors = task
    if not label_tensors:
        raise ValueError(f"Global AUC output {name!r} has no labels")
    if not score_tensors:
        raise ValueError(f"Global AUC output {name!r} has no scores")
    labels = torch.cat(label_tensors)
    scores = torch.cat(score_tensors)
    return name, binary_auc(labels, scores)


def _compute_global_auc_by_name(
    auc_labels: dict[str, list[torch.Tensor]],
    auc_scores: dict[str, list[torch.Tensor]],
    *,
    requested_workers: int | None,
) -> tuple[dict[str, float | None], int, float]:
    tasks = [
        (name, auc_labels.get(name, []), auc_scores[name])
        for name in sorted(auc_scores)
    ]
    if not tasks:
        return {}, 0, 0.0
    if requested_workers is not None and requested_workers <= 0:
        raise ValueError("global_auc_workers must be positive when provided")
    available_cpus = _available_cpu_count()
    worker_count = min(
        len(tasks),
        available_cpus,
        requested_workers if requested_workers is not None else len(tasks),
    )
    started_at = perf_counter()
    if worker_count == 1:
        results = [_compute_global_auc_task(task) for task in tasks]
    else:
        # Eval inference and DataLoader work are complete at this point. Split
        # the process CPU budget across independent task-level exact sorts so
        # the workers do not each create a full-size intra-op thread pool.
        previous_intraop_threads = torch.get_num_threads()
        per_worker_threads = max(
            1,
            min(previous_intraop_threads, available_cpus // worker_count),
        )
        changed_intraop_threads = per_worker_threads != previous_intraop_threads
        if changed_intraop_threads:
            torch.set_num_threads(per_worker_threads)
        try:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="global-auc",
            ) as executor:
                results = list(executor.map(_compute_global_auc_task, tasks))
        finally:
            if changed_intraop_threads:
                torch.set_num_threads(previous_intraop_threads)
    elapsed = perf_counter() - started_at
    return dict(results), worker_count, elapsed


# ===== Result Assembly =====


def _build_eval_result(
    *,
    checkpoint: Path,
    loss_sums: dict[str, float],
    counts: dict[str, int],
    auc_labels: dict[str, list[torch.Tensor]],
    auc_scores: dict[str, list[torch.Tensor]],
    request_auc_sums: dict[str, float],
    request_auc_counts: dict[str, int],
    request_hr_sums: dict[str, float],
    request_hr_counts: dict[str, int],
    request_ndcg_sums: dict[str, float],
    request_ndcg_counts: dict[str, int],
    batches: int,
    request_groups: int,
    examples: int,
    torch_device: torch.device,
    use_amp: bool,
    metrics_path: Path,
    log_path: Path,
    data_paths: Any,
    status: str,
    global_eval: bool = False,
    global_auc_workers: int | None = None,
    distributed_context: DistributedContext,
    torchrec_mode_value: str,
    torchrec_dmp_enabled: bool,
    negative_mask_fields: tuple[str, ...] = (),
    candidate_counts: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    losses = _losses_from_sums(loss_sums, counts)
    eval_loss = sum(losses.values()) / len(losses) if losses else 0.0
    if counts and not any(value > 0 for value in counts.values()):
        status = "no_valid_samples"
        eval_loss = None
    candidate_counts = candidate_counts or {}
    auc: dict[str, float | None] = {}
    effective_global_auc_workers = 0
    global_auc_compute_sec = 0.0
    if global_eval:
        (
            auc,
            effective_global_auc_workers,
            global_auc_compute_sec,
        ) = _compute_global_auc_by_name(
            auc_labels,
            auc_scores,
            requested_workers=global_auc_workers,
        )
    request_auc = {
        name: (
            request_auc_sums[name] / request_auc_counts[name]
            if request_auc_counts.get(name, 0) > 0
            else None
        )
        for name in sorted(set(counts) | set(auc_scores) | set(request_auc_sums))
    }
    request_hr = _averages_from_sums(request_hr_sums, request_hr_counts)
    request_ndcg = _averages_from_sums(request_ndcg_sums, request_ndcg_counts)
    return {
        "status": status,
        "checkpoint_path": str(checkpoint),
        "eval_loss": eval_loss,
        "losses": losses,
        "auc": auc,
        "auc_mode": "exact" if global_eval else "skipped",
        "global_auc_workers": effective_global_auc_workers,
        "global_auc_compute_sec": global_auc_compute_sec,
        "request_auc": request_auc,
        "request_auc_counts": {
            name: request_auc_counts.get(name, 0) for name in sorted(request_auc)
        },
        "request_auc_mode": "per_request_exact",
        "request_hr": request_hr,
        "request_hr_counts": {
            name: request_hr_counts.get(name, 0) for name in sorted(request_hr)
        },
        "request_ndcg": request_ndcg,
        "request_ndcg_counts": {
            name: request_ndcg_counts.get(name, 0) for name in sorted(request_ndcg)
        },
        "request_ranking_mode": "per_request_exact",
        "global_eval": global_eval,
        "batches": batches,
        "requests": request_groups,
        "request_groups": request_groups,
        "examples": examples,
        "predictions": sum(counts.values()),
        "negative_mask_fields": list(negative_mask_fields),
        "valid_candidates": int(candidate_counts.get(
            "valid_candidates", max(counts.values(), default=0),
        )),
        "masked_candidates": int(candidate_counts.get("masked_candidates", 0)),
        "empty_requests": int(candidate_counts.get("empty_requests", 0)),
        "masked_positive_counts": {
            key.removeprefix("masked_positive/"): int(value)
            for key, value in sorted(candidate_counts.items())
            if key.startswith("masked_positive/")
        },
        "device": str(torch_device),
        "amp_enabled": use_amp,
        "distributed": distributed_context.enabled,
        "rank": distributed_context.rank,
        "world_size": distributed_context.world_size,
        "torchrec_mode": torchrec_mode_value,
        "torchrec_dmp_enabled": torchrec_dmp_enabled,
        "data_paths": [str(path) for path in data_paths],
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }


# ===== Request-Level Ranking Metrics =====


def request_level_metric_sums(
    labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    name: str,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[
    float,
    int,
    dict[str, float],
    dict[str, int],
    dict[str, float],
    dict[str, int],
]:
    """Accumulate per-request AUC, HR, and NDCG sums for one output head."""
    sums, counts = _vectorized_request_metric_tensors(
        labels, scores, name=name, candidate_valid_mask=candidate_valid_mask,
    )
    auc_key = f"request_auc/{name}"
    prefixes = ("request_hr/", "request_ndcg/")
    host = torch.stack(
        [
            *[sums[key] for key in sorted(sums)],
            *[counts[key].to(torch.float64) for key in sorted(counts)],
        ]
    ).detach().cpu().tolist()
    ordered_sum_keys = sorted(sums)
    ordered_count_keys = sorted(counts)
    sum_values = dict(
        zip(ordered_sum_keys, map(float, host[: len(ordered_sum_keys)]), strict=True)
    )
    count_values = dict(
        zip(
            ordered_count_keys,
            map(int, host[len(ordered_sum_keys) :]),
            strict=True,
        )
    )
    metric_dicts: list[dict[str, float]] = []
    count_dicts: list[dict[str, int]] = []
    for prefix in prefixes:
        metric_dicts.append(
            {
                key.removeprefix(prefix): value
                for key, value in sum_values.items()
                if key.startswith(prefix)
            }
        )
        count_dicts.append(
            {
                key.removeprefix(prefix): value
                for key, value in count_values.items()
                if key.startswith(prefix)
            }
        )
    return (
        sum_values.get(auc_key, 0.0),
        count_values.get(auc_key, 0),
        metric_dicts[0],
        count_dicts[0],
        metric_dicts[1],
        count_dicts[1],
    )


def _request_ranking_ks(candidate_count: int) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            k for k in (1, 3, 5, 10, candidate_count) if 0 < k <= candidate_count
        )
    )


def _averages_from_sums(
    sums: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float | None]:
    return {
        name: sums[name] / counts[name] if counts.get(name, 0) > 0 else None
        for name in sorted(set(sums) | set(counts))
    }


# ===== IO And Config Helpers =====


def _write_eval_metrics(path: Path, result: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _eval_train_config(config: RecAgoraConfig) -> TrainConfig:
    return replace(
        config.train,
        batch_size=config.eval.batch_size or config.train.batch_size,
        num_workers=(
            config.train.num_workers
            if config.eval.num_workers is None
            else config.eval.num_workers
        ),
        pin_memory=(
            config.train.pin_memory
            if config.eval.pin_memory is None
            else config.eval.pin_memory
        ),
        persistent_workers=(
            config.train.persistent_workers
            if config.eval.persistent_workers is None
            else config.eval.persistent_workers
        ),
        device=config.eval.device or config.train.device,
        amp=config.train.amp if config.eval.amp is None else config.eval.amp,
        negative_mask_fields=config.eval.negative_mask_fields,
    )


def _eval_data_config(config: RecAgoraConfig) -> DataConfig:
    if config.data.root is not None:
        if config.eval.paths is not None:
            raise ValueError("HotUser evaluation uses eval.split, not eval.paths")
        return replace(config.data, split=config.eval.split or "valid")
    if config.eval.split is not None:
        raise ValueError("eval.split requires a HotUser data config")
    if config.eval.paths is None:
        return config.data
    return replace(
        config.data,
        paths=config.eval.paths,
        parquet_start_file=None,
    )


def _resolve_checkpoint_path(
    config: RecAgoraConfig,
    checkpoint_path: str | Path | None,
) -> Path:
    path = checkpoint_path or config.eval.checkpoint_path
    if path is None:
        path = Path(config.train.output_dir) / "checkpoints" / "last"
    path = Path(path)
    if path.suffix == ".pt":
        raise ValueError(f"DCP checkpoints must be directories, not .pt files: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"DCP checkpoint path is not a directory: {path}")
    return path


def _validate_ranking_checkpoint_interface(
    checkpoint: Mapping[str, Any],
) -> None:
    contract = checkpoint.get("arena_model_contract")
    if contract is None:
        # Ranking checkpoints published before the additive interface contract.
        return
    if not isinstance(contract, Mapping) or contract.get("interface") != "ranking":
        raise ValueError(
            "ranking evaluation cannot load a non-ranking model checkpoint"
        )


def _resolve_output_dir(
    config: RecAgoraConfig,
    output_dir: str | Path | None,
) -> str | Path:
    return output_dir or config.eval.output_dir or config.train.output_dir


def _eval_start_lines(
    *,
    config: RecAgoraConfig,
    model: torch.nn.Module,
    runtime_data: DataConfig,
    runtime_train: TrainConfig,
    torch_device: torch.device,
    use_amp: bool,
    checkpoint: Path,
    batch_limit: int | None,
    global_eval: bool,
    global_auc_workers: int | None,
    distributed_context: DistributedContext,
) -> list[str]:
    return [
        f"  config_path={config.source_config_path}",
        *environment_diagnostic_lines(),
        f"  model={config.model.name}",
        f"  model_class_path={config.model.class_path}",
        f"  torchrec_mode={torchrec_mode(config.model)} torchrec_dmp_enabled={uses_torchrec_dmp(config.model, distributed_context)}",
        *torchrec_model_diagnostic_lines(
            model,
            config=config.model,
            distributed_context=distributed_context,
        ),
        f"  distributed_enabled={distributed_context.enabled} backend={distributed_context.backend} rank={distributed_context.rank} local_rank={distributed_context.local_rank} world_size={distributed_context.world_size}",
        f"  checkpoint={checkpoint}",
        f"  cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        f"  requested_device={runtime_train.device} resolved_device={torch_device}",
        f"  amp_requested={runtime_train.amp} amp_enabled={use_amp}",
        f"  data_format={runtime_data.data_format} list_len={runtime_data.list_len} history_len={runtime_data.history_len}",
        f"  data_paths={_summarize_paths(runtime_data.paths)}",
        f"  batch_size={runtime_train.batch_size} max_batches={batch_limit}",
        f"  negative_mask_fields={list(runtime_train.negative_mask_fields)}",
        f"  progress_every_steps={runtime_train.progress_every_steps}",
        f"  num_workers={runtime_train.num_workers} pin_memory={runtime_train.pin_memory} persistent_workers={runtime_train.persistent_workers}",
        f"  global_eval={global_eval}",
        f"  global_auc_workers={global_auc_workers if global_auc_workers is not None else 'auto'}",
        f"  auc_mode={'exact' if global_eval else 'skipped'}",
        "  request_auc_mode=per_request_exact",
        "  request_ranking_mode=per_request_exact",
    ]


def _summarize_paths(paths: Any, *, limit: int = 3) -> str:
    rendered = [str(path) for path in paths]
    if len(rendered) <= limit:
        return "[" + ", ".join(rendered) + "]"
    head = ", ".join(rendered[:limit])
    return f"[{head}, ...] (total={len(rendered)})"

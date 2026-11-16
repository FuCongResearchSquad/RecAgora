from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.distributed as dist

from .factory import build_model
from utils.compiler.config import RecAgoraConfig
from utils.runtime.data.request_batch import RequestBatch
from utils.runtime.candidate_mask import (
    accumulate_candidate_mask_counts,
    negative_mask_resume_warning,
    prepare_candidate_mask_step,
    require_candidate_mask_support,
    skip_empty_candidate_update,
)
from utils.runtime.distributed import (
    DistributedContext,
    all_gather_object as _all_gather_object,
    barrier as _dist_barrier,
    destroy_distributed as _destroy_distributed,
    init_distributed as _init_distributed,
    reduce_mean as _reduce_mean,
    reduce_sum as _reduce_sum,
    reduce_min_int as _reduce_min_int,
    resolve_device_for_context as _resolve_device_for_context,
    unwrap_model as _unwrap_model,
)
from utils.runtime.plots import write_training_curves
from utils.runtime.execution import (
    append_jsonl as _append_jsonl,
    autocast as _autocast,
    build_data_loader,
    grad_scaler as _grad_scaler,
    iter_with_progress as _iter_with_progress,
    load_model_weights as _load_model_weights,
    prepare_result_dirs as _prepare_result_dirs,
    write_log as _write_log,
)
from utils.runtime.diagnostics import (
    environment_diagnostic_lines as _environment_diagnostic_lines,
    run_start_line as _run_start_line,
    torchrec_model_diagnostic_lines as _torchrec_model_diagnostic_lines,
)
from utils.runtime.checkpoint.dcp import (
    load_training_checkpoint as _load_training_checkpoint,
    read_model_checkpoint_metadata as _read_model_checkpoint_metadata,
    read_training_checkpoint_metadata as _read_training_checkpoint_metadata,
    save_model_checkpoint as _save_model_state_checkpoint,
    save_training_checkpoint as _save_training_checkpoint,
)
from utils.runtime.checkpoint.retention import (
    CheckpointRetentionResult,
    prune_step_checkpoints_after_completion,
    prune_step_checkpoints as _prune_step_checkpoints,
)
from utils.runtime.checkpoint.resume_contract import (
    RESUME_INPUT_CONTRACT_KEY,
    build_resume_input_contract as _build_resume_input_contract,
    validate_resume_input_contract as _validate_resume_input_contract,
)
from utils.compiler.model_input_contract import (
    build_model_input_contract as _build_model_input_contract,
    data_loader_contract_kwargs as _data_loader_contract_kwargs,
)
from utils.runtime.torchrec import (
    build_torchrec_optimizer as _build_torchrec_optimizer,
    torchrec_mode as _torchrec_mode,
    torchrec_sparse_optimizer_policy as _torchrec_sparse_optimizer_policy,
    uses_torchrec_dmp as _uses_torchrec_dmp,
    wrap_torchrec_dmp as _wrap_torchrec_dmp,
)
from utils.runtime.timing import (
    PendingTimingTrace,
    RankLocalTimingTracer,
    format_timing_trace as _format_timing_trace,
)
from utils.runtime.communication_trace import (
    RankLocalCommunicationProfiler,
    format_communication_trace as _format_communication_trace,
)


PROGRESS_DIAGNOSTIC_PREFIXES = (
    "dedup",
    "neg_",
    "negative_",
    "pos_",
    "positive_",
    "recall@",
    "temperature",
    "valid",
)

# ===== Public API =====


def train(config: RecAgoraConfig) -> dict[str, Any]:
    """Run the configured training job and persist checkpoints, logs, and metrics."""
    started_at = datetime.now().astimezone()
    dist_context = _init_distributed()
    try:
        # Runtime setup: device, output paths, and optional DDP metadata.
        torch.manual_seed(config.train.seed)
        torch_device = _resolve_device_for_context(config.train.device, dist_context)
        use_amp = config.train.amp and torch_device.type == "cuda"

        result_dirs = _prepare_result_dirs(config.train.output_dir)
        model_checkpoint_path = result_dirs["checkpoints"] / "last"
        log_path = result_dirs["logs"] / "train.log"
        hotuser_identity = None
        if config.data.root is not None:
            from utils.runtime.data.hotuser import build_hotuser_index

            hotuser_identity = build_hotuser_index(config.data).identity
        if dist_context.is_main:
            _write_log(
                log_path,
                _run_start_line("train", started_at=started_at),
            )
            if hotuser_identity is not None:
                _write_log(log_path, f"hotuser_data_identity={json.dumps(hotuser_identity, sort_keys=True)}")
        metrics_path = result_dirs["outputs"] / "metrics.jsonl"
        timing_trace_path = result_dirs["outputs"] / "timing_trace.jsonl"
        rank_diagnostics_path = result_dirs["outputs"] / "rank_diagnostics.jsonl"
        communication_trace_path = result_dirs["outputs"] / "communication_trace.jsonl"
        epoch_metrics_path = result_dirs["outputs"] / "epoch_metrics.jsonl"
        runtime_config = config
        timing_tracer = RankLocalTimingTracer(
            enabled=config.train.timing_trace.enabled,
            rank=dist_context.rank,
            target_rank=config.train.timing_trace.rank,
            every_steps=config.train.timing_trace.every_steps,
            resolve_delay_steps=config.train.timing_trace.resolve_delay_steps,
        )
        communication_profiler = RankLocalCommunicationProfiler(
            enabled=config.train.communication_trace.enabled,
            rank=dist_context.rank,
            target_rank=config.train.communication_trace.rank,
            every_steps=config.train.communication_trace.every_steps,
            active_steps=config.train.communication_trace.active_steps,
            export_chrome_trace=config.train.communication_trace.export_chrome_trace,
            output_dir=result_dirs["outputs"],
        )

        # Compile the ranking feature and embedding input contract.
        input_contract = _build_model_input_contract(
            runtime_config.data,
            runtime_config.model,
        )
        resume_input_contract = _build_resume_input_contract(
            input_contract,
            runtime_config.model,
        )
        model = build_model(config.model, input_contract=input_contract)
        require_candidate_mask_support(model, config.train.negative_mask_fields)
        public_model = getattr(model, "model", None)
        if (
            bool(getattr(public_model, "requires_init_from", False))
            and config.train.resume_from is None
            and config.train.init_from is None
        ):
            raise ValueError(
                "this model configuration requires train.init_from for a cold start"
            )
        model = model.to(torch_device)
        model = _wrap_torchrec_dmp(
            model,
            config=config.model,
            context=dist_context,
            device=torch_device,
        )
        optimizer = _build_optimizer(model, config)
        scaler = _grad_scaler(use_amp)

        losses: list[float] = []
        last_loss_value: float | None = None
        last_loss_tensor: torch.Tensor | None = None
        last_metrics_step = 0
        global_step = 0
        optimizer_updates_this_run = 0
        skipped_empty_batches = 0
        candidate_mask_totals: dict[str, Any] = {}
        last_checkpoint_step: int | None = None
        start_epoch = 1
        last_epoch = 0
        last_epoch_step = 0
        resume_rank_batch_offset = 0
        resume_global_batch_id: int | None = None
        resume_global_batch_stop: int | None = None
        resume_previous_world_size: int | None = None
        resume_uses_global_cursor = False
        elastic_world_size_changed = False
        if config.train.resume_from is not None:
            resume_metadata = _read_training_checkpoint_metadata(
                config.train.resume_from
            )
            mask_resume_warning = negative_mask_resume_warning(
                resume_metadata, config.train.negative_mask_fields
            )
            if mask_resume_warning and dist_context.is_main:
                _write_log(log_path, f"candidate_mask_resume_warning={mask_resume_warning}")
            resume_contract_validation = _validate_resume_input_contract(
                resume_metadata,
                current_contract=resume_input_contract,
            )
            if dist_context.is_main:
                _write_log(
                    log_path,
                    "resume_input_contract "
                    f"status={resume_contract_validation.status}",
                )
                for warning in resume_contract_validation.warnings:
                    _write_log(
                        log_path,
                        f"resume_input_contract_warning={warning}",
                    )
            _validate_checkpoint_model_contract(
                resume_metadata,
                config=runtime_config,
                model=model,
                require_execution_match=True,
            )
            _validate_checkpoint_data_state(
                resume_metadata,
                config=runtime_config,
                distributed_context=dist_context,
                hotuser_identity=hotuser_identity,
            )
            checkpoint_world_size = _checkpoint_recorded_world_size(resume_metadata)
            elastic_world_size_changed = (
                config.train.resume_world_size_policy == "elastic"
                and checkpoint_world_size is not None
                and checkpoint_world_size != int(dist_context.world_size)
            )
            resume_state = _load_checkpoint(
                config.train.resume_from,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                restore_rng=not elastic_world_size_changed,
                distributed_context=dist_context,
                rank_state_rank=0 if elastic_world_size_changed else None,
            )
            losses = [float(value) for value in resume_state.get("losses", [])]
            resumed_last_loss = resume_state.get("last_loss")
            last_loss_value = (
                float(resumed_last_loss)
                if resumed_last_loss is not None
                else (losses[-1] if losses else None)
            )
            global_step = int(resume_state.get("global_step", 0))
            start_epoch = _checkpoint_next_epoch(resume_state)
            resume_uses_global_cursor = (
                elastic_world_size_changed
                or _checkpoint_has_stable_global_cursor(resume_state)
            )
            if resume_uses_global_cursor:
                resume_global_batch_id = _checkpoint_next_global_batch_id(
                    resume_state
                )
                resume_global_batch_stop = _checkpoint_epoch_global_batch_stop(
                    resume_state
                )
                resume_previous_world_size = checkpoint_world_size
                resume_rank_batch_offset = (
                    resume_global_batch_id // int(dist_context.world_size)
                )
            else:
                resume_rank_batch_offset = _checkpoint_next_rank_batch_offset(
                    resume_state
                )
            if elastic_world_size_changed:
                elastic_seed = _elastic_resume_seed(
                    base_seed=config.train.seed,
                    global_step=global_step,
                    rank=dist_context.rank,
                )
            last_epoch = start_epoch
            last_epoch_step = resume_rank_batch_offset
            if dist_context.is_main:
                _write_log(
                    log_path,
                    f"resumed checkpoint={config.train.resume_from} "
                    f"epoch={start_epoch} global_step={global_step} "
                    f"rank_batch_offset={resume_rank_batch_offset} "
                    f"parquet_start_file={runtime_config.data.parquet_start_file}",
                )
                if resume_uses_global_cursor:
                    _write_log(
                        log_path,
                        "stable_global_cursor_resume "
                        f"next_global_batch_id={resume_global_batch_id} "
                        f"epoch_global_batch_stop={resume_global_batch_stop} "
                        f"checkpoint_world_size={checkpoint_world_size} "
                        f"current_world_size={dist_context.world_size}",
                    )
                if elastic_world_size_changed:
                    _write_log(
                        log_path,
                        "elastic_resume "
                        f"old_world_size={checkpoint_world_size} "
                        f"new_world_size={dist_context.world_size} "
                        f"next_global_batch_id={resume_global_batch_id} "
                        f"epoch_global_batch_stop={resume_global_batch_stop} "
                        "rng_restored=false rng_reseeded=true "
                        f"rng_seed={elastic_seed} scaler_source_rank=0 "
                        "global_batch_size_changed=true",
                    )
        elif config.train.init_from is not None:
            init_metadata = _read_model_checkpoint_metadata(config.train.init_from)
            source_model = init_metadata.get("model_config", {})
            source_params = source_model.get("params", {}) if isinstance(source_model, Mapping) else {}
            source_objectives = source_params.get("objectives", {}) if isinstance(source_params, Mapping) else {}
            source_is_v2_p0 = (
                isinstance(source_model, Mapping)
                and source_model.get("name") == "onerankv2"
                and isinstance(source_objectives, Mapping)
                and source_objectives.get("mode") == "next_event"
            )
            _validate_checkpoint_model_contract(
                init_metadata,
                config=runtime_config,
                model=model,
                require_execution_match=False,
            )
            _load_model_weights(
                config.train.init_from,
                model=model,
                distributed_context=dist_context,
                allow_missing_acr=(
                    bool(getattr(public_model, "acr_enabled", False))
                    and source_is_v2_p0
                ),
            )
            if dist_context.is_main:
                _write_log(
                    log_path,
                    f"initialized model checkpoint={config.train.init_from}",
                )

        # Main epoch/step loop. The loader is rebuilt per epoch so resume offsets
        # can be applied without mutating the dataset contract.
        model.train()
        run_start_logged = False
        loader = None
        min_loader_len = 0
        current_epoch_global_batch_stop = 0
        current_global_full_batches = 0
        epoch_start_offset = 0
        last_data_cursor: dict[str, Any] | None = None
        max_steps_reached = (
            runtime_config.train.max_steps is not None
            and global_step >= runtime_config.train.max_steps
        )
        for epoch in range(start_epoch, runtime_config.train.epochs + 1):
            if max_steps_reached:
                break
            epoch_start_offset = (
                resume_rank_batch_offset if epoch == start_epoch else 0
            )
            loader = build_data_loader(
                runtime_config.data,
                runtime_config.train,
                torch_device=torch_device,
                distributed_context=dist_context,
                start_rank_batch_index=epoch_start_offset,
                start_global_batch_id=(
                    resume_global_batch_id
                    if resume_uses_global_cursor and epoch == start_epoch
                    else None
                ),
                resume_global_batch_stop=(
                    resume_global_batch_stop
                    if resume_uses_global_cursor and epoch == start_epoch
                    else None
                ),
                resume_previous_world_size=(
                    resume_previous_world_size
                    if resume_uses_global_cursor and epoch == start_epoch
                    else None
                ),
                **_data_loader_contract_kwargs(input_contract),
            )
            if hotuser_identity is not None and loader.dataset.data_identity != hotuser_identity:
                raise ValueError("HotUser data changed after training startup")
            local_loader_plan = _loader_batch_plan(
                loader,
                num_workers=runtime_config.train.num_workers,
            )
            local_loader_len = int(local_loader_plan["planned_batches"])
            current_epoch_global_batch_stop = int(
                local_loader_plan.get(
                    "epoch_global_batch_stop",
                    local_loader_plan.get("usable_global_batches", 0),
                )
            )
            current_global_full_batches = int(
                local_loader_plan.get(
                    "global_full_batches",
                    current_epoch_global_batch_stop,
                )
            )
            gathered_loader_plans = _all_gather_object(local_loader_plan, dist_context)
            min_loader_len = _reduce_min_int(
                local_loader_len,
                dist_context,
                device=torch_device,
            )
            if min_loader_len == 0:
                if epoch_start_offset > 0:
                    if dist_context.is_main:
                        _write_log(
                            log_path,
                            f"resume_epoch_already_complete epoch={epoch} "
                            f"rank_batch_offset={epoch_start_offset}",
                        )
                    continue
                raise ValueError("Training DataLoader is empty on at least one rank")

            if dist_context.is_main:
                if not run_start_logged:
                    for line in _train_start_lines(
                        config=runtime_config,
                        model=model,
                        torch_device=torch_device,
                        use_amp=use_amp,
                        model_checkpoint_path=model_checkpoint_path,
                        distributed_context=dist_context,
                        local_loader_len=local_loader_len,
                        min_loader_len=min_loader_len,
                        input_contract=input_contract,
                    ):
                        _write_log(log_path, line)
                    run_start_logged = True
                _write_log(
                    log_path,
                    f"epoch_loader_start epoch={epoch} "
                    f"rank_batch_offset={epoch_start_offset} "
                    f"local_loader_len={local_loader_len} "
                    f"min_loader_len={min_loader_len}",
                )
                for line in _loader_batch_plan_lines(
                    gathered_loader_plans,
                    min_planned_batches=min_loader_len,
                ):
                    _write_log(log_path, line)

            epoch_loader = _iter_with_progress(
                loader,
                enabled=runtime_config.data.index_progress and dist_context.is_main,
                desc=f"train epoch {epoch}",
            )
            epoch_local_loss_sum = torch.zeros(
                (),
                dtype=torch.float64,
                device=torch_device,
            )
            epoch_local_sample_count = torch.zeros(
                (),
                dtype=torch.float64,
                device=torch_device,
            )
            epoch_local_named_loss_sums: dict[str, torch.Tensor] = {}
            max_epoch_step = epoch_start_offset + min_loader_len

            def save_due_checkpoints(
                epoch_step: int, data_cursor: Mapping[str, Any] | None
            ) -> None:
                nonlocal last_checkpoint_step
                if _should_save_step_checkpoint(
                    global_step,
                    runtime_config.train.checkpoint_every_steps,
                ):
                    step_checkpoint_path = (
                        result_dirs["checkpoints"] / f"step_{global_step}"
                    )
                    _save_resume_checkpoint(
                        step_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=runtime_config,
                        device=torch_device,
                        use_amp=use_amp,
                        losses=losses,
                        last_loss=last_loss_value,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        global_step=global_step,
                        next_epoch=epoch,
                        next_rank_batch_offset=_next_rank_batch_offset(
                            epoch_step,
                            data_cursor,
                        ),
                        next_global_batch_id=_next_global_batch_id(
                            epoch_step,
                            data_cursor,
                            distributed_context=dist_context,
                        ),
                        epoch_global_batch_stop=current_epoch_global_batch_stop,
                        resume_input_contract=resume_input_contract,
                        distributed_context=dist_context,
                        hotuser_identity=hotuser_identity,
                    )
                    _save_model_checkpoint(
                        model_checkpoint_path,
                        model=model,
                        config=runtime_config,
                        device=torch_device,
                        use_amp=use_amp,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        global_step=global_step,
                        distributed_context=dist_context,
                    )
                    retention = _prune_saved_checkpoints(
                        result_dirs["checkpoints"],
                        current_step=global_step,
                        max_to_keep=runtime_config.train.checkpoint_max_to_keep,
                        distributed_context=dist_context,
                    )
                    last_checkpoint_step = global_step
                    if dist_context.is_main:
                        _write_log(
                            log_path,
                            f"resume_checkpoint_saved={step_checkpoint_path} "
                            f"model_checkpoint_saved={model_checkpoint_path}",
                        )
                        _write_checkpoint_retention_log(log_path, retention)

            previous_step_finished_at = time.perf_counter()
            for epoch_step, batch in enumerate(
                epoch_loader,
                start=epoch_start_offset + 1,
            ):
                step_started_at = time.perf_counter()
                fetch_batch_sec = step_started_at - previous_step_finished_at
                if epoch_step > max_epoch_step:
                    break
                last_epoch = epoch
                last_epoch_step = epoch_step
                global_step += 1
                communication_session = communication_profiler.start_step(global_step)
                should_write_timing_trace = timing_tracer.should_record(global_step)
                batch, data_cursor = _strip_batch_train_metadata(batch)
                last_data_cursor = data_cursor
                loss_owner = _unwrap_model(model)
                _set_model_communication_trace_enabled(
                    loss_owner,
                    enabled=communication_session is not None,
                )
                _set_batch_prepare_timing_enabled(
                    loss_owner,
                    enabled=should_write_timing_trace,
                )
                prepare_model_batch_started_at = (
                    time.perf_counter() if should_write_timing_trace else None
                )
                model_batch = _prepare_model_batch(loss_owner, batch)
                model_batch, mask_counts = prepare_candidate_mask_step(
                    model_batch,
                    context=dist_context,
                    device=torch_device,
                    task_names=runtime_config.data.label_names,
                )
                if mask_counts is not None:
                    accumulate_candidate_mask_counts(candidate_mask_totals, mask_counts)
                    if skip_empty_candidate_update(loss_owner, model_batch):
                        # All ranks make this decision from the same global count,
                        # before DMP forward or fused optimizer-in-backward work.
                        skipped_empty_batches += 1
                        _set_model_communication_trace_enabled(loss_owner, enabled=False)
                        empty_trace = communication_profiler.finish_step(communication_session)
                        if empty_trace is not None:
                            empty_trace["skipped_empty_candidate_batch"] = True
                            _append_jsonl(communication_trace_path, empty_trace)
                        if dist_context.is_main:
                            _append_jsonl(metrics_path, {
                                "event": "masked_empty_batch",
                                "epoch": epoch,
                                "step": global_step,
                                "optimizer_updated": False,
                                "candidate_mask": mask_counts,
                            })
                            _write_log(log_path, f"masked_empty_batch step={global_step} optimizer_updated=false")
                        save_due_checkpoints(epoch_step, data_cursor)
                        if runtime_config.train.max_steps is not None and global_step >= runtime_config.train.max_steps:
                            max_steps_reached = True
                            break
                        previous_step_finished_at = time.perf_counter()
                        continue
                batch_prepare_trace = (
                    _batch_prepare_timing_trace(model_batch)
                    if should_write_timing_trace
                    else {}
                )
                prepare_model_batch_sec = (
                    time.perf_counter() - prepare_model_batch_started_at
                    if prepare_model_batch_started_at is not None
                    else 0.0
                )
                device_step_timer = timing_tracer.new_timer(
                    torch_device,
                    enabled=should_write_timing_trace,
                )
                h2d_timer = timing_tracer.new_timer(
                    torch_device,
                    enabled=should_write_timing_trace,
                )
                model_batch = model_batch.to(
                    torch_device,
                    non_blocking=runtime_config.train.pin_memory
                    and torch_device.type == "cuda",
                )
                h2d_timer.stop()
                forward_timer = timing_tracer.new_timer(
                    torch_device,
                    enabled=should_write_timing_trace,
                )
                with _autocast(torch_device, use_amp):
                    _set_model_forward_timing_enabled(
                        loss_owner,
                        enabled=should_write_timing_trace,
                    )
                    with _communication_profile_phase(
                        communication_session,
                        "recagora.model_forward",
                    ):
                        model_output = model(model_batch)
                    embedding_timing_span = _consume_embedding_timing_span(
                        loss_owner
                    )
                    forward_timer.stop()
                    loss_timer = timing_tracer.new_timer(
                        torch_device,
                        enabled=should_write_timing_trace,
                    )
                    compute_loss = getattr(loss_owner, "compute_loss", None)
                    if not callable(compute_loss):
                        raise AttributeError(
                            f"Model {type(loss_owner).__name__} must define compute_loss"
                        )
                    loss, named_losses = compute_loss(
                        model_output,
                        model_batch,
                    )
                    loss_timer.stop()
                    epoch_metric_loss = loss
                    epoch_metric_named_losses = named_losses
                _require_finite_main_loss(
                    loss,
                    step=global_step,
                    distributed_context=dist_context,
                )
                last_loss_tensor = loss.detach()
                request_count = _model_batch_request_count(model_batch)
                epoch_loss_weight = float(request_count)
                epoch_count = float(request_count)
                candidate_mean = (
                    getattr(model_batch, "candidate_loss_context", None) is not None
                    and getattr(public_model, "candidate_mask_loss_reduction", "model_defined") == "candidate_mean"
                )
                if candidate_mean:
                    count_context = model_batch.candidate_loss_context
                    # Undo the backward-only DDP scale to accumulate actual local
                    # numerators; epoch denominator is the retained candidate count.
                    epoch_loss_weight = count_context.global_valid_count / count_context.world_size
                    epoch_count = float(count_context.local_valid_count)
                epoch_local_loss_sum.add_(
                    epoch_metric_loss.detach().to(dtype=torch.float64)
                    * epoch_loss_weight
                )
                epoch_local_sample_count.add_(epoch_count)
                for name, value in epoch_metric_named_losses.items():
                    metric_name = f"loss_{name}"
                    if not _is_progress_loss_name(metric_name):
                        continue
                    if metric_name not in epoch_local_named_loss_sums:
                        epoch_local_named_loss_sums[metric_name] = torch.zeros(
                            (),
                            dtype=torch.float64,
                            device=torch_device,
                        )
                    epoch_local_named_loss_sums[metric_name].add_(
                        _loss_metric_tensor(
                            value,
                            device=torch_device,
                        ).to(dtype=torch.float64)
                        * epoch_loss_weight
                    )

                optimizer.zero_grad(set_to_none=True)
                backward_timer = timing_tracer.new_timer(
                    torch_device,
                    enabled=should_write_timing_trace,
                )
                with _communication_profile_phase(
                    communication_session,
                    "recagora.backward",
                ):
                    scaler.scale(loss).backward()
                if runtime_config.train.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        runtime_config.train.grad_clip_norm,
                    )
                backward_timer.stop()
                optimizer_timer = timing_tracer.new_timer(
                    torch_device,
                    enabled=should_write_timing_trace,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates_this_run += 1
                optimizer_timer.stop()
                device_step_timer.stop()

                should_write_progress = (
                    dist_context.is_main
                    and _should_write_train_metrics(
                        global_step,
                        runtime_config.train.progress_every_steps,
                    )
                )
                should_write_global_metrics = (
                    runtime_config.train.global_metrics.enabled
                    and _should_write_train_metrics(
                        global_step,
                        runtime_config.train.global_metrics.every_steps,
                    )
                )
                should_write_rank_diagnostics = (
                    runtime_config.train.rank_diagnostics.enabled
                    and dist_context.enabled
                    and _should_write_train_metrics(
                        global_step,
                        runtime_config.train.rank_diagnostics.every_steps,
                    )
                )
                reduce_log_sec = 0.0
                metric: dict[str, Any] | None = None
                local_loss_metrics: dict[str, float] | None = None
                global_loss_metrics: dict[str, float] | None = None
                if should_write_global_metrics:
                    reduce_log_started_at = time.perf_counter()
                    with _communication_profile_phase(
                        communication_session,
                        "recagora.global_metrics",
                    ):
                        local_loss_metrics, global_loss_metrics = _reduce_loss_metrics(
                            loss,
                            named_losses,
                            dist_context,
                            torch_device,
                        )
                    reduce_log_sec = time.perf_counter() - reduce_log_started_at

                if (should_write_progress or should_write_rank_diagnostics) and (
                    local_loss_metrics is None
                ):
                    local_loss_metrics = _materialize_local_loss_metrics(
                        loss,
                        named_losses,
                        torch_device,
                    )

                if should_write_global_metrics:
                    if global_loss_metrics is None:
                        raise RuntimeError("global metric reduction was not initialized")
                    loss_value = global_loss_metrics["loss"]
                    losses.append(loss_value)
                    last_loss_value = loss_value
                    last_metrics_step = global_step
                    metric = {
                        "event": "global_metrics",
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "step": global_step,
                        "loss_scope": "global_mean",
                        "loss_global": loss_value,
                        "lr": runtime_config.train.lr,
                        "world_size": dist_context.world_size,
                    }
                    if mask_counts is not None:
                        metric["negative_mask_fields"] = list(runtime_config.train.negative_mask_fields)
                        metric["candidate_mask"] = mask_counts
                        metric["optimizer_updates_this_run"] = optimizer_updates_this_run
                        if candidate_mean:
                            metric["loss_scope"] = "global_valid_candidate_mean"
                    metric.update(
                        {
                            f"{name}_global": value
                            for name, value in global_loss_metrics.items()
                            if name != "loss"
                        }
                    )

                if should_write_rank_diagnostics:
                    if local_loss_metrics is None:
                        raise RuntimeError("rank diagnostics lack local loss metrics")
                    with _communication_profile_phase(
                        communication_session,
                        "recagora.rank_diagnostics",
                    ):
                        gathered_trace = _all_gather_object(
                            _rank_diagnostic_record(
                                rank=dist_context.rank,
                                local_losses=local_loss_metrics,
                                loader=loader,
                            ),
                            dist_context,
                        )
                    if dist_context.is_main:
                        diagnostic_record = {
                            "event": "rank_diagnostics",
                            "step": global_step,
                            "world_size": dist_context.world_size,
                            "ranks": gathered_trace,
                        }
                        _append_jsonl(rank_diagnostics_path, diagnostic_record)
                        _write_log(
                            log_path,
                            f"rank_diagnostics step={global_step} "
                            f"world_size={dist_context.world_size} "
                            f"records={len(gathered_trace)}",
                            mirror_stdout=False,
                        )

                _set_model_communication_trace_enabled(loss_owner, enabled=False)
                communication_trace = communication_profiler.finish_step(
                    communication_session
                )
                if communication_trace is not None:
                    epoch_loader.exclude_sampled_step(
                        time.perf_counter() - step_started_at
                    )
                    communication_trace["excluded_from_steady_state_rate"] = True
                    _append_jsonl(communication_trace_path, communication_trace)
                    _write_log(
                        log_path,
                        _format_communication_trace(communication_trace),
                    )

                if should_write_global_metrics and dist_context.is_main:
                    if metric is None:
                        raise RuntimeError("global metric record was not initialized")
                    _append_jsonl(metrics_path, metric)

                if should_write_progress:
                    if local_loss_metrics is None:
                        raise RuntimeError("rank-0 progress lacks local loss metrics")
                    local_loss_value = local_loss_metrics["loss"]
                    local_named_losses = {
                        name: value
                        for name, value in local_loss_metrics.items()
                        if name != "loss"
                    }
                    epoch_loader.set_postfix(
                        _progress_loss_postfix(local_loss_value, local_named_losses)
                    )
                    timing_text = _format_log_fields(epoch_loader.timing_log_fields())
                    named_text = "".join(
                        f" {name}_local_rank0={value:.6f}"
                        for name, value in sorted(local_named_losses.items())
                    )
                    _write_log(
                        log_path,
                        f"epoch={epoch} step={global_step} epoch_step={epoch_step} "
                        f"loss_scope={'rank0_ddp_contribution' if candidate_mean else 'rank0_local'} "
                        f"loss_local_rank0={local_loss_value:.6f}{named_text} "
                        f"lr={runtime_config.train.lr}{timing_text}",
                        mirror_stdout=False,
                    )

                if should_write_timing_trace:
                    device_timers = {
                        "h2d_ms": h2d_timer,
                        "forward_ms": forward_timer,
                        "loss_ms": loss_timer,
                        "backward_ms": backward_timer,
                        "optimizer_ms": optimizer_timer,
                        "device_step_ms": device_step_timer,
                    }
                    if embedding_timing_span is not None:
                        device_timers["embedding_lookup_ms"] = embedding_timing_span
                    cpu_ms = {
                        "fetch_batch_ms": fetch_batch_sec * 1000.0,
                        "prepare_batch_ms": prepare_model_batch_sec * 1000.0,
                        "global_metric_comm_ms": reduce_log_sec * 1000.0,
                    }
                    cpu_ms.update(
                        {
                            name.removesuffix("_sec") + "_ms": value * 1000.0
                            for name, value in batch_prepare_trace.items()
                        }
                    )
                    timing_tracer.submit(
                        PendingTimingTrace(
                            step=global_step,
                            rank=dist_context.rank,
                            cpu_ms=cpu_ms,
                            device_timers=device_timers,
                            metadata=(
                                _data_timing_trace_fields(data_cursor)
                                if data_cursor is not None
                                else {}
                            ),
                        )
                    )

                for resolved_timing_trace in timing_tracer.resolve_ready(global_step):
                    _append_jsonl(timing_trace_path, resolved_timing_trace)
                    _write_log(
                        log_path,
                        _format_timing_trace(resolved_timing_trace),
                    )
                save_due_checkpoints(epoch_step, data_cursor)
                if (
                    runtime_config.train.max_steps is not None
                    and global_step >= runtime_config.train.max_steps
                ):
                    max_steps_reached = True
                    break
                previous_step_finished_at = time.perf_counter()
            epoch_named_loss_names = sorted(epoch_local_named_loss_sums)
            epoch_global = _reduce_sum(
                torch.stack(
                    (
                        epoch_local_loss_sum,
                        *(
                            epoch_local_named_loss_sums[name]
                            for name in epoch_named_loss_names
                        ),
                        epoch_local_sample_count,
                    )
                ),
                dist_context,
            )
            epoch_values = epoch_global.detach().cpu().tolist()
            epoch_loss_sum_value = epoch_values[0]
            epoch_sample_count_value = epoch_values[-1]
            if dist_context.is_main and epoch_sample_count_value > 0:
                epoch_loss_global = epoch_loss_sum_value / epoch_sample_count_value
                epoch_complete = last_epoch_step >= max_epoch_step
                epoch_record = {
                    "event": "epoch_global_metrics",
                    "epoch": epoch,
                    "step": global_step,
                    "complete": epoch_complete,
                    "loss_scope": "global_sample_weighted_mean",
                    "loss_global": epoch_loss_global,
                    "sample_count_global": int(epoch_sample_count_value),
                }
                if runtime_config.train.negative_mask_fields:
                    epoch_record["negative_mask_fields"] = list(runtime_config.train.negative_mask_fields)
                    if getattr(public_model, "candidate_mask_loss_reduction", "model_defined") == "candidate_mean":
                        epoch_record["loss_scope"] = "global_valid_candidate_mean"
                        epoch_record["valid_candidate_count_global"] = int(epoch_sample_count_value)
                        del epoch_record["sample_count_global"]
                    else:
                        epoch_record["loss_scope"] = "global_sample_weighted_mean"
                epoch_record.update(
                    {
                        f"{name}_global": value / epoch_sample_count_value
                        for name, value in zip(
                            epoch_named_loss_names,
                            epoch_values[1:-1],
                            strict=True,
                        )
                    }
                )
                _append_jsonl(epoch_metrics_path, epoch_record)
                _write_log(
                    log_path,
                    f"epoch_global_metrics epoch={epoch} step={global_step} "
                    f"complete={epoch_complete} "
                    f"loss_scope={epoch_record['loss_scope']} "
                    f"loss_global={epoch_loss_global:.6f} "
                    f"{'valid_candidate_count_global' if epoch_record['loss_scope'] == 'global_valid_candidate_mean' else 'sample_count_global'}={int(epoch_sample_count_value)}",
                    mirror_stdout=False,
                )
            if max_steps_reached:
                if dist_context.is_main:
                    _write_log(
                        log_path,
                        f"max_steps_reached={runtime_config.train.max_steps}",
                    )
                break

        # Final resume and model-only DCP checkpoints are collective; reporting
        # remains rank0-only.
        final_next_epoch, final_next_rank_batch_offset = _final_data_position(
            epoch=last_epoch,
            epoch_step=last_epoch_step,
            epoch_start_offset=epoch_start_offset,
            min_loader_len=min_loader_len,
            stopped_mid_epoch=max_steps_reached,
        )
        if last_loss_tensor is not None and last_metrics_step != global_step:
            _, final_loss_metrics = _reduce_loss_metrics(
                last_loss_tensor,
                {},
                dist_context,
                torch_device,
            )
            last_loss_value = final_loss_metrics["loss"]
        resume_checkpoint_path = (
            result_dirs["checkpoints"] / f"step_{global_step}"
        )
        if last_checkpoint_step != global_step:
            _save_resume_checkpoint(
                resume_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=runtime_config,
                device=torch_device,
                use_amp=use_amp,
                losses=losses,
                last_loss=last_loss_value,
                epoch=last_epoch,
                epoch_step=last_epoch_step,
                global_step=global_step,
                next_epoch=final_next_epoch,
                next_rank_batch_offset=final_next_rank_batch_offset,
                next_global_batch_id=(
                    0
                    if final_next_epoch > last_epoch
                    else _next_global_batch_id(
                        last_epoch_step,
                        last_data_cursor,
                        distributed_context=dist_context,
                    )
                ),
                epoch_global_batch_stop=(
                    current_global_full_batches
                    if final_next_epoch > last_epoch
                    else current_epoch_global_batch_stop
                ),
                resume_input_contract=resume_input_contract,
                distributed_context=dist_context,
                hotuser_identity=hotuser_identity,
            )
            _save_model_checkpoint(
                model_checkpoint_path,
                model=model,
                config=runtime_config,
                device=torch_device,
                use_amp=use_amp,
                epoch=last_epoch,
                epoch_step=last_epoch_step,
                global_step=global_step,
                distributed_context=dist_context,
            )
            retention = _prune_saved_checkpoints(
                result_dirs["checkpoints"],
                current_step=global_step,
                max_to_keep=runtime_config.train.checkpoint_max_to_keep,
                distributed_context=dist_context,
            )
            if dist_context.is_main:
                _write_log(
                    log_path,
                    f"resume_checkpoint_saved={resume_checkpoint_path} "
                    f"model_checkpoint_saved={model_checkpoint_path}",
                )
                _write_checkpoint_retention_log(log_path, retention)
        if dist_context.is_main:
            curve_paths = _write_training_curves(
                metrics_path=metrics_path,
                output_dir=result_dirs["outputs"],
                log_path=log_path,
            )
        else:
            curve_paths = {
                "plot_path": None,
                "summary_path": None,
            }

        data_stats = _loader_stats(loader) if loader is not None else None
        if data_stats is not None and dist_context.is_main:
            stats_text = "data_stats " + " ".join(
                f"{key}={value}" for key, value in sorted(data_stats.items())
            )
            _write_log(
                log_path,
                stats_text,
            )
        for resolved_timing_trace in timing_tracer.resolve_ready(
            global_step + runtime_config.train.timing_trace.resolve_delay_steps
        ):
            _append_jsonl(timing_trace_path, resolved_timing_trace)
            _write_log(log_path, _format_timing_trace(resolved_timing_trace))
        if timing_tracer.pending_count > 0:
            _write_log(
                log_path,
                f"timing_trace_unresolved count={timing_tracer.pending_count} "
                "reason=cuda_events_not_ready no_synchronize=true",
                mirror_stdout=False,
            )
        completion_cleanup = _prune_completed_checkpoints(
            result_dirs["checkpoints"],
            current_step=global_step,
            distributed_context=dist_context,
        )
        if dist_context.is_main:
            _write_checkpoint_retention_log(
                log_path,
                completion_cleanup,
                event="checkpoint_completion_cleanup",
            )
        _dist_barrier(dist_context)

        if runtime_config.train.negative_mask_fields and dist_context.is_main:
            summary = {
                "event": "candidate_mask_summary",
                "step": global_step,
                "negative_mask_fields": list(runtime_config.train.negative_mask_fields),
                "candidate_mask": candidate_mask_totals,
                "optimizer_updates_this_run": optimizer_updates_this_run,
                "skipped_empty_batches": skipped_empty_batches,
            }
            _append_jsonl(metrics_path, summary)
            _write_log(
                log_path,
                f"candidate_mask_summary optimizer_updates={optimizer_updates_this_run} "
                f"skipped_empty_batches={skipped_empty_batches} "
                f"valid_candidates={candidate_mask_totals.get('valid_candidates', 0)} "
                f"masked_candidates={candidate_mask_totals.get('masked_candidates', 0)}",
            )

        return {
            "steps": global_step,
            **({
                "negative_mask_fields": list(runtime_config.train.negative_mask_fields),
                "candidate_mask": candidate_mask_totals,
                "optimizer_updates_this_run": optimizer_updates_this_run,
                "skipped_empty_batches": skipped_empty_batches,
            } if runtime_config.train.negative_mask_fields else {}),
            "epochs": runtime_config.train.epochs,
            "last_loss": last_loss_value,
            "checkpoint_path": str(model_checkpoint_path),
            "resume_checkpoint_path": None,
            "resume_checkpoints_deleted": [
                str(path) for path in completion_cleanup.deleted
            ],
            "resume_checkpoints_skipped": [
                str(path) for path in completion_cleanup.skipped
            ],
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "timing_trace_path": str(timing_trace_path),
            "rank_diagnostics_path": str(rank_diagnostics_path),
            "communication_trace_path": str(communication_trace_path),
            "epoch_metrics_path": str(epoch_metrics_path),
            "device": str(torch_device),
            "amp_enabled": use_amp,
            "global_step": global_step,
            "curve_path": curve_paths["plot_path"],
            "summary_path": curve_paths["summary_path"],
            "data_stats": data_stats,
            "distributed": dist_context.enabled,
            "rank": dist_context.rank,
            "world_size": dist_context.world_size,
        }
    finally:
        _destroy_distributed(dist_context)


# ===== Batch And Loss Helpers =====


def _require_finite_main_loss(
    loss: torch.Tensor,
    *,
    step: int,
    distributed_context: DistributedContext,
) -> None:
    """Abort the process group before backward on a local NaN or Inf loss."""

    if not isinstance(loss, torch.Tensor):
        raise TypeError(
            f"main loss must be a Tensor before backward, got {type(loss).__name__}"
        )
    if bool(torch.isfinite(loss.detach()).all().item()):
        return

    error = FloatingPointError(
        "non-finite main loss before backward: "
        f"step={int(step)} rank={distributed_context.rank} "
        f"dtype={loss.dtype} shape={tuple(loss.shape)}"
    )
    if (
        distributed_context.enabled
        and dist.is_available()
        and dist.is_initialized()
    ):
        print(str(error), file=sys.stderr, flush=True)
        abort = getattr(dist.group.WORLD, "abort", None)
        if not callable(abort):
            raise RuntimeError(
                "initialized process group does not support abort()"
            ) from error
        try:
            abort()
        except Exception as abort_error:
            raise error from abort_error
    raise error


def _should_save_step_checkpoint(
    global_step: int,
    checkpoint_every_steps: int | None,
) -> bool:
    return (
        checkpoint_every_steps is not None
        and global_step > 0
        and global_step % checkpoint_every_steps == 0
    )


def _prune_saved_checkpoints(
    checkpoints_dir: Path,
    *,
    current_step: int,
    max_to_keep: int,
    distributed_context: DistributedContext,
) -> CheckpointRetentionResult:
    return _prune_step_checkpoints(
        checkpoints_dir,
        current_step=current_step,
        max_to_keep=max_to_keep,
        distributed_context=distributed_context,
    )


def _prune_completed_checkpoints(
    checkpoints_dir: Path,
    *,
    current_step: int,
    distributed_context: DistributedContext,
) -> CheckpointRetentionResult:
    return prune_step_checkpoints_after_completion(
        checkpoints_dir,
        current_step=current_step,
        distributed_context=distributed_context,
    )


def _write_checkpoint_retention_log(
    log_path: Path,
    result: CheckpointRetentionResult,
    *,
    event: str = "checkpoint_retention",
) -> None:
    kept = ",".join(path.name for path in result.kept) or "none"
    deleted = ",".join(path.name for path in result.deleted) or "none"
    skipped = ",".join(path.name for path in result.skipped) or "none"
    _write_log(
        log_path,
        f"{event} kept={kept} deleted={deleted} skipped={skipped}",
    )


def _loader_stats(loader: Any) -> dict[str, int] | None:
    dataset = getattr(loader, "dataset", None)
    stats = getattr(dataset, "stats", None)
    if isinstance(stats, dict):
        return {str(key): int(value) for key, value in stats.items()}
    return None


def _strip_batch_train_metadata(
    batch: Any,
) -> tuple[Any, dict[str, Any] | None]:
    if isinstance(batch, RequestBatch):
        cursor = batch.metadata.cursor
        return (
            batch,
            dict(cursor) if isinstance(cursor, dict) else None,
        )
    if not isinstance(batch, dict):
        return batch, None
    cursor = batch.get("_data_cursor")
    out = dict(batch)
    out.pop("_data_cursor", None)
    out.pop("_source_path", None)
    return out, cursor if isinstance(cursor, dict) else None


def _prepare_model_batch(model: torch.nn.Module, batch: Any) -> Any:
    prepare = getattr(model, "prepare_model_batch", None)
    if not callable(prepare):
        raise AttributeError(
            f"Model {type(model).__name__} must define prepare_model_batch"
        )
    return prepare(batch)


def _model_batch_request_count(batch: Any) -> int:
    labels = getattr(batch, "labels", None)
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("prepared model batch must expose non-empty labels")
    first = next(iter(labels.values()))
    if not isinstance(first, torch.Tensor) or first.ndim == 0:
        raise ValueError("prepared model batch labels must have a batch dimension")
    return int(first.shape[0])


def _set_batch_prepare_timing_enabled(
    model: torch.nn.Module,
    *,
    enabled: bool,
) -> None:
    configure = getattr(model, "set_batch_prepare_timing_enabled", None)
    if callable(configure):
        configure(enabled)


def _batch_prepare_timing_trace(batch: Any) -> dict[str, float]:
    raw_trace = getattr(batch, "preparation_timing", None)
    if not isinstance(raw_trace, Mapping):
        return {}
    return {
        str(name): float(value)
        for name, value in raw_trace.items()
        if str(name).startswith("prepare_")
        and str(name).endswith("_sec")
        and isinstance(value, (float, int))
        and not isinstance(value, bool)
    }


def _set_model_forward_timing_enabled(
    model: torch.nn.Module,
    *,
    enabled: bool,
) -> None:
    configure = getattr(model, "set_forward_timing_enabled", None)
    if callable(configure):
        configure(enabled)


def _set_model_communication_trace_enabled(
    model: torch.nn.Module,
    *,
    enabled: bool,
) -> None:
    configure = getattr(model, "set_communication_trace_enabled", None)
    if callable(configure):
        configure(enabled)


def _communication_profile_phase(session: Any | None, name: str):
    if session is None:
        return nullcontext()
    return torch.profiler.record_function(name)


def _consume_embedding_timing_span(model: torch.nn.Module) -> Any | None:
    consume = getattr(model, "consume_embedding_timing_span", None)
    if not callable(consume):
        return None
    return consume()


def _next_rank_batch_offset(
    epoch_step: int,
    data_cursor: dict[str, Any] | None,
) -> int:
    if data_cursor is None:
        return int(epoch_step)
    return int(data_cursor.get("next_rank_batch_index", epoch_step))


def _next_global_batch_id(
    epoch_step: int,
    data_cursor: dict[str, Any] | None,
    *,
    distributed_context: DistributedContext,
) -> int:
    if data_cursor is None or data_cursor.get("global_batch_id") is None:
        return int(epoch_step) * int(distributed_context.world_size)
    return (
        int(data_cursor["global_batch_id"])
        - int(distributed_context.rank)
        + int(distributed_context.world_size)
    )


def _checkpoint_next_epoch(checkpoint: dict[str, Any]) -> int:
    data_state = checkpoint.get("data_state")
    if isinstance(data_state, dict) and data_state.get("next_epoch") is not None:
        return max(1, int(data_state["next_epoch"]))
    return max(1, int(checkpoint.get("epoch", 1)))


def _checkpoint_next_rank_batch_offset(checkpoint: dict[str, Any]) -> int:
    data_state = checkpoint.get("data_state")
    if isinstance(data_state, dict):
        for key in ("next_rank_batch_offset", "next_rank_batch_index"):
            if data_state.get(key) is not None:
                return max(0, int(data_state[key]))
    return max(0, int(checkpoint.get("epoch_step", 0)))


def _checkpoint_recorded_world_size(checkpoint: Mapping[str, Any]) -> int | None:
    data_state = checkpoint.get("data_state")
    data_world_size = (
        data_state.get("world_size") if isinstance(data_state, Mapping) else None
    )
    values = {
        int(value)
        for value in (
            checkpoint.get("rank_state_world_size"),
            data_world_size,
        )
        if value is not None
    }
    if len(values) > 1:
        raise ValueError(
            "Checkpoint world_size metadata is inconsistent: "
            f"{sorted(values)}"
        )
    return next(iter(values), None)


def _checkpoint_next_global_batch_id(checkpoint: Mapping[str, Any]) -> int:
    data_state = checkpoint.get("data_state")
    if isinstance(data_state, Mapping):
        value = data_state.get("next_global_batch_id")
        if value is not None:
            return max(0, int(value))
    world_size = _checkpoint_recorded_world_size(checkpoint)
    if world_size is None:
        raise ValueError(
            "Elastic resume requires checkpoint world_size metadata to infer "
            "the global data cursor"
        )
    return _checkpoint_next_rank_batch_offset(dict(checkpoint)) * world_size


def _checkpoint_epoch_global_batch_stop(
    checkpoint: Mapping[str, Any],
) -> int | None:
    data_state = checkpoint.get("data_state")
    if not isinstance(data_state, Mapping):
        return None
    value = data_state.get("epoch_global_batch_stop")
    return None if value is None else max(0, int(value))


def _checkpoint_has_stable_global_cursor(checkpoint: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint carries the restart-stable epoch cursor pair."""

    data_state = checkpoint.get("data_state")
    if not isinstance(data_state, Mapping):
        return False
    has_cursor = data_state.get("next_global_batch_id") is not None
    has_stop = data_state.get("epoch_global_batch_stop") is not None
    if has_cursor != has_stop:
        raise ValueError(
            "Checkpoint stable global cursor metadata is incomplete: "
            "next_global_batch_id and epoch_global_batch_stop must be saved together"
        )
    return has_cursor


def _elastic_resume_seed(*, base_seed: int, global_step: int, rank: int) -> int:
    seed = int(base_seed) + int(global_step) + int(rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _validate_checkpoint_data_state(
    checkpoint: dict[str, Any],
    *,
    config: RecAgoraConfig,
    distributed_context: DistributedContext,
    hotuser_identity: Mapping[str, Any] | None = None,
) -> None:
    data_state = checkpoint.get("data_state")
    if hotuser_identity is not None:
        saved_identity = data_state.get("hotuser_identity") if isinstance(data_state, Mapping) else None
        if saved_identity != hotuser_identity:
            raise ValueError("HotUser full resume requires the same root, split and file identity; use init_from for model-only weights")
    has_stable_global_cursor = _checkpoint_has_stable_global_cursor(checkpoint)
    checkpoint_world_size = _checkpoint_recorded_world_size(checkpoint)
    if (
        config.train.resume_world_size_policy == "elastic"
        and checkpoint_world_size is None
    ):
        raise ValueError("Elastic resume requires checkpoint world_size metadata")
    world_size_changed = (
        checkpoint_world_size is not None
        and checkpoint_world_size != int(distributed_context.world_size)
    )
    if world_size_changed and config.train.resume_world_size_policy == "strict":
        raise ValueError(
            "Cannot resume checkpoint state with a different world_size: "
            f"checkpoint={checkpoint_world_size} "
            f"current={distributed_context.world_size}"
        )
    if not isinstance(data_state, dict):
        if config.train.resume_world_size_policy == "elastic":
            _validate_elastic_resume_data_identity(checkpoint, config=config)
        return
    batch_size = data_state.get("batch_size")
    if batch_size is not None and int(batch_size) != int(config.train.batch_size):
        raise ValueError(
            "Cannot resume global-sample data cursor with a different batch_size: "
            f"checkpoint={batch_size} current={config.train.batch_size}"
        )
    if (
        config.train.resume_world_size_policy == "elastic"
        or has_stable_global_cursor
    ):
        _validate_elastic_resume_data_identity(checkpoint, config=config)


def _validate_elastic_resume_data_identity(
    checkpoint: Mapping[str, Any],
    *,
    config: RecAgoraConfig,
) -> None:
    data_state = checkpoint.get("data_state")
    saved_data_config = checkpoint.get("data_config")
    saved_paths: Any = None
    saved_start_file: Any = None
    if isinstance(data_state, Mapping):
        saved_paths = data_state.get("data_paths")
        saved_start_file = data_state.get("parquet_start_file")
    if isinstance(saved_data_config, Mapping):
        if saved_paths is None:
            saved_paths = saved_data_config.get("paths")
        if saved_start_file is None:
            saved_start_file = saved_data_config.get("parquet_start_file")
    if not isinstance(saved_paths, (list, tuple)):
        raise ValueError(
            "Elastic resume requires checkpoint data path metadata"
        )
    checkpoint_paths = tuple(str(value) for value in saved_paths)
    current_paths = tuple(str(value) for value in config.data.paths)
    if checkpoint_paths != current_paths:
        raise ValueError(
            "Elastic resume requires identical ordered data paths: "
            f"checkpoint={checkpoint_paths!r} current={current_paths!r}"
        )
    checkpoint_start = None if saved_start_file is None else str(saved_start_file)
    current_start = (
        None
        if config.data.parquet_start_file is None
        else str(config.data.parquet_start_file)
    )
    if checkpoint_start != current_start:
        raise ValueError(
            "Elastic resume requires identical parquet_start_file: "
            f"checkpoint={checkpoint_start!r} current={current_start!r}"
        )


def _validate_checkpoint_model_contract(
    checkpoint: Mapping[str, Any],
    *,
    config: RecAgoraConfig,
    model: torch.nn.Module,
    require_execution_match: bool = False,
) -> None:
    del config, model, require_execution_match
    contract = checkpoint.get("arena_model_contract")
    if contract is None:
        # Ranking checkpoints published before this additive metadata remain valid.
        return
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint arena_model_contract must be a mapping")
    if contract.get("interface") != "ranking":
        raise ValueError(
            "checkpoint model interface does not match current ranking model"
        )


def _final_data_position(
    *,
    epoch: int,
    epoch_step: int,
    epoch_start_offset: int,
    min_loader_len: int,
    stopped_mid_epoch: bool,
) -> tuple[int, int]:
    if epoch <= 0:
        return 1, 0
    epoch_stop = epoch_start_offset + min_loader_len
    if not stopped_mid_epoch and min_loader_len > 0 and epoch_step >= epoch_stop:
        return epoch + 1, 0
    return epoch, max(0, int(epoch_step))


def _loader_batch_plan(loader: Any, *, num_workers: int) -> dict[str, Any]:
    advertised_len = len(loader)
    dataset = getattr(loader, "dataset", None)
    planned_batch_stats = getattr(dataset, "planned_batch_stats", None)
    if callable(planned_batch_stats):
        plan = dict(planned_batch_stats(num_workers))
        plan["advertised_len"] = int(plan.get("advertised_len", advertised_len))
        plan["planned_batches"] = int(plan["planned_batches"])
        plan["batch_count_source"] = "planned_batch_stats"
        return plan
    return {
        "rank": 0,
        "world_size": 1,
        "batch_size": None,
        "num_workers": max(1, int(num_workers)),
        "sharding": None,
        "advertised_len": int(advertised_len),
        "rank_rows": None,
        "planned_batches": int(advertised_len),
        "planned_rows_emitted": None,
        "planned_rows_dropped_tail": None,
        "worker_stats": [],
        "batch_count_source": "len_loader",
    }


def _loader_batch_plan_lines(
    plans_by_rank: list[Any],
    *,
    min_planned_batches: int,
) -> list[str]:
    plans = [item for item in plans_by_rank if isinstance(item, dict)]
    if not plans:
        return [
            "loader_batch_plan_summary ranks=0 "
            f"min_planned_batches={min_planned_batches}"
        ]
    planned_values = [int(item["planned_batches"]) for item in plans]
    advertised_values = [int(item["advertised_len"]) for item in plans]
    adjusted = any(
        int(item["planned_batches"]) != int(item["advertised_len"])
        for item in plans
    )
    lines = [
        (
            "loader_batch_plan_summary "
            f"ranks={len(plans)} "
            f"min_planned_batches={min_planned_batches} "
            f"planned_mismatch={len(set(planned_values)) > 1} "
            f"advertised_mismatch={len(set(advertised_values)) > 1} "
            f"batch_count_adjusted={adjusted}"
        )
    ]
    for item in sorted(plans, key=lambda value: int(value.get("rank", 0))):
        worker_batches = [
            int(worker["planned_batches"])
            for worker in item.get("worker_stats", [])
        ]
        worker_tail_rows = [
            int(worker["planned_rows_dropped_tail"])
            for worker in item.get("worker_stats", [])
        ]
        lines.append(
            "loader_batch_plan_rank "
            f"rank={item.get('rank')} "
            f"world_size={item.get('world_size')} "
            f"batch_size={item.get('batch_size')} "
            f"num_workers={item.get('num_workers')} "
            f"sharding={item.get('sharding')} "
            f"source={item.get('batch_count_source')} "
            f"advertised_len={item.get('advertised_len')} "
            f"planned_batches={item.get('planned_batches')} "
            f"rank_rows={item.get('rank_rows')} "
            f"rank_batch_start_offset={item.get('rank_batch_start_offset')} "
            f"total_rank_batches={item.get('total_rank_batches')} "
            f"planned_rows_emitted={item.get('planned_rows_emitted')} "
            f"planned_rows_dropped_tail={item.get('planned_rows_dropped_tail')} "
            f"worker_batches={worker_batches} "
            f"worker_tail_rows={worker_tail_rows}"
        )
    return lines


# ===== Step Diagnostics =====


def _should_write_train_metrics(
    global_step: int,
    every_steps: int,
) -> bool:
    return global_step > 0 and global_step % every_steps == 0


def _data_timing_trace_fields(data_cursor: dict[str, Any]) -> dict[str, Any]:
    spans = data_cursor.get("spans")
    span_items = (
        [span for span in spans if isinstance(span, dict)]
        if isinstance(spans, list)
        else []
    )
    row_groups = [
        int(span["row_group_index"])
        for span in span_items
        if span.get("row_group_index") is not None
    ]
    row_counts = [
        int(span["row_count"])
        for span in span_items
        if span.get("row_count") is not None
    ]
    return {
        "worker_id": _optional_int(data_cursor.get("worker_id")),
        "num_workers": _optional_int(data_cursor.get("num_workers")),
        "global_batch_id": _optional_int(data_cursor.get("global_batch_id")),
        "rank_batch_index": _optional_int(data_cursor.get("rank_batch_index")),
        "source_path": data_cursor.get("source_path"),
        "read_row_group_sec": _optional_float(data_cursor.get("read_row_group_sec")),
        "convert_record_batch_sec": _optional_float(
            data_cursor.get("convert_record_batch_sec")
        ),
        "span_count": _optional_int(data_cursor.get("span_count")),
        "row_group_indexes": row_groups,
        "row_counts": row_counts,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _rank_diagnostic_record(
    *,
    rank: int,
    local_losses: dict[str, float],
    loader: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rank": rank,
        "local_losses": local_losses,
    }
    stats = _loader_stats(loader)
    if stats is not None:
        out["data_stats"] = stats
    return out


def _reduce_loss_metrics(
    loss: torch.Tensor,
    named_losses: Mapping[str, Any],
    context: DistributedContext,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Materialize all model losses with one all-reduce and one D2H sync."""
    sorted_names = sorted(named_losses)
    ordered_names = ["loss", *(f"loss_{name}" for name in sorted_names)]
    local_tensor = torch.stack(
        [
            _loss_metric_tensor(loss, device=device),
            *(
                _loss_metric_tensor(
                    named_losses[name],
                    device=device,
                )
                for name in sorted_names
            ),
        ]
    )
    global_tensor = _reduce_mean(local_tensor, context)
    if global_tensor.data_ptr() == local_tensor.data_ptr():
        values = local_tensor.detach().cpu().tolist()
        local_values = values
        global_values = values
    else:
        values = torch.cat((local_tensor, global_tensor)).detach().cpu().tolist()
        split = len(ordered_names)
        local_values = values[:split]
        global_values = values[split:]
    return (
        dict(zip(ordered_names, map(float, local_values), strict=True)),
        dict(zip(ordered_names, map(float, global_values), strict=True)),
    )


def _materialize_local_loss_metrics(
    loss: torch.Tensor,
    named_losses: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Copy rank-local progress values once without a distributed collective."""
    sorted_names = sorted(named_losses)
    ordered_names = ["loss", *(f"loss_{name}" for name in sorted_names)]
    local_tensor = torch.stack(
        [
            _loss_metric_tensor(loss, device=device),
            *(
                _loss_metric_tensor(
                    named_losses[name],
                    device=device,
                )
                for name in sorted_names
            ),
        ]
    )
    values = local_tensor.detach().cpu().tolist()
    return dict(zip(ordered_names, map(float, values), strict=True))


def _loss_metric_tensor(
    value: Any,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Training loss must contain one scalar")
        return value.detach().to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _progress_loss_postfix(
    loss_value: float,
    named_losses: dict[str, float],
) -> dict[str, str]:
    postfix = {"loss": f"{loss_value:.6f}"}
    for name, value in sorted(named_losses.items()):
        if _is_progress_loss_name(name):
            postfix[name] = f"{value:.6f}"
    return postfix


def _is_progress_loss_name(name: str) -> bool:
    raw = name.removeprefix("loss_")
    leaf = raw.rsplit("/", maxsplit=1)[-1]
    return not leaf.startswith(PROGRESS_DIAGNOSTIC_PREFIXES)


def _format_log_fields(fields: dict[str, str]) -> str:
    if not fields:
        return ""
    return "".join(f" {name}={value}" for name, value in fields.items())


# ===== Optimizer And Curves =====


def _build_optimizer(
    model: torch.nn.Module,
    config: RecAgoraConfig,
) -> Any:
    return _build_torchrec_optimizer(
        model,
        optimizer_name=config.train.optimizer_name,
        optimizer_params=config.train.optimizer_params,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        sparse_optimizer=_torchrec_sparse_optimizer_policy(config.model),
    )


def _write_training_curves(
    *,
    metrics_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, str | None]:
    try:
        curve_paths = write_training_curves(metrics_path, output_dir)
    except Exception as exc:  # pragma: no cover - diagnostic path
        _write_log(log_path, f"training_curves_failed={type(exc).__name__}: {exc}")
        return {
            "plot_path": None,
            "summary_path": None,
        }
    if curve_paths["plot_path"] is not None:
        _write_log(
            log_path,
            f"training_curves_saved={curve_paths['plot_path']} "
            f"training_summary={curve_paths['summary_path']}",
        )
    return curve_paths


# ===== Checkpointing =====


def _save_resume_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: RecAgoraConfig,
    device: torch.device,
    use_amp: bool,
    losses: list[float],
    last_loss: float | None,
    epoch: int,
    epoch_step: int,
    global_step: int,
    next_epoch: int,
    next_rank_batch_offset: int,
    next_global_batch_id: int,
    epoch_global_batch_stop: int,
    resume_input_contract: Mapping[str, Any],
    distributed_context: DistributedContext,
    hotuser_identity: Mapping[str, Any] | None = None,
) -> None:
    metadata = {
        "train_config": _checkpoint_safe_value(config.train.__dict__),
        **_checkpoint_model_metadata(config, model=model),
        RESUME_INPUT_CONTRACT_KEY: _checkpoint_safe_value(resume_input_contract),
        "data_config": _checkpoint_safe_value(config.data.__dict__),
        "data_state": {
            **({"hotuser_identity": dict(hotuser_identity)} if hotuser_identity is not None else {}),
            "parquet_start_file": _checkpoint_safe_value(config.data.parquet_start_file),
            "sharding": "global_sample",
            "epoch": int(epoch),
            "epoch_step": int(epoch_step),
            "next_epoch": int(next_epoch),
            "next_rank_batch_offset": int(next_rank_batch_offset),
            "next_global_batch_id": int(next_global_batch_id),
            "epoch_global_batch_stop": int(epoch_global_batch_stop),
            "global_step": int(global_step),
            "batch_size": int(config.train.batch_size),
            "rank": int(distributed_context.rank),
            "world_size": int(distributed_context.world_size),
            "data_paths": [str(path) for path in config.data.paths],
        },
        "device": str(device),
        "amp_enabled": use_amp,
        "losses": losses,
        "last_loss": last_loss,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "global_step": global_step,
    }
    _save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        metadata=metadata,
        distributed_context=distributed_context,
    )


def _save_model_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: RecAgoraConfig,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epoch_step: int,
    global_step: int,
    distributed_context: DistributedContext,
) -> None:
    metadata = {
        **_checkpoint_model_metadata(config, model=model),
        "device": str(device),
        "amp_enabled": use_amp,
        "source_epoch": int(epoch),
        "source_epoch_step": int(epoch_step),
        "source_global_step": int(global_step),
        **({"negative_mask_fields": list(config.train.negative_mask_fields)}
           if config.train.negative_mask_fields else {}),
    }
    _save_model_state_checkpoint(
        path,
        model=model,
        metadata=metadata,
        distributed_context=distributed_context,
    )


# ===== Logging And Serialization Helpers =====


def _train_start_lines(
    *,
    config: RecAgoraConfig,
    model: torch.nn.Module,
    torch_device: torch.device,
    use_amp: bool,
    model_checkpoint_path: Path,
    distributed_context: DistributedContext,
    local_loader_len: int,
    min_loader_len: int,
    input_contract: object,
) -> list[str]:
    owner = _unwrap_model(model)
    public_model = getattr(owner, "model", owner)
    candidate_mean = bool(config.train.negative_mask_fields) and (
        getattr(public_model, "candidate_mask_loss_reduction", "model_defined")
        == "candidate_mean"
    )
    local_scope = "rank0_ddp_contribution" if candidate_mean else "rank0_local"
    global_scope = "global_valid_candidate_mean" if candidate_mean else "global_mean"
    return [
        f"  config_path={config.source_config_path}",
        *_environment_diagnostic_lines(),
        f"  model={config.model.name}",
        f"  model_class_path={config.model.class_path}",
        f"  torchrec_mode={_torchrec_mode(config.model)} torchrec_dmp_enabled={_uses_torchrec_dmp(config.model, distributed_context)}",
        *_torchrec_model_diagnostic_lines(
            model,
            config=config.model,
            distributed_context=distributed_context,
        ),
        f"  distributed_enabled={distributed_context.enabled} backend={distributed_context.backend} rank={distributed_context.rank} local_rank={distributed_context.local_rank} world_size={distributed_context.world_size}",
        f"  progress_every_steps={config.train.progress_every_steps} progress_loss_scope={local_scope}",
        f"  global_metrics_enabled={config.train.global_metrics.enabled} global_metrics_every_steps={config.train.global_metrics.every_steps} global_metrics_loss_scope={global_scope}",
        f"  rank_diagnostics_enabled={config.train.rank_diagnostics.enabled} rank_diagnostics_every_steps={config.train.rank_diagnostics.every_steps}",
        f"  timing_trace_enabled={config.train.timing_trace.enabled} timing_trace_rank={config.train.timing_trace.rank} timing_trace_every_steps={config.train.timing_trace.every_steps} timing_trace_distributed_sync={config.train.timing_trace.distributed_sync} timing_trace_resolve_delay_steps={config.train.timing_trace.resolve_delay_steps}",
        f"  communication_trace_enabled={config.train.communication_trace.enabled} communication_trace_rank={config.train.communication_trace.rank} communication_trace_every_steps={config.train.communication_trace.every_steps} communication_trace_active_steps={config.train.communication_trace.active_steps} communication_trace_distributed_sync={config.train.communication_trace.distributed_sync}",
        *_input_prepack_start_lines(config, input_contract),
        f"  cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        f"  requested_device={config.train.device} resolved_device={torch_device}",
        f"  amp_requested={config.train.amp} amp_enabled={use_amp}",
        f"  data_format={config.data.data_format} list_len={config.data.list_len} history_len={config.data.history_len}",
        f"  data_paths={_summarize_paths(config.data.paths)}",
        f"  batch_size={config.train.batch_size} epochs={config.train.epochs} max_steps={config.train.max_steps}",
        f"  optimizer={config.train.optimizer_name} lr={config.train.lr} weight_decay={config.train.weight_decay} grad_clip_norm={config.train.grad_clip_norm}",
        *([f"  negative_mask_fields={list(config.train.negative_mask_fields)!r} "
           f"candidate_mask_counts_scope=global candidate_mask_count_collectives_per_step={int(distributed_context.enabled)} "
           "global_step_counts=read_batches"] if config.train.negative_mask_fields else []),
        f"  local_loader_len={local_loader_len} min_loader_len={min_loader_len}",
        f"  num_workers={config.train.num_workers} pin_memory={config.train.pin_memory} persistent_workers={config.train.persistent_workers}",
        f"  parquet_start_file={config.data.parquet_start_file}",
        f"  model_checkpoint={model_checkpoint_path}",
    ]


def _input_prepack_start_lines(
    config: RecAgoraConfig,
    input_contract: object,
) -> list[str]:
    del input_contract
    return [
        "  categorical_prepack_mode=worker_required",
        "  categorical_prepack_preserve_sources="
        f"{list(config.train.categorical_prepack_preserve_sources)}",
    ]


def _summarize_paths(paths: Any, *, limit: int = 3) -> str:
    rendered = [str(path) for path in paths]
    if len(rendered) <= limit:
        return "[" + ", ".join(rendered) + "]"
    head = ", ".join(rendered[:limit])
    return f"[{head}, ...] (total={len(rendered)})"


def _checkpoint_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _checkpoint_safe_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _checkpoint_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_safe_value(item) for item in value]
    return value


def _checkpoint_model_metadata(
    config: RecAgoraConfig,
    *,
    model: torch.nn.Module,
) -> dict[str, Any]:
    del model
    model_config: dict[str, Any] = {
        "name": config.model.name,
        "class_path": config.model.class_path,
        "interface": "ranking",
        "params": _checkpoint_safe_value(dict(config.model.params)),
    }
    recagora_contract: dict[str, Any] = {
        "version": 1,
        "interface": "ranking",
        "model_name": config.model.name,
        "class_path": config.model.class_path,
    }
    return {
        "model_config": model_config,
        # Persisted checkpoint key: keep stable across Python API renames.
        "arena_model_contract": recagora_contract,
    }


# ===== Checkpoint Loading Helpers =====


def _load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    restore_rng: bool,
    distributed_context: DistributedContext,
    rank_state_rank: int | None = None,
) -> dict[str, Any]:
    return _load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        restore_rng=restore_rng,
        distributed_context=distributed_context,
        rank_state_rank=rank_state_rank,
    )

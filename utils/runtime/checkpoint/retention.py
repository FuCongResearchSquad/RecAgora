from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from utils.runtime.checkpoint.dcp import (
    CHECKPOINT_FORMAT_VERSION,
    DCP_MODEL_OPTIMIZER_OWNER,
    DCP_MODEL_OWNER,
    METADATA_FILENAME,
)
from utils.runtime.distributed import DistributedContext, all_gather_object


_STEP_CHECKPOINT_PATTERN = re.compile(r"step_(\d+)")


@dataclass(frozen=True)
class CheckpointRetentionResult:
    kept: tuple[Path, ...]
    deleted: tuple[Path, ...]
    skipped: tuple[Path, ...]


def prune_step_checkpoints(
    checkpoints_dir: str | Path,
    *,
    current_step: int,
    max_to_keep: int,
    distributed_context: DistributedContext,
) -> CheckpointRetentionResult:
    """Keep the newest complete ``step_N`` checkpoints and preserve ``last``.

    Only rank 0 mutates the shared checkpoint directory. The result or failure
    is then propagated to every rank so no worker waits at a later collective
    after a rank-0-only deletion error.
    """

    _validate_current_step(current_step)
    if max_to_keep <= 0:
        raise ValueError("max_to_keep must be positive")

    return _run_collective_retention(
        operation="checkpoint retention",
        distributed_context=distributed_context,
        action=lambda: _prune_step_checkpoints(
            Path(checkpoints_dir),
            current_step=current_step,
            max_to_keep=max_to_keep,
        ),
    )


def prune_step_checkpoints_after_completion(
    checkpoints_dir: str | Path,
    *,
    current_step: int,
    distributed_context: DistributedContext,
) -> CheckpointRetentionResult:
    """Delete complete resumable checkpoints after validating final ``last``.

    The cleanup is collective so every rank observes a rank-0 filesystem
    failure. Partial, unknown, and symlinked ``step_N`` entries remain skipped
    rather than being deleted as if they were framework-owned checkpoints.
    """

    _validate_current_step(current_step)
    return _run_collective_retention(
        operation="checkpoint completion cleanup",
        distributed_context=distributed_context,
        action=lambda: _prune_step_checkpoints_after_completion(
            Path(checkpoints_dir),
            current_step=current_step,
        ),
    )


def _validate_current_step(current_step: int) -> None:
    if current_step < 0:
        raise ValueError("current_step must be non-negative")


def _run_collective_retention(
    *,
    operation: str,
    distributed_context: DistributedContext,
    action: Callable[[], CheckpointRetentionResult],
) -> CheckpointRetentionResult:
    payload: dict[str, Any] | None = None
    if distributed_context.is_main:
        try:
            result = action()
            payload = {
                "kept": [str(path) for path in result.kept],
                "deleted": [str(path) for path in result.deleted],
                "skipped": [str(path) for path in result.skipped],
            }
        except Exception as exc:  # Propagate rank-0 filesystem failures safely.
            payload = {"error": f"{type(exc).__name__}: {exc}"}

    leader_payload = all_gather_object(payload, distributed_context)[0]
    if not isinstance(leader_payload, dict):
        raise RuntimeError(f"{operation} did not receive a rank-0 result")
    error = leader_payload.get("error")
    if error is not None:
        raise RuntimeError(f"{operation} failed on rank 0: {error}")
    return CheckpointRetentionResult(
        kept=tuple(Path(path) for path in leader_payload.get("kept", ())),
        deleted=tuple(Path(path) for path in leader_payload.get("deleted", ())),
        skipped=tuple(Path(path) for path in leader_payload.get("skipped", ())),
    )


def _prune_step_checkpoints(
    checkpoints_dir: Path,
    *,
    current_step: int,
    max_to_keep: int,
) -> CheckpointRetentionResult:
    _require_checkpoint_root(checkpoints_dir)

    complete, skipped = _classify_step_checkpoints(checkpoints_dir)
    _validate_checkpoint_boundary(
        complete,
        current_step=current_step,
        missing_prefix="newly saved checkpoint",
        missing_suffix="refusing to prune older checkpoints",
        newer_context="the current training step",
        newer_action="use a new output_dir for a forked training run",
    )

    delete_count = max(0, len(complete) - max_to_keep)
    to_delete = complete[:delete_count]
    kept = complete[delete_count:]
    deleted: list[Path] = []
    for _, path in to_delete:
        shutil.rmtree(path)
        deleted.append(path)

    return CheckpointRetentionResult(
        kept=tuple(path for _, path in kept),
        deleted=tuple(deleted),
        skipped=tuple(sorted(skipped, key=lambda path: path.name)),
    )


def _prune_step_checkpoints_after_completion(
    checkpoints_dir: Path,
    *,
    current_step: int,
) -> CheckpointRetentionResult:
    _require_checkpoint_root(checkpoints_dir)
    last = checkpoints_dir / "last"
    if not _is_complete_model_checkpoint(last, expected_step=current_step):
        raise RuntimeError(
            "final last checkpoint is not a complete DCP v3 model-only checkpoint "
            f"for step {current_step}; refusing to delete resumable checkpoints"
        )

    complete, skipped = _classify_step_checkpoints(checkpoints_dir)
    _validate_checkpoint_boundary(
        complete,
        current_step=current_step,
        missing_prefix="final checkpoint",
        missing_suffix="refusing completion cleanup",
        newer_context="the completed training step",
        newer_action="refusing completion cleanup",
    )

    deleted: list[Path] = []
    for _, path in complete:
        shutil.rmtree(path)
        deleted.append(path)
    return CheckpointRetentionResult(
        kept=(),
        deleted=tuple(deleted),
        skipped=tuple(sorted(skipped, key=lambda path: path.name)),
    )


def _require_checkpoint_root(checkpoints_dir: Path) -> None:
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint root does not exist or is not a directory: {checkpoints_dir}"
        )


def _validate_checkpoint_boundary(
    complete: list[tuple[int, Path]],
    *,
    current_step: int,
    missing_prefix: str,
    missing_suffix: str,
    newer_context: str,
    newer_action: str,
) -> None:
    complete_steps = {step for step, _ in complete}
    if current_step not in complete_steps:
        raise RuntimeError(
            f"{missing_prefix} step_{current_step} is not a complete DCP v3 "
            f"training checkpoint; {missing_suffix}"
        )
    higher_steps = [step for step, _ in complete if step > current_step]
    if higher_steps:
        raise RuntimeError(
            f"checkpoint root contains steps newer than {newer_context}: "
            f"current={current_step} newer={higher_steps}; {newer_action}"
        )


def _classify_step_checkpoints(
    checkpoints_dir: Path,
) -> tuple[list[tuple[int, Path]], list[Path]]:
    complete: list[tuple[int, Path]] = []
    skipped: list[Path] = []
    for path in checkpoints_dir.iterdir():
        match = _STEP_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if path.is_symlink() or not path.is_dir():
            skipped.append(path)
            continue
        if _is_complete_step_checkpoint(path, expected_step=step):
            complete.append((step, path))
        else:
            skipped.append(path)
    complete.sort(key=lambda item: item[0])
    return complete, skipped


def _is_complete_step_checkpoint(path: Path, *, expected_step: int) -> bool:
    metadata = _checkpoint_metadata(path)
    return metadata is not None and (
        metadata.get("checkpoint_format") == "dcp"
        and metadata.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
        and metadata.get("checkpoint_state_owner") == DCP_MODEL_OPTIMIZER_OWNER
        and metadata.get("global_step") == expected_step
    )


def _is_complete_model_checkpoint(path: Path, *, expected_step: int) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    metadata = _checkpoint_metadata(path)
    return metadata is not None and (
        metadata.get("checkpoint_format") == "dcp"
        and metadata.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
        and metadata.get("checkpoint_state_owner") == DCP_MODEL_OWNER
        and metadata.get("source_global_step") == expected_step
    )


def _checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    return metadata

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from .optimizer_state import (
    initialize_keyed_optimizer_state_for_dcp_load,
    initialize_torch_optimizer_state_for_dcp_load,
)
from utils.runtime.distributed import DistributedContext, barrier


METADATA_FILENAME = "metadata.json"
RANK_STATE_TEMPLATE = "rank_{rank:05d}.pt"
CHECKPOINT_FORMAT_VERSION = 3
DCP_MODEL_OPTIMIZER_OWNER = "model_and_optimizer.state_dict"
DCP_MODEL_OWNER = "model.state_dict"


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scaler: Any,
    metadata: dict[str, Any],
    distributed_context: DistributedContext,
) -> None:
    checkpoint_dir = _prepare_checkpoint_dir(path)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    _save_state_dict(state, checkpoint_dir)
    _save_rank_state(
        checkpoint_dir,
        scaler=scaler,
        context=distributed_context,
    )
    if distributed_context.is_main:
        _write_metadata(
            checkpoint_dir,
            {
                **metadata,
                "checkpoint_format": "dcp",
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "checkpoint_state_owner": DCP_MODEL_OPTIMIZER_OWNER,
                "rank_state_pattern": RANK_STATE_TEMPLATE,
                "rank_state_world_size": int(distributed_context.world_size),
            },
        )
    barrier(distributed_context)


def save_model_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    distributed_context: DistributedContext,
) -> None:
    """Save model weights without optimizer or rank-local training state."""

    checkpoint_dir = _prepare_checkpoint_dir(path)
    if distributed_context.is_main:
        _clear_checkpoint_dir(checkpoint_dir)
    barrier(distributed_context)
    _save_state_dict({"model": model.state_dict()}, checkpoint_dir)
    if distributed_context.is_main:
        _write_metadata(
            checkpoint_dir,
            {
                **metadata,
                "checkpoint_format": "dcp",
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "checkpoint_state_owner": DCP_MODEL_OWNER,
            },
        )
    barrier(distributed_context)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scaler: Any,
    restore_rng: bool,
    distributed_context: DistributedContext,
    rank_state_rank: int | None = None,
) -> dict[str, Any]:
    metadata = read_training_checkpoint_metadata(path)
    checkpoint_dir = Path(path)
    model_state = model.state_dict()
    keyed_state_initialized = initialize_keyed_optimizer_state_for_dcp_load(
        optimizer,
        checkpoint_dir,
        model_state_keys=tuple(model_state),
    )
    if not keyed_state_initialized and not initialize_torch_optimizer_state_for_dcp_load(
        optimizer, checkpoint_dir
    ):
        _initialize_optimizer_state_for_checkpoint_load(optimizer)
    state = {"model": model_state, "optimizer": optimizer.state_dict()}
    _load_state_dict(state, checkpoint_dir)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rank_state = _load_rank_state(
        checkpoint_dir,
        distributed_context,
        rank_override=rank_state_rank,
    )
    scaler_state = rank_state.get("scaler")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        _restore_rng(rank_state)
    return metadata


def load_model_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    distributed_context: DistributedContext | None = None,
    allow_missing_acr: bool = False,
) -> dict[str, Any]:
    metadata = read_model_checkpoint_metadata(path)
    checkpoint_dir = Path(path)
    model_state = model.state_dict()
    if allow_missing_acr:
        available = set(FileSystemReader(str(checkpoint_dir)).read_metadata().state_dict_metadata)
        missing = [key for key in model_state if f"model.{key}" not in available]
        incompatible = [key for key in missing if ".backbone.acr." not in f".{key}"]
        if incompatible:
            raise ValueError(
                "P0 checkpoint is missing non-ACR model parameters: "
                + ", ".join(incompatible[:10])
            )
        state = {"model": {key: value for key, value in model_state.items() if key not in missing}}
    else:
        state = {"model": model_state}
    _load_state_dict(state, checkpoint_dir)
    model.load_state_dict({**model_state, **state["model"]})
    return metadata


def read_training_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Validate a resume checkpoint kind and return metadata without state I/O."""

    checkpoint_dir = _validate_checkpoint_dir(path)
    metadata = _read_metadata(checkpoint_dir)
    _validate_checkpoint_format(
        metadata,
        accepted_state_owners=(DCP_MODEL_OPTIMIZER_OWNER,),
    )
    return metadata


def read_model_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Validate an init/eval checkpoint kind and return metadata without state I/O."""

    checkpoint_dir = _validate_checkpoint_dir(path)
    metadata = _read_metadata(checkpoint_dir)
    _validate_checkpoint_format(
        metadata,
        accepted_state_owners=(DCP_MODEL_OWNER, DCP_MODEL_OPTIMIZER_OWNER),
    )
    return metadata


def _save_state_dict(
    state: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    dcp.save(
        state_dict=state,
        storage_writer=FileSystemWriter(str(checkpoint_dir), overwrite=True),
    )


def _load_state_dict(
    state: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    dcp.load(
        state_dict=state,
        storage_reader=FileSystemReader(str(checkpoint_dir)),
    )


def _save_rank_state(
    checkpoint_dir: Path,
    *,
    scaler: Any,
    context: DistributedContext,
) -> None:
    torch.save(
        {
            "scaler": scaler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": _cuda_rng_state_all(),
        },
        _rank_state_path(checkpoint_dir, context),
    )


def _load_rank_state(
    checkpoint_dir: Path,
    context: DistributedContext,
    *,
    rank_override: int | None = None,
) -> dict[str, Any]:
    rank = int(context.rank) if rank_override is None else int(rank_override)
    if rank < 0:
        raise ValueError("rank_state_rank must be non-negative")
    rank_state_path = checkpoint_dir / RANK_STATE_TEMPLATE.format(rank=rank)
    if not rank_state_path.exists():
        raise FileNotFoundError(
            f"Rank-local checkpoint state is missing: {rank_state_path}"
        )
    return torch.load(
        rank_state_path,
        map_location="cpu",
        weights_only=False,
    )


def _rank_state_path(checkpoint_dir: Path, context: DistributedContext) -> Path:
    return checkpoint_dir / RANK_STATE_TEMPLATE.format(rank=int(context.rank))


def _validate_checkpoint_format(
    metadata: dict[str, Any],
    *,
    accepted_state_owners: tuple[str, ...] = (DCP_MODEL_OPTIMIZER_OWNER,),
) -> None:
    actual = metadata.get("checkpoint_format_version")
    owner = metadata.get("checkpoint_state_owner")
    if actual != CHECKPOINT_FORMAT_VERSION:
        _raise_incompatible_checkpoint_format(
            actual=actual,
            owner=owner,
            accepted_state_owners=accepted_state_owners,
        )
    if owner in accepted_state_owners:
        return
    if owner == DCP_MODEL_OWNER and accepted_state_owners == (
        DCP_MODEL_OPTIMIZER_OWNER,
    ):
        raise ValueError(
            "cannot resume training from a model-only checkpoint; "
            "use init_from or eval checkpoint loading instead"
        )
    _raise_incompatible_checkpoint_format(
        actual=actual,
        owner=owner,
        accepted_state_owners=accepted_state_owners,
    )


def _raise_incompatible_checkpoint_format(
    *,
    actual: object,
    owner: object,
    accepted_state_owners: tuple[str, ...],
) -> NoReturn:
    expected_owners = ", ".join(repr(value) for value in accepted_state_owners)
    raise ValueError(
        "checkpoint format is incompatible with this publish runtime: "
        f"expected version {CHECKPOINT_FORMAT_VERSION} with state owner in "
        f"({expected_owners}), got version {actual!r} and owner "
        f"{owner!r}."
    )


def _initialize_optimizer_state_for_checkpoint_load(optimizer: Any) -> None:
    """Materialize lazy optimizer tensors before strict checkpoint loading."""
    init_state = getattr(optimizer, "init_state", None)
    if callable(init_state):
        init_state()
        return

    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None:
        raise TypeError(
            "optimizer must expose init_state() or param_groups for DCP loading"
        )
    for group in param_groups:
        for parameter in group["params"]:
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _prepare_checkpoint_dir(path: str | Path) -> Path:
    checkpoint_dir = _checkpoint_dir_path(path, require_existing=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _clear_checkpoint_dir(checkpoint_dir: Path) -> None:
    resolved = checkpoint_dir.resolve()
    if resolved == resolved.parent or resolved.name in {"", ".", ".."}:
        raise ValueError(f"Refusing to clear unsafe checkpoint path: {resolved}")
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _validate_checkpoint_dir(path: str | Path) -> Path:
    checkpoint_dir = _checkpoint_dir_path(path, require_existing=True)
    metadata_path = checkpoint_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"DCP checkpoint metadata is missing: {metadata_path}")
    return checkpoint_dir


def _checkpoint_dir_path(
    path: str | Path,
    *,
    require_existing: bool,
) -> Path:
    checkpoint_dir = Path(path)
    if checkpoint_dir.suffix == ".pt":
        raise ValueError(
            f"DCP checkpoints must be directories, not .pt files: {checkpoint_dir}"
        )
    if require_existing and not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise ValueError(f"DCP checkpoint path is not a directory: {checkpoint_dir}")
    return checkpoint_dir


def _write_metadata(checkpoint_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = checkpoint_dir / METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe_value(metadata), f, indent=2, sort_keys=True)
        f.write("\n")


def _read_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    with (checkpoint_dir / METADATA_FILENAME).open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {checkpoint_dir}")
    return value


def _restore_rng(metadata: dict[str, Any]) -> None:
    rng_state = metadata.get("rng_state")
    if rng_state is not None:
        torch.set_rng_state(_as_byte_tensor(rng_state))
    cuda_rng_state_all = metadata.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_as_byte_tensor(value) for value in cuda_rng_state_all]
        )


def _cuda_rng_state_all() -> list[list[int]] | None:
    if not torch.cuda.is_available():
        return None
    return [_tensor_to_list(value) for value in torch.cuda.get_rng_state_all()]


def _tensor_to_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.cpu().tolist()]


def _as_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.uint8, device="cpu")
    if isinstance(value, list):
        return torch.tensor([int(item) for item in value], dtype=torch.uint8)
    raise ValueError("Checkpoint RNG state must be a byte tensor or JSON list")


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_value(asdict(value))
    if isinstance(value, torch.Tensor):
        return _tensor_to_list(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value

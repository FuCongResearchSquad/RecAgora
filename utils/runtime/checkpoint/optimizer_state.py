from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader


def initialize_torch_optimizer_state_for_dcp_load(
    optimizer: Any,
    checkpoint_dir: Path,
) -> bool:
    """Build regular optimizer templates only for states actually saved in DCP.

    A masked run may save a full checkpoint without performing any update, or
    after using only some parameters.  Initializing every AdamW parameter would
    invent state entries absent from that checkpoint and break strict DCP load.
    LocalCombinedOptimizer members retain their native nested state prefixes.
    Unknown optimizer wrappers return false for the existing fallback path;
    TorchRec keyed optimizers continue to use their separate initializer.
    """
    members = _torch_optimizer_members(optimizer, prefix="optimizer")
    if members is None:
        return False
    metadata_keys = tuple(
        FileSystemReader(str(checkpoint_dir)).read_metadata().state_dict_metadata
    )
    for member, prefix in members:
        _initialize_saved_torch_states(member, prefix=prefix, metadata_keys=metadata_keys)
    return True


def _torch_optimizer_members(
    optimizer: Any,
    *,
    prefix: str,
) -> list[tuple[torch.optim.Optimizer, str]] | None:
    if isinstance(optimizer, torch.optim.Optimizer):
        return [(optimizer, prefix)]
    children = getattr(optimizer, "optimizers", None)
    if not isinstance(children, (list, tuple)) or not children:
        return None
    members: list[tuple[torch.optim.Optimizer, str]] = []
    names: list[str] = []
    for child in children:
        if not isinstance(child, (list, tuple)) or len(child) != 2:
            return None
        name, child_optimizer = child
        if not isinstance(name, str):
            return None
        nested = _torch_optimizer_members(
            child_optimizer, prefix=f"{prefix}.optimizers.{name}"
        )
        if nested is None:
            return None
        names.append(name)
        members.extend(nested)
    # Only recognize the local wrapper's published state structure, not an
    # arbitrary optimizer exposing an unrelated ``optimizers`` attribute.
    serialized = optimizer.state_dict()
    child_states = serialized.get("optimizers")
    if not isinstance(child_states, Mapping) or set(child_states) != set(names):
        return None
    return members


def _initialize_saved_torch_states(
    optimizer: torch.optim.Optimizer,
    *,
    prefix: str,
    metadata_keys: tuple[str, ...],
) -> None:
    serialized = optimizer.state_dict()
    parameters: dict[str, torch.Tensor] = {}
    for saved_group, live_group in zip(
        serialized["param_groups"], optimizer.param_groups, strict=True
    ):
        for parameter_id, parameter in zip(
            saved_group["params"], live_group["params"], strict=True
        ):
            parameters[str(parameter_id)] = parameter
    state_prefix = f"{prefix}.state."
    expected = {
        key[len(state_prefix):].split(".", 1)[0]
        for key in metadata_keys
        if key.startswith(state_prefix)
    }
    unknown = expected - parameters.keys()
    if unknown:
        raise ValueError(
            f"DCP optimizer {prefix} contains unknown parameter state IDs: {sorted(unknown)}"
        )
    materialized = {str(key) for key in serialized["state"]}
    unexpected = materialized - expected
    if unexpected:
        raise ValueError(
            f"DCP optimizer {prefix} has unexpected initialized state IDs: {sorted(unexpected)}"
        )
    missing = expected - materialized
    if missing:
        # Only saved parameters receive synthetic gradients.  A zero-update
        # checkpoint performs no initialization step at all.  During resume the
        # model weights and these temporary states are overwritten by DCP next.
        optimizer.zero_grad(set_to_none=True)
        for parameter_id in missing:
            parameters[parameter_id].grad = torch.zeros_like(parameters[parameter_id])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    actual = {str(key) for key in optimizer.state_dict()["state"]}
    if actual != expected:
        raise ValueError(
            f"DCP optimizer {prefix} state template does not match checkpoint: "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
        )


def initialize_keyed_optimizer_state_for_dcp_load(
    optimizer: Any,
    checkpoint_dir: Path,
    *,
    model_state_keys: tuple[str, ...] = (),
) -> bool:
    """Materialize exactly the keyed optimizer states present in DCP.

    Returns false for a regular torch optimizer so the caller can use its
    generic initialization path. TorchRec KeyedOptimizer state is strict and
    must have the same lazy parameter-state set as the checkpoint before DCP
    performs its in-place load. DCP metadata is global, while a table-wise
    fused optimizer only owns the parameters placed on the current rank.
    States for model parameters owned by another rank are therefore ignored
    when constructing this rank's local optimizer template.
    """
    params = getattr(optimizer, "params", None)
    state = getattr(optimizer, "state", None)
    if not isinstance(params, Mapping) or not isinstance(state, Mapping):
        return False

    metadata_keys = tuple(
        FileSystemReader(str(checkpoint_dir)).read_metadata().state_dict_metadata
    )
    optimizer_state_keys = tuple(
        key for key in metadata_keys if key.startswith("optimizer.state.")
    )
    param_keys = tuple(str(key) for key in params)
    checkpoint_param_keys: set[str] = set()
    unmatched_state_keys: list[str] = []
    for state_key in optimizer_state_keys:
        matches = [
            key
            for key in param_keys
            if state_key.startswith(f"optimizer.state.{key}.")
        ]
        if not matches:
            if _is_remote_fused_optimizer_state(
                state_key,
                model_state_keys=model_state_keys,
            ):
                continue
            unmatched_state_keys.append(state_key)
            continue
        checkpoint_param_keys.add(max(matches, key=len))
    if unmatched_state_keys:
        raise ValueError(
            "DCP optimizer checkpoint contains unknown state keys: "
            f"{sorted(unmatched_state_keys)}"
        )

    for key, parameter in params.items():
        if (
            str(key) in checkpoint_param_keys
            and parameter not in state
            and parameter.requires_grad
        ):
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    materialized = {
        str(key) for key in optimizer.state_dict().get("state", {})
    }
    if materialized != checkpoint_param_keys:
        raise ValueError(
            "DCP optimizer state template does not match checkpoint parameters: "
            f"missing={sorted(checkpoint_param_keys - materialized)} "
            f"unexpected={sorted(materialized - checkpoint_param_keys)}"
        )

    missing = sorted(set(param_keys) - checkpoint_param_keys)
    if missing:
        warnings.warn(
            "DCP optimizer checkpoint has no lazy state for current parameters; "
            f"leaving optimizer state absent for: {missing}",
            UserWarning,
            stacklevel=2,
        )
    return True


def _is_remote_fused_optimizer_state(
    state_key: str,
    *,
    model_state_keys: tuple[str, ...],
) -> bool:
    """Return whether a global fused state belongs to another DMP rank.

    KeyedOptimizer parameter names omit only the DCP state roots. For example,
    ``optimizer.state.fused.model.table.weight.table.momentum1`` maps to the
    model key ``model.table.weight``. A key is considered remote only when it
    names a parameter in the current model; genuinely stale or foreign keys
    remain errors.
    """
    prefix = "optimizer.state.fused."
    if not state_key.startswith(prefix):
        return False
    payload = state_key[len(prefix):]
    return any(
        payload.startswith(f"{model_key}.")
        for model_key in model_state_keys
    )

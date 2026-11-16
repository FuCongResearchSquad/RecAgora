from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedContext:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    launched_by_torchrun = "WORLD_SIZE" in os.environ
    if world_size <= 1 and not launched_by_torchrun:
        return DistributedContext(
            enabled=False,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device_id: torch.device | None = None
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        init_kwargs: dict[str, object] = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
        }
        if device_id is not None:
            init_kwargs["device_id"] = device_id
        dist.init_process_group(**init_kwargs)
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def resolve_device_for_context(
    requested_device: str,
    context: DistributedContext,
) -> torch.device:
    if context.enabled and torch.cuda.is_available():
        return torch.device(f"cuda:{context.local_rank}")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested_device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return resolved


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if not context.enabled or not dist.is_initialized():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= context.world_size
    return out


def reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if not context.enabled or not dist.is_initialized():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def reduce_min_int(value: int, context: DistributedContext, *, device: torch.device) -> int:
    if not context.enabled or not dist.is_initialized():
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def all_gather_object(value: object, context: DistributedContext) -> list[object]:
    if not context.enabled or not dist.is_initialized():
        return [value]
    gathered: list[object | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return list(gathered)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if isinstance(wrapped, torch.nn.Module):
            current = wrapped
            continue
        wrapped = getattr(current, "_dmp_wrapped_module", None)
        if isinstance(wrapped, torch.nn.Module):
            current = wrapped
            continue
        break
    return current


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)

from __future__ import annotations

from datetime import datetime
import importlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from utils.compiler.config import ModelConfig
from utils.runtime.distributed import DistributedContext
from utils.runtime.torchrec import (
    torchrec_model_table_embedding_dims,
    torchrec_model_table_num_embeddings,
    torchrec_model_table_names,
    torchrec_table_storage_policies,
    uses_torchrec_dmp,
)


def run_start_line(mode: str, *, started_at: datetime) -> str:
    normalized_mode = str(mode).strip()
    if not normalized_mode:
        raise ValueError("run mode must be a non-empty string")
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("run start time must include a timezone")
    timestamp = started_at.isoformat(timespec="seconds")
    return f"run_start mode={normalized_mode} started_at={timestamp}"


def environment_diagnostic_lines(*, cwd: str | Path = ".") -> list[str]:
    versions = runtime_version_info()
    return [
        f"  git_dirty={_git_dirty(cwd=cwd)}",
        f"  torch_version={versions['torch']} torchrec_version={versions['torchrec']} fbgemm_gpu_version={versions['fbgemm_gpu']}",
    ]


def torchrec_model_diagnostic_lines(
    model: torch.nn.Module,
    *,
    config: ModelConfig,
    distributed_context: DistributedContext,
) -> list[str]:
    storage_lines = _torchrec_storage_lines(config, model=model)
    model_lines = _model_parallel_lines(
        model,
        config=config,
        distributed_context=distributed_context,
    )
    return [*storage_lines, *model_lines]


def runtime_version_info() -> dict[str, str]:
    return {
        "torch": getattr(torch, "__version__", "unknown"),
        "torchrec": _module_version("torchrec"),
        "fbgemm_gpu": _module_version("fbgemm_gpu"),
    }


def _torchrec_storage_lines(
    config: ModelConfig,
    *,
    model: torch.nn.Module,
) -> list[str]:
    params = dict(config.params)
    torchrec = params.get("torchrec", {})
    if torchrec is None:
        torchrec = {}
    if not isinstance(torchrec, Mapping):
        return ["  torchrec_config_error=model.params.torchrec must be a mapping"]

    try:
        table_names = getattr(model, "_recagora_torchrec_table_names", None)
        if table_names is None:
            table_names = torchrec_model_table_names(model)
        table_num_embeddings = getattr(
            model,
            "_recagora_torchrec_table_num_embeddings",
            None,
        )
        if table_num_embeddings is None:
            table_num_embeddings = torchrec_model_table_num_embeddings(model)
        table_embedding_dims = getattr(
            model,
            "_recagora_torchrec_table_embedding_dims",
            None,
        )
        if table_embedding_dims is None:
            table_embedding_dims = torchrec_model_table_embedding_dims(model)
        policies = torchrec_table_storage_policies(
            config,
            table_names=table_names,
            table_num_embeddings=table_num_embeddings,
        )
    except Exception as exc:
        return [
            f"  torchrec_storage_policy_error={type(exc).__name__}: {_one_line(str(exc), limit=300)}",
        ]

    kernel_counts: dict[str, int] = {}
    sharding_counts: dict[str, int] = {}
    cache_tables: list[str] = []
    for table_name, policy in policies.items():
        kernel_counts[policy.compute_kernel] = kernel_counts.get(policy.compute_kernel, 0) + 1
        sharding_label = (
            ",".join(policy.sharding_types)
            if policy.sharding_types is not None
            else "planner_default"
        )
        sharding_counts[sharding_label] = sharding_counts.get(sharding_label, 0) + 1
        if policy.cache_load_factor is not None:
            cache_tables.append(f"{table_name}:{policy.cache_load_factor:g}")

    uvm_tables = sorted(
        table_name
        for table_name, policy in policies.items()
        if "uvm" in policy.compute_kernel
    )
    storage_summary = (
        "  torchrec_storage_policy "
        f"table_count={len(policies)} "
        f"sharding_counts={_format_counts(sharding_counts)} "
        f"kernel_counts={_format_counts(kernel_counts)} "
        f"uvm_tables={_format_sequence(uvm_tables)} "
        f"cache_tables={_format_sequence(cache_tables)}"
    )
    try:
        total_rows = sum(int(rows) for rows in table_num_embeddings.values())
        total_fp32_bytes = sum(
            int(table_num_embeddings[table_name])
            * int(table_embedding_dims[table_name])
            * 4
            for table_name in table_names
        )
        table_lines: list[str] = []
        for table_name in sorted(
            table_names,
            key=lambda name: (-int(table_num_embeddings[name]), str(name)),
        ):
            rows = int(table_num_embeddings[table_name])
            embedding_dim = int(table_embedding_dims[table_name])
            policy = policies.get(table_name)
            cache_load_factor = (
                f"{policy.cache_load_factor:g}"
                if policy is not None and policy.cache_load_factor is not None
                else "none"
            )
            table_lines.append(
                "  torchrec_embedding_table "
                f"name={table_name} "
                f"rows={rows} "
                f"dim={embedding_dim} "
                f"estimated_fp32_gib={_gib(rows * embedding_dim * 4)} "
                "sharding_types="
                f"{','.join(policy.sharding_types) if policy is not None and policy.sharding_types is not None else 'planner_default'} "
                f"kernel={policy.compute_kernel if policy is not None else 'planner_default'} "
                f"cache_load_factor={cache_load_factor}"
            )
    except Exception as exc:
        return [
            storage_summary,
            "  torchrec_embedding_table_error="
            f"{type(exc).__name__}: {_one_line(str(exc), limit=300)}",
        ]
    return [
        storage_summary,
        "  torchrec_embedding_summary "
        f"table_count={len(table_names)} "
        f"total_rows={total_rows} "
        f"estimated_fp32_gib={_gib(total_fp32_bytes)}",
        *table_lines,
    ]


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.6f}"


def _model_parallel_lines(
    model: torch.nn.Module,
    *,
    config: ModelConfig,
    distributed_context: DistributedContext,
) -> list[str]:
    fused_optimizer = getattr(model, "fused_optimizer", None)
    fused_params = (
        getattr(fused_optimizer, "params", {})
        if fused_optimizer is not None
        else {}
    )
    plan_attr, plan_value = _first_present_attr(
        model,
        ("_recagora_torchrec_plan", "plan", "_plan", "_sharding_plan"),
    )
    constraints = getattr(model, "_recagora_torchrec_planner_constraints", {})
    sharder_count = getattr(model, "_recagora_torchrec_sharder_count", None)
    module_samples = _torchrec_module_samples(model)
    actual_plan = _torchrec_plan_counts(plan_value)
    table_names = getattr(model, "_recagora_torchrec_table_names", ())
    return [
        "  torchrec_runtime "
        f"dmp_enabled={uses_torchrec_dmp(config, distributed_context)} "
        f"rank={distributed_context.rank} world_size={distributed_context.world_size} "
        f"model_type={_qualified_type_name(model)} "
        f"plan_attr={plan_attr or 'none'} "
        f"sharder_count={sharder_count if sharder_count is not None else 'unknown'} "
        f"has_fused_optimizer={fused_optimizer is not None} "
        f"fused_param_count={len(fused_params)}",
        f"  torchrec_planner_constraint_keys={_format_sequence(sorted(str(key) for key in constraints), limit=40) if isinstance(constraints, Mapping) else 'unknown'}",
        f"  torchrec_actual_table_names={_format_sequence(table_names, limit=60)}",
        "  torchrec_actual_plan "
        f"table_count={actual_plan['table_count']} "
        f"sharding_counts={_format_counts(actual_plan['sharding_counts'])} "
        f"kernel_counts={_format_counts(actual_plan['kernel_counts'])}",
        f"  torchrec_fused_param_keys={_format_sequence(list(fused_params)[:20])}",
        f"  torchrec_module_sample={_format_sequence(module_samples, limit=30)}",
        f"  torchrec_plan_summary={_one_line(repr(plan_value), limit=2000) if plan_value is not None else 'none'}",
    ]


def _torchrec_plan_counts(plan: Any) -> dict[str, Any]:
    table_count = 0
    sharding_counts: dict[str, int] = {}
    kernel_counts: dict[str, int] = {}
    raw_plan = getattr(plan, "plan", None)
    if not isinstance(raw_plan, Mapping):
        return {
            "table_count": 0,
            "sharding_counts": sharding_counts,
            "kernel_counts": kernel_counts,
        }
    for module_plan in raw_plan.values():
        if not isinstance(module_plan, Mapping):
            continue
        for parameter_sharding in module_plan.values():
            table_count += 1
            sharding_type = str(
                getattr(parameter_sharding, "sharding_type", "unknown")
            )
            compute_kernel = str(
                getattr(parameter_sharding, "compute_kernel", "unknown")
            )
            sharding_counts[sharding_type] = sharding_counts.get(sharding_type, 0) + 1
            kernel_counts[compute_kernel] = kernel_counts.get(compute_kernel, 0) + 1
    return {
        "table_count": table_count,
        "sharding_counts": sharding_counts,
        "kernel_counts": kernel_counts,
    }


def _first_present_attr(obj: Any, names: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return name, value
    return None, None


def _torchrec_module_samples(model: torch.nn.Module, *, limit: int = 30) -> list[str]:
    samples: list[str] = []
    for name, module in model.named_modules():
        type_name = _qualified_type_name(module)
        if (
            "torchrec" in type_name
            or "Sharded" in type_name
        ):
            samples.append(f"{name or '<root>'}:{type_name}")
            if len(samples) >= limit:
                break
    return samples


def _module_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "not_installed"
    return str(getattr(module, "__version__", "unknown"))


def _run_git(args: list[str], *, cwd: str | Path) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_dirty(*, cwd: str | Path) -> str:
    status = _run_git(["git", "status", "--porcelain"], cwd=cwd)
    if status is None:
        return "unknown"
    return str(bool(status)).lower()


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "[]"
    return "[" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)) + "]"


def _format_sequence(values: list[Any], *, limit: int = 20) -> str:
    rendered = [str(value) for value in values[:limit]]
    suffix = f",... total={len(values)}" if len(values) > limit else ""
    return "[" + ",".join(rendered) + suffix + "]"


def _one_line(value: str, *, limit: int) -> str:
    rendered = " ".join(str(value).split())
    if len(rendered) <= limit:
        return rendered
    return rendered[: max(0, limit - 3)] + "..."

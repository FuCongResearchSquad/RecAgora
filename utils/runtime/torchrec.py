from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn

from utils.compiler.config import ModelConfig
from utils.runtime.distributed import DistributedContext


_VALID_COMPUTE_KERNELS = {
    "dense",
    "dram_virtual_table",
    "fused",
    "fused_uvm",
    "fused_uvm_caching",
    "key_value",
    "ssd_virtual_table",
}

_VALID_SHARDING_TYPES = {
    "column_wise",
    "data_parallel",
    "grid_shard",
    "row_wise",
    "table_column_wise",
    "table_row_wise",
    "table_wise",
}


@dataclass(frozen=True)
class TorchRecTableStoragePolicy:
    compute_kernel: str
    sharding_types: tuple[str, ...] | None = None
    cache_load_factor: float | None = None
    enforce_hbm: bool | None = None


@dataclass(frozen=True)
class TorchRecSparseOptimizerPolicy:
    name: str
    lr: float
    eps: float = 1e-8
    weight_decay: float = 0.0


def torchrec_mode(config: ModelConfig) -> str:
    torchrec = _torchrec_config(dict(config.params))
    mode = str(torchrec.get("mode", "dmp"))
    if mode not in {"disabled", "dmp"}:
        raise ValueError(
            "model.params.torchrec.mode must be one of: disabled, dmp"
        )
    return mode


def torchrec_output_dtensor(config: ModelConfig) -> bool:
    """Whether TorchRec DMP state_dict values should use DTensor.

    DTensor is the default distributed state representation. Configs may set
    ``output_dtensor: false`` to fall back to the legacy ShardedTensor path.
    """
    torchrec = _torchrec_config(dict(config.params))
    value = torchrec.get("output_dtensor", True)
    if not isinstance(value, bool):
        raise ValueError("model.params.torchrec.output_dtensor must be a boolean")
    return value


def torchrec_table_storage_policies(
    config: ModelConfig,
    *,
    table_names: Sequence[str],
    table_num_embeddings: Mapping[str, int] | None = None,
) -> dict[str, TorchRecTableStoragePolicy]:
    params = dict(config.params)
    torchrec = _torchrec_config(params)
    if "storage" not in torchrec:
        return {}
    storage = torchrec.get("storage", {})
    if storage is None:
        return {}
    if not isinstance(storage, Mapping):
        raise ValueError("model.params.torchrec.storage must be a mapping")

    if not table_names:
        raise ValueError(
            "torchrec.storage is declared but the compiled model has no embedding tables"
        )
    table_names = tuple(dict.fromkeys(str(name) for name in table_names))

    default_kernel = str(storage.get("default_compute_kernel", "fused"))
    _validate_compute_kernel(default_kernel)
    default_sharding_types = _optional_sharding_types(
        storage.get("default_sharding_types"),
        field="model.params.torchrec.storage.default_sharding_types",
    )

    raw_tables = storage.get("tables", {})
    if raw_tables is None:
        raw_tables = {}
    if not isinstance(raw_tables, Mapping):
        raise ValueError("model.params.torchrec.storage.tables must be a mapping")
    unknown_tables = sorted(
        str(name) for name in raw_tables if str(name) not in table_names
    )
    if unknown_tables:
        raise ValueError(
            "model.params.torchrec.storage.tables contains unknown embedding "
            f"tables: {unknown_tables}"
        )
    raw_capacity_rules = storage.get("capacity_rules", ())
    if not isinstance(raw_capacity_rules, Sequence) or isinstance(
        raw_capacity_rules, (str, bytes)
    ):
        raise ValueError(
            "model.params.torchrec.storage.capacity_rules must be a list"
        )
    capacity_rules: list[Mapping[str, Any]] = []
    for index, raw_rule in enumerate(raw_capacity_rules):
        if not isinstance(raw_rule, Mapping):
            raise ValueError(
                "model.params.torchrec.storage.capacity_rules"
                f"[{index}] must be a mapping"
            )
        capacity_rules.append(raw_rule)
    if capacity_rules and table_num_embeddings is None:
        raise ValueError(
            "torchrec storage capacity_rules require compiled table capacities"
        )

    policies: dict[str, TorchRecTableStoragePolicy] = {}
    for table_name in table_names:
        resolved: dict[str, Any] = {
            "compute_kernel": default_kernel,
            "sharding_types": default_sharding_types,
            "enforce_hbm": storage.get("enforce_hbm"),
        }
        if capacity_rules:
            if table_name not in table_num_embeddings:
                raise ValueError(
                    f"missing num_embeddings for table {table_name!r}"
                )
            rows = int(table_num_embeddings[table_name])
            for raw_rule in capacity_rules:
                threshold = int(raw_rule["min_vocab_size_exclusive"])
                if rows > threshold:
                    resolved.update(dict(raw_rule))
        raw_table = raw_tables.get(table_name, {})
        if raw_table is None:
            raw_table = {}
        if not isinstance(raw_table, Mapping):
            raise ValueError(
                f"model.params.torchrec.storage.tables.{table_name} must be a mapping"
            )
        resolved.update(dict(raw_table))
        compute_kernel = str(resolved.get("compute_kernel", default_kernel))
        _validate_compute_kernel(compute_kernel)
        sharding_types = _optional_sharding_types(
            resolved.get("sharding_types"),
            field=f"torchrec storage policy for table {table_name!r}.sharding_types",
        )
        cache_load_factor = resolved.get("cache_load_factor")
        if cache_load_factor is not None:
            cache_load_factor = float(cache_load_factor)
            if not 0.0 < cache_load_factor <= 1.0:
                raise ValueError(
                    f"cache_load_factor for table {table_name!r} must be in (0, 1]"
                )
        enforce_hbm = resolved.get("enforce_hbm")
        if enforce_hbm is not None:
            enforce_hbm = bool(enforce_hbm)
        policies[table_name] = TorchRecTableStoragePolicy(
            compute_kernel=compute_kernel,
            sharding_types=sharding_types,
            cache_load_factor=cache_load_factor,
            enforce_hbm=enforce_hbm,
        )
    return policies


def torchrec_sparse_optimizer_policy(
    config: ModelConfig,
) -> TorchRecSparseOptimizerPolicy | None:
    torchrec = _torchrec_config(dict(config.params))
    raw = torchrec.get("sparse_optimizer")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("model.params.torchrec.sparse_optimizer must be a mapping")
    name = str(raw.get("name", "exact_adagrad"))
    if name not in {"exact_adagrad", "exact_row_wise_adagrad"}:
        raise ValueError(
            "torchrec sparse_optimizer.name must be one of: "
            "exact_adagrad, exact_row_wise_adagrad"
        )
    lr = float(raw.get("lr", 0.01))
    eps = float(raw.get("eps", 1e-8))
    weight_decay = float(raw.get("weight_decay", 0.0))
    if lr <= 0:
        raise ValueError("torchrec sparse_optimizer.lr must be positive")
    if eps <= 0:
        raise ValueError("torchrec sparse_optimizer.eps must be positive")
    if weight_decay < 0:
        raise ValueError("torchrec sparse_optimizer.weight_decay must be non-negative")
    return TorchRecSparseOptimizerPolicy(
        name=name,
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
    )


def uses_torchrec_dmp(config: ModelConfig, context: DistributedContext) -> bool:
    return torchrec_mode(config) == "dmp" and context.enabled


def wrap_torchrec_dmp(
    model: nn.Module,
    *,
    config: ModelConfig,
    context: DistributedContext,
    device: torch.device,
) -> nn.Module:
    if not uses_torchrec_dmp(config, context):
        return model
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("TorchRec DMP requires an initialized process group")

    try:
        from torchrec.distributed.model_parallel import DistributedModelParallel
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.sharding_plan import get_default_sharders
        from torchrec.distributed.types import ShardingEnv
    except ImportError as exc:
        raise ImportError(
            "TorchRec DMP requires torchrec distributed modules on the remote "
            "Linux/GPU environment."
        ) from exc

    process_group = dist.group.WORLD
    env = ShardingEnv(
        world_size=dist.get_world_size(process_group),
        rank=dist.get_rank(process_group),
        pg=process_group,
        output_dtensor=torchrec_output_dtensor(config),
    )
    table_names = torchrec_model_table_names(model)
    table_specs = torchrec_model_table_specs(model)
    table_num_embeddings = {
        table_name: spec[0]
        for table_name, spec in table_specs.items()
    }
    table_embedding_dims = {
        table_name: spec[1]
        for table_name, spec in table_specs.items()
    }
    constraints = _build_torchrec_planner_constraints(
        config,
        table_names=table_names,
        table_num_embeddings=table_num_embeddings,
    )
    sparse_optimizer = torchrec_sparse_optimizer_policy(config)
    plan = None
    sharders = None
    if constraints or sparse_optimizer is not None:
        sharders = _torchrec_sharders(
            get_default_sharders(),
            sparse_optimizer=sparse_optimizer,
        )
    if constraints:
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=_torchrec_local_world_size(context.world_size),
                world_size=context.world_size,
                compute_device=device.type,
            ),
            constraints=constraints,
        )
        plan = planner.collective_plan(model, sharders, dist.group.WORLD)
    wrapped = DistributedModelParallel(
        module=model,
        env=env,
        device=device,
        plan=plan,
        sharders=sharders,
        init_data_parallel=True,
    )
    setattr(wrapped, "_recagora_torchrec_plan", plan)
    setattr(wrapped, "_recagora_torchrec_planner_constraints", constraints)
    setattr(wrapped, "_recagora_torchrec_sharder_count", len(sharders or ()))
    setattr(wrapped, "_recagora_torchrec_table_names", table_names)
    setattr(
        wrapped,
        "_recagora_torchrec_table_num_embeddings",
        table_num_embeddings,
    )
    setattr(
        wrapped,
        "_recagora_torchrec_table_embedding_dims",
        table_embedding_dims,
    )
    return wrapped


def build_torchrec_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    optimizer_params: Mapping[str, Any] | None = None,
    lr: float,
    weight_decay: float,
    sparse_optimizer: TorchRecSparseOptimizerPolicy | None = None,
) -> Any:
    fused_optimizer = getattr(model, "fused_optimizer", None)
    if fused_optimizer is None:
        if sparse_optimizer is not None:
            return _build_local_split_optimizer(
                model,
                optimizer_name=optimizer_name,
                optimizer_params=optimizer_params,
                lr=lr,
                weight_decay=weight_decay,
                sparse_optimizer=sparse_optimizer,
            )
        return _build_torch_optimizer(
            model.parameters(),
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    try:
        from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
        from torchrec.optim.optimizers import in_backward_optimizer_filter
    except ImportError as exc:
        raise ImportError(
            "TorchRec DMP fused optimizer support requires torchrec.optim.keyed."
        ) from exc

    fused_params = getattr(fused_optimizer, "params", {})
    dense_params = dict(in_backward_optimizer_filter(model.named_parameters()))
    if not dense_params:
        return fused_optimizer
    dense_optimizer = KeyedOptimizerWrapper(
        dense_params,
        lambda params: _build_torch_optimizer(
            params,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
        ),
    )

    if not fused_params:
        return dense_optimizer
    return CombinedOptimizer(
        [
            ("dense", dense_optimizer),
            ("fused", fused_optimizer),
        ]
    )


def _build_torch_optimizer(
    params,
    *,
    optimizer_name: str,
    optimizer_params: Mapping[str, Any] | None = None,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    kwargs = dict(optimizer_params or {})
    protected = {"lr", "weight_decay"}.intersection(kwargs)
    if protected:
        raise ValueError(
            "optimizer.params must not override lr or weight_decay: "
            f"{sorted(protected)}"
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")


def _build_local_split_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    optimizer_params: Mapping[str, Any] | None,
    lr: float,
    weight_decay: float,
    sparse_optimizer: TorchRecSparseOptimizerPolicy,
) -> Any:
    if sparse_optimizer.name != "exact_adagrad":
        raise ValueError(
            "non-DMP execution only supports exact_adagrad sparse optimizer; "
            f"got {sparse_optimizer.name!r}"
        )
    named_params = dict(model.named_parameters())
    sparse_params = {
        name: param
        for name, param in named_params.items()
        if ".embedding_collection." in f".{name}"
    }
    if not sparse_params:
        raise RuntimeError(
            "torchrec sparse_optimizer is configured but the model has no "
            "FeatureEmbedding embedding_collection parameters"
        )
    dense_params = {
        name: param for name, param in named_params.items() if name not in sparse_params
    }
    optimizers: list[tuple[str, torch.optim.Optimizer]] = []
    if dense_params:
        optimizers.append(
            (
                "dense",
                _build_torch_optimizer(
                    dense_params.values(),
                    optimizer_name=optimizer_name,
                    optimizer_params=optimizer_params,
                    lr=lr,
                    weight_decay=weight_decay,
                ),
            )
        )
    optimizers.append(
        (
            "sparse",
            torch.optim.Adagrad(
                sparse_params.values(),
                lr=sparse_optimizer.lr,
                eps=sparse_optimizer.eps,
                weight_decay=sparse_optimizer.weight_decay,
            ),
        )
    )
    return _LocalCombinedOptimizer(optimizers)


class _LocalCombinedOptimizer:
    """Small non-DMP wrapper retaining native optimizer checkpoint semantics."""

    def __init__(
        self,
        optimizers: list[tuple[str, torch.optim.Optimizer]],
    ) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            group
            for _name, optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _name, optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        for _name, optimizer in self.optimizers:
            optimizer.step(closure=closure)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers
            }
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        states = state_dict.get("optimizers")
        if not isinstance(states, Mapping):
            raise ValueError("local combined optimizer checkpoint is malformed")
        expected = {name for name, _optimizer in self.optimizers}
        if set(states) != expected:
            raise ValueError(
                "local combined optimizer members differ: "
                f"expected={sorted(expected)} actual={sorted(states)}"
            )
        for name, optimizer in self.optimizers:
            optimizer.load_state_dict(states[name])


def _torchrec_sharders(
    sharders: list[Any],
    *,
    sparse_optimizer: TorchRecSparseOptimizerPolicy | None,
) -> list[Any]:
    if sparse_optimizer is None:
        return sharders
    try:
        from fbgemm_gpu.split_embedding_configs import EmbOptimType
        from torchrec.distributed.embedding import EmbeddingCollectionSharder
    except ImportError as exc:
        raise ImportError(
            "TorchRec sparse optimizer configuration requires fbgemm_gpu and "
            "EmbeddingCollectionSharder."
        ) from exc

    optimizer_by_name = {
        "exact_adagrad": EmbOptimType.EXACT_ADAGRAD,
        "exact_row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
    }
    fused_params = {
        "optimizer": optimizer_by_name[sparse_optimizer.name],
        "learning_rate": sparse_optimizer.lr,
        "eps": sparse_optimizer.eps,
        "weight_decay": sparse_optimizer.weight_decay,
    }
    replaced = False
    configured: list[Any] = []
    for sharder in sharders:
        if isinstance(sharder, EmbeddingCollectionSharder):
            configured.append(EmbeddingCollectionSharder(fused_params=fused_params))
            replaced = True
        else:
            configured.append(sharder)
    if not replaced:
        raise RuntimeError(
            "TorchRec default sharders contain no EmbeddingCollectionSharder"
        )
    return configured


def _build_torchrec_planner_constraints(
    config: ModelConfig,
    *,
    table_names: Sequence[str],
    table_num_embeddings: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    policies = torchrec_table_storage_policies(
        config,
        table_names=table_names,
        table_num_embeddings=table_num_embeddings,
    )
    if not policies:
        return {}
    from torchrec.distributed.planner import ParameterConstraints
    try:
        from torchrec.distributed.types import CacheParams
    except ImportError as exc:
        raise ImportError(
            "TorchRec UVM caching storage policies require CacheParams from "
            "torchrec.distributed.types."
        ) from exc

    constraints: dict[str, Any] = {}
    for table_name, policy in policies.items():
        cache_params = (
            CacheParams(load_factor=policy.cache_load_factor)
            if policy.cache_load_factor is not None
            else None
        )
        constraints[table_name] = ParameterConstraints(
            sharding_types=(
                list(policy.sharding_types)
                if policy.sharding_types is not None
                else None
            ),
            compute_kernels=[policy.compute_kernel],
            cache_params=cache_params,
            enforce_hbm=policy.enforce_hbm,
        )
    return constraints


def torchrec_model_table_names(model: nn.Module) -> tuple[str, ...]:
    """Return the actual compiled FeatureEmbedding tables owned by a model."""
    names: list[str] = []
    seen: set[str] = set()
    for module in model.modules():
        table_names = getattr(module, "table_names", None)
        if table_names is None or not hasattr(module, "embedding_collection"):
            continue
        for raw_name in table_names:
            name = str(raw_name)
            if name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


def torchrec_model_table_num_embeddings(model: nn.Module) -> dict[str, int]:
    """Return physical table capacities from actual FeatureEmbedding owners."""
    return {
        table_name: spec[0]
        for table_name, spec in torchrec_model_table_specs(model).items()
    }


def torchrec_model_table_embedding_dims(model: nn.Module) -> dict[str, int]:
    """Return physical table dimensions from actual FeatureEmbedding owners."""
    return {
        table_name: spec[1]
        for table_name, spec in torchrec_model_table_specs(model).items()
    }


def torchrec_model_table_specs(
    model: nn.Module,
) -> dict[str, tuple[int, int]]:
    """Return physical table ``(num_embeddings, embedding_dim)`` pairs."""
    specs: dict[str, tuple[int, int]] = {}
    for module in model.modules():
        explicit = getattr(module, "acr_table_specs", None)
        if explicit is not None:
            for table_name, spec in explicit.items():
                specs[str(table_name)] = (int(spec[0]), int(spec[1]))
        input_plan = getattr(module, "input_plan", None)
        if input_plan is None or not hasattr(module, "embedding_collection"):
            continue
        features = getattr(input_plan, "categorical_features", ())
        for feature in features:
            table_name = str(feature.resolved_embedding_name)
            num_embeddings = int(feature.vocab_size)
            embedding_dim = int(feature.embedding_dim)
            spec = (num_embeddings, embedding_dim)
            existing = specs.get(table_name)
            if existing is not None and existing != spec:
                raise ValueError(
                    f"table {table_name!r} has inconsistent shape: "
                    f"{existing!r} != {spec!r}"
                )
            specs[table_name] = spec
    return specs


def _torchrec_config(params: Mapping[str, Any]) -> Mapping[str, Any]:
    torchrec = params.get("torchrec", {})
    if torchrec is None:
        return {}
    if not isinstance(torchrec, Mapping):
        raise ValueError("model.params.torchrec must be a mapping")
    return torchrec


def _validate_compute_kernel(value: str) -> None:
    if value not in _VALID_COMPUTE_KERNELS:
        raise ValueError(
            f"Unsupported TorchRec compute kernel {value!r}; expected one of "
            f"{sorted(_VALID_COMPUTE_KERNELS)}"
        )


def _optional_sharding_types(
    value: Any,
    *,
    field: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"{field} must be a non-empty list")
    resolved = tuple(str(item) for item in value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} contains duplicate values: {resolved}")
    unsupported = sorted(set(resolved) - _VALID_SHARDING_TYPES)
    if unsupported:
        raise ValueError(
            f"Unsupported TorchRec sharding types {unsupported}; expected values from "
            f"{sorted(_VALID_SHARDING_TYPES)}"
        )
    return resolved


def _torchrec_local_world_size(world_size: int) -> int:
    from torchrec.distributed.comm import get_local_size

    return int(get_local_size(world_size))

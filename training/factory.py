from __future__ import annotations

from typing import cast

from torch import nn

from recagora import RecAgoraRankingModel
from training._dynamic_import import resolve_local_symbol
from utils.compiler.config import ModelConfig
from utils.compiler.feature_contract import FeatureContractPlans
from utils.runtime.model.adapter import compile_recagora_model_adapter
from utils.runtime.model.lookup import CompiledSlotLookup
from utils.runtime.model.runner import RecAgoraRuntimeShell


def _build_recagora_model(
    config: ModelConfig,
    *,
    input_contract: FeatureContractPlans | None,
    model_cls: type[RecAgoraRankingModel],
) -> nn.Module:
    if input_contract is None:
        raise ValueError(
            f"{config.name} requires a compiled FeatureContractPlans input_contract"
        )
    if not isinstance(input_contract, FeatureContractPlans):
        raise TypeError("ranking models require a FeatureContractPlans input_contract")
    adapter = compile_recagora_model_adapter(input_contract)
    params = dict(config.params)
    params.pop("inputs", None)
    params.pop("feature", None)
    params.pop("input_view", None)
    params.pop("input_reduction", None)
    params.pop("torchrec", None)
    model = model_cls(input_spec=adapter.input_spec, **params)
    model.bind_slot_lookup(
        CompiledSlotLookup(
            input_plan=input_contract.embedding_input_plan,
            adapter=adapter,
        )
    )
    return RecAgoraRuntimeShell(model=model, plans=input_contract)


def _load_model_class(
    class_path: str,
    *,
    base_class: type[nn.Module],
) -> type[nn.Module]:
    value = resolve_local_symbol(class_path, field="model.class_path")
    if not isinstance(value, type) or not issubclass(value, base_class):
        raise TypeError(
            f"model.class_path {class_path!r} must resolve to an "
            f"{base_class.__name__} subclass"
        )
    return value


def _load_recagora_model_class(class_path: str) -> type[RecAgoraRankingModel]:
    return cast(
        type[RecAgoraRankingModel],
        _load_model_class(class_path, base_class=RecAgoraRankingModel),
    )


def build_model(
    config: ModelConfig,
    *,
    input_contract: FeatureContractPlans | None = None,
) -> nn.Module:
    if config.class_path is None:
        raise ValueError(
            f"model {config.name!r} requires an explicit model.class_path"
        )
    model_cls = _load_recagora_model_class(config.class_path)
    return _build_recagora_model(
        config,
        input_contract=input_contract,
        model_cls=model_cls,
    )

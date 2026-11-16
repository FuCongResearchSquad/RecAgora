from __future__ import annotations

from .config import DataConfig, ModelConfig
from .feature_contract import FeatureContractPlans, build_feature_contract


ModelInputContract = FeatureContractPlans


def build_model_input_contract(
    data_config: DataConfig,
    model_config: ModelConfig,
) -> ModelInputContract:
    """Compile the ranking feature and embedding input contract."""

    return build_feature_contract(data_config, model_config).build_all_plans()


def data_loader_contract_kwargs(
    contract: ModelInputContract,
) -> dict[str, object]:
    """Return explicit worker-plan arguments without task-name branching."""

    return {
        "read_plan": contract.read_plan,
        "parse_plan": contract.parse_plan,
        "adapter_plan": contract.adapter_plan,
        "embedding_input_plan": contract.embedding_input_plan,
    }

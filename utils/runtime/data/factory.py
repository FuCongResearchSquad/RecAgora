from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset, IterableDataset

from .dataset import RequestDataset
from utils.compiler.config import DataConfig

if TYPE_CHECKING:
    from utils.runtime.data.prepack import CategoricalPrepackPlan


DatasetType = Dataset | IterableDataset


def build_request_dataset(
    config: DataConfig,
    *,
    batch_size: int | None = None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    start_rank_batch_index: int = 0,
    start_global_batch_id: int | None = None,
    resume_global_batch_stop: int | None = None,
    resume_previous_world_size: int | None = None,
    read_plan: object | None = None,
    parse_plan: object | None = None,
    categorical_prepack_plan: "CategoricalPrepackPlan | None" = None,
) -> DatasetType:
    return RequestDataset(
        config,
        batch_size=batch_size or 128,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        start_rank_batch_index=start_rank_batch_index,
        start_global_batch_id=start_global_batch_id,
        resume_global_batch_stop=resume_global_batch_stop,
        resume_previous_world_size=resume_previous_world_size,
        read_plan=read_plan,
        parse_plan=parse_plan,
        categorical_prepack_plan=categorical_prepack_plan,
    )

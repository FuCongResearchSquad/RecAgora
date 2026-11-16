from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence, cast

from .config import (
    DataConfig,
    HistoryGroupConfig,
    ModelConfig,
    _mapping_or_empty,
    _validated_dataset_feature_bindings,
    load_yaml_mapping,
    normalize_slot_id,
)
from .manifest import (
    ResolvedModelFeatureManifest,
    compile_model_feature_manifest,
)


LOGGER = logging.getLogger(__name__)


FeatureKind = Literal["sparse", "varlen_sparse", "dense"]
FeatureRole = Literal["request", "item", "sequence"]
EncodingKind = Literal["embedding", "direct"]
ReductionKind = Literal["mean", "sum", "max", "none"]
SharingType = Literal["semantic_share", "offset_merge"]
MappingType = Literal["modulo_bucket", "dense_map"]


@dataclass(frozen=True)
class DataFeatureSpec:
    name: str
    source_path: str
    source_level: str
    reader_group: str | None
    reader_shape: str | None
    parser_kind: str | None
    dtype: str
    parse: Mapping[str, Any] | None = None
    max_outer_len: int | None = None
    max_inner_len: int | None = None
    representation: str | None = None
    role: FeatureRole | None = None
    encoding: EncodingKind | None = None
    reduction: ReductionKind | None = None
    namespace_id: int | None = None
    history_group: str | None = None
    slot_id: int | None = None
    view: str | None = None


@dataclass(frozen=True)
class DataFeatureUniverse:
    version: str
    features_by_name: Mapping[str, DataFeatureSpec]
    features_by_source_path: Mapping[str, DataFeatureSpec]
    features_by_slot_id: Mapping[int, DataFeatureSpec]
    id_fields: Mapping[str, str]
    label_fields: tuple[str, ...]
    candidate_metadata: Mapping[str, str]
    target: str
    list_len: int
    history_len: int
    history_groups: Mapping[str, HistoryGroupConfig]
    missing_slot_policy: str
    bad_length_policy: str


@dataclass(frozen=True)
class ModelFeatureSpec:
    name: str
    kind: FeatureKind
    bucket_size: int
    embedding_dim: int
    table_name: str
    sharing_type: SharingType
    id_mapping: Mapping[str, Any]
    source_path: str | None = None
    maxlen: int | None = None
    length_name: str | None = None
    active: bool = True
    namespace_id: int | None = None
    representation: str | None = None
    slot_id: int | None = None
    mapped_ids_exposed: bool = False


@dataclass(frozen=True)
class ModelFeatureRequest:
    input_view: str
    active_features: tuple[ModelFeatureSpec, ...]
    dense_features: tuple[str, ...]
    history_feature_list: tuple[str, ...]
    torchrec_runtime: Mapping[str, Any]
    loss_config: Mapping[str, Any]
    input_reduction: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadPlanFeature:
    feature_name: str
    source_path: str
    reader_group: str | None
    reader_shape: str | None
    parser_kind: str | None
    dtype: str
    parse: Mapping[str, Any] = field(default_factory=dict)
    max_outer_len: int | None = None
    max_inner_len: int | None = None
    representation: str | None = None
    namespace_id: int | None = None
    history_group: str | None = None
    history_rule: HistoryGroupConfig | None = None
    # Compiled plans always set role explicitly. ``None`` remains supported for
    # hand-built/legacy plans so the reader can infer it from the physical
    # parser metadata without reviving an ``is_varlen`` switch.
    role: FeatureRole | None = None
    encoding: EncodingKind = "embedding"


@dataclass(frozen=True)
class ReadPlan:
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ParsePlan:
    id_fields: Mapping[str, str]
    labels: tuple[str, ...]
    target: str
    features: tuple[ReadPlanFeature, ...]
    missing_slot_policy: str
    bad_length_policy: str
    candidate_metadata: Mapping[str, str] = field(default_factory=dict)
    history_groups: Mapping[str, HistoryGroupConfig] = field(default_factory=dict)
    history_length_features: tuple[ReadPlanFeature, ...] = ()


@dataclass(frozen=True)
class AdapterPlan:
    input_view: str
    list_len: int
    categorical_features: Mapping[str, Mapping[str, str]]
    direct_features: Mapping[str, Mapping[str, str]]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "list_len": self.list_len,
            "categorical_features": dict(self.categorical_features),
            "direct_features": dict(self.direct_features),
            "target_label_source": "target_label",
        }


@dataclass(frozen=True)
class EmbeddingTable:
    table_name: str
    num_embeddings: int
    embedding_dim: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingLookup:
    feature_name: str
    table_name: str
    bucket_size: int
    mapping_type: MappingType


@dataclass(frozen=True)
class EmbeddingPlan:
    """Single source of truth for ID feature to embedding-table lookup."""

    tables: Mapping[str, EmbeddingTable]
    feature_to_table: Mapping[str, str]
    lookups: Mapping[str, EmbeddingLookup]


@dataclass(frozen=True)
class EmbeddingInputFeature:
    """Fully compiled feature contract shared by the KJT adapter and lookup layer."""

    name: str
    source_path: str
    source_level: str
    role: FeatureRole
    encoding: EncodingKind
    reduction: ReductionKind
    representation: str
    history_group: str | None
    max_sequence_length: int | None
    length_source: str | None
    table_name: str | None
    vocab_size: int
    embedding_dim: int
    id_mapping: Mapping[str, Any]
    slot_id: int
    mapped_ids_exposed: bool = False

    @property
    def key(self) -> str:
        return self.name

    @property
    def resolved_embedding_name(self) -> str:
        return str(self.table_name or self.name)


@dataclass(frozen=True)
class EmbeddingInputPlan:
    """Single compiled input specification consumed by adapter and embedding lookup."""

    input_view: str
    list_len: int
    features: tuple[EmbeddingInputFeature, ...]

    @property
    def id_features(self) -> tuple[EmbeddingInputFeature, ...]:
        return self.categorical_features

    @property
    def categorical_features(self) -> tuple[EmbeddingInputFeature, ...]:
        return tuple(feature for feature in self.features if feature.encoding == "embedding")

    @property
    def direct_features(self) -> tuple[EmbeddingInputFeature, ...]:
        return tuple(feature for feature in self.features if feature.encoding == "direct")

    @property
    def mapped_id_features(self) -> tuple[EmbeddingInputFeature, ...]:
        return tuple(feature for feature in self.features if feature.mapped_ids_exposed)

    @property
    def tables(self) -> tuple[str, ...]:
        """Actual lookup tables after namespace sharing has been compiled."""
        return tuple(
            dict.fromkeys(
                feature.resolved_embedding_name
                for feature in self.categorical_features
            )
        )


@dataclass(frozen=True)
class FeatureContractPlans:
    read_plan: ReadPlan
    parse_plan: ParsePlan
    adapter_plan: AdapterPlan
    embedding_input_plan: EmbeddingInputPlan
    resolved_manifest: ResolvedModelFeatureManifest


@dataclass(frozen=True)
class FeatureContract:
    data: DataFeatureUniverse
    model: ModelFeatureRequest
    resolved_manifest: ResolvedModelFeatureManifest

    def _validate_model_and_adapter(self, adapter: Mapping[str, Any]) -> None:
        if not self.model.active_features:
            raise ValueError("FeatureContract requires at least one active model feature")
        seen_features: set[str] = set()
        for feature in self.model.active_features:
            if feature.name in seen_features:
                raise ValueError(f"duplicate model feature request: {feature.name!r}")
            seen_features.add(feature.name)
            if feature.kind == "dense":
                continue
            if feature.bucket_size <= 0:
                raise ValueError(
                    f"feature {feature.name!r} bucket_size must be positive"
                )
            if feature.embedding_dim <= 0:
                raise ValueError(
                    f"feature {feature.name!r} embedding_dim must be positive"
                )
            mapping_type = str(feature.id_mapping.get("type", "modulo_bucket"))
            if mapping_type not in {"modulo_bucket", "dense_map"}:
                raise ValueError(
                    f"feature {feature.name!r} mapping type must be one of: "
                    "modulo_bucket, dense_map"
                )

        adapter_features = _adapter_feature_sources(adapter)
        missing_adapter = [
            feature.name
            for feature in self.model.active_features
            if feature.name not in adapter_features
        ]
        if missing_adapter:
            raise ValueError(
                "missing adapter sources for active features: "
                f"{sorted(missing_adapter)!r}"
            )
        for feature_name, source_path in adapter_features.items():
            if feature_name not in seen_features:
                continue
            if source_path not in self.data.features_by_source_path:
                raise ValueError(
                    f"feature {feature_name!r} source_path {source_path!r} "
                    "is not in the data feature contract"
                )

    def _build_read_plan(self, parse_plan: ParsePlan) -> ReadPlan:
        columns = [
            _physical_column(source)
            for source in parse_plan.id_fields.values()
        ]
        columns.extend(_physical_column(label) for label in parse_plan.labels)
        columns.extend(
            _physical_column(name) for name in parse_plan.candidate_metadata
        )
        columns.extend(
            _physical_column(feature.source_path)
            for feature in parse_plan.features
        )
        columns.extend(
            _physical_column(feature.source_path)
            for feature in parse_plan.history_length_features
        )
        return ReadPlan(columns=tuple(dict.fromkeys(columns)))

    def _build_parse_plan(self, adapter: Mapping[str, Any]) -> ParsePlan:
        adapter_features = _adapter_feature_sources(adapter)
        features: list[ReadPlanFeature] = []
        for model_feature in self.model.active_features:
            source_path = adapter_features[model_feature.name]
            data_feature = self.data.features_by_source_path[source_path]
            features.append(
                _read_plan_feature(
                    data_feature,
                    feature_name=model_feature.name,
                    encoding=_feature_encoding(data_feature, model_feature),
                    history_groups=self.data.history_groups,
                )
            )
        history_length_features_by_source: dict[str, ReadPlanFeature] = {}
        for data_feature in self.data.features_by_slot_id.values():
            group_name = data_feature.history_group
            if group_name is None:
                continue
            group = self.data.history_groups[group_name]
            if not group.validate_aligned_lengths:
                continue
            history_length_features_by_source[data_feature.source_path] = (
                _read_plan_feature(
                    data_feature,
                    feature_name=data_feature.name,
                    encoding=cast(
                        EncodingKind, data_feature.encoding or "embedding"
                    ),
                    history_groups=self.data.history_groups,
                )
            )
        history_length_features = tuple(history_length_features_by_source.values())
        return ParsePlan(
            id_fields=dict(self.data.id_fields),
            labels=self.data.label_fields,
            target=self.data.target,
            features=tuple(features),
            missing_slot_policy=self.data.missing_slot_policy,
            bad_length_policy=self.data.bad_length_policy,
            candidate_metadata=dict(self.data.candidate_metadata),
            history_groups=dict(self.data.history_groups),
            history_length_features=history_length_features,
        )

    def build_embedding_plan(self) -> EmbeddingPlan:
        tables: dict[str, EmbeddingTable] = {}
        lookups: dict[str, EmbeddingLookup] = {}
        for feature in self.model.active_features:
            if feature.kind == "dense":
                continue
            if feature.sharing_type != "semantic_share":
                raise ValueError(
                    f"embedding feature {feature.name!r} has unsupported sharing "
                    f"type {feature.sharing_type!r}"
                )
            existing = tables.get(feature.table_name)
            if existing is not None:
                if existing.num_embeddings != feature.bucket_size:
                    raise ValueError(
                        f"embedding table {feature.table_name!r} has inconsistent "
                        "num_embeddings"
                    )
                if existing.embedding_dim != feature.embedding_dim:
                    raise ValueError(
                        f"embedding table {feature.table_name!r} has inconsistent "
                        "embedding_dim"
                    )
                tables[feature.table_name] = EmbeddingTable(
                    table_name=existing.table_name,
                    num_embeddings=existing.num_embeddings,
                    embedding_dim=existing.embedding_dim,
                    features=(*existing.features, feature.name),
                )
            else:
                tables[feature.table_name] = EmbeddingTable(
                    table_name=feature.table_name,
                    num_embeddings=feature.bucket_size,
                    embedding_dim=feature.embedding_dim,
                    features=(feature.name,),
                )
            lookups[feature.name] = EmbeddingLookup(
                feature_name=feature.name,
                table_name=feature.table_name,
                bucket_size=feature.bucket_size,
                mapping_type=cast(
                    MappingType,
                    str(feature.id_mapping.get("type", "modulo_bucket")),
                ),
            )
        return EmbeddingPlan(
            tables=tables,
            feature_to_table={name: lookup.table_name for name, lookup in lookups.items()},
            lookups=lookups,
        )

    def _build_adapter_plan(self, adapter: Mapping[str, Any]) -> AdapterPlan:
        return AdapterPlan(
            input_view=self.model.input_view,
            list_len=self.data.list_len,
            categorical_features={
                str(name): dict(value)
                for name, value in _mapping_or_empty(adapter.get("categorical_features")).items()
            },
            direct_features={
                str(name): dict(value)
                for name, value in _mapping_or_empty(adapter.get("direct_features")).items()
            },
        )

    def build_embedding_input_plan(
        self,
        embedding_plan: EmbeddingPlan,
        adapter_plan: AdapterPlan,
    ) -> EmbeddingInputPlan:
        adapter_sources = {
            **adapter_plan.categorical_features,
            **adapter_plan.direct_features,
        }
        features: list[EmbeddingInputFeature] = []
        for feature in self.model.active_features:
            source = adapter_sources.get(feature.name)
            if source is None:
                raise ValueError(
                    f"embedding input plan has no adapter source for {feature.name!r}"
                )
            source_path = str(source["source"])
            source_level = str(source.get("level", "request"))
            data_feature = self.data.features_by_source_path[source_path]
            role = _feature_role(data_feature)
            encoding = _feature_encoding(data_feature, feature)
            reduction = _feature_reduction(
                self.model.input_reduction,
                role=role,
                source_path=source_path,
                feature_name=feature.name,
                universe_default=data_feature.reduction,
            )
            table_name: str | None = None
            if encoding == "embedding":
                table_name = embedding_plan.feature_to_table.get(feature.name)
                if table_name is None:
                    raise ValueError(
                        f"embedding input plan has no table for {feature.name!r}"
                    )
                table = embedding_plan.tables[table_name]
                if (
                    table.num_embeddings != feature.bucket_size
                    or table.embedding_dim != feature.embedding_dim
                ):
                    raise ValueError(
                        f"embedding input plan table mismatch for {feature.name!r}"
                    )
            features.append(
                EmbeddingInputFeature(
                    name=feature.name,
                    source_path=source_path,
                    source_level=source_level,
                    role=role,
                    encoding=encoding,
                    reduction=reduction,
                    representation=(
                        data_feature.representation
                        or feature.representation
                        or ("time" if encoding == "direct" else "sparse_id")
                    ),
                    history_group=data_feature.history_group,
                    max_sequence_length=_embedding_max_sequence_length(
                        data_feature,
                        history_groups=self.data.history_groups,
                    ),
                    length_source=(
                        None
                        if data_feature.history_group is None
                        else self.data.history_groups[data_feature.history_group].anchor
                    ),
                    table_name=table_name,
                    vocab_size=feature.bucket_size,
                    embedding_dim=feature.embedding_dim,
                    id_mapping=dict(feature.id_mapping),
                    slot_id=int(feature.slot_id),
                    mapped_ids_exposed=feature.mapped_ids_exposed,
                )
            )
        return EmbeddingInputPlan(
            input_view=adapter_plan.input_view,
            list_len=adapter_plan.list_len,
            features=tuple(features),
        )

    def build_adapter(self) -> Mapping[str, Any]:
        categorical_features: dict[str, dict[str, Any]] = {}
        direct_features: dict[str, dict[str, Any]] = {}
        for feature in self.model.active_features:
            source_path = feature.source_path
            data_feature = None
            if source_path is not None:
                source_path = _normalize_source_path(source_path)
                data_feature = self.data.features_by_source_path.get(source_path)
                if data_feature is None and source_path == "item_id":
                    if "item_id" not in self.data.id_fields:
                        raise ValueError(
                            f"model feature {feature.name!r} requires a top-level item_id column"
                        )
                    source_path = self.data.id_fields["item_id"]
                    data_feature = self.data.features_by_source_path.get(source_path)
            if source_path is None:
                data_feature = self.data.features_by_name.get(feature.name)
                source_path = data_feature.source_path if data_feature is not None else feature.name
            feature_spec = {
                "source": source_path,
                "level": data_feature.source_level if data_feature is not None else "request",
            }
            if feature.kind == "dense":
                direct_features[feature.name] = feature_spec
            else:
                categorical_features[feature.name] = feature_spec

        adapter: dict[str, Any] = {
            "categorical_features": categorical_features,
            "direct_features": direct_features,
        }
        return adapter

    def build_all_plans(self) -> FeatureContractPlans:
        adapter = self.build_adapter()
        self._validate_model_and_adapter(adapter)
        embedding_plan = self.build_embedding_plan()
        adapter_plan = self._build_adapter_plan(adapter)
        parse_plan = self._build_parse_plan(adapter)
        read_plan = self._build_read_plan(parse_plan)
        embedding_input_plan = self.build_embedding_input_plan(
            embedding_plan,
            adapter_plan,
        )
        return FeatureContractPlans(
            read_plan=read_plan,
            parse_plan=parse_plan,
            adapter_plan=adapter_plan,
            embedding_input_plan=embedding_input_plan,
            resolved_manifest=self.resolved_manifest,
        )


def _read_plan_feature(
    data_feature: DataFeatureSpec,
    *,
    feature_name: str,
    encoding: EncodingKind,
    history_groups: Mapping[str, HistoryGroupConfig],
) -> ReadPlanFeature:
    history_rule = None
    if data_feature.history_group is not None:
        history_rule = history_groups[data_feature.history_group]
    return ReadPlanFeature(
        feature_name=feature_name,
        source_path=data_feature.source_path,
        reader_group=data_feature.reader_group,
        reader_shape=data_feature.reader_shape,
        parser_kind=data_feature.parser_kind,
        dtype=data_feature.dtype,
        parse=dict(data_feature.parse or {}),
        max_outer_len=data_feature.max_outer_len,
        max_inner_len=data_feature.max_inner_len,
        representation=data_feature.representation,
        namespace_id=data_feature.namespace_id,
        history_group=data_feature.history_group,
        history_rule=history_rule,
        role=_feature_role(data_feature),
        encoding=encoding,
    )


def build_feature_contract(
    data_config: DataConfig,
    model_config: ModelConfig,
) -> FeatureContract:
    data = build_data_feature_universe(data_config)
    if data_config.slot_registry is None:
        raise ValueError("model declaration requires a feature contract")
    if data_config.sparse_id_contract.layout is None:
        raise ValueError("model declaration requires a dataset contract")
    resolved_manifest = compile_model_feature_manifest(
        load_yaml_mapping(data_config.slot_registry),
        load_yaml_mapping(data_config.sparse_id_contract.layout),
        model_config.feature_declaration,
    )
    LOGGER.info(
        "resolved_model_feature_manifest feature_version=%s feature_count=%d "
        "slot_ids=%s",
        resolved_manifest.universe_version,
        resolved_manifest.feature_count,
        list(resolved_manifest.slot_ids),
    )
    model = _model_feature_request_from_resolved_manifest(
        config=model_config,
        data=data,
        manifest=resolved_manifest,
    )
    return FeatureContract(
        data=data,
        model=model,
        resolved_manifest=resolved_manifest,
    )


def _model_feature_request_from_resolved_manifest(
    *,
    config: ModelConfig,
    data: DataFeatureUniverse,
    manifest: ResolvedModelFeatureManifest,
) -> ModelFeatureRequest:
    params = dict(config.params)
    table_by_namespace = {
        table.namespace: table for table in manifest.embedding_tables
    }
    features: list[ModelFeatureSpec] = []
    dense_features: list[str] = []
    history_features: list[str] = []
    for resolved in manifest.features:
        data_feature = data.features_by_slot_id.get(resolved.slot_id)
        if data_feature is None:
            raise ValueError(
                f"resolved manifest slot {resolved.slot_id} is missing from data universe"
            )
        name = data_feature.name
        kind = _feature_kind_from_data_feature(data_feature)
        if kind == "dense":
            dense_features.append(name)
            bucket_size = 0
            embedding_dim = 0
            table_name = name
            id_mapping: Mapping[str, Any] = {"type": "direct"}
        else:
            namespace = data_feature.namespace_id
            if namespace is None:
                raise ValueError(
                    f"categorical slot {resolved.slot_id} has no namespace"
                )
            table = table_by_namespace.get(namespace)
            if table is None:
                raise ValueError(
                    f"categorical slot {resolved.slot_id} has no compiled table"
                )
            bucket_size = table.num_embeddings
            embedding_dim = table.embedding_dim
            table_name = table.table_name
            id_mapping = {
                "type": "dense_map",
                "namespace": namespace,
                "mapping_version": table.mapping_version,
                "padding_id": table.padding_id,
                "oov_id": table.oov_id,
            }
            if kind == "varlen_sparse":
                history_features.append(name)
        features.append(
            ModelFeatureSpec(
                name=name,
                kind=kind,
                bucket_size=bucket_size,
                embedding_dim=embedding_dim,
                table_name=table_name,
                sharing_type="semantic_share",
                id_mapping=id_mapping,
                source_path=data_feature.source_path,
                maxlen=data_feature.max_outer_len,
                active=True,
                namespace_id=data_feature.namespace_id,
                representation=data_feature.representation,
                slot_id=resolved.slot_id,
                mapped_ids_exposed=resolved.mapped_ids_exposed,
            )
        )
    return ModelFeatureRequest(
        input_view=str(params.get("input_view", "listwise")),
        active_features=tuple(features),
        dense_features=tuple(dense_features),
        history_feature_list=tuple(history_features),
        torchrec_runtime=_mapping_or_empty(params.get("torchrec")),
        loss_config={
            "output_label_names": tuple(
                str(name) for name in params.get("output_label_names", ())
            ),
            "loss_weights": _mapping_or_empty(params.get("loss_weights")),
        },
        input_reduction=_mapping_or_empty(params.get("input_reduction")),
    )


def _feature_kind_from_data_feature(feature: DataFeatureSpec) -> FeatureKind:
    if feature.encoding == "direct" or str(feature.representation) == "time":
        return "dense"
    parser_kind = str((feature.parse or {}).get("kind") or feature.parser_kind or "")
    if parser_kind in {
        "request_sequence",
        "item_sequence",
        "request_sequence_pad_truncate",
        "item_sequence_pad_inner",
    }:
        return "varlen_sparse"
    return "sparse"


def build_data_feature_universe(config: DataConfig) -> DataFeatureUniverse:
    if config.slot_registry is None:
        raise ValueError("data feature contract path is required")
    registry_raw = load_yaml_mapping(config.slot_registry)
    if registry_raw.get("config_kind") != "feature":
        raise ValueError("data feature contract must have config_kind=feature")
    if registry_raw.get("features") is None:
        raise ValueError("data feature contract requires a features list")
    return _build_v1_data_feature_universe(config, registry_raw)


def _build_v1_data_feature_universe(
    config: DataConfig,
    universe_raw: Mapping[str, Any],
) -> DataFeatureUniverse:
    if config.sparse_id_contract.mapping_type != "dense_map":
        raise ValueError("feature v1 requires a dense_map dataset contract")
    if config.sparse_id_contract.layout is None:
        raise ValueError("feature v1 requires a dataset layout")
    dataset_raw = load_yaml_mapping(config.sparse_id_contract.layout)
    if dataset_raw.get("config_kind") != "dataset":
        raise ValueError("feature v1 layout must have config_kind=dataset")
    universe_version = str(universe_raw.get("version", ""))
    dataset_version = str(dataset_raw.get("version", ""))
    if not universe_version or universe_version != dataset_version:
        raise ValueError(
            "feature and dataset versions must match: "
            f"feature={universe_version!r} dataset={dataset_version!r}"
        )

    bindings_by_slot = {
        int(raw_binding["slot_id"]): raw_binding
        for raw_binding in _validated_dataset_feature_bindings(dataset_raw)
    }

    raw_features = universe_raw.get("features")
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise ValueError("feature.features must be a feature list")
    features_by_name: dict[str, DataFeatureSpec] = {}
    features_by_source_path: dict[str, DataFeatureSpec] = {}
    features_by_slot_id: dict[int, DataFeatureSpec] = {}
    for raw_feature in raw_features:
        if not isinstance(raw_feature, Mapping):
            raise ValueError("feature.features entries must be mappings")
        slot_id = int(raw_feature["slot_id"])
        if slot_id in features_by_slot_id:
            raise ValueError(f"feature.features duplicates slot_id {slot_id}")
        binding = bindings_by_slot.get(slot_id)
        if binding is None:
            raise ValueError(f"dataset has no physical binding for slot_id {slot_id}")
        parse = _mapping_or_empty(binding.get("parse"))
        encoding = str(raw_feature.get("encoding", "embedding"))
        namespace_id = (
            int(binding["namespace_id"])
            if binding.get("namespace_id") is not None
            else None
        )
        if encoding == "embedding" and namespace_id is None:
            raise ValueError(f"embedding slot {slot_id} requires namespace_id")
        if encoding == "direct" and namespace_id is not None:
            raise ValueError(f"direct slot {slot_id} must not define namespace_id")
        raw_max_inner_len = binding.get(
            "max_inner_len", parse.get("max_inner_len")
        )
        max_inner_len = (
            int(raw_max_inner_len) if raw_max_inner_len is not None else None
        )
        if max_inner_len is not None and max_inner_len <= 0:
            raise ValueError(f"slot {slot_id} max_inner_len must be positive")
        name = f"slot_{slot_id}"
        source_path = str(binding["source_path"])
        spec = DataFeatureSpec(
            name=name,
            source_path=source_path,
            source_level=str(binding["source_level"]),
            reader_group=source_path.split(".", 1)[0] if "." in source_path else None,
            reader_shape=str(parse.get("kind")) if parse.get("kind") is not None else None,
            parser_kind=str(parse.get("kind")) if parse.get("kind") is not None else None,
            dtype=str(binding["dtype"]),
            parse=dict(parse),
            max_outer_len=(
                config.history_groups[str(binding["history_group"])].max_len
                if binding.get("history_group") is not None
                else config.list_len
                if str(binding["source_level"]) == "item"
                else None
            ),
            max_inner_len=max_inner_len,
            representation=str(raw_feature["representation"]),
            role=cast(FeatureRole, str(raw_feature["role"])),
            encoding=cast(EncodingKind, encoding),
            reduction=cast(ReductionKind, str(raw_feature["reduction"])),
            namespace_id=namespace_id,
            history_group=(
                str(binding["history_group"])
                if binding.get("history_group") is not None
                else None
            ),
            slot_id=slot_id,
            view=str(raw_feature["view"]),
        )
        features_by_name[name] = spec
        features_by_source_path[source_path] = spec
        features_by_slot_id[slot_id] = spec

    extra_bindings = sorted(set(bindings_by_slot) - set(features_by_slot_id))
    if extra_bindings:
        raise ValueError(
            f"dataset has bindings outside the feature contract: {extra_bindings}"
        )
    id_fields = {"request_id": config.request_id_field}
    if config.item_id_field is not None:
        id_fields["item_id"] = config.item_id_field
    if config.user_id_field is not None:
        id_fields["user_id"] = config.user_id_field
    _add_v1_top_level_features(
        features_by_name=features_by_name,
        features_by_source_path=features_by_source_path,
        config=config,
    )
    _validate_history_feature_groups(
        features_by_name=features_by_name,
        history_groups=config.history_groups,
    )
    return DataFeatureUniverse(
        version=universe_version,
        features_by_name=features_by_name,
        features_by_source_path=features_by_source_path,
        features_by_slot_id=features_by_slot_id,
        id_fields=id_fields,
        label_fields=tuple(str(name) for name in config.label_names),
        candidate_metadata={
            str(name): str(config.candidate_metadata_dtypes.get(name, "int64"))
            for name in config.candidate_metadata_fields
        },
        target=config.target,
        list_len=config.list_len,
        history_len=config.history_len,
        history_groups=dict(config.history_groups),
        missing_slot_policy=config.missing_slot_policy,
        bad_length_policy=config.bad_length_policy,
    )


def _add_v1_top_level_features(
    *,
    features_by_name: dict[str, DataFeatureSpec],
    features_by_source_path: dict[str, DataFeatureSpec],
    config: DataConfig,
) -> None:
    if config.item_id_field is not None:
        _add_top_level_feature(
            features_by_name,
            features_by_source_path,
            name="item_id",
            source_path=config.item_id_field,
            source_level="item",
            dtype=config.id_dtype,
            max_outer_len=config.list_len,
        )
    _add_top_level_feature(
        features_by_name,
        features_by_source_path,
        name="request_id",
        source_path=config.request_id_field,
        source_level="request",
        dtype=config.id_dtype,
    )
    if config.user_id_field is not None:
        _add_top_level_feature(
            features_by_name,
            features_by_source_path,
            name="user_id",
            source_path=config.user_id_field,
            source_level="request",
            dtype=config.id_dtype,
        )
    for label in config.label_names:
        _add_top_level_feature(
            features_by_name,
            features_by_source_path,
            name=str(label),
            source_path=str(label),
            source_level="item",
            dtype=config.label_dtype,
            max_outer_len=config.list_len,
        )
    _add_top_level_feature(
        features_by_name,
        features_by_source_path,
        name="target_label",
        source_path="target_label",
        source_level="item",
        dtype=config.label_dtype,
        max_outer_len=config.list_len,
    )


def _validate_history_feature_groups(
    *,
    features_by_name: Mapping[str, DataFeatureSpec],
    history_groups: Mapping[str, HistoryGroupConfig],
) -> None:
    for feature in features_by_name.values():
        if feature.history_group is not None:
            if feature.history_group not in history_groups:
                raise ValueError(
                    f"feature {feature.name!r} references unknown history_group "
                    f"{feature.history_group!r}"
                )
            if feature.parser_kind != "request_sequence":
                raise ValueError(
                    f"feature {feature.name!r} history_group requires "
                    "parse.kind=request_sequence"
                )


def _adapter_feature_sources(adapter: Mapping[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for section_name in ("categorical_features", "direct_features"):
        section = _mapping_or_empty(adapter.get(section_name))
        for feature_name, raw_spec in section.items():
            sources[str(feature_name)] = _adapter_source_path(raw_spec)
    return sources


def _feature_role(feature: DataFeatureSpec) -> FeatureRole:
    if feature.role is not None:
        return feature.role
    if feature.history_group is not None:
        return "sequence"
    if feature.source_level == "item":
        return "item"
    if feature.source_level == "request":
        return "request"
    raise ValueError(
        f"feature {feature.name!r} has unsupported source_level "
        f"{feature.source_level!r}"
    )


def _embedding_max_sequence_length(
    feature: DataFeatureSpec,
    *,
    history_groups: Mapping[str, HistoryGroupConfig],
) -> int | None:
    if feature.history_group is not None:
        return int(history_groups[feature.history_group].max_len)
    if feature.parser_kind in {"item_sequence", "item_sequence_pad_inner"}:
        # max_outer_len is the candidate axis. A lookup bag contains the
        # inner values for one candidate, so only an explicit inner bound
        # applies to the jagged length checked by the embedding adapter.
        return feature.max_inner_len
    return feature.max_outer_len


def _feature_encoding(
    feature: DataFeatureSpec,
    model_feature: ModelFeatureSpec,
) -> EncodingKind:
    inferred: EncodingKind = "direct" if model_feature.kind == "dense" else "embedding"
    if feature.encoding is None:
        return inferred
    if feature.encoding != inferred:
        raise ValueError(
            f"feature {feature.source_path!r} contract encoding {feature.encoding!r} "
            f"does not match compiled model kind {model_feature.kind!r}"
        )
    return feature.encoding


def _feature_reduction(
    config: Mapping[str, Any],
    *,
    role: FeatureRole,
    source_path: str,
    feature_name: str,
    universe_default: ReductionKind | None = None,
) -> ReductionKind:
    defaults = {
        "request": "mean",
        "item": "mean",
        "sequence": "none",
    }
    configured_defaults = _mapping_or_empty(config.get("defaults"))
    overrides = _mapping_or_empty(config.get("overrides"))
    default = configured_defaults.get(role, universe_default or defaults[role])
    raw = overrides.get(source_path, overrides.get(feature_name, default))
    reduction = str(raw)
    if reduction not in {"mean", "sum", "max", "none"}:
        raise ValueError(
            f"feature {source_path!r} has unsupported input reduction {reduction!r}"
        )
    if role in {"request", "item"} and reduction == "none":
        raise ValueError(
            f"{role} feature {source_path!r} requires a segment reduction"
        )
    return reduction  # type: ignore[return-value]


def _adapter_source_path(raw_spec: Any) -> str:
    if isinstance(raw_spec, str):
        return _normalize_source_path(raw_spec)
    if isinstance(raw_spec, Mapping):
        source = raw_spec.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("adapter feature source must be a non-empty string")
        return _normalize_source_path(source)
    raise ValueError("adapter feature source must be a string or mapping")


def _normalize_source_path(source_path: str) -> str:
    parts = str(source_path).split(".")
    if len(parts) == 2 and parts[0] in {"user_slots", "context_user_slots", "item_slots"}:
        return f"{parts[0]}.{normalize_slot_id(parts[1])}"
    return source_path


def _physical_column(source_path: str) -> str:
    parts = str(source_path).split(".")
    if len(parts) == 2 and parts[0] in {
        "user_slots",
        "context_user_slots",
        "item_slots",
    }:
        return f"slot_{normalize_slot_id(parts[1])}"
    return str(source_path)


def _add_top_level_feature(
    features_by_name: dict[str, DataFeatureSpec],
    features_by_source_path: dict[str, DataFeatureSpec],
    *,
    name: str,
    source_path: str,
    source_level: str,
    dtype: str,
    max_outer_len: int | None = None,
) -> None:
    spec = DataFeatureSpec(
        name=name,
        source_path=source_path,
        source_level=source_level,
        reader_group=None,
        reader_shape=f"{source_level}_scalar",
        parser_kind=f"{source_level}_scalar",
        dtype=dtype,
        max_outer_len=max_outer_len,
    )
    features_by_name.setdefault(name, spec)
    features_by_source_path.setdefault(source_path, spec)

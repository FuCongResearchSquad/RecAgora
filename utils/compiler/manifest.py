from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias


NamespaceKey: TypeAlias = int | str

_FORBIDDEN_PUBLIC_FEATURE_KEYS = {
    "feature_name",
    "column",
    "source_path",
    "namespace_id",
    "reader_group",
    "reader_shape",
    "parser_kind",
    "parse",
    "kjt_key",
    "packing_index",
}
_NON_CATEGORICAL_REPRESENTATIONS = {"direct", "dense", "float", "time"}
_PUBLIC_SHAPE_KEYS = {"layout", "axes", "candidate_count", "max_length"}
_ALLOWED_VIEWS = {"candidate", "history", "query"}
_ALLOWED_ROLES = {"item", "request", "sequence"}
_ALLOWED_REDUCTIONS = {"mean", "none"}


@dataclass(frozen=True)
class ResolvedFeature:
    """One model-visible feature with no physical reader or packing details."""

    slot_id: int
    view: str
    role: str
    representation: str
    encoding: str
    reduction: str
    shape: Mapping[str, Any]
    history_group: str | None
    output_dim: int | None
    mapped_ids_exposed: bool = False

    def public_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "slot_id": self.slot_id,
            "view": self.view,
            "role": self.role,
            "representation": self.representation,
            "encoding": self.encoding,
            "reduction": self.reduction,
            "shape": _thaw_json(self.shape),
            "output_dim": self.output_dim,
        }
        if self.history_group is not None:
            result["history_group"] = self.history_group
        if self.mapped_ids_exposed:
            result["mapped_ids_exposed"] = True
        return result


@dataclass(frozen=True)
class ResolvedEmbeddingTable:
    """Private physical table plan compiled from selected categorical slots."""

    namespace: NamespaceKey
    table_name: str
    num_embeddings: int
    embedding_dim: int
    padding_id: int
    oov_id: int
    mapping_version: str
    slot_ids: tuple[int, ...]
    dim_source: str

    def private_summary(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "table_name": self.table_name,
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "padding_id": self.padding_id,
            "oov_id": self.oov_id,
            "mapping_version": self.mapping_version,
            "slot_ids": list(self.slot_ids),
            "dim_source": self.dim_source,
        }

    def layout_summary(self) -> dict[str, Any]:
        """Return only physical checkpoint-compatibility fields."""

        result = self.private_summary()
        result.pop("dim_source")
        return result


@dataclass(frozen=True)
class PrivateFeatureBinding:
    """Physical source information deliberately excluded from public summaries."""

    slot_id: int
    column: str
    source_path: str
    source_level: str
    namespace: NamespaceKey | None
    physical_contract: Mapping[str, Any]

    def private_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "slot_id": self.slot_id,
            "column": self.column,
            "source_path": self.source_path,
            "source_level": self.source_level,
        }
        if self.namespace is not None:
            result["namespace"] = self.namespace
        result["physical_contract"] = _thaw_json(self.physical_contract)
        return result


@dataclass(frozen=True)
class ResolvedModelFeatureManifest:
    """Immutable result of model feature selection and embedding compilation."""

    universe_version: str
    features: tuple[ResolvedFeature, ...]
    embedding_tables: tuple[ResolvedEmbeddingTable, ...]
    private_bindings: Mapping[int, PrivateFeatureBinding]

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(feature.slot_id for feature in self.features)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def mapped_id_slot_ids(self) -> tuple[int, ...]:
        return tuple(
            feature.slot_id
            for feature in self.features
            if feature.mapped_ids_exposed
        )

    def public_summary(self) -> dict[str, Any]:
        counts = _feature_counts(self.features)
        return {
            "universe_version": self.universe_version,
            "feature_count": self.feature_count,
            "resolved_slot_ids": list(self.slot_ids),
            "counts": counts,
            "features": [feature.public_summary() for feature in self.features],
        }

    def private_layout_summary(self) -> dict[str, Any]:
        return {
            "tables": [table.private_summary() for table in self.embedding_tables],
        }


def compile_model_feature_manifest(
    universe: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model_declaration: Mapping[str, Any],
) -> ResolvedModelFeatureManifest:
    """Compile the Phase 2 model manifest without touching runtime objects.

    ``universe`` and ``dataset`` must already be parsed mappings.  This function
    intentionally performs no YAML/path loading and imports no training,
    DataLoader, TorchRec, or model implementation modules.
    """

    universe = _require_mapping(universe, "universe")
    dataset = _require_mapping(dataset, "dataset")
    declaration = _require_mapping(model_declaration, "model_declaration")
    _reject_unknown_keys(
        declaration,
        allowed={"feature", "features", "embedding"},
        field="model_declaration",
    )

    universe_version = _require_non_empty_string(
        universe.get("version"), "universe.version"
    )
    requested_version = _require_non_empty_string(
        declaration.get("feature"),
        "model_declaration.feature",
    )
    if requested_version != universe_version:
        raise ValueError(
            "model feature version mismatch: "
            f"requested={requested_version!r} actual={universe_version!r}"
        )
    dataset_version = _require_non_empty_string(dataset.get("version"), "dataset.version")
    if dataset_version != universe_version:
        raise ValueError(
            "dataset/feature version mismatch: "
            f"dataset={dataset_version!r} feature={universe_version!r}"
        )

    universe_features = _parse_universe_features(universe)
    raw_selection = declaration.get("features", {})
    selected_features = _resolve_selection(
        universe_features,
        raw_selection,
    )
    mapped_id_slots = _resolve_mapped_id_slots(
        universe_features,
        selected_features,
        raw_selection,
    )
    bindings = _parse_bindings(dataset)
    namespace_specs = _parse_namespace_specs(dataset)
    embedding = _parse_embedding_declaration(declaration.get("embedding", {}))
    default_dim = int(embedding["default_dim"])
    default_dim_source = "model_default"

    selected_namespaces: list[NamespaceKey] = []
    slots_by_namespace: dict[NamespaceKey, list[int]] = {}
    binding_by_slot: dict[int, Mapping[str, Any]] = {}
    for feature in selected_features:
        slot_id = int(feature["slot_id"])
        binding = bindings.get(slot_id)
        if binding is None:
            raise ValueError(f"dataset has no feature binding for selected slot {slot_id}")
        binding_by_slot[slot_id] = binding
        public_history_group = feature.get("history_group")
        private_history_group = binding.get("history_group")
        if private_history_group != public_history_group:
            raise ValueError(
                f"slot {slot_id} history_group conflict: "
                f"feature={public_history_group!r} dataset={private_history_group!r}"
            )
        if not _is_categorical(feature):
            if binding.get("namespace_id") is not None:
                raise ValueError(
                    f"non-categorical slot {slot_id} must not bind an embedding namespace"
                )
            continue
        if binding.get("namespace_id") is None:
            raise ValueError(
                f"categorical slot {slot_id} is missing dataset namespace_id"
            )
        namespace = _normalize_namespace(
            binding["namespace_id"],
            f"dataset.feature_bindings[{slot_id}].namespace_id",
        )
        spec = namespace_specs.get(namespace)
        if spec is None:
            raise ValueError(
                f"slot {slot_id} references unknown dataset namespace {namespace!r}"
            )
        _validate_binding_namespace_contract(slot_id, binding, spec)
        if namespace not in slots_by_namespace:
            selected_namespaces.append(namespace)
            slots_by_namespace[namespace] = []
        slots_by_namespace[namespace].append(slot_id)

    overrides = _normalize_namespace_overrides(
        embedding["namespace_overrides"],
        known_namespaces=set(namespace_specs),
        selected_namespaces=set(selected_namespaces),
    )
    tables: list[ResolvedEmbeddingTable] = []
    dim_by_namespace: dict[NamespaceKey, int] = {}
    sparse_contract = _require_mapping(
        dataset.get("sparse_id_contract"), "dataset.sparse_id_contract"
    )
    _require_enum(
        sparse_contract.get("type"),
        {"dense_map", "modulo_bucket"},
        "dataset.sparse_id_contract.type",
    )
    mapping_version = _require_non_empty_string(
        sparse_contract.get("mapping_version"),
        "dataset.sparse_id_contract.mapping_version",
    )
    padding_id = _require_non_negative_int(
        sparse_contract.get("padding_id"),
        "dataset.sparse_id_contract.padding_id",
    )
    oov_id = _require_non_negative_int(
        sparse_contract.get("oov_id"),
        "dataset.sparse_id_contract.oov_id",
    )
    if padding_id == oov_id:
        raise ValueError("dataset padding_id and oov_id must be different")
    for namespace in selected_namespaces:
        spec = namespace_specs[namespace]
        if namespace in overrides:
            embedding_dim = overrides[namespace]
            dim_source = "model_namespace_override"
        else:
            embedding_dim = default_dim
            dim_source = default_dim_source
        dim_by_namespace[namespace] = embedding_dim
        num_embeddings = _require_positive_int(
            spec.get("vocab_size"),
            f"dataset namespace {namespace!r} vocab_size",
        )
        if padding_id >= num_embeddings or oov_id >= num_embeddings:
            raise ValueError(
                f"dataset namespace {namespace!r} padding/OOV IDs must be smaller "
                f"than vocab_size={num_embeddings}"
            )
        tables.append(
            ResolvedEmbeddingTable(
                namespace=namespace,
                table_name=_table_name(namespace),
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_id=padding_id,
                oov_id=oov_id,
                mapping_version=mapping_version,
                slot_ids=tuple(slots_by_namespace[namespace]),
                dim_source=dim_source,
            )
        )

    resolved_features: list[ResolvedFeature] = []
    private_bindings: dict[int, PrivateFeatureBinding] = {}
    for feature in selected_features:
        slot_id = int(feature["slot_id"])
        binding = binding_by_slot[slot_id]
        namespace: NamespaceKey | None = None
        output_dim: int | None = None
        if _is_categorical(feature):
            namespace = _normalize_namespace(
                binding["namespace_id"],
                f"dataset.feature_bindings[{slot_id}].namespace_id",
            )
            output_dim = dim_by_namespace[namespace]
        resolved_features.append(
            ResolvedFeature(
                slot_id=slot_id,
                view=_require_non_empty_string(
                    feature.get("view"), f"universe slot {slot_id} view"
                ),
                role=_require_non_empty_string(
                    feature.get("role"), f"universe slot {slot_id} role"
                ),
                representation=_require_non_empty_string(
                    feature.get("representation"),
                    f"universe slot {slot_id} representation",
                ),
                encoding=_require_non_empty_string(
                    feature.get("encoding"), f"universe slot {slot_id} encoding"
                ),
                reduction=_require_non_empty_string(
                    feature.get("reduction"), f"universe slot {slot_id} reduction"
                ),
                shape=_freeze_mapping(
                    _require_mapping(
                        feature.get("shape"), f"universe slot {slot_id} shape"
                    )
                ),
                history_group=(
                    None
                    if feature.get("history_group") is None
                    else _require_non_empty_string(
                        feature["history_group"],
                        f"universe slot {slot_id} history_group",
                    )
                ),
                output_dim=output_dim,
                mapped_ids_exposed=slot_id in mapped_id_slots,
            )
        )
        private_bindings[slot_id] = PrivateFeatureBinding(
            slot_id=slot_id,
            column=_require_non_empty_string(
                binding.get("column"), f"dataset binding {slot_id} column"
            ),
            source_path=_require_non_empty_string(
                binding.get("source_path"), f"dataset binding {slot_id} source_path"
            ),
            source_level=_require_non_empty_string(
                binding.get("source_level"), f"dataset binding {slot_id} source_level"
            ),
            namespace=namespace,
            physical_contract=_freeze_mapping(binding),
        )

    return ResolvedModelFeatureManifest(
        universe_version=universe_version,
        features=tuple(resolved_features),
        embedding_tables=tuple(tables),
        private_bindings=MappingProxyType(private_bindings),
    )


def _parse_universe_features(
    universe: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_features = _require_sequence(universe.get("features"), "universe.features")
    features: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for index, raw_feature in enumerate(raw_features):
        feature = _require_mapping(raw_feature, f"universe.features[{index}]")
        forbidden = _FORBIDDEN_PUBLIC_FEATURE_KEYS.intersection(feature)
        if forbidden:
            raise ValueError(
                "public universe feature contains private fields: "
                f"slot_index={index} fields={sorted(forbidden)!r}"
            )
        slot_id = _require_slot_id(feature.get("slot_id"), f"universe.features[{index}].slot_id")
        if slot_id in seen:
            raise ValueError(f"duplicate universe slot_id: {slot_id}")
        seen.add(slot_id)
        _validate_public_feature(feature, slot_id)
        features.append(feature)
    return tuple(features)


def _resolve_selection(
    universe_features: tuple[Mapping[str, Any], ...],
    raw_selection: Any,
) -> tuple[Mapping[str, Any], ...]:
    selection = _require_mapping(raw_selection, "model_declaration.features")
    _reject_unknown_keys(
        selection,
        allowed={"use", "include", "exclude", "mapped_id_slots"},
        field="model_declaration.features",
    )
    use = selection.get("use", "all")
    if use not in {"all", "selected"}:
        raise ValueError(
            "model_declaration.features.use must be 'all' or 'selected'"
        )
    include = _slot_id_list(selection.get("include", ()), "features.include")
    exclude = _slot_id_list(selection.get("exclude", ()), "features.exclude")
    overlap = sorted(set(include).intersection(exclude))
    if overlap:
        raise ValueError(f"features.include/exclude overlap: {overlap!r}")
    by_slot = {int(feature["slot_id"]): feature for feature in universe_features}
    unknown = sorted((set(include) | set(exclude)) - set(by_slot))
    if unknown:
        raise ValueError(f"model feature selection references unknown slots: {unknown!r}")
    if use == "selected" and not include:
        raise ValueError(
            "model feature selection with use='selected' requires features.include"
        )
    default_slots = (
        {int(feature["slot_id"]) for feature in universe_features}
        if use == "all"
        else set()
    )
    selected = (default_slots | set(include)) - set(exclude)
    if not selected:
        raise ValueError("model feature selection resolved to zero slots")
    return tuple(
        feature
        for feature in universe_features
        if int(feature["slot_id"]) in selected
    )


def _resolve_mapped_id_slots(
    universe_features: tuple[Mapping[str, Any], ...],
    selected_features: tuple[Mapping[str, Any], ...],
    raw_selection: Any,
) -> frozenset[int]:
    selection = _require_mapping(raw_selection, "model_declaration.features")
    requested = _slot_id_list(
        selection.get("mapped_id_slots", ()),
        "features.mapped_id_slots",
    )
    if not requested:
        return frozenset()
    universe_by_slot = {
        int(feature["slot_id"]): feature for feature in universe_features
    }
    unknown = sorted(set(requested) - set(universe_by_slot))
    if unknown:
        raise ValueError(
            f"features.mapped_id_slots references unknown slots: {unknown!r}"
        )
    selected_slots = {int(feature["slot_id"]) for feature in selected_features}
    unselected = sorted(set(requested) - selected_slots)
    if unselected:
        raise ValueError(
            "features.mapped_id_slots must be selected by include/use and not "
            f"excluded: {unselected!r}"
        )
    for slot_id in requested:
        feature = universe_by_slot[slot_id]
        contract = (
            str(feature.get("view")),
            str(feature.get("role")),
            str(feature.get("representation")),
            str(feature.get("encoding")),
            str(feature.get("reduction")),
        )
        history_contract = ("history", "sequence", "sparse_id", "embedding", "none")
        candidate_contract = ("candidate", "item", "sparse_id", "embedding", "mean")
        if contract not in {history_contract, candidate_contract}:
            raise ValueError(
                "features.mapped_id_slots supports history sequences or "
                "candidate item embeddings: "
                f"slot={slot_id} contract={contract!r}"
            )
    return frozenset(requested)


def _parse_bindings(dataset: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw_bindings = _require_sequence(
        dataset.get("feature_bindings"), "dataset.feature_bindings"
    )
    bindings: dict[int, Mapping[str, Any]] = {}
    for index, raw_binding in enumerate(raw_bindings):
        binding = _require_mapping(
            raw_binding, f"dataset.feature_bindings[{index}]"
        )
        slot_id = _require_slot_id(
            binding.get("slot_id"), f"dataset.feature_bindings[{index}].slot_id"
        )
        if slot_id in bindings:
            raise ValueError(f"duplicate dataset feature binding for slot {slot_id}")
        bindings[slot_id] = binding
    return bindings


def _parse_namespace_specs(
    dataset: Mapping[str, Any],
) -> dict[NamespaceKey, Mapping[str, Any]]:
    sparse_contract = _require_mapping(
        dataset.get("sparse_id_contract"), "dataset.sparse_id_contract"
    )
    raw_namespaces = _require_mapping(
        sparse_contract.get("namespaces"),
        "dataset.sparse_id_contract.namespaces",
    )
    namespaces: dict[NamespaceKey, Mapping[str, Any]] = {}
    for raw_namespace, raw_spec in raw_namespaces.items():
        namespace = _normalize_namespace(
            raw_namespace, "dataset.sparse_id_contract.namespaces key"
        )
        if namespace in namespaces:
            raise ValueError(
                f"duplicate normalized dataset namespace: {namespace!r}"
            )
        spec = _require_mapping(
            raw_spec, f"dataset namespace {namespace!r} spec"
        )
        _require_positive_int(
            spec.get("vocab_size"), f"dataset namespace {namespace!r} vocab_size"
        )
        namespaces[namespace] = spec
    return namespaces


def _parse_embedding_declaration(raw_embedding: Any) -> dict[str, Any]:
    embedding = _require_mapping(raw_embedding, "model_declaration.embedding")
    _reject_unknown_keys(
        embedding,
        allowed={"default_dim", "namespace_overrides"},
        field="model_declaration.embedding",
    )
    default_dim = _require_positive_int(
        embedding.get("default_dim"), "model_declaration.embedding.default_dim"
    )
    overrides = embedding.get("namespace_overrides", {})
    overrides = _require_mapping(
        overrides, "model_declaration.embedding.namespace_overrides"
    )
    return {"default_dim": default_dim, "namespace_overrides": overrides}


def _normalize_namespace_overrides(
    raw_overrides: Mapping[Any, Any],
    *,
    known_namespaces: set[NamespaceKey],
    selected_namespaces: set[NamespaceKey],
) -> dict[NamespaceKey, int]:
    overrides: dict[NamespaceKey, int] = {}
    for raw_namespace, raw_dim in raw_overrides.items():
        namespace = _normalize_namespace(
            raw_namespace, "embedding.namespace_overrides key"
        )
        if namespace in overrides:
            raise ValueError(
                f"duplicate normalized namespace override: {namespace!r}"
            )
        if namespace not in known_namespaces:
            raise ValueError(f"embedding override references unknown namespace {namespace!r}")
        if namespace not in selected_namespaces:
            raise ValueError(
                "embedding override references a namespace not selected by this model: "
                f"{namespace!r}"
            )
        overrides[namespace] = _require_positive_int(
            raw_dim, f"embedding namespace override {namespace!r}"
        )
    return overrides


def _validate_binding_namespace_contract(
    slot_id: int,
    binding: Mapping[str, Any],
    namespace_spec: Mapping[str, Any],
) -> None:
    for key in ("vocab_size", "mapping_version", "padding_id", "oov_id"):
        if key not in binding:
            continue
        expected = namespace_spec.get(key)
        if expected is None or binding[key] != expected:
            raise ValueError(
                f"slot {slot_id} conflicts with its shared namespace {key}: "
                f"binding={binding[key]!r} namespace={expected!r}"
            )


def _validate_public_feature(feature: Mapping[str, Any], slot_id: int) -> None:
    view = _require_enum(
        feature.get("view"), _ALLOWED_VIEWS, f"universe slot {slot_id} view"
    )
    role = _require_enum(
        feature.get("role"), _ALLOWED_ROLES, f"universe slot {slot_id} role"
    )
    representation = _require_enum(
        feature.get("representation"),
        {"sparse_id", *_NON_CATEGORICAL_REPRESENTATIONS},
        f"universe slot {slot_id} representation",
    )
    encoding = _require_enum(
        feature.get("encoding"),
        {"embedding", "direct"},
        f"universe slot {slot_id} encoding",
    )
    reduction = _require_enum(
        feature.get("reduction"),
        _ALLOWED_REDUCTIONS,
        f"universe slot {slot_id} reduction",
    )
    if (encoding == "embedding") != (representation == "sparse_id"):
        raise ValueError(
            f"universe slot {slot_id} has incompatible representation/encoding: "
            f"{representation!r}/{encoding!r}"
        )
    if view == "history":
        if role != "sequence" or reduction != "none":
            raise ValueError(
                f"universe history slot {slot_id} must use sequence role and none reduction"
            )
        _require_non_empty_string(
            feature.get("history_group"),
            f"universe slot {slot_id} history_group",
        )
    elif feature.get("history_group") is not None:
        raise ValueError(
            f"universe non-history slot {slot_id} must not define history_group"
        )
    elif view == "candidate" and role != "item":
        raise ValueError(f"universe candidate slot {slot_id} must use item role")
    elif view == "query" and role != "request":
        raise ValueError(f"universe query slot {slot_id} must use request role")

    shape = _require_mapping(feature.get("shape"), f"universe slot {slot_id} shape")
    unknown_shape_keys = set(shape) - _PUBLIC_SHAPE_KEYS
    if unknown_shape_keys:
        raise ValueError(
            f"universe slot {slot_id} shape contains unsupported fields: "
            f"{sorted(unknown_shape_keys)!r}"
        )
    _reject_private_keys_recursively(shape, f"universe slot {slot_id} shape")
    _require_non_empty_string(shape.get("layout"), f"universe slot {slot_id} shape.layout")
    axes = _require_sequence(shape.get("axes"), f"universe slot {slot_id} shape.axes")
    if not axes or any(not isinstance(axis, str) or not axis for axis in axes):
        raise ValueError(f"universe slot {slot_id} shape.axes must be non-empty strings")
    for size_key in ("candidate_count", "max_length"):
        if size_key in shape:
            _require_positive_int(
                shape[size_key], f"universe slot {slot_id} shape.{size_key}"
            )


def _reject_private_keys_recursively(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_PUBLIC_FEATURE_KEYS.intersection(
            str(key) for key in value
        )
        if forbidden:
            raise ValueError(
                f"{field} contains private fields: {sorted(forbidden)!r}"
            )
        for key, item in value.items():
            _reject_private_keys_recursively(item, f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_private_keys_recursively(item, f"{field}[{index}]")


def _feature_counts(features: tuple[ResolvedFeature, ...]) -> dict[str, Any]:
    views: dict[str, int] = {}
    encodings: dict[str, int] = {}
    for feature in features:
        views[feature.view] = views.get(feature.view, 0) + 1
        encodings[feature.encoding] = encodings.get(feature.encoding, 0) + 1
    return {"views": views, "encodings": encodings}


def _is_categorical(feature: Mapping[str, Any]) -> bool:
    # _validate_public_feature has already established the exact
    # representation/encoding equivalence for every parsed universe feature.
    return str(feature.get("encoding", "")) == "embedding"


def _slot_id_list(raw: Any, field: str) -> tuple[int, ...]:
    values = _require_sequence(raw, field)
    result = tuple(
        _require_slot_id(value, f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    if len(result) != len(set(result)):
        duplicates = sorted(
            slot_id for slot_id in set(result) if result.count(slot_id) > 1
        )
        raise ValueError(f"{field} contains duplicate slot IDs: {duplicates!r}")
    return result


def _normalize_namespace(value: Any, field: str) -> NamespaceKey:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, str))
        or (isinstance(value, str) and not value)
    ):
        raise TypeError(f"{field} must be an integer or canonical string namespace")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
        return value
    raw = value
    if raw.startswith("namespace_") and raw.removeprefix("namespace_").isdigit():
        raw = raw.removeprefix("namespace_")
    if raw.isdigit():
        return int(raw)
    return value


def _table_name(namespace: NamespaceKey) -> str:
    return f"namespace_{namespace}"


def _require_slot_id(value: Any, field: str) -> int:
    return _require_bounded_int(
        value,
        field,
        minimum=0,
        type_description="a numeric slot_id",
        range_description="non-negative",
    )


def _require_positive_int(value: Any, field: str) -> int:
    return _require_bounded_int(
        value,
        field,
        minimum=1,
        type_description="a positive integer",
        range_description="a positive integer",
    )


def _require_non_negative_int(value: Any, field: str) -> int:
    return _require_bounded_int(
        value,
        field,
        minimum=0,
        type_description="a non-negative integer",
        range_description="a non-negative integer",
    )


def _require_bounded_int(
    value: Any,
    field: str,
    *,
    minimum: int,
    type_description: str,
    range_description: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be {type_description}")
    if value < minimum:
        raise ValueError(f"{field} must be {range_description}")
    return value


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_enum(value: Any, allowed: set[str], field: str) -> str:
    value = _require_non_empty_string(value, field)
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)!r}, got {value!r}")
    return value


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unsupported fields: {unknown!r}")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_json(item)
            for key, item in value.items()
        }
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item) for item in value)
    return _require_json_scalar(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")

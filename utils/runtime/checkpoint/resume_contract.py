from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from utils.compiler.config import ModelConfig
from utils.compiler.feature_contract import FeatureContractPlans
from utils.runtime.model.adapter import compile_recagora_input_spec


RESUME_INPUT_CONTRACT_KEY = "resume_input_contract"
RESUME_INPUT_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ResumeInputContractValidation:
    status: Literal["verified", "legacy_unverified"]
    warnings: tuple[str, ...] = ()


def build_resume_input_contract(
    plans: FeatureContractPlans,
    model_config: ModelConfig,
) -> dict[str, Any]:
    """Build the active-input contract used only by full training resume."""

    input_spec = compile_recagora_input_spec(plans)
    manifest = plans.resolved_manifest
    active_features = []
    for feature in manifest.features:
        slot = input_spec.slot(feature.slot_id)
        active_feature = {
            "slot_id": slot.slot_id,
            "view": slot.view,
            "role": slot.role,
            "representation": slot.representation,
            "encoding": slot.encoding,
            "reduction": slot.reduction,
            "output_dim": slot.output_dim,
            "history_group": feature.history_group,
        }
        if slot.mapped_ids_exposed:
            active_feature["mapped_ids_exposed"] = True
        active_features.append(active_feature)

    payload = _canonical_json_value(
        {
            "input_view": plans.embedding_input_plan.input_view,
            "list_len": input_spec.list_len,
            "task_names": input_spec.task_names,
            "query_slot_ids": input_spec.query.slot_ids,
            "candidate_slot_ids": input_spec.candidate.slot_ids,
            "active_features": active_features,
            "histories": {
                name: {
                    "categorical_slot_ids": spec.categorical_slot_ids,
                    "direct_slot_ids": spec.direct_slot_ids,
                    "event_total_dim": spec.event_total_dim,
                    "max_length": spec.max_length,
                    "event_order": spec.event_order,
                    "truncate_side": spec.truncate_side,
                    "padding_side": spec.padding_side,
                    "direct_padding_side": spec.direct_padding_side,
                    "padding_value": spec.padding_value,
                    "length_anchor_slot_id": spec.length_anchor_slot_id,
                }
                for name, spec in sorted(input_spec.histories.items())
            },
            "embedding_tables": [
                table.layout_summary() for table in manifest.embedding_tables
            ],
            "embedding_inputs": [
                {
                    "name": feature.name,
                    "role": feature.role,
                    "encoding": feature.encoding,
                    "reduction": feature.reduction,
                    "representation": feature.representation,
                    "history_group": feature.history_group,
                    "max_sequence_length": feature.max_sequence_length,
                    "table_name": feature.table_name,
                    "vocab_size": feature.vocab_size,
                    "embedding_dim": feature.embedding_dim,
                    "id_mapping": feature.id_mapping,
                }
                for feature in plans.embedding_input_plan.features
            ],
        }
    )
    diagnostics = _canonical_json_value(
        {
            "model_name": model_config.name,
            "class_path": model_config.class_path,
        }
    )
    return {
        "version": RESUME_INPUT_CONTRACT_VERSION,
        "signature_sha256": _sha256(payload),
        "payload": payload,
        "diagnostics": diagnostics,
    }


def validate_resume_input_contract(
    checkpoint: Mapping[str, Any],
    *,
    current_contract: Mapping[str, Any],
) -> ResumeInputContractValidation:
    """Validate active input semantics without gating legacy checkpoints."""

    saved = checkpoint.get(RESUME_INPUT_CONTRACT_KEY)
    if saved is None:
        return ResumeInputContractValidation(
            status="legacy_unverified",
            warnings=(
                "checkpoint has no resume_input_contract; active input semantics "
                "were not verified",
            ),
        )
    saved_contract = _validated_contract(saved, label="checkpoint")
    current = _validated_contract(current_contract, label="current")
    if saved_contract["signature_sha256"] != current["signature_sha256"]:
        differing_sections = sorted(
            key
            for key in set(saved_contract["payload"]) | set(current["payload"])
            if saved_contract["payload"].get(key) != current["payload"].get(key)
        )
        raise ValueError(
            "Cannot resume checkpoint with a different active input contract: "
            f"differing_sections={differing_sections}"
        )

    saved_diagnostics = saved_contract["diagnostics"]
    current_diagnostics = current["diagnostics"]
    warnings = tuple(
        f"{key} differs: checkpoint={saved_diagnostics.get(key)!r} "
        f"current={current_diagnostics.get(key)!r}"
        for key in sorted(set(saved_diagnostics) | set(current_diagnostics))
        if saved_diagnostics.get(key) != current_diagnostics.get(key)
    )
    return ResumeInputContractValidation(status="verified", warnings=warnings)


def _validated_contract(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} resume_input_contract must be a mapping")
    version = value.get("version")
    if version != RESUME_INPUT_CONTRACT_VERSION:
        raise ValueError(
            f"{label} resume_input_contract version must be "
            f"{RESUME_INPUT_CONTRACT_VERSION}, got {version!r}"
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} resume_input_contract payload must be a mapping")
    canonical_payload = _canonical_json_value(payload)
    signature = value.get("signature_sha256")
    if not isinstance(signature, str) or signature != _sha256(canonical_payload):
        raise ValueError(
            f"{label} resume_input_contract signature does not match its payload"
        )
    diagnostics = value.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise ValueError(
            f"{label} resume_input_contract diagnostics must be a mapping"
        )
    return {
        "version": version,
        "signature_sha256": signature,
        "payload": canonical_payload,
        # Ignore retired diagnostic extensions in older checkpoints. Only these
        # readable identifiers participate in diagnostic warnings.
        "diagnostics": _canonical_json_value(
            {
                key: diagnostics[key]
                for key in ("model_name", "class_path")
                if key in diagnostics
            }
        ),
    }


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _canonical_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "resume input contract values must be JSON-compatible, got "
        f"{type(value).__name__}"
    )

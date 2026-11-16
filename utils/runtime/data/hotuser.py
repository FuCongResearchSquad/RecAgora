"""HotUser ordered-split index. Metadata only; request payload stays streaming."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from utils.compiler.config import DataConfig
from utils.runtime.data.errors import DataContractError


_PART = re.compile(r"part-(\d{19})-(\d{19})\.zstd\.parquet")


@dataclass(frozen=True)
class HotUserFile:
    path: Path
    first_id: int
    last_id: int
    row_groups: tuple[int, ...]
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class HotUserIndex:
    files: tuple[HotUserFile, ...]
    identity: dict[str, Any]


def build_hotuser_index(config: DataConfig) -> HotUserIndex:
    import pyarrow.parquet as pq
    import yaml

    root = Path(config.root).expanduser().resolve()
    directory = root / config.split
    if not directory.is_dir():
        raise DataContractError(f"HotUser split directory not found: {directory}")
    expected = None
    if config.sparse_id_contract.layout is not None:
        with Path(config.sparse_id_contract.layout).open() as source:
            layout = yaml.safe_load(source)
        expected = layout.get("physical_contract", {}).get("columns")
    files = []
    schema_signature = None
    for path in sorted(directory.iterdir()):
        if path.is_file() and (path.name == "_SUCCESS" or path.name.startswith(".")):
            continue
        match = _PART.fullmatch(path.name)
        if path.is_symlink() or not path.is_file() or match is None:
            raise DataContractError(f"HotUser requires flat part-<19 digit min>-<19 digit max>.zstd.parquet files: {path}")
        first, last = map(int, match.groups())
        if first < 1 or last < first or last >= 2**63:
            raise DataContractError(f"Invalid HotUser request ID range: {path.name}")
        if files and first != files[-1].last_id + 1:
            raise DataContractError(f"HotUser file ranges contain a gap or overlap: {path.name}")
        with pq.ParquetFile(path) as parquet:
            meta = parquet.metadata
            if meta.num_rows != last - first + 1:
                raise DataContractError(f"HotUser footer row count disagrees with filename: {path.name}")
            signature = [(field.name, _arrow_type(str(field.type))) for field in parquet.schema_arrow]
            if expected is not None:
                declared = [(field["name"], _arrow_type(field["arrow_type"])) for field in expected]
                if signature != declared:
                    raise DataContractError(f"HotUser schema differs from dataset physical contract: {path.name}")
            if schema_signature is not None and schema_signature != signature:
                raise DataContractError(f"HotUser schemas differ between files: {path.name}")
            schema_signature = signature
            groups = tuple(meta.row_group(i).num_rows for i in range(meta.num_row_groups))
        stat = path.stat()
        files.append(HotUserFile(path, first, last, groups, stat.st_size, stat.st_mtime_ns))
    if not files:
        raise DataContractError(f"HotUser split has no Parquet files: {directory}")
    if config.split == "train" and files[0].first_id != 1:
        raise DataContractError("HotUser train split must begin with request_id=1")
    plan_id = None
    if config.manifest_path is not None:
        with Path(config.manifest_path).open() as source:
            manifest = json.load(source)
        records = [record for record in manifest["files"] if record["split"] == config.split]
        records.sort(key=lambda record: int(record["min_id"]))
        actual = [(f.path.name, f.first_id, f.last_id, sum(f.row_groups), f.size) for f in files]
        published = [(r["name"], int(r["min_id"]), int(r["max_id"]), int(r["rows"]), int(r["bytes"])) for r in records]
        if actual != published or sum(sum(f.row_groups) for f in files) != int(manifest["split_rows"][config.split]):
            raise DataContractError("HotUser split is incomplete or differs from published manifest")
        plan_id = manifest.get("plan_id")
    fingerprint_input = {
        "schema": schema_signature,
        "files": [(f.path.name, f.first_id, f.last_id, f.row_groups, f.size, f.mtime_ns) for f in files],
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    return HotUserIndex(tuple(files), {
        "contract": "large_arrival20_ordered811_v1",
        "root": str(root), "split": config.split,
        "first_id": files[0].first_id, "last_id": files[-1].last_id,
        "rows": sum(sum(f.row_groups) for f in files), "file_count": len(files),
        "metadata_fingerprint": fingerprint, "plan_id": plan_id,
    })


def _arrow_type(value: str) -> str:
    # Arrow versions differ only in the displayed list child name.
    return value.replace("element:", "item:").replace(" ", "")


def validate_request_ids(table: Any, *, field: str, first_id: int) -> None:
    import numpy as np

    column = table.column(field)
    if column.null_count or not np.array_equal(
        column.to_numpy(), np.arange(first_id, first_id + table.num_rows, dtype=np.int64)
    ):
        raise DataContractError(f"HotUser {field} must be contiguous in physical row order; expected first ID {first_id}")

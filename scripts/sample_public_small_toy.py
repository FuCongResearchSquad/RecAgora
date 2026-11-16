#!/usr/bin/env python3
"""Sample a tiny public Small dataset from the first Parquet in DAY01/DAY02."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from bisect import bisect_left
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PARTITIONS = ("DAY01", "DAY02")
FORBIDDEN_COLUMNS = {"user_id", "item_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/work/datasets/data"),
        help="Directory containing dt=DAY01 and dt=DAY02 (default: /home/work/datasets/data)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets/toy-small-public",
        help="New directory for the sampled dataset; must not already exist",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--denominator", type=int, default=20)
    parser.add_argument("--batch-rows", type=int, default=8192)
    args = parser.parse_args()
    if args.denominator < 1 or args.batch_rows < 1:
        parser.error("--denominator and --batch-rows must be positive")
    return args


def select_first_parquet(source_root: Path, partition: str) -> Path:
    directory = source_root / f"dt={partition}"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing partition directory: {directory}")
    files = sorted(path for path in directory.glob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files in: {directory}")
    return files[0]


def sample_partition(
    source: Path, destination: Path, *, denominator: int, seed: int, batch_rows: int
) -> dict[str, object]:
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(schema.names))
    if forbidden:
        raise ValueError(f"Public source still contains forbidden columns {forbidden}: {source}")

    total_rows = parquet.metadata.num_rows
    if total_rows == 0:
        raise ValueError(f"Source Parquet is empty: {source}")
    sample_rows = (total_rows + denominator - 1) // denominator
    selected = sorted(random.Random(seed).sample(range(total_rows), sample_rows))

    destination.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    position = 0
    with pq.ParquetWriter(destination, schema=schema, compression="zstd") as writer:
        for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
            next_position = position + batch.num_rows
            begin = bisect_left(selected, position)
            end = bisect_left(selected, next_position, lo=begin)
            if end > begin:
                offsets = pa.array(
                    [index - position for index in selected[begin:end]], type=pa.int64()
                )
                writer.write_table(pa.Table.from_batches([batch]).take(offsets))
                emitted += end - begin
            position = next_position

    if position != total_rows or emitted != sample_rows:
        raise RuntimeError(
            f"Sampling count mismatch for {source}: read={position}/{total_rows}, "
            f"written={emitted}/{sample_rows}"
        )
    output_rows = pq.ParquetFile(destination).metadata.num_rows
    if output_rows != sample_rows:
        raise RuntimeError(f"Output row count mismatch for {destination}: {output_rows}")
    return {
        "source": f"{source.parent.name}/{source.name}",
        "output": str(destination.relative_to(destination.parents[1])),
        "source_rows": total_rows,
        "sample_rows": sample_rows,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    sources = {
        partition: select_first_parquet(args.source_root, partition)
        for partition in PARTITIONS
    }
    output_root = args.output_root.expanduser().absolute()
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        manifest: dict[str, object] = {
            "description": "Public Small toy: one source Parquet per partition, sampled by row",
            "denominator": args.denominator,
            "partitions": {},
        }
        for day_number, partition in enumerate(PARTITIONS, start=1):
            source = sources[partition]
            destination = temporary_root / f"dt={partition}" / source.name
            info = sample_partition(
                source,
                destination,
                denominator=args.denominator,
                seed=args.seed + day_number,
                batch_rows=args.batch_rows,
            )
            manifest["partitions"][partition] = info
            print(
                f"{partition}: {info['sample_rows']}/{info['source_rows']} rows "
                f"from {source.name}",
                flush=True,
            )
        (temporary_root / "sampling_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_root.exists():
            raise FileExistsError(f"Output directory appeared during sampling: {output_root}")
        temporary_root.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(f"Toy dataset ready: {output_root}")


if __name__ == "__main__":
    main()

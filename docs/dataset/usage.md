# Data usage

## Layout

Small/Large use anonymous DAY partitions; HotUser has only two precomputed splits, `train` and `valid`:

```text
small/dt=DAY01/*.parquet ... small/dt=DAY08/*.parquet
large/dt=DAY01/*.parquet ... large/dt=DAY35/*.parquet
hotuser/train/*.parquet
hotuser/valid/*.parquet
```

## Shell entry

Activate your Python environment and set the distributed environment described in [Quick Start](../../README.md#quick-start). Run from the repository root:

```bash
bash run.sh --dataset small --exp rankmixer_small \
  --config configs/models/rankmixer/rankmixer.yaml \
  --data-root /path/to/small --train-days 01-07 --eval-days 08-08
```

Defaults are DAY01–DAY07 / DAY08 for Small and DAY01–DAY34 / DAY35 for Large. The ranges are inclusive. CLI ranges override TRAIN_DAY_RANGE/EVAL_DAY_RANGE. Only dt=DAYxx is accepted; missing partitions and empty Parquet directories fail before launch. Relative data-root paths resolve from the calling directory.

HotUser selects train/valid and rejects day ranges:

```bash
bash run.sh --dataset hotuser --exp rankmixer_hotuser \
  --config configs/models/rankmixer/rankmixer.yaml --data-root /path/to/hotuser
```

Use `--mode train`, `eval`, or `all` (default). `--data-config` overrides the profile contract explicitly. Model/data configuration paths in run.sh are relative to the project root.

## Python CLI

Use explicit paths for custom layouts and single-node multi-GPU jobs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/models/rankmixer/rankmixer.yaml \
  --data-config configs/data/large.yaml \
  --data-path '/path/to/large/dt=DAY01/*.parquet' \
  --output-dir ./results/custom --device cuda --batch-size 512 --no-amp
```

Repeat --data-path for multiple partitions; the example above selects only DAY01. Quote globs. A directory path reads its immediate Parquet files, not nested partitions. Data configs override model defaults, and YAML include paths resolve relative to their owning YAML.

For HotUser, pass --data-root and --split train or valid with hotuser.yaml. If provided, use the same --data-manifest for training and evaluation. The reader validates filenames, metadata, and request ranges; a manifest additionally specifies the expected file inventory.

The Python CLI retains --date-range for custom real-date datasets. It is not used for public DAY partitions. --data-path is mutually exclusive with --data-root/--date-range; in date mode, root and range must be supplied together.

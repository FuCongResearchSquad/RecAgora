from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import _resolve_data_paths
from training import load_recagora_config
from training.eval import evaluate
from utils.compiler.config import RecAgoraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a request-level ranking model.")
    parser.add_argument("--config", required=True, help="Path to a model YAML config.")
    parser.add_argument(
        "--data-config",
        default=None,
        help="Override the model YAML data_config include with a RequestData YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint directory to evaluate. Accepts a model-only last "
            "checkpoint or a full step_N checkpoint."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Base train/eval output directory.")
    parser.add_argument(
        "--seed",
        default=None,
        help="Single seed or comma-separated seeds. Multiple values enable multi-seed eval.",
    )
    parser.add_argument("--device", default=None, help="Override eval/train device.")
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help="Override eval.paths. Repeat this option for multiple paths.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory containing dt=YYYY-MM-DD parquet partitions for eval.",
    )
    parser.add_argument(
        "--date-range",
        default=None,
        help="Inclusive compact date range for --data-root, e.g. 20260512-20260518.",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default=None, help="HotUser evaluation split (default valid).")
    parser.add_argument("--data-manifest", default=None, help="Optional HotUser published.json manifest.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override eval workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit eval batches.")
    parser.add_argument(
        "--global-eval",
        action="store_true",
        help="Also compute exact full-dataset global AUC.",
    )
    parser.add_argument(
        "--global-auc-workers",
        type=int,
        default=None,
        help=(
            "CPU workers for exact global AUC. Defaults to one worker per "
            "output task, capped by available CPUs."
        ),
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override eval pin_memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override eval persistent_workers.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override eval amp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = _apply_eval_overrides(
        load_recagora_config(
            args.config,
            data_config=args.data_config,
            data_root=args.data_root if args.split is not None else None,
            data_manifest=args.data_manifest,
        ),
        args,
    )
    seeds = parse_seeds(args.seed, default_seed=base_config.train.seed)
    base_output_dir = _resolve_base_output_dir(base_config, args.output_dir)

    results: list[dict[str, Any]] = []
    for seed in seeds:
        run_output_dir = _run_output_dir(
            base_output_dir=base_output_dir,
            seed=seed,
            seeds=seeds,
        )
        config = replace(base_config, train=replace(base_config.train, seed=seed))
        result = evaluate(
            config,
            checkpoint_path=args.checkpoint,
            output_dir=run_output_dir,
            global_eval=args.global_eval,
        )
        if not _is_main_result(result):
            continue
        result["seed"] = seed
        results.append(result)
        print(_format_result(config, result), flush=True)

    if len(seeds) > 1 and results:
        summary = build_eval_summary(results, seeds=seeds, output_dir=base_output_dir)
        summary_path = base_output_dir / "outputs" / "eval_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(_format_summary(summary, summary_path), flush=True)


def _is_main_result(result: dict[str, Any]) -> bool:
    return int(result.get("rank", 0)) == 0 and result.get("status") != "done_non_main"


def _apply_eval_overrides(config: RecAgoraConfig, args: argparse.Namespace) -> RecAgoraConfig:
    if args.split is not None or args.data_manifest is not None:
        if args.date_range is not None or args.data_path is not None:
            raise ValueError("HotUser --split cannot be combined with --date-range or --data-path")
        data_paths = None
    else:
        data_paths = _resolve_data_paths(args)
    eval_config = config.eval
    train_config = config.train

    if data_paths is not None:
        eval_config = replace(eval_config, paths=tuple(data_paths))
    if args.split is not None:
        eval_config = replace(eval_config, split=args.split)
    if args.batch_size is not None:
        eval_config = replace(eval_config, batch_size=args.batch_size)
    if args.num_workers is not None:
        eval_config = replace(eval_config, num_workers=args.num_workers)
    if args.pin_memory is not None:
        eval_config = replace(eval_config, pin_memory=args.pin_memory)
    if args.persistent_workers is not None:
        eval_config = replace(eval_config, persistent_workers=args.persistent_workers)
    if args.device is not None:
        eval_config = replace(eval_config, device=args.device)
        train_config = replace(train_config, device=args.device)
    if args.amp is not None:
        eval_config = replace(eval_config, amp=args.amp)
        train_config = replace(train_config, amp=args.amp)
    if args.max_batches is not None:
        eval_config = replace(eval_config, max_batches=args.max_batches)
    if args.global_auc_workers is not None:
        if args.global_auc_workers <= 0:
            raise ValueError("--global-auc-workers must be positive")
        eval_config = replace(
            eval_config,
            global_auc_workers=args.global_auc_workers,
        )
    return replace(config, eval=eval_config, train=train_config)


def _resolve_base_output_dir(config: RecAgoraConfig, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if config.eval.output_dir is not None:
        return Path(config.eval.output_dir)
    return Path(config.train.output_dir)


def _run_output_dir(
    *,
    base_output_dir: Path,
    seed: int,
    seeds: tuple[int, ...],
) -> Path:
    if len(seeds) == 1:
        return base_output_dir
    return base_output_dir / f"seed_{seed}"


def parse_seeds(raw: str | None, *, default_seed: int) -> tuple[int, ...]:
    if raw is None or str(raw).strip() == "":
        return (int(default_seed),)
    seeds = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not seeds:
        raise ValueError("--seed must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seed values must be unique")
    return seeds


def build_eval_summary(
    results: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> dict[str, Any]:
    metric_names = sorted(
        {
            metric
            for result in results
            for metric in result.get("metrics", {})
            if isinstance(result.get("metrics"), dict)
        }
    )
    metrics: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = [
            float(result["metrics"][name])
            for result in results
            if isinstance(result.get("metrics"), dict)
            and name in result["metrics"]
            and result["metrics"][name] is not None
        ]
        if not values:
            continue
        metrics[name] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {
        "seeds": list(seeds),
        "output_dir": str(output_dir),
        "runs": [
            {
                "seed": result.get("seed"),
                "metrics_path": result.get("metrics_path"),
                "metrics": result.get("metrics", {}),
            }
            for result in results
        ],
        "metrics": metrics,
    }


def _format_result(config: RecAgoraConfig, result: dict[str, Any]) -> str:
    loss = result["eval_loss"]
    loss_text = "n/a" if loss is None else f"{loss:.6f}"
    return (
        f"seed={config.train.seed} batches={result['batches']} "
        f"requests={result['request_groups']} examples={result['examples']} "
        f"eval_loss={loss_text} metrics={result['metrics_path']}"
        + (f" status={result['status']}" if loss is None else "")
    )


def _format_summary(summary: dict[str, Any], summary_path: Path) -> str:
    metrics = summary.get("metrics", {})
    metric_text = " ".join(
        f"{name}_mean={value['mean']:.6f}"
        for name, value in sorted(metrics.items())
        if isinstance(value, dict) and "mean" in value
    )
    return f"summary={summary_path} {metric_text}".strip()


if __name__ == "__main__":
    main()

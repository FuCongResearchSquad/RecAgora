from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import load_recagora_config
from training.train import train


# ===== CLI Arguments =====


def parse_args() -> argparse.Namespace:
    """Parse train CLI arguments and runtime overrides."""
    parser = argparse.ArgumentParser(description="Train a request-level ranking model.")
    parser.add_argument("--config", required=True, help="Path to a model YAML config.")
    parser.add_argument(
        "--data-config",
        default=None,
        help="Override the model YAML data_config include with a RequestData YAML.",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help=(
            "Override data.paths. Repeat this option to pass multiple files, "
            "directories, or glob patterns."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory containing dt=YYYY-MM-DD parquet partitions.",
    )
    parser.add_argument(
        "--date-range",
        default=None,
        help=(
            "Inclusive compact date range for --data-root, e.g. "
            "20260512-20260518."
        ),
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default=None, help="HotUser split (default train).")
    parser.add_argument("--data-manifest", default=None, help="Optional HotUser published.json manifest.")
    parser.add_argument("--output-dir", default=None, help="Override train.output_dir.")
    parser.add_argument("--device", default=None, help="Override train.device.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override train.max_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override train.lr.")
    grad_clip_group = parser.add_mutually_exclusive_group()
    grad_clip_group.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Override train.grad_clip_norm.",
    )
    grad_clip_group.add_argument(
        "--no-grad-clip",
        action="store_true",
        help="Disable gradient clipping regardless of the shared config.",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=None,
        help="Override train.progress_every_steps for rank-0 local progress.",
    )
    parser.add_argument(
        "--global-metrics-every-steps",
        type=int,
        default=None,
        help="Override train.global_metrics.every_steps.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Override train.num_workers.")
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help="Override train.checkpoint_every_steps.",
    )
    parser.add_argument(
        "--checkpoint-max-to-keep",
        type=int,
        default=None,
        help="Override train.checkpoint_max_to_keep.",
    )
    parser.add_argument(
        "--timing-trace-every-steps",
        type=int,
        default=None,
        help="Override train.timing_trace.every_steps.",
    )
    parser.add_argument(
        "--timing-trace-rank",
        type=int,
        default=None,
        help="Override the single rank that records timing traces.",
    )
    parser.add_argument(
        "--timing-trace",
        "--trace",
        dest="timing_trace",
        action="store_true",
        help="Enable rank-local asynchronous timing traces.",
    )
    parser.add_argument(
        "--no-timing-trace-distributed-sync",
        dest="timing_trace_distributed_sync",
        action="store_false",
        default=None,
        help="Keep timing trace collection free of distributed synchronization.",
    )
    parser.add_argument(
        "--communication-trace",
        action="store_true",
        help="Enable low-frequency rank-local communication profiler sampling.",
    )
    parser.add_argument(
        "--communication-trace-rank",
        type=int,
        default=None,
        help="Override the single rank sampled by the communication profiler.",
    )
    parser.add_argument(
        "--communication-trace-every-steps",
        type=int,
        default=None,
        help="Override train.communication_trace.every_steps.",
    )
    parser.add_argument(
        "--communication-trace-active-steps",
        type=int,
        default=None,
        help="Override profiler active steps per sampling window.",
    )
    parser.add_argument(
        "--rank-diagnostics",
        action="store_true",
        help="Enable explicit cross-rank diagnostic snapshots.",
    )
    parser.add_argument(
        "--rank-diagnostics-every-steps",
        type=int,
        default=None,
        help="Override train.rank_diagnostics.every_steps.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Override train.resume_from. Requires a full step_N training "
            "checkpoint and restores model, optimizer, scaler, RNG, and cursor."
        ),
    )
    parser.add_argument(
        "--resume-world-size-policy",
        choices=("strict", "elastic"),
        default=None,
        help=(
            "Override train.resume.world_size_policy. strict requires the "
            "checkpoint world size; elastic reloads a DCP v3 step checkpoint "
            "under a new world size with a global data cursor and RNG reseed."
        ),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Override train.init_from. Loads model weights only from a model-only "
            "last checkpoint or a full step_N checkpoint."
        ),
    )
    parser.add_argument(
        "--parquet-start-file",
        default=None,
        help="Override data.streaming.start_file for Parquet streaming.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override train.pin_memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override train.persistent_workers.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override train.amp.",
    )
    return parser.parse_args()


# ===== Main Entrypoint =====


def main() -> None:
    """Load config, apply CLI overrides, run training, and print result paths."""
    args = parse_args()
    data_paths = _resolve_data_paths(args)
    config = load_recagora_config(
        args.config,
        data_config=args.data_config,
        data_paths=data_paths,
        data_root=args.data_root if args.split is not None else None,
        data_split=args.split,
        data_manifest=args.data_manifest,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip_norm=args.grad_clip_norm,
        disable_grad_clip=args.no_grad_clip,
        progress_every_steps=args.progress_every_steps,
        global_metrics_every_steps=args.global_metrics_every_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        amp=args.amp,
        checkpoint_every_steps=args.checkpoint_every_steps,
        checkpoint_max_to_keep=args.checkpoint_max_to_keep,
        timing_trace_enabled=True if args.timing_trace else None,
        timing_trace_every_steps=args.timing_trace_every_steps,
        timing_trace_rank=args.timing_trace_rank,
        timing_trace_distributed_sync=args.timing_trace_distributed_sync,
        communication_trace_enabled=True if args.communication_trace else None,
        communication_trace_every_steps=args.communication_trace_every_steps,
        communication_trace_rank=args.communication_trace_rank,
        communication_trace_active_steps=args.communication_trace_active_steps,
        rank_diagnostics_enabled=True if args.rank_diagnostics else None,
        rank_diagnostics_every_steps=args.rank_diagnostics_every_steps,
        resume_from=args.resume_from,
        resume_world_size_policy=args.resume_world_size_policy,
        init_from=args.init_from,
        parquet_start_file=args.parquet_start_file,
    )
    result = train(config)
    last_loss = result["last_loss"]
    last_loss_text = "None" if last_loss is None else f"{last_loss:.6f}"
    print(
        f"device={result['device']} amp={result['amp_enabled']} "
        f"steps={result['steps']} last_loss={last_loss_text} "
        f"checkpoint={result['checkpoint_path']} "
        f"log={result['log_path']} metrics={result['metrics_path']} "
        f"curves={result['curve_path']} summary={result['summary_path']}"
    )


# ===== Data Path Helpers =====


def _resolve_data_paths(args: argparse.Namespace) -> list[str] | None:
    if args.split is not None or args.data_manifest is not None:
        if args.date_range is not None or args.data_path is not None:
            raise ValueError("HotUser --split cannot be combined with --date-range or --data-path")
        if args.data_root is not None and args.split is None:
            raise ValueError("Use --split when overriding the HotUser --data-root")
        return None
    if args.data_root is None and args.date_range is None:
        return args.data_path
    if args.data_root is None or args.date_range is None:
        raise ValueError("--data-root and --date-range must be provided together")
    if args.data_path is not None:
        raise ValueError("Use either --data-path or --data-root/--date-range")
    return _date_range_data_paths(args.data_root, args.date_range)


def _date_range_data_paths(data_root: str, date_range: str) -> list[str]:
    try:
        start_raw, end_raw = date_range.split("-", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            "--date-range must use compact format YYYYMMDD-YYYYMMDD"
        ) from exc

    start = _parse_compact_date(start_raw, "--date-range start")
    end = _parse_compact_date(end_raw, "--date-range end")
    if end < start:
        raise ValueError("--date-range end date must be >= start date")

    paths: list[str] = []
    day = start
    root = Path(data_root).expanduser()
    while day <= end:
        date = day.strftime("%Y-%m-%d")
        hive_partition = root / f"dt={date}"
        hive_exists = hive_partition.is_dir()
        if not hive_exists:
            raise FileNotFoundError(
                f"Data partition for {date} was not found: "
                f"{hive_partition.as_posix()!r}"
            )
        paths.append((hive_partition / "*.parquet").as_posix())
        day += timedelta(days=1)
    return paths


def _parse_compact_date(raw: str, field_name: str):
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYYMMDD format: {raw!r}") from exc


if __name__ == "__main__":
    main()

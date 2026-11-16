from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


LOSS_PLOT_WARMUP_STEPS = 100
LOSS_PLOT_SMOOTH_POINTS = 25
LOSS_COLORS = {
    "loss": "#2563eb",
    "loss_global": "#2563eb",
    "loss_click": "#16a34a",
    "loss_click_global": "#16a34a",
    "loss_atc": "#d97706",
    "loss_atc_global": "#d97706",
    "loss_order": "#dc2626",
    "loss_order_global": "#dc2626",
}
RANK_COLORS = ("#64748b", "#8b5cf6", "#0f766e", "#be123c", "#0369a1", "#9333ea")


def write_training_curves(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str | None]:
    metrics_file = Path(metrics_path)
    output = Path(output_dir)
    records = _read_metrics(metrics_file)
    loss_keys = _loss_keys(records)
    if not records or not loss_keys:
        return {
            "plot_path": None,
            "summary_path": None,
        }

    output.mkdir(parents=True, exist_ok=True)
    plot_path = output / "training_curves.png"
    summary_path = output / "training_summary.json"

    plot_info = _loss_plot_info(loss_keys)
    summary = _build_summary(records, loss_keys)
    summary["plots"] = plot_info
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_loss_comparison_figure(
        records,
        loss_keys,
        plot_path,
        title=(
            "Training losses "
            f"(rolling mean={LOSS_PLOT_SMOOTH_POINTS}, "
            f"scale step > {LOSS_PLOT_WARMUP_STEPS})"
        ),
        y_scale_min_step=LOSS_PLOT_WARMUP_STEPS,
    )
    return {
        "plot_path": str(plot_path),
        "summary_path": str(summary_path),
    }


def _read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _loss_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key, value in record.items():
            if key not in {"loss", "loss_global"} and not (
                key.startswith("loss_") and key.endswith("_global")
            ):
                continue
            if key in seen or _finite_float(value) is None:
                continue
            seen.add(key)
            keys.append(str(key))
    return sorted(keys, key=lambda item: (item not in {"loss", "loss_global"}, item))


def _build_summary(records: list[dict[str, Any]], loss_keys: list[str]) -> dict[str, Any]:
    steps = [_step_value(record, index) for index, record in enumerate(records)]
    summary: dict[str, Any] = {
        "steps": {
            "count": len(records),
            "first": steps[0] if steps else None,
            "last": steps[-1] if steps else None,
        },
        "losses": {},
    }
    for key in loss_keys:
        values = [
            value
            for record in records
            if (value := _finite_float(record.get(key))) is not None
        ]
        if not values:
            continue
        summary["losses"][key] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    lr_values = [
        value
        for record in records
        if (value := _finite_float(record.get("lr"))) is not None
    ]
    if lr_values:
        summary["lr"] = {
            "first": lr_values[0],
            "last": lr_values[-1],
            "min": min(lr_values),
            "max": max(lr_values),
        }
    return summary


def _write_loss_comparison_figure(
    records: list[dict[str, Any]],
    loss_keys: list[str],
    output_path: Path,
    *,
    title: str,
    y_scale_min_step: int | None = None,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    steps = [_step_value(record, index) for index, record in enumerate(records)]
    rows = len(loss_keys)
    fig_height = max(3.4, min(3.4 * rows, 18.0))
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(13.5, fig_height),
        dpi=145,
        squeeze=False,
        sharex=True,
    )
    fig.suptitle(title, fontsize=13, y=0.995)

    for row_index, key in enumerate(loss_keys):
        ax = axes[row_index][0]
        global_xs: list[int] = []
        global_ys: list[float] = []
        for step, record in zip(steps, records):
            value = _finite_float(record.get(key))
            if value is None:
                continue
            global_xs.append(step)
            global_ys.append(value)
        if global_ys:
            smooth_xs, smooth_ys = _rolling_mean_series(
                global_xs,
                global_ys,
                window=LOSS_PLOT_SMOOTH_POINTS,
            )
            ax.plot(
                smooth_xs,
                smooth_ys,
                linewidth=1.8,
                label=f"mean_ma{_effective_smooth_window(global_ys)}",
                color=_loss_color(key, row_index),
                alpha=0.96,
                solid_capstyle="round",
            )

        for rank_index, (rank, series) in enumerate(
            _rank_loss_series(records, steps, key).items()
        ):
            if not series:
                continue
            xs = [step for step, _ in series]
            ys = [value for _, value in series]
            color = RANK_COLORS[rank_index % len(RANK_COLORS)]
            if len(ys) >= 3:
                smooth_xs, smooth_ys = _rolling_mean_series(
                    xs,
                    ys,
                    window=LOSS_PLOT_SMOOTH_POINTS,
                )
                ax.plot(
                    smooth_xs,
                    smooth_ys,
                    linewidth=1.0,
                    alpha=0.58,
                    label=f"rank_{rank}_ma{_effective_smooth_window(ys)}",
                    color=color,
                    solid_capstyle="round",
                )
            else:
                ax.scatter(
                    xs,
                    ys,
                    s=14,
                    alpha=0.46,
                    label=f"rank_{rank}",
                    color=color,
                    linewidths=0,
                )

        ax.set_title(key)
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y_limits = _loss_y_limits(
            records,
            steps,
            key,
            y_scale_min_step=y_scale_min_step,
        )
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.legend(
            loc="upper right",
            ncol=min(4, 1 + len(_rank_loss_series(records, steps, key))),
            frameon=False,
            fontsize=8,
            handlelength=1.8,
            columnspacing=0.9,
        )

    axes[-1][0].set_xlabel("step")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def _configure_matplotlib() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    user_token = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user_token and hasattr(os, "getuid"):
        user_token = str(os.getuid())
    user_token = user_token or "user"
    safe_token = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in user_token
    )
    cache_dir = Path(tempfile.gettempdir()) / f"recagora-matplotlib-{safe_token}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def _rank_loss_series(
    records: list[dict[str, Any]],
    steps: list[int],
    loss_key: str,
) -> dict[int, list[tuple[int, float]]]:
    series: dict[int, list[tuple[int, float]]] = {}
    for step, record in zip(steps, records):
        traces = record.get("ddp_rank_trace")
        if not isinstance(traces, list):
            continue
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            rank = trace.get("rank")
            if not isinstance(rank, int):
                continue
            local_losses = trace.get("local_losses")
            if not isinstance(local_losses, dict):
                continue
            value = _finite_float(local_losses.get(loss_key))
            if value is None:
                continue
            series.setdefault(rank, []).append((step, value))
    return dict(sorted(series.items()))


def _loss_plot_info(loss_keys: list[str]) -> dict[str, Any]:
    return {
        "training_curves": "training_curves.png",
        "loss_plot_warmup_steps": LOSS_PLOT_WARMUP_STEPS,
        "loss_plot_smooth_points": LOSS_PLOT_SMOOTH_POINTS,
        "loss_keys": loss_keys,
        "rank_local_curves": False,
    }


def _loss_y_limits(
    records: list[dict[str, Any]],
    steps: list[int],
    loss_key: str,
    *,
    y_scale_min_step: int | None,
) -> tuple[float, float] | None:
    if y_scale_min_step is None:
        return None
    values = [
        value
        for step, record in zip(steps, records)
        if step > y_scale_min_step
        if (value := _finite_float(record.get(loss_key))) is not None
    ]
    for series in _rank_loss_series(records, steps, loss_key).values():
        values.extend(value for step, value in series if step > y_scale_min_step)
    if len(values) < 2:
        return None
    low = min(values)
    high = max(values)
    spread = high - low
    if spread <= 0:
        padding = max(abs(high) * 0.05, 1e-6)
    else:
        padding = spread * 0.12
    y_min = low - padding
    if low >= 0:
        y_min = max(0.0, y_min)
    return y_min, high + padding


def _rolling_mean_series(
    xs: list[int],
    ys: list[float],
    *,
    window: int,
) -> tuple[list[int], list[float]]:
    if not ys:
        return [], []
    effective_window = _effective_smooth_window(ys, window)
    if effective_window <= 1:
        return xs, ys
    out_xs: list[int] = []
    out_ys: list[float] = []
    total = 0.0
    for index, value in enumerate(ys):
        total += value
        if index >= effective_window:
            total -= ys[index - effective_window]
        divisor = min(index + 1, effective_window)
        out_xs.append(xs[index])
        out_ys.append(total / divisor)
    return out_xs, out_ys


def _effective_smooth_window(
    values: list[float],
    requested_window: int = LOSS_PLOT_SMOOTH_POINTS,
) -> int:
    if len(values) < 3:
        return 1
    return max(1, min(requested_window, len(values)))


def _loss_color(loss_key: str, index: int) -> str:
    if loss_key in LOSS_COLORS:
        return LOSS_COLORS[loss_key]
    palette = ("#7c3aed", "#0891b2", "#ca8a04", "#db2777", "#475569")
    return palette[index % len(palette)]


def _step_value(record: dict[str, Any], index: int) -> int:
    value = record.get("step")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return index + 1


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None

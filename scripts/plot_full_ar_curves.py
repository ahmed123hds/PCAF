#!/usr/bin/env python3
"""Plot full-autoregressive validation perplexity curves from JSONL logs.

The script recursively scans a log root, groups runs by dataset/model/sequence
length, and emits one two-panel figure per dataset. If multiple seeds are found
for the same dataset/model/sequence length, it plots the mean curve and a light
standard-deviation band.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

MODEL_ORDER = [
    "transformer_dense",
    "linear_attention",
    "local_transformer_w128",
    "global_local_transformer_w128_g16",
    "local_conv",
    "pcaf_no_gate",
    "pcaf_semantic",
    "pcaf_hybrid",
    "pcaf_context",
]

COLORS = {
    "transformer_dense": "#434343",
    "linear_attention": "#5f5f5f",
    "local_transformer_w128": "#7f7f7f",
    "global_local_transformer_w128_g16": "#adadad",
    "local_conv": "#cc3333",
    "pcaf_no_gate": "#e69138",
    "pcaf_semantic": "#3d85c8",
    "pcaf_hybrid": "#7b5ea7",
    "pcaf_context": "#2e7d32",
}

LABELS = {
    "transformer_dense": "Dense Transformer",
    "linear_attention": "Linear Attention",
    "local_transformer_w128": "Local Transformer (w=128)",
    "global_local_transformer_w128_g16": "Global-Local Trans. (w=128, g=16)",
    "local_conv": "Local Conv (No Cache)",
    "pcaf_no_gate": "PCAF No Gate",
    "pcaf_semantic": "PCAF Semantic Hash",
    "pcaf_hybrid": "PCAF Hybrid Hash",
    "pcaf_context": "PCAF Context Hash",
}

LINESTYLES = {
    "transformer_dense": (0, (5, 2)),
    "linear_attention": (0, (4, 1, 1, 1)),
    "local_transformer_w128": (0, (3, 2)),
    "global_local_transformer_w128_g16": (0, (1, 2)),
    "local_conv": "-",
    "pcaf_no_gate": "-",
    "pcaf_semantic": "-",
    "pcaf_hybrid": "-",
    "pcaf_context": "-",
}

MARKERS = {
    "transformer_dense": "^",
    "linear_attention": "P",
    "local_transformer_w128": "v",
    "global_local_transformer_w128_g16": ">",
    "local_conv": "x",
    "pcaf_no_gate": "d",
    "pcaf_semantic": "o",
    "pcaf_hybrid": "*",
    "pcaf_context": "s",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def model_from_path(path: Path, first_row: dict) -> str | None:
    stem = path.stem
    for candidate in sorted(MODEL_ORDER, key=len, reverse=True):
        if stem.startswith(candidate):
            return candidate

    model = first_row.get("model")
    if isinstance(model, str) and model:
        return model
    return None


def seq_len_from_path(path: Path, first_row: dict) -> int | None:
    value = first_row.get("seq_len")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"(?:seq|seqlen)(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def dataset_from_path(path: Path, first_row: dict) -> tuple[str, str]:
    haystack = " ".join(
        [
            str(path).lower(),
            str(first_row.get("dataset", "")).lower(),
            str(first_row.get("dataset_config", "")).lower(),
        ]
    )
    if "pg19" in haystack:
        return "pg19", "PG-19"
    if "wikitext-103" in haystack or "wikitext103" in haystack:
        return "wikitext103", "WikiText-103"
    if "wikitext-2" in haystack or "wikitext2" in haystack:
        return "wikitext2", "WikiText-2"
    return "unknown", "Unknown Dataset"


def curve_from_rows(rows: list[dict], min_step: int, max_ppl: float) -> dict[int, float]:
    points = {}
    for row in rows:
        if "step" not in row or "eval_ppl" not in row:
            continue
        step = int(row["step"])
        ppl = float(row["eval_ppl"])
        if step < min_step or not math.isfinite(ppl) or ppl > max_ppl:
            continue
        points[step] = ppl
    return points


def running_best_curve(curve: dict[int, float]) -> dict[int, float]:
    best = float("inf")
    out = {}
    for step in sorted(curve):
        best = min(best, curve[step])
        out[step] = best
    return out


def aggregate_curves(curves: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for curve in curves:
        for step, ppl in curve.items():
            by_step[step].append(ppl)

    steps, means, stds = [], [], []
    for step in sorted(by_step):
        values = np.asarray(by_step[step], dtype=np.float64)
        steps.append(step)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)) if len(values) > 1 else 0.0)
    return np.asarray(steps), np.asarray(means), np.asarray(stds)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.convolve(values, kernel, mode="same")
    half = window // 2
    out[:half] = values[:half]
    out[-half:] = values[-half:]
    return out


def collect_runs(log_roots: list[Path], min_step: int, max_ppl: float):
    grouped = defaultdict(list)
    dataset_titles = {}
    scanned = 0

    for log_root in log_roots:
        for path in sorted(log_root.rglob("*.jsonl")):
            rows = read_jsonl(path)
            if not rows:
                continue
            first = rows[0]
            model = model_from_path(path, first)
            seq_len = seq_len_from_path(path, first)
            if model is None or seq_len is None:
                continue

            dataset_key, dataset_title = dataset_from_path(path, first)
            curve = curve_from_rows(rows, min_step=min_step, max_ppl=max_ppl)
            if not curve:
                continue

            grouped[(dataset_key, model, seq_len)].append(curve)
            dataset_titles[dataset_key] = dataset_title
            scanned += 1

    return grouped, dataset_titles, scanned


def plot_dataset(
    grouped,
    dataset_key: str,
    dataset_title: str,
    output_dir: Path,
    seq_lens: list[int],
    max_ppl: float,
    smooth_window: int,
    y_max: float | None,
    y_min: float | None,
    running_best: bool,
):
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(6.5 * len(seq_lens), 5.8), sharey=False)
    if len(seq_lens) == 1:
        axes = [axes]

    all_plotted_values = []
    for ax, seq_len in zip(axes, seq_lens):
        ax.set_title(f"Sequence Length $T = {seq_len}$", fontweight="bold", pad=10)
        for model in MODEL_ORDER:
            curves = grouped.get((dataset_key, model, seq_len), [])
            if not curves:
                continue
            if running_best:
                curves = [running_best_curve(curve) for curve in curves]

            steps, means, stds = aggregate_curves(curves)
            if len(steps) == 0:
                continue

            means = smooth(means, smooth_window)
            all_plotted_values.extend(means.tolist())
            step_gap = int(steps[1] - steps[0]) if len(steps) > 1 else 1000
            markevery = max(1, 2500 // max(1, step_gap))

            ax.plot(
                steps,
                means,
                color=COLORS.get(model, "#333333"),
                label=LABELS.get(model, model),
                linestyle=LINESTYLES.get(model, "-"),
                linewidth=2.5 if model in {"pcaf_context", "pcaf_semantic"} else 1.7,
                marker=MARKERS.get(model, "o"),
                markersize=4.5,
                markevery=markevery,
                zorder=10 if model.startswith("pcaf") else 5,
                alpha=0.95,
            )
            if np.max(stds) > 0:
                ax.fill_between(
                    steps,
                    means - stds,
                    means + stds,
                    color=COLORS.get(model, "#333333"),
                    alpha=0.12,
                    linewidth=0,
                )

        ax.set_xlabel("Training Step", labelpad=8)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ylabel = "Best-so-far Full-AR Validation Perplexity" if running_best else "Full-AR Validation Perplexity"
    axes[0].set_ylabel(f"{ylabel} ({dataset_title})", labelpad=8)

    if all_plotted_values:
        ymin = y_min if y_min is not None else max(0.0, min(all_plotted_values) * 0.85)
        ymax = y_max if y_max is not None else min(max_ppl, max(all_plotted_values) * 1.10)
        if ymax <= ymin:
            ymax = ymin + 10
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    seen = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            seen.setdefault(label, handle)

    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.86])

    pdf_path = output_dir / f"full_ar_curves_{dataset_key}.pdf"
    png_path = output_dir / f"full_ar_curves_{dataset_key}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--extra-log-root",
        action="append",
        type=Path,
        default=[],
        help="Additional log root to merge into the same figure.",
    )
    parser.add_argument("--seq-lens", default="1024,2048")
    parser.add_argument(
        "--dataset",
        choices=["wikitext103", "pg19", "wikitext2", "unknown"],
        default=None,
        help="Only plot one dataset key.",
    )
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument(
        "--running-best",
        action="store_true",
        help="Plot best-so-far validation perplexity instead of raw validation perplexity.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Lower y-axis limit for showing the competitive region.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Upper y-axis limit for showing the competitive region.",
    )
    parser.add_argument(
        "--max-ppl",
        type=float,
        default=1000.0,
        help="Drop very early high-perplexity points above this value.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seq_lens = [int(part) for part in args.seq_lens.split(",") if part.strip()]
    log_roots = [args.log_root, *args.extra_log_root]
    grouped, dataset_titles, scanned = collect_runs(log_roots, args.min_step, args.max_ppl)

    if scanned == 0:
        roots = ", ".join(str(path) for path in log_roots)
        raise SystemExit(f"No plottable JSONL curves found under {roots}")

    outputs = []
    for dataset_key, dataset_title in sorted(dataset_titles.items()):
        if args.dataset is not None and dataset_key != args.dataset:
            continue
        has_dataset = any(key[0] == dataset_key for key in grouped)
        if not has_dataset:
            continue
        outputs.extend(
            plot_dataset(
                grouped,
                dataset_key,
                dataset_title,
                args.output_dir,
                seq_lens=seq_lens,
                max_ppl=args.max_ppl,
                smooth_window=args.smooth_window,
                y_max=args.y_max,
                y_min=args.y_min,
                running_best=args.running_best,
            )
        )

    print(f"Scanned {scanned} JSONL files from {', '.join(str(path) for path in log_roots)}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

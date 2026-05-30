#!/usr/bin/env python3
"""Create paper-ready Markdown and LaTeX tables from TPU JSONL logs."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt_num(val: float, digits: int = 2) -> str:
    if math.isnan(val):
        return "--"
    return f"{val:.{digits}f}"


def fmt_mean_std(vals: list[float], digits: int = 2) -> str:
    mean, std = mean_std(vals)
    if math.isnan(mean):
        return "--"
    if len(vals) <= 1:
        return fmt_num(mean, digits)
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def dataset_label(run: dict[str, Any]) -> str:
    ds = str(run.get("dataset", ""))
    cfg = str(run.get("dataset_config", ""))
    if "pg19" in ds.lower():
        return "PG-19"
    if "wikitext-103" in cfg:
        return "WikiText-103"
    if "wikitext-2" in cfg:
        return "WikiText-2"
    return cfg or ds or "unknown"


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        last = dict(rows[-1])
        best = min((float(row["eval_ppl"]) for row in rows if "eval_ppl" in row), default=float("nan"))
        best_row = min(
            (row for row in rows if "eval_ppl" in row),
            key=lambda row: float(row["eval_ppl"]),
            default=last,
        )
        last["run_name"] = path.stem
        last["log_group"] = str(path.parent.relative_to(root))
        last["dataset_label"] = dataset_label(last)
        last["best_ppl"] = best
        last["best_acc"] = float(best_row.get("eval_acc", last.get("eval_acc", 0.0)))
        last["last_step"] = int(last.get("step", 0))
        last["tok_per_sec"] = float(last.get("train_tokens_per_sec", 0.0))
        runs.append(last)
    return runs


def model_label(name: str) -> str:
    labels = {
        "pcaf_context": "PCAF-context",
        "pcaf_semantic": "PCAF-semantic",
        "pcaf_hybrid": "PCAF-hybrid",
        "pcaf_no_gate": "PCAF no gate",
        "local_conv": "Local conv",
        "transformer_dense": "Dense Transformer",
        "linear_attention": "Linear attention",
        "local_transformer_w128": "Local Transformer",
        "global_local_transformer_w128_g16": "Global-local Transformer",
    }
    for key, label in labels.items():
        if name == key or name.startswith(f"{key}_"):
            return label
    if name.startswith("pcaf_topk"):
        return "PCAF top-k=" + name.removeprefix("pcaf_topk").split("_")[0]
    if name.startswith("pcaf_buckets"):
        return "PCAF buckets=" + name.removeprefix("pcaf_buckets").split("_")[0]
    if name.startswith("pcaf_context_order"):
        return "PCAF order=" + name.removeprefix("pcaf_context_order").split("_")[0]
    return name.replace("_", "\\_")


def grouped_stats(runs: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["dataset_label"], model_label(run["run_name"]), int(run.get("seq_len", 0)))].append(run)
    return groups


def write_markdown(root: Path, groups: dict[tuple[str, str, int], list[dict[str, Any]]]) -> None:
    out = root / "paper_ready_results.md"
    lines = [
        "# Paper-Ready Results",
        "",
        "| Dataset | Model | Seq | Seeds | Params | Best PPL | Acc @ Best | Tok/s | Final Step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, model, seq), rows in sorted(groups.items()):
        params = [float(r.get("params", 0)) for r in rows]
        ppls = [float(r.get("best_ppl", float("nan"))) for r in rows]
        accs = [float(r.get("best_acc", 0.0)) * 100.0 for r in rows]
        toks = [float(r.get("tok_per_sec", 0.0)) / 1_000_000.0 for r in rows]
        steps = [float(r.get("last_step", 0.0)) for r in rows]
        lines.append(
            f"| {dataset} | {model} | {seq} | {len(rows)} | "
            f"{fmt_num(statistics.mean(params) / 1_000_000.0, 2)}M | "
            f"{fmt_mean_std(ppls, 2)} | {fmt_mean_std(accs, 2)}\\% | "
            f"{fmt_mean_std(toks, 2)}M | {fmt_num(statistics.mean(steps), 0)} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(root: Path, groups: dict[tuple[str, str, int], list[dict[str, Any]]]) -> None:
    out = root / "paper_ready_tables.tex"
    lines = [
        "% Auto-generated by associative_field_lab/scripts/make_paper_ready_tables.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Submission-ready TPU results aggregated over available seeds. Throughput is reported across the full TPU pod.}",
        "  \\label{tab:submission_ready_results}",
        "  \\resizebox{\\linewidth}{!}{",
        "  \\begin{tabular}{llrrrrrr}",
        "    \\toprule",
        "    Dataset & Model & Seq. & Seeds & Params & Best PPL & Acc. @ Best & Throughput \\\\",
        "    \\midrule",
    ]
    current_dataset = None
    for (dataset, model, seq), rows in sorted(groups.items()):
        if current_dataset is not None and dataset != current_dataset:
            lines.append("    \\midrule")
        current_dataset = dataset
        params = [float(r.get("params", 0)) for r in rows]
        ppls = [float(r.get("best_ppl", float("nan"))) for r in rows]
        accs = [float(r.get("best_acc", 0.0)) * 100.0 for r in rows]
        toks = [float(r.get("tok_per_sec", 0.0)) / 1_000_000.0 for r in rows]
        lines.append(
            "    "
            + " & ".join(
                [
                    dataset,
                    model,
                    str(seq),
                    str(len(rows)),
                    f"{statistics.mean(params) / 1_000_000.0:.2f}M",
                    fmt_mean_std(ppls, 2),
                    fmt_mean_std(accs, 2) + "\\%",
                    fmt_mean_std(toks, 2) + "M tok/s",
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  }",
            "\\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_paper_ready_tables.py LOG_ROOT")
    root = Path(sys.argv[1])
    runs = load_runs(root)
    groups = grouped_stats(runs)
    write_markdown(root, groups)
    write_latex(root, groups)
    print(f"Wrote {root / 'paper_ready_results.md'}")
    print(f"Wrote {root / 'paper_ready_tables.tex'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def best_ppl(rows: list[dict[str, Any]]) -> float:
    vals = [row["eval_ppl"] for row in rows if "eval_ppl" in row]
    return min(vals) if vals else float("nan")


def pct(val: float) -> str:
    return f"{100.0 * val:.2f}%"


def fmt_float(val: Any, digits: int = 4) -> str:
    if val is None:
        return ""
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_iclr_results.py LOG_ROOT")
    root = Path(sys.argv[1])
    paths = sorted(root.rglob("*.jsonl"))
    runs: list[dict[str, Any]] = []

    for path in paths:
        rows = read_rows(path)
        if not rows:
            continue
        last = rows[-1]
        run = dict(last)
        run["run_name"] = path.stem
        run["log_dir"] = str(path.parent.relative_to(root))
        run["best_ppl"] = best_ppl(rows)
        run["last_step"] = last.get("step", 0)
        runs.append(run)

    print("# ICLR Experiment Audit")
    print()
    print(f"Log root: `{root}`")
    print(f"JSONL runs found: {len(runs)}")
    print()

    if not runs:
        return

    print("## Run Summary")
    print()
    print(
        "| Group | Run | Dataset | Seq | Batch | Params | Step | Final PPL | Best PPL | Acc | Tok/s |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in sorted(runs, key=lambda r: (r["log_dir"], r.get("seq_len", 0), r["run_name"])):
        dataset = run.get("dataset", "")
        config = run.get("dataset_config", "")
        dataset_label = f"{dataset}/{config}" if config else dataset
        print(
            "| "
            + " | ".join(
                [
                    str(run["log_dir"]),
                    str(run["run_name"]),
                    dataset_label,
                    str(run.get("seq_len", "")),
                    str(run.get("global_batch_size", run.get("batch_size", ""))),
                    str(run.get("params", "")),
                    str(run.get("last_step", "")),
                    fmt_float(run.get("eval_ppl"), 2),
                    fmt_float(run.get("best_ppl"), 2),
                    pct(float(run.get("eval_acc", 0.0))),
                    fmt_float(run.get("train_tokens_per_sec"), 1),
                ]
            )
            + " |"
        )
    print()

    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run.get("log_dir"), run.get("dataset_config", ""), run.get("seq_len"))].append(run)

    print("## Fairness Checks")
    print()
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        params = [int(run.get("params", 0)) for run in group if run.get("params")]
        steps = sorted({int(run.get("last_step", 0)) for run in group})
        train_tokens = sorted({int(run.get("train_tokens", 0)) for run in group if run.get("train_tokens")})
        eval_tokens = sorted({int(run.get("eval_tokens", 0)) for run in group if run.get("eval_tokens")})
        models = ", ".join(sorted(run["run_name"] for run in group))
        print(f"- `{key}`")
        print(f"  Models: {models}")
        if params:
            ratio = max(params) / max(min(params), 1)
            status = "OK" if ratio <= 1.25 else "CHECK"
            print(f"  Param range: {min(params):,} to {max(params):,} ratio={ratio:.3f} [{status}]")
        print(f"  Final steps seen: {steps}")
        if len(train_tokens) == 1:
            print(f"  Train tokens: {train_tokens[0]:,} [OK]")
        elif train_tokens:
            print(f"  Train tokens vary: {train_tokens} [CHECK]")
        if len(eval_tokens) == 1:
            print(f"  Eval tokens: {eval_tokens[0]:,} [OK]")
        elif eval_tokens:
            print(f"  Eval tokens vary: {eval_tokens} [CHECK]")
    print()

    print("## Reviewer Risk Notes")
    print()
    print("- Treat TPU throughput as XLA/JAX throughput, not CUDA/Triton throughput.")
    print("- Claims should be scoped to the implemented baselines and training budget.")
    print("- If a Transformer curve is still descending at the final step, extend that run before submission.")
    print("- For final paper tables, report mean/std over seeds for the headline WikiText-103 rows if budget allows.")


if __name__ == "__main__":
    main()

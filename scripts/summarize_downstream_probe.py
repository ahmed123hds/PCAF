#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kv_recall_eval"):
                rows.append(row)
    return rows


def model_name(path: Path, row: dict[str, Any]) -> str:
    stem = path.stem
    if stem:
        return stem
    return str(row.get("model", "unknown"))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_downstream_probe.py LOG_DIR_OR_JSONL")

    root = Path(sys.argv[1])
    files = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    runs: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for path in files:
        rows = load_rows(path)
        if not rows:
            continue
        best_lm = min(rows, key=lambda r: float(r.get("eval_ppl", float("inf"))))
        best_kv = max(rows, key=lambda r: float(r.get("kv_recall_acc", float("-inf"))))
        final = rows[-1]
        runs.append((model_name(path, final), best_lm, best_kv, final))

    header = (
        "run\tseq_len\tparams\tbest_lm_ppl\tbest_lm_step\t"
        "best_kv_acc\tbest_kv_step\tfinal_kv_acc\tkv_pairs"
    )
    print(header)
    for name, best_lm, best_kv, final in runs:
        print(
            "\t".join(
                [
                    name,
                    str(final.get("seq_len", "")),
                    str(final.get("params", "")),
                    fmt(float(best_lm.get("eval_ppl", 0.0)), 2),
                    str(best_lm.get("step", "")),
                    fmt(100.0 * float(best_kv.get("kv_recall_acc", 0.0)), 2) + "%",
                    str(best_kv.get("step", "")),
                    fmt(100.0 * float(final.get("kv_recall_acc", 0.0)), 2) + "%",
                    str(final.get("kv_recall_pairs", "")),
                ]
            )
        )


if __name__ == "__main__":
    main()

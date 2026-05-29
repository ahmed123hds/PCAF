from __future__ import annotations

import json
import sys
from pathlib import Path


def read_last_jsonl(path: Path) -> dict | None:
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def best_ppl(path: Path) -> float | None:
    best = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            val = row.get("eval_ppl")
            if val is not None:
                best = val if best is None else min(best, val)
    return best


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_logs.py LOG_DIR")
    log_dir = Path(sys.argv[1])
    print(
        "run\tseq_len\tbatch\tparams\tfinal_ppl\tbest_ppl\tfinal_acc\t"
        "tok_per_sec\ttrain_step_sec\telapsed_min\tpeak_mem_mb"
    )
    for path in sorted(log_dir.glob("*.jsonl")):
        row = read_last_jsonl(path)
        if row is None:
            continue
        print(
            "\t".join(
                [
                    path.stem,
                    str(row.get("seq_len", "")),
                    str(row.get("batch_size", "")),
                    str(row.get("params", "")),
                    f"{row.get('eval_ppl', 0.0):.4f}",
                    f"{best_ppl(path) or 0.0:.4f}",
                    f"{row.get('eval_acc', 0.0):.4f}",
                    f"{row.get('train_tokens_per_sec', 0.0):.1f}",
                    f"{row.get('train_step_sec', 0.0):.4f}",
                    f"{row.get('elapsed_sec', 0.0) / 60.0:.2f}",
                    f"{row.get('peak_cuda_mem_mb', 0.0):.1f}",
                ]
            )
        )


if __name__ == "__main__":
    main()

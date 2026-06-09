from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from pcaf import SparseAssociativeField, TransformerClassifier, make_batch, task_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train associative field experiments")
    parser.add_argument("--model", choices=["pcaf", "transformer"], default="pcaf")
    parser.add_argument("--task", choices=["kv_recall", "induction"], default="kv_recall")

    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--eval-n-pairs", type=int, default=0)
    parser.add_argument("--n-keys", type=int, default=2048)
    parser.add_argument("--n-values", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--eval-seq-len", type=int, default=0)
    parser.add_argument("--symbol-vocab", type=int, default=2048)

    parser.add_argument(
        "--candidate-mode",
        choices=["hash", "full", "oracle", "triton_hash"],
        default="hash",
    )
    parser.add_argument("--num-buckets", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=4096)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--log-jsonl", default="")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(args: argparse.Namespace, vocab_size: int, num_classes: int) -> torch.nn.Module:
    if args.model == "pcaf":
        return SparseAssociativeField(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            num_buckets=args.num_buckets,
            top_k=args.top_k,
            candidate_mode=args.candidate_mode,
            dropout=args.dropout,
        )

    return TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    )


def sample(args: argparse.Namespace, device: torch.device, eval_mode: bool = False):
    n_pairs = args.eval_n_pairs if eval_mode and args.eval_n_pairs > 0 else args.n_pairs
    seq_len = args.eval_seq_len if eval_mode and args.eval_seq_len > 0 else args.seq_len
    return make_batch(
        args.task,
        args.batch_size,
        device,
        n_pairs=n_pairs,
        n_keys=args.n_keys,
        n_values=args.n_values,
        seq_len=seq_len,
        symbol_vocab=args.symbol_vocab,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for _ in range(args.eval_batches):
        batch = sample(args, device, eval_mode=True)
        logits = model(batch.tokens)
        loss = F.cross_entropy(logits, batch.targets)
        pred = logits.argmax(dim=-1)
        correct += int((pred == batch.targets).sum().item())
        total += int(batch.targets.numel())
        loss_sum += float(loss.item()) * int(batch.targets.numel())

    return loss_sum / total, correct / total


def append_jsonl(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    info = task_info(args.task, args.n_keys, args.n_values, args.symbol_vocab)
    model = build_model(args, info.vocab_size, info.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model={args.model} task={args.task} device={device} "
        f"params={n_params:,} chance={info.chance_accuracy:.5f}"
    )
    if args.task == "kv_recall":
        eval_pairs = args.eval_n_pairs if args.eval_n_pairs > 0 else args.n_pairs
        print(
            f"train_pairs={args.n_pairs} eval_pairs={eval_pairs} "
            f"keys={args.n_keys} values={args.n_values}"
        )
    else:
        eval_len = args.eval_seq_len if args.eval_seq_len > 0 else args.seq_len
        print(
            f"train_seq_len={args.seq_len} eval_seq_len={eval_len} "
            f"symbol_vocab={args.symbol_vocab}"
        )

    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = sample(args, device, eval_mode=False)
        logits = model(batch.tokens)
        loss = F.cross_entropy(logits, batch.targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            train_acc = float((logits.argmax(dim=-1) == batch.targets).float().mean().item())
            eval_loss, eval_acc = evaluate(model, args, device)
            elapsed = time.time() - start
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "elapsed_sec": elapsed,
            }
            print(
                f"step={step:05d} loss={loss.item():.4f} train_acc={train_acc:.4f} "
                f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                f"time={elapsed:.1f}s"
            )
            append_jsonl(args.log_jsonl, row)


if __name__ == "__main__":
    main()

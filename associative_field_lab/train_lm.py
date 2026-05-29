from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from datasets import load_dataset

from pcaf import (
    ContextAssociativeLM,
    LocalConvLM,
    SparseAssociativeField,
    SparseAttentionClassifier,
    TransformerClassifier,
)

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover
    Mamba = None


PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(tok, unk) for tok in TOKEN_RE.findall(text)]


class RandomTokenBatcher:
    def __init__(
        self,
        ids: list[int],
        *,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seed: int,
    ):
        if len(ids) <= seq_len + 1:
            raise ValueError("not enough tokens for requested seq_len")
        self.data = torch.tensor(ids, dtype=torch.long, device=device)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.aranges = torch.arange(seq_len + 1, device=device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0,
            self.data.numel() - self.seq_len - 1,
            (self.batch_size,),
            device=self.device,
            generator=self.generator,
        )
        block = self.data[starts.unsqueeze(1) + self.aranges.unsqueeze(0)]
        return block[:, :-1], block[:, -1]


class MambaClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        num_layers: int,
        d_state: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise RuntimeError("mamba_ssm is not installed in this environment")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "mixer": Mamba(d_model=d_model, d_state=d_state),
                        "norm2": nn.LayerNorm(d_model),
                        "mlp": nn.Sequential(
                            nn.Linear(d_model, d_hidden),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_hidden, d_model),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = x + layer["mixer"](layer["norm1"](x))
            x = x + layer["mlp"](layer["norm2"](x))
        return self.head(self.final_norm(x[:, -1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real text LM benchmark for PCAF")
    parser.add_argument(
        "--model",
        choices=[
            "pcaf",
            "pcaf_context",
            "local_conv",
            "transformer",
            "local_transformer",
            "global_local_transformer",
            "mamba",
        ],
        default="pcaf",
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-train-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-eval-tokens", type=int, default=200_000)

    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-sample-seed", type=int, default=10_001)
    parser.add_argument("--eval-sample-seed", type=int, default=20_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-jsonl", default="")

    parser.add_argument(
        "--candidate-mode",
        choices=[
            "hash",
            "full",
            "oracle",
            "triton_hash",
            "semantic_hash",
            "hybrid_semantic_hash",
        ],
        default="triton_hash",
    )
    parser.add_argument("--num-buckets", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--context-order", type=int, default=3)
    parser.add_argument("--semantic-buckets", type=int, default=256)
    parser.add_argument("--semantic-temperature", type=float, default=0.2)
    parser.add_argument("--semantic-score-scale", type=float, default=1.0)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--fixed-cache-weight", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--attention-window", type=int, default=128)
    parser.add_argument("--global-tokens", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def iter_texts(dataset_split) -> list[str]:
    return [row["text"] for row in dataset_split if row.get("text", "").strip()]


def build_vocab(texts: list[str], *, max_vocab: int, min_freq: int) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(TOKEN_RE.findall(text))

    itos = [PAD, UNK, EOS]
    for token, freq in counter.most_common(max_vocab - len(itos)):
        if freq < min_freq:
            break
        itos.append(token)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def encode_corpus(texts: list[str], vocab: Vocab, *, limit: int) -> list[int]:
    eos = vocab.stoi[EOS]
    ids: list[int] = []
    for text in texts:
        ids.extend(vocab.encode(text))
        ids.append(eos)
        if 0 < limit <= len(ids):
            return ids[:limit]
    return ids


def build_model(args: argparse.Namespace, vocab_size: int) -> nn.Module:
    if args.model == "pcaf":
        return SparseAssociativeField(
            vocab_size=vocab_size,
            num_classes=vocab_size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            num_buckets=args.num_buckets,
            top_k=args.top_k,
            candidate_mode=args.candidate_mode,
            dropout=args.dropout,
        )
    if args.model == "pcaf_context":
        return ContextAssociativeLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            num_buckets=args.num_buckets,
            top_k=args.top_k,
            candidate_mode=args.candidate_mode,
            context_order=args.context_order,
            local_layers=args.local_layers,
            local_kernel_size=args.local_kernel_size,
            dropout=args.dropout,
            use_cache=not args.no_cache,
            use_gate=not args.no_gate,
            fixed_cache_weight=args.fixed_cache_weight,
            semantic_buckets=args.semantic_buckets,
            semantic_temperature=args.semantic_temperature,
            semantic_score_scale=args.semantic_score_scale,
        )
    if args.model == "local_conv":
        return LocalConvLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            local_layers=args.local_layers,
            local_kernel_size=args.local_kernel_size,
            dropout=args.dropout,
        )
    if args.model == "transformer":
        return TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=vocab_size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            max_seq_len=args.seq_len,
        )
    if args.model in {"local_transformer", "global_local_transformer"}:
        return SparseAttentionClassifier(
            vocab_size=vocab_size,
            num_classes=vocab_size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            max_seq_len=args.seq_len,
            attention_mode=(
                "local" if args.model == "local_transformer" else "global_local"
            ),
            window_size=args.attention_window,
            global_tokens=args.global_tokens,
        )
    return MambaClassifier(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        num_layers=args.layers,
        d_state=args.d_state,
        dropout=args.dropout,
    )


def append_jsonl(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(model: nn.Module, batcher: RandomTokenBatcher, *, batches: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for _ in range(batches):
        tokens, targets = batcher.sample()
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        total_loss += float(loss.item()) * targets.numel()
        total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total += targets.numel()
    return total_loss / total, total_correct / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    raw = load_dataset(args.dataset, args.dataset_config, cache_dir=args.cache_dir)
    train_texts = iter_texts(raw["train"])
    eval_split = "validation" if "validation" in raw else "test"
    eval_texts = iter_texts(raw[eval_split])

    vocab = build_vocab(train_texts, max_vocab=args.max_vocab, min_freq=args.min_freq)
    train_ids = encode_corpus(train_texts, vocab, limit=args.max_train_tokens)
    eval_ids = encode_corpus(eval_texts, vocab, limit=args.max_eval_tokens)

    train_batcher = RandomTokenBatcher(
        train_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        seed=args.train_sample_seed,
    )
    eval_batcher = RandomTokenBatcher(
        eval_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        seed=args.eval_sample_seed,
    )

    model = build_model(args, vocab.size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    n_params = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"dataset={args.dataset}/{args.dataset_config} split={eval_split} "
        f"model={args.model} device={device}"
    )
    print(
        f"vocab={vocab.size} train_tokens={len(train_ids):,} "
        f"eval_tokens={len(eval_ids):,} params={n_params:,}"
    )

    run_start = time.perf_counter()
    last_log_time = run_start
    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        model.train()
        tokens, targets = train_batcher.sample()
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_elapsed = time.perf_counter() - step_start

        if step == 1 or step % args.eval_every == 0:
            eval_start = time.perf_counter()
            train_acc = float((logits.argmax(dim=-1) == targets).float().mean().item())
            eval_loss, eval_acc = evaluate(
                model, eval_batcher, batches=args.eval_batches
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            eval_elapsed = time.perf_counter() - eval_start
            now = time.perf_counter()
            total_elapsed = now - run_start
            interval_elapsed = now - last_log_time
            interval_steps = 1 if step == 1 else args.eval_every
            train_step_sec = max(
                (interval_elapsed - eval_elapsed) / interval_steps,
                0.0,
            )
            train_tokens_per_sec = (
                args.batch_size * args.seq_len / train_step_sec
                if train_step_sec > 0
                else 0.0
            )
            peak_cuda_mem_mb = (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )
            ppl = math.exp(min(20.0, eval_loss))
            print(
                f"step={step:05d} loss={loss.item():.4f} train_acc={train_acc:.4f} "
                f"eval_loss={eval_loss:.4f} eval_ppl={ppl:.2f} eval_acc={eval_acc:.4f} "
                f"train_step_sec={train_step_sec:.4f} tok_per_sec={train_tokens_per_sec:.1f} "
                f"eval_sec={eval_elapsed:.2f} elapsed_min={total_elapsed / 60.0:.2f} "
                f"peak_mem_mb={peak_cuda_mem_mb:.1f}"
            )
            append_jsonl(
                args.log_jsonl,
                {
                    "step": step,
                    "model": args.model,
                    "params": n_params,
                    "train_loss": float(loss.item()),
                    "train_acc": train_acc,
                    "eval_loss": eval_loss,
                    "eval_ppl": ppl,
                    "eval_acc": eval_acc,
                    "last_step_sec": step_elapsed,
                    "train_step_sec": train_step_sec,
                    "train_tokens_per_sec": train_tokens_per_sec,
                    "eval_sec": eval_elapsed,
                    "elapsed_sec": total_elapsed,
                    "peak_cuda_mem_mb": peak_cuda_mem_mb,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "d_model": args.d_model,
                    "d_hidden": args.d_hidden,
                    "layers": args.layers,
                    "heads": args.heads,
                    "attention_window": args.attention_window,
                    "global_tokens": args.global_tokens,
                    "top_k": args.top_k,
                    "num_buckets": args.num_buckets,
                    "candidate_mode": args.candidate_mode,
                    "context_order": args.context_order,
                    "semantic_buckets": args.semantic_buckets,
                    "semantic_temperature": args.semantic_temperature,
                    "semantic_score_scale": args.semantic_score_scale,
                    "local_layers": args.local_layers,
                    "local_kernel_size": args.local_kernel_size,
                    "use_cache": not args.no_cache,
                    "use_gate": not args.no_gate,
                    "fixed_cache_weight": args.fixed_cache_weight,
                },
            )
            last_log_time = now


if __name__ == "__main__":
    main()

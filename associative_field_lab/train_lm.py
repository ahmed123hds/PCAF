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
    causal_ngram_hash,
)

try:
    from mamba_ssm import Mamba, Mamba2, Mamba3
except Exception:  # pragma: no cover
    Mamba = None
    Mamba2 = None
    Mamba3 = None


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
        full_ar: bool = False,
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
        self.full_ar = full_ar

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0,
            self.data.numel() - self.seq_len - 1,
            (self.batch_size,),
            device=self.device,
            generator=self.generator,
        )
        block = self.data[starts.unsqueeze(1) + self.aranges.unsqueeze(0)]
        tokens = block[:, :-1]
        targets = block[:, 1:] if self.full_ar else block[:, -1]
        return tokens, targets


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
        variant: str = "mamba",
    ) -> None:
        super().__init__()
        mixer_cls = {
            "mamba": Mamba,
            "mamba2": Mamba2,
            "mamba3": Mamba3,
        }[variant]
        if mixer_cls is None:
            raise RuntimeError(f"{variant} is not installed in this environment")

        self.embedding = nn.Embedding(vocab_size, d_model)
        layers = []
        for layer_idx in range(num_layers):
            mixer_kwargs = {"d_model": d_model, "d_state": d_state}
            if variant == "mamba3":
                mixer_kwargs.update({"layer_idx": layer_idx, "n_layer": num_layers})
            else:
                mixer_kwargs.update({"layer_idx": layer_idx})
            layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "mixer": mixer_cls(**mixer_kwargs),
                        "norm2": nn.LayerNorm(d_model),
                        "mlp": nn.Sequential(
                            nn.Linear(d_model, d_hidden),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_hidden, d_model),
                        ),
                    }
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            mixed = layer["mixer"](layer["norm1"](x))
            if isinstance(mixed, tuple):
                mixed = mixed[0]
            x = x + mixed
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
            "mamba2",
            "mamba3",
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
    parser.add_argument(
        "--loss-mode",
        choices=["last_token", "full_ar"],
        default="last_token",
        help="last_token predicts one token after the sampled window; full_ar "
             "predicts every next token inside the sampled window.",
    )
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
        variant=args.model,
    )


def target_log_probs_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return target_logits - torch.logsumexp(logits, dim=-1)


def transformer_full_ar_logits(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    seq_len = tokens.size(1)
    if seq_len > model.pos_emb.size(0):
        raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={model.pos_emb.size(0)}")

    x = model.token_emb(tokens) + model.pos_emb[:seq_len].to(tokens.device)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device),
        diagonal=1,
    )
    encoded = model.encoder(x, mask=causal_mask)
    return model.head(encoded)


def pcaf_context_full_ar_target_log_probs(
    model: ContextAssociativeLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = model.token_emb(tokens)
    for block in model.local_blocks:
        x = block(x)

    states = x
    param_logits = model.param_head(states)
    target_param_log_probs = target_log_probs_from_logits(param_logits, targets)
    pred = param_logits.argmax(dim=-1)
    acc = (pred == targets).float().mean()
    if not model.use_cache:
        return target_param_log_probs, acc

    batch, seq_len = tokens.shape
    value_tokens = torch.cat(
        [tokens[:, 1:], torch.zeros(batch, 1, dtype=tokens.dtype, device=tokens.device)],
        dim=1,
    )
    record_keys = F.normalize(model.key_proj(states), dim=-1)
    query = F.normalize(model.query_proj(states), dim=-1)
    pos = torch.arange(seq_len, device=tokens.device)
    causal = pos.unsqueeze(0) < pos.unsqueeze(1)
    recency = pos.float() / max(float(seq_len - 1), 1.0)

    semantic_scores = None
    semantic_idx = None
    semantic_route_scores = None
    if model.candidate_mode in {"semantic_hash", "hybrid_semantic_hash"}:
        semantic_logits = model.semantic_router(states)
        temperature = max(float(model.semantic_temperature), 1.0e-4)
        semantic_probs = torch.softmax(semantic_logits / temperature, dim=-1)
        semantic_scores = torch.einsum("btc,bsc->bts", semantic_probs, semantic_probs)
        route_scores = semantic_scores + 1.0e-4 * model.recency_scale * recency.view(1, 1, -1)
        route_scores = route_scores.masked_fill(~causal.view(1, seq_len, seq_len), -1.0e9)
        k = seq_len if model.top_k <= 0 else min(model.top_k, seq_len)
        _, semantic_idx = route_scores.topk(k, dim=2)
        semantic_route_scores = semantic_scores.gather(dim=2, index=semantic_idx)

    if model.candidate_mode in {"hash", "triton_hash", "oracle", "full", "hybrid_semantic_hash"}:
        context_hashes = causal_ngram_hash(tokens, model.context_order)
        if model.candidate_mode == "oracle":
            candidate_mask = context_hashes[:, :, None] == context_hashes[:, None, :]
        elif model.candidate_mode == "full":
            candidate_mask = torch.ones(batch, seq_len, seq_len, dtype=torch.bool, device=tokens.device)
        else:
            buckets = torch.remainder(context_hashes.long() * 1_000_003 + 97_531, model.num_buckets)
            candidate_mask = buckets[:, :, None] == buckets[:, None, :]
        candidate_mask = candidate_mask & causal.view(1, seq_len, seq_len)
        scores = torch.einsum("btd,bsd->bts", query, record_keys) * model.scale
        scores = scores + model.recency_scale * recency.view(1, 1, -1)
        scores = scores.masked_fill(~candidate_mask, -1.0e9)
        k = seq_len if model.top_k <= 0 else min(model.top_k, seq_len)
        top_scores, token_idx = scores.topk(k, dim=2)
        token_valid = top_scores > -1.0e8
    else:
        token_idx = torch.zeros(batch, seq_len, model.top_k, dtype=torch.long, device=tokens.device)
        token_valid = torch.zeros_like(token_idx, dtype=torch.bool)

    if model.candidate_mode == "semantic_hash":
        candidate_idx = semantic_idx
        valid = torch.ones_like(candidate_idx, dtype=torch.bool)
        candidate_route_scores = semantic_route_scores
    elif model.candidate_mode == "hybrid_semantic_hash":
        token_route_scores = semantic_scores.gather(dim=2, index=token_idx)
        candidate_idx = torch.cat([token_idx, semantic_idx], dim=2)
        valid = torch.cat([token_valid, torch.ones_like(semantic_idx, dtype=torch.bool)], dim=2)
        candidate_route_scores = torch.cat([token_route_scores, semantic_route_scores], dim=2)
    else:
        candidate_idx = token_idx
        valid = token_valid
        candidate_route_scores = None

    safe_idx = candidate_idx.clamp_min(0)
    gather_key_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, record_keys.size(-1))
    cand_keys = record_keys.unsqueeze(1).expand(-1, seq_len, -1, -1).gather(
        dim=2, index=gather_key_idx
    )
    cand_tokens = value_tokens.unsqueeze(1).expand(-1, seq_len, -1).gather(
        dim=2, index=safe_idx
    )

    cache_scores = torch.einsum("btkd,btd->btk", cand_keys, query) * model.scale
    cache_scores = cache_scores + model.recency_scale * (
        safe_idx.float() / max(float(seq_len - 1), 1.0)
    )
    if candidate_route_scores is not None:
        cache_scores = cache_scores + model.semantic_score_scale * torch.log(
            candidate_route_scores.clamp_min(1.0e-6)
        )
    cache_scores = cache_scores.masked_fill(~valid, -1.0e9)
    weights = torch.softmax(cache_scores, dim=2) * valid.float()
    weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
    target_cache_probs = (
        weights * (cand_tokens == targets.unsqueeze(-1)).float()
    ).sum(dim=2)

    if model.use_gate:
        gate = torch.sigmoid(model.gate(states)).squeeze(-1)
    else:
        gate = torch.full_like(target_cache_probs, model.fixed_cache_weight)
    has_cache = valid.any(dim=2)
    gate = (gate * has_cache.float()).clamp(1.0e-5, 1.0 - 1.0e-5)
    target_mix_probs = (
        (1.0 - gate) * target_param_log_probs.exp() + gate * target_cache_probs
    )
    return torch.log(target_mix_probs.clamp_min(1.0e-8)), acc


def full_ar_loss_and_acc(model: nn.Module, tokens: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, ContextAssociativeLM):
        target_log_probs, acc = pcaf_context_full_ar_target_log_probs(model, tokens, targets)
        return -target_log_probs.mean(), acc
    if isinstance(model, TransformerClassifier):
        logits = transformer_full_ar_logits(model, tokens)
        loss = -target_log_probs_from_logits(logits, targets).mean()
        acc = (logits.argmax(dim=-1) == targets).float().mean()
        return loss, acc
    logits = model(tokens)
    loss = F.cross_entropy(logits, targets[:, -1])
    acc = (logits.argmax(dim=-1) == targets[:, -1]).float().mean()
    return loss, acc


def append_jsonl(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(model: nn.Module, batcher: RandomTokenBatcher, *, batches: int, loss_mode: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    for _ in range(batches):
        tokens, targets = batcher.sample()
        if loss_mode == "full_ar":
            loss, acc = full_ar_loss_and_acc(model, tokens, targets)
        else:
            logits = model(tokens)
            loss = F.cross_entropy(logits, targets)
            acc = (logits.argmax(dim=-1) == targets).float().mean()
        weight = targets.numel() if loss_mode == "full_ar" else targets.size(0)
        total_loss += float(loss.item()) * weight
        total_acc += float(acc.item()) * weight
        total += weight
    return total_loss / total, total_acc / total


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
        full_ar=args.loss_mode == "full_ar",
    )
    eval_batcher = RandomTokenBatcher(
        eval_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        seed=args.eval_sample_seed,
        full_ar=args.loss_mode == "full_ar",
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
        f"model={args.model} loss_mode={args.loss_mode} device={device}"
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
        if args.loss_mode == "full_ar":
            loss, train_acc_tensor = full_ar_loss_and_acc(model, tokens, targets)
            train_acc_value = float(train_acc_tensor.detach().item())
        else:
            logits = model(tokens)
            loss = F.cross_entropy(logits, targets)
            train_acc_value = float((logits.argmax(dim=-1) == targets).float().mean().item())

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
            eval_loss, eval_acc = evaluate(
                model, eval_batcher, batches=args.eval_batches, loss_mode=args.loss_mode
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
                f"step={step:05d} loss={loss.item():.4f} train_acc={train_acc_value:.4f} "
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
                    "loss_mode": args.loss_mode,
                    "params": n_params,
                    "train_loss": float(loss.item()),
                    "train_acc": train_acc_value,
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

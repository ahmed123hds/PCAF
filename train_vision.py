from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
HASH_MOD = 2_147_483_647


class PatchMixerBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        y = self.ff(y)
        return x + self.dropout(y)


class PCAFVisionClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        d_model: int,
        d_hidden: int,
        local_layers: int,
        local_kernel_size: int,
        top_k: int,
        num_buckets: int,
        routing_mode: str,
        semantic_buckets: int,
        semantic_temperature: float,
        dropout: float,
        use_cache: bool = True,
        use_gate: bool = True,
        fixed_cache_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must divide patch_size")
        if routing_mode not in {"token_hash", "semantic_hash", "hybrid_semantic_hash"}:
            raise ValueError("unsupported routing mode")

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.top_k = top_k
        self.num_buckets = num_buckets
        self.routing_mode = routing_mode
        self.semantic_temperature = semantic_temperature
        self.use_cache = use_cache
        self.use_gate = use_gate
        self.fixed_cache_weight = fixed_cache_weight
        self.scale = d_model**-0.5

        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.blocks = nn.ModuleList(
            [
                PatchMixerBlock(d_model, d_hidden, local_kernel_size, dropout)
                for _ in range(local_layers)
            ]
        )
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.semantic_router = nn.Linear(d_model, semantic_buckets)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(d_model // 2, 1)),
            nn.GELU(),
            nn.Linear(max(d_model // 2, 1), 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )
        self.register_buffer(
            "hash_coeff",
            torch.arange(1, self.patch_dim + 1, dtype=torch.long),
            persistent=False,
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        return F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

    def patch_hash(self, patches: torch.Tensor) -> torch.Tensor:
        # CIFAR tensors are normalized; map back into a stable small integer range.
        q = torch.clamp(((patches + 2.5) * 16.0).round(), 0, 127).long()
        coeff = self.hash_coeff.to(q.device)
        return torch.remainder((q * coeff).sum(dim=-1) + 97_531, HASH_MOD)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw_patches = self.patchify(images)
        x = self.patch_proj(raw_patches) + self.pos
        for block in self.blocks:
            x = block(x)

        if not self.use_cache:
            return self.head(x.mean(dim=1))

        query = F.normalize(self.query_proj(x), dim=-1)
        keys = F.normalize(self.key_proj(x), dim=-1)
        values = self.value_proj(x)
        batch, seq_len, _ = x.shape
        not_self = ~torch.eye(seq_len, dtype=torch.bool, device=x.device).view(1, seq_len, seq_len)

        route_scores = None
        if self.routing_mode in {"semantic_hash", "hybrid_semantic_hash"}:
            logits = self.semantic_router(x)
            temperature = max(float(self.semantic_temperature), 1.0e-4)
            probs = torch.softmax(logits / temperature, dim=-1)
            route_scores = torch.einsum("btc,bsc->bts", probs, probs)

        if self.routing_mode == "semantic_hash":
            scores = route_scores.masked_fill(~not_self, -1.0e9)
        else:
            buckets = torch.remainder(self.patch_hash(raw_patches), self.num_buckets)
            mask = (buckets[:, :, None] == buckets[:, None, :]) & not_self
            scores = torch.einsum("btd,bsd->bts", query, keys) * self.scale
            scores = scores.masked_fill(~mask, -1.0e9)
            if self.routing_mode == "hybrid_semantic_hash":
                scores = torch.maximum(scores, route_scores.masked_fill(~not_self, -1.0e9))

        k = seq_len if self.top_k <= 0 else min(self.top_k, seq_len)
        top_scores, top_idx = scores.topk(k, dim=2)
        valid = top_scores > -1.0e8
        safe_idx = top_idx.clamp_min(0)
        gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, values.size(-1))
        cand_values = values.unsqueeze(1).expand(-1, seq_len, -1, -1).gather(2, gather_idx)
        weights = torch.softmax(top_scores, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        context = torch.einsum("btk,btkd->btd", weights, cand_values)

        if self.use_gate:
            gate = torch.sigmoid(self.gate(x))
        else:
            gate = torch.full_like(context[..., :1], self.fixed_cache_weight)
        mixed = (1.0 - gate) * x + gate * context
        return self.head(mixed.mean(dim=1))


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        d_model: int,
        d_hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must divide patch_size")
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch_proj(patches)
        cls = self.cls.expand(images.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.encoder(x)
        return self.head(x[:, 0])


def build_loaders(args: argparse.Namespace):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )

    root = args.data_dir
    dataset_name = args.dataset.lower()
    if dataset_name == "cifar10":
        ds_cls = datasets.CIFAR10
        total_classes = 10
    elif dataset_name in {"cifar50", "cifar100"}:
        ds_cls = datasets.CIFAR100
        total_classes = 100
    else:
        raise ValueError("dataset must be cifar10, cifar50, or cifar100")

    train_set = ds_cls(root=root, train=True, download=True, transform=train_tf)
    eval_set = ds_cls(root=root, train=False, download=True, transform=eval_tf)
    num_classes = args.class_subset or (50 if dataset_name == "cifar50" else total_classes)
    if num_classes < total_classes:
        train_idx = [i for i, y in enumerate(train_set.targets) if y < num_classes]
        eval_idx = [i for i, y in enumerate(eval_set.targets) if y < num_classes]
        train_set = Subset(train_set, train_idx)
        eval_set = Subset(eval_set, eval_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )
    return train_loader, eval_loader, num_classes


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model in {"pcaf_context", "pcaf_semantic", "pcaf_hybrid", "local_conv", "pcaf_no_gate"}:
        routing_mode = "token_hash"
        use_cache = True
        use_gate = True
        if args.model == "pcaf_semantic":
            routing_mode = "semantic_hash"
        elif args.model == "pcaf_hybrid":
            routing_mode = "hybrid_semantic_hash"
        elif args.model == "local_conv":
            use_cache = False
        elif args.model == "pcaf_no_gate":
            use_gate = False
        return PCAFVisionClassifier(
            image_size=32,
            patch_size=args.patch_size,
            num_classes=num_classes,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            local_layers=args.local_layers,
            local_kernel_size=args.local_kernel_size,
            top_k=args.top_k,
            num_buckets=args.num_buckets,
            routing_mode=routing_mode,
            semantic_buckets=args.semantic_buckets,
            semantic_temperature=args.semantic_temperature,
            dropout=args.dropout,
            use_cache=use_cache,
            use_gate=use_gate,
            fixed_cache_weight=args.fixed_cache_weight,
        )
    if args.model == "transformer":
        return VisionTransformerClassifier(
            image_size=32,
            patch_size=args.patch_size,
            num_classes=num_classes,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model: {args.model}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += labels.numel()
    return total_loss / total, total_correct / total


def append_jsonl(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision PCAF classifier for CIFAR")
    parser.add_argument("--model", choices=["local_conv", "pcaf_no_gate", "pcaf_context", "pcaf_semantic", "pcaf_hybrid", "transformer"], default="pcaf_context")
    parser.add_argument("--dataset", choices=["cifar10", "cifar50", "cifar100"], default="cifar50")
    parser.add_argument("--class-subset", type=int, default=0, help="Use labels [0, N). For CIFAR-50, use --dataset cifar50 or --class-subset 50.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--local-layers", type=int, default=6)
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-buckets", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--semantic-buckets", type=int, default=256)
    parser.add_argument("--semantic-temperature", type=float, default=0.2)
    parser.add_argument("--fixed-cache-weight", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_loader, eval_loader, num_classes = build_loaders(args)
    model = build_model(args, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"dataset={args.dataset} classes={num_classes} model={args.model} "
        f"device={device} params={n_params:,}"
    )

    global_step = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.perf_counter()
        for images, labels in train_loader:
            global_step += 1
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.numel()
            running_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            running_total += labels.numel()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        img_per_sec = running_total / max(elapsed, 1.0e-9)

        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            eval_start = time.perf_counter()
            eval_loss, eval_acc = evaluate(model, eval_loader, device, args.eval_batches)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            eval_sec = time.perf_counter() - eval_start
            peak_mem = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            total_elapsed = time.perf_counter() - start
            print(
                f"epoch={epoch:03d} step={global_step:05d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                f"img_per_sec={img_per_sec:.1f} epoch_sec={elapsed:.2f} "
                f"eval_sec={eval_sec:.2f} elapsed_min={total_elapsed / 60.0:.2f} "
                f"peak_mem_mb={peak_mem:.1f}"
            )
            append_jsonl(
                args.log_jsonl,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "dataset": args.dataset,
                    "classes": num_classes,
                    "model": args.model,
                    "params": n_params,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "img_per_sec": img_per_sec,
                    "epoch_sec": elapsed,
                    "eval_sec": eval_sec,
                    "elapsed_min": total_elapsed / 60.0,
                    "peak_mem_mb": peak_mem,
                },
            )


if __name__ == "__main__":
    main()

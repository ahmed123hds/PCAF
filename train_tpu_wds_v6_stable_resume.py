"""
CS-Mamba V6 — Characteristic Mamba TPU Training Script
======================================================
Selective 2D characteristic transport with learned divergence-free flow.
No torch.fft, no torch.complex, TPU/XLA-friendly.

Recommended usage:
  python train_tpu_wds_v6.py --dataset tiny-imagenet

If you want global XLA BF16 remapping, set it outside the script before launch:
  export XLA_USE_BF16=1
"""

import copy
import os
import sys
import threading
import time
import math
import argparse
import random
import gc
import signal
from itertools import islice
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

_orig_excepthook = threading.excepthook


def _silent_dl_excepthook(args):
    if issubclass(args.exc_type, KeyError):
        return
    _orig_excepthook(args)


threading.excepthook = _silent_dl_excepthook
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds

from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v1_2 import ContinuousSpatialMambaClassifier_V12 as CSMamba_V12
from models.characteristic_mamba_v6 import CSMamba_V6
from models.vmamba_4d import VMamba4D


@dataclass
class EmptyConfig:
    pass


_TERMINATING = False


def _make_process_group():
    """Keep spawned TPU ranks in a process group the launcher can clean up."""
    try:
        if os.getpid() != os.getpgrp():
            os.setpgrp()
    except OSError:
        pass


def _terminate_process_group(signum, _frame):
    """Forward Ctrl-C/SIGTERM to all PJRT child ranks before exiting."""
    global _TERMINATING
    if _TERMINATING:
        os._exit(128 + signum)
    _TERMINATING = True
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        os.killpg(os.getpgrp(), signum)
    except ProcessLookupError:
        pass
    os._exit(128 + signum)


def _install_parent_signal_handlers():
    signal.signal(signal.SIGINT, _terminate_process_group)
    signal.signal(signal.SIGTERM, _terminate_process_group)


def _set_parent_death_signal():
    """Ask Linux to SIGTERM this rank if the xmp launcher exits unexpectedly."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        pr_set_pdeathsig = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
    except Exception:
        pass


def _forkserver_warmup():
    pass


def _ensure_forkserver_started():
    """Start the DataLoader forkserver before this rank initializes XLA/PJRT."""
    import multiprocessing as mp

    ctx = mp.get_context("forkserver")
    proc = ctx.Process(target=_forkserver_warmup)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"forkserver warmup failed with exit code {proc.exitcode}")


class _XLANodeSplitter:
    """Module-level picklable node splitter for WebDataset.

    A local closure cannot be pickled by the forkserver/spawn multiprocessing
    context used by xmp.spawn. This class is defined at module level so it is
    fully picklable and can be serialized to forkserver worker processes.
    """
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def __call__(self, urls):
        yield from islice(urls, self.rank, None, self.world_size)


def parse_args():
    p = argparse.ArgumentParser("CS-Mamba V6 — Characteristic Mamba TPU Training")

    # Dataset
    p.add_argument("--dataset", choices=["imagenet1k", "tiny-imagenet"], default="tiny-imagenet")
    p.add_argument("--train_shards", type=str, default="")
    p.add_argument("--val_shards", type=str, default="")

    # Model
    p.add_argument("--model_version", choices=["v1", "v1_2", "v6", "vmamba"], default="v6")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--d_embed", type=int, default=512)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--n_mamba_layers", type=int, default=8)
    p.add_argument("--K_steps", type=int, default=4)
    p.add_argument("--n_classes", type=int, default=1000)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--spatial_op", choices=["laplacian", "conv2d", "conv1d"], default="laplacian",
                   help="V1.2 spatial operator ablation")
    p.add_argument("--n_flow_groups", type=int, default=8, help="Number of flow groups for transport")

    # Optimizer
    p.add_argument("--batch_size", type=int, default=128, help="Per TPU core")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--base_lr", type=float, default=1e-3, help="Reference LR for GlobalBS=1024")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--amp_bf16", action="store_true", help="Use torch.autocast('xla', dtype=torch.bfloat16)")

    # Regularization
    p.add_argument("--mixup_alpha", type=float, default=0.8)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--label_smooth", type=float, default=0.1)
    # ── VMamba paper additions ──────────────────────────────────────────
    p.add_argument("--randaug", action="store_true",
                   help="Enable RandAugment (rand-m9) in training pipeline [CPU-side]")
    p.add_argument("--reprob", type=float, default=0.25,
                   help="RandomErasing probability (VMamba=0.25, 0=disabled) [CPU-side]")
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA shadow model decay (VMamba=0.9999, 0=disabled) [TPU static graph]")

    # Checkpoint
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume_model_only", action="store_true", help="Resume only the model weights, reset optimizer/scheduler and start from epoch 1")
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--save_every", type=int, default=10)

    # Infra
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2, help="Batches prefetched per WebDataset worker when num_workers > 0")
    p.add_argument("--loader_prefetch_size", type=int, default=4, help="Host-side PyTorch/XLA MpDeviceLoader queue size")
    p.add_argument("--device_prefetch_size", type=int, default=1, help="Device-side PyTorch/XLA MpDeviceLoader queue size")
    p.add_argument("--host_to_device_transfer_threads", type=int, default=1, help="Host-to-device transfer threads used by MpDeviceLoader")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xla_metrics_every", type=int, default=0, help="Print selected cumulative XLA metrics every N epochs; 0 disables")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Mixup / CutMix — ALL randomness (beta, randperm, randint) lives on CPU.
# These run inside DataLoader workers before MpDeviceLoader ships tensors
# to the TPU, so the XLA graph sees static-shape tensors only.
# ─────────────────────────────────────────────────────────────────────
def mixup_data(images, labels, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    # CPU randperm — index never touches XLA; avoids dynamic graph recompile
    index = torch.randperm(images.size(0))      # CPU tensor
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], float(lam)


def cutmix_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bsz, _, h, w = images.shape
    index = torch.randperm(bsz)                 # CPU tensor
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    # np.random.randint — CPU scalar, not an XLA op
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / max(h * w, 1)
    return mixed, labels, labels[index], float(lam)


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def mixed_top1(logits, labels_a, labels_b, lam):
    pred = logits.argmax(dim=1)
    correct_a = pred.eq(labels_a).float()
    correct_b = pred.eq(labels_b).float()
    return (lam * correct_a + (1.0 - lam) * correct_b).sum().item()


def build_lr_scheduler(optimizer, flags, scaled_lr):
    warmup = flags.warmup_epochs
    total = flags.epochs
    min_lr = flags.min_lr

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / scaled_lr, cosine_factor)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



# ─────────────────────────────────────────────────────────────────────
# EMA — shadow model lives on the TPU device but its update is a
# fixed-shape linear blend: s = decay*s + (1-decay)*m.
# XLA sees the same static arithmetic graph every epoch → zero recompile.
# ─────────────────────────────────────────────────────────────────────
class ModelEMA:
    """EMA shadow model for TPU. Update is a static graph; no recompile."""
    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        # deepcopy on CPU first, then move to same device as model
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Call after xm.optimizer_step(); flush with xm.mark_step() outside."""
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            # Static linear blend — same graph shape every call.
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


class _ApplyTransforms:
    """Picklable per-sample transform mapper for WebDataset."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform(sample[0]), sample[1]


class _ApplyMix:
    """Picklable batch-level Mixup/CutMix mapper for WebDataset."""
    def __init__(self, mixup_prob, mixup_alpha, cutmix_alpha):
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, sample):
        images, labels = sample
        labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        if np.random.random() < self.mixup_prob:
            mixed, la, lb, lam = mixup_data(images, labels, self.mixup_alpha)
        else:
            mixed, la, lb, lam = cutmix_data(images, labels, self.cutmix_alpha)
        return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)


class _ApplyStackVal:
    """Picklable batch-level validation label cast mapper for WebDataset."""
    def __call__(self, sample):
        images, labels = sample
        labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        return images, labels


def build_wds_loader(shards_url, batch_size, flags, is_training=True):
    import torch_xla.core.xla_model as xm
    try:
        global_rank = xm.get_ordinal()
        global_world_size = xm.xrt_world_size()
    except Exception:
        global_rank = 0
        global_world_size = 1

    # Use module-level picklable class — local closures cannot be pickled by
    # the forkserver multiprocessing context used with start_method='spawn'.
    node_splitter = _XLANodeSplitter(global_rank, global_world_size)

    if is_training:
        train_aug = [
            T.RandomResizedCrop(flags.img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
        ]
        # VMamba recipe: RandAugment (opt-in via --randaug) — CPU-side, safe.
        if getattr(flags, 'randaug', False):
            train_aug.append(T.RandAugment(num_ops=2, magnitude=9))
        train_aug += [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        # VMamba recipe: RandomErasing p=0.25 — CPU-side in worker, safe.
        if getattr(flags, 'reprob', 0.0) > 0.0:
            train_aug.append(T.RandomErasing(p=flags.reprob, scale=(0.02, 0.33)))
        transform = T.Compose(train_aug)
    else:
        transform = T.Compose([
            T.Resize(int(flags.img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(flags.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # All mapper callables are module-level picklable classes.
    apply_transforms_fn = _ApplyTransforms(transform)
    apply_mix_fn = _ApplyMix(flags.mixup_prob, flags.mixup_alpha, flags.cutmix_alpha)
    apply_stack_val_fn = _ApplyStackVal()

    if is_training:
        dataset = (
            wds.WebDataset(
                shards_url,
                resampled=True,
                shardshuffle=1000,
                nodesplitter=node_splitter,
                workersplitter=wds.split_by_worker,
                empty_check=False,
            )
            .shuffle(5000)
            .decode("pil")
            .to_tuple("jpg;png;jpeg", "cls")
            .map(apply_transforms_fn)
            .batched(batch_size, partial=False)
            .map(apply_mix_fn)
            .with_epoch(math.ceil(1_281_167 / (batch_size * max(global_world_size, 1))))
        )
    else:
        dataset = (
            wds.WebDataset(
                shards_url,
                resampled=False,
                shardshuffle=False,
                nodesplitter=node_splitter,
                workersplitter=wds.split_by_worker,
                empty_check=False,
            )
            .decode("pil")
            .to_tuple("jpg;png;jpeg", "cls")
            .map(apply_transforms_fn)
            .batched(batch_size, partial=False)
            .map(apply_stack_val_fn)
        )

    mp_ctx = "forkserver" if flags.num_workers > 0 else None
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=flags.num_workers,
        pin_memory=False,
        prefetch_factor=flags.prefetch_factor if flags.num_workers > 0 else None,
        persistent_workers=False,
        multiprocessing_context=mp_ctx,
    )




class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image, label = item["image"], item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image) if self.transform else image, label


class _TinyMixupCollate:
    """Picklable Tiny-ImageNet training collate for spawn/forkserver workers."""
    def __init__(self, mixup_prob, mixup_alpha, cutmix_alpha):
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if np.random.random() < self.mixup_prob:
            mixed, la, lb, lam = mixup_data(images, labels, self.mixup_alpha)
        else:
            mixed, la, lb, lam = cutmix_data(images, labels, self.cutmix_alpha)
        return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)


def build_tiny_loader(flags, is_training=True):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, DistributedSampler
    import torch_xla.core.xla_model as xm

    ds = load_dataset("Maysee/tiny-imagenet")

    if is_training:
        train_aug = [
            T.RandomResizedCrop(flags.img_size, scale=(0.3, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
        ]
        # VMamba recipe: RandAugment (opt-in via --randaug) — CPU-side, safe.
        if getattr(flags, 'randaug', False):
            train_aug.append(T.RandAugment(num_ops=2, magnitude=9))
        train_aug.append(T.ColorJitter(0.4, 0.4, 0.4))
        train_aug += [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        # VMamba recipe: RandomErasing p=0.25 — CPU-side in worker, safe.
        if getattr(flags, 'reprob', 0.0) > 0.0:
            train_aug.append(T.RandomErasing(p=flags.reprob, scale=(0.02, 0.33)))
        transform = T.Compose(train_aug)
    else:
        transform = T.Compose([
            T.Resize(int(flags.img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(flags.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    dataset = TinyImageNetDataset(ds["train"] if is_training else ds["valid"], transform)
    mixup_collate = _TinyMixupCollate(flags.mixup_prob, flags.mixup_alpha, flags.cutmix_alpha)
    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=is_training,
    )
    return DataLoader(
        dataset,
        batch_size=flags.batch_size,
        sampler=sampler,
        num_workers=flags.num_workers,
        drop_last=is_training,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=mixup_collate if is_training else None,
    )


def _maybe_autocast(flags):
    if flags.amp_bf16:
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)
    return torch.autocast(device_type="xla", enabled=False)


def _print_xla_metrics(tag):
    """Print a compact subset of cumulative XLA metrics for compile diagnosis."""
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        report = met.metrics_report()
    except Exception as exc:
        try:
            xm.master_print(f"[xla-metrics:{tag}] unavailable: {exc}")
        except Exception:
            print(f"[xla-metrics:{tag}] unavailable: {exc}")
        return

    wanted = {
        "CompileTime",
        "ExecuteTime",
        "MarkStep",
        "CachedCompile",
        "CreateCompileHandles",
        "CompileCacheMiss",
        "UncachedCompile",
    }
    lines = report.splitlines()
    selected = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(("Metric: ", "Counter: ")):
            name = line.split(":", 1)[1].strip()
            if name in wanted:
                selected.append(line.strip())
                j = i + 1
                while j < len(lines) and not lines[j].startswith(("Metric: ", "Counter: ")):
                    text = lines[j].strip()
                    if text.startswith(("TotalSamples:", "Value:", "Rate:", "Accumulator:", "Percentiles:")):
                        selected.append("  " + text)
                    j += 1
                i = j
                continue
        i += 1

    if selected:
        xm.master_print(f"[xla-metrics:{tag}]\n" + "\n".join(selected))
    else:
        xm.master_print(f"[xla-metrics:{tag}] no selected metrics found")


def _mp_fn(index, flags):
    _set_parent_death_signal()
    _ensure_forkserver_started()

    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    device = xm.xla_device()
    torch.manual_seed(flags.seed + index)
    np.random.seed(flags.seed + index)
    random.seed(flags.seed + index)

    cfg = EmptyConfig()
    cfg.img_size = flags.img_size
    cfg.patch_size = flags.patch_size
    cfg.d_embed = flags.d_embed
    cfg.d_state = flags.d_state
    cfg.n_mamba_layers = flags.n_mamba_layers
    cfg.K_steps = flags.K_steps
    cfg.n_classes = flags.n_classes
    cfg.canvas_size = flags.img_size
    cfg.drop_path = flags.drop_path
    cfg.spatial_op = flags.spatial_op
    cfg.n_flow_groups = flags.n_flow_groups

    if flags.model_version == "v1":
        model = CSMamba_V1(cfg).to(device)
        model_name = "CS-Mamba V1 (Continuous Spatial)"
    elif flags.model_version == "v1_2":
        model = CSMamba_V12(cfg).to(device)
        model_name = f"CS-Mamba V1.2 ({flags.spatial_op} 3D Continuous Spatial)"
    elif flags.model_version == "vmamba":
        model = VMamba4D(cfg).to(device)
        model_name = "VMamba4D (TPU-safe Cross-Scan)"
    else:
        model = CSMamba_V6(cfg).to(device)
        model_name = "CS-Mamba V6 (Characteristic Mamba)"

    world_size = xm.xrt_world_size()
    global_bs = flags.batch_size * world_size
    scaled_lr = flags.base_lr * (global_bs / 1024.0)

    criterion = nn.CrossEntropyLoss(label_smoothing=flags.label_smooth)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=flags.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = build_lr_scheduler(optimizer, flags, scaled_lr)

    # ── VMamba recipe: EMA — shadow model on TPU device, static graph ──
    ema_decay = getattr(flags, 'ema_decay', 0.0)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        xm.master_print(f"  EMA enabled (decay={ema_decay}) — static XLA graph, no recompile")

    start_epoch = 1
    best_acc = 0.0
    if flags.resume:
        ckpt = torch.load(flags.resume, map_location="cpu")
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            ckpt_epoch = ckpt.get("epoch", 0)
            best_acc = ckpt.get("best_acc", 0.0)
            if flags.resume_model_only:
                start_epoch = 1
                xm.master_print(f"Loading MODEL ONLY from: {flags.resume}")
                xm.master_print(f"Optimizer/scheduler reset. Checkpoint came from epoch {ckpt_epoch}; new training starts at epoch 1 with best_acc carried as {best_acc:.4f}")
            else:
                start_epoch = ckpt_epoch + 1
                xm.master_print(f"Resuming FULL state from: {flags.resume}")
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                xm.master_print(f"Optimizer LR after load: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            xm.master_print(f"Loading raw model weights from: {flags.resume}")
            model.load_state_dict(ckpt)

    is_imagenet = flags.dataset == "imagenet1k"
    if is_imagenet:
        train_loader = build_wds_loader(flags.train_shards, flags.batch_size, flags, True)
        val_loader = build_wds_loader(flags.val_shards, flags.batch_size, flags, False)
        train_steps = math.ceil(1_281_167 / global_bs)
        val_steps = math.ceil(50_000 / global_bs)
    else:
        train_loader = build_tiny_loader(flags, True)
        val_loader = build_tiny_loader(flags, False)
        train_steps = math.ceil(100_000 / global_bs)
        val_steps = math.ceil(10_000 / global_bs)

    n_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"\n{'='*72}")
    xm.master_print(f"{model_name} | Params: {n_params/1e6:.1f}M")
    xm.master_print(f"World Size: {world_size} TPU cores | Global BS: {global_bs}")
    xm.master_print(f"Scaled LR: {scaled_lr:.6f} | AMP BF16: {flags.amp_bf16} | Flow Groups: {flags.n_flow_groups}")
    print(f"Actual optimizer LR now: {optimizer.param_groups[0]['lr']:.6f}")
    xm.master_print(f"{'='*72}\n")

    autocast_ctx = _maybe_autocast(flags)

    xla_loader_kwargs = {
        "loader_prefetch_size": flags.loader_prefetch_size,
        "device_prefetch_size": flags.device_prefetch_size,
        "host_to_device_transfer_threads": flags.host_to_device_transfer_threads,
    }
    para_train = pl.MpDeviceLoader(train_loader, device, **xla_loader_kwargs)
    para_val = pl.MpDeviceLoader(val_loader, device, **xla_loader_kwargs)

    if flags.eval_only:
        xm.master_print("=== RUNNING EVAL ONLY ===")
        model.eval()
        v_correct, v_total = 0, 0
        t1 = time.time()
        for step, batch in enumerate(para_val):
            if step >= val_steps:
                break
            images, labels = batch
            with torch.no_grad():
                with autocast_ctx:
                    logits = model(images)
            v_correct += logits.argmax(1).eq(labels).sum().item()
            v_total += images.size(0)
            xm.mark_step()
            
        val_time = time.time() - t1
        g_v_total = int(max(xm.mesh_reduce("v_n", v_total, np.sum), 1))
        v_acc = 100.0 * xm.mesh_reduce("v_c", v_correct, np.sum) / g_v_total
        xm.master_print(f"Eval results: Val Acc: {v_acc:.2f}% | Total Processed: {g_v_total} | Time: {val_time:.0f}s")
        return

    for epoch in range(start_epoch, flags.epochs + 1):
        if not is_imagenet:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        tracker = xm.RateTracker()
        total_loss, total_correct, total = 0.0, 0.0, 0
        t0 = time.time()

        for step, batch in enumerate(para_train):
            if step >= train_steps:
                break
            images, labels_a, labels_b, lam = batch
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(images)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss.backward()
            xm.reduce_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), flags.grad_clip)
            xm.optimizer_step(optimizer)

            # ── EMA update (static blend graph, flushed with mark_step) ──
            if ema is not None:
                ema.update(model)

            tracker.add(flags.batch_size)

            total_loss += loss.item() * images.size(0)
            total_correct += mixed_top1(logits, labels_a, labels_b, float(lam))
            total += images.size(0)

            if step > 0 and step % 100 == 0:
                xm.master_print(
                    f"E{epoch:03d} | Step {step}/{train_steps} | "
                    f"Loss {loss.item():.4f} | Rate {tracker.global_rate():.1f} img/s"
                )

        train_time = time.time() - t0
        g_total = max(xm.mesh_reduce("tr_n", total, np.sum), 1)
        g_loss = xm.mesh_reduce("tr_loss", total_loss, np.sum) / g_total
        g_acc = 100.0 * xm.mesh_reduce("tr_c", total_correct, np.sum) / g_total

        # ── Always evaluate on LIVE model for logging ───────────────────
        # EMA decay=0.999 needs many steps to converge. Logging live model
        # gives accurate training curves. EMA runs silently per-step.
        # NOTE: Do NOT call xm.mark_step() here before validation —
        # it causes XLA to donate the EMA shadow buffer, making it
        # unreadable when checkpoint saving tries to access it.
        model.eval()
        v_correct, v_total = 0, 0
        t1 = time.time()
        for step, batch in enumerate(para_val):
            if step >= val_steps:
                break
            images, labels = batch
            with torch.no_grad():
                with autocast_ctx:
                    logits = model(images)
            v_correct += logits.argmax(1).eq(labels).sum().item()
            v_total += images.size(0)
            xm.mark_step()

        val_time = time.time() - t1
        v_acc = 100.0 * xm.mesh_reduce("v_c", v_correct, np.sum) / max(xm.mesh_reduce("v_n", v_total, np.sum), 1)
        scheduler.step()

        xm.master_print(
            f"\nEpoch {epoch:03d}/{flags.epochs} | "
            f"Train {g_acc:.1f}% (loss {g_loss:.4f}) | Val {v_acc:.1f}% | "
            f"Train {train_time:.0f}s | Val {val_time:.0f}s | LR {scheduler.get_last_lr()[0]:.2e}\n"
        )
        if flags.xla_metrics_every > 0 and epoch % flags.xla_metrics_every == 0:
            _print_xla_metrics(f"epoch-{epoch:03d}")

        # ALL workers must evaluate this block to prevent XLA save-barriers from deadlocking!
        # Always save LIVE model weights — EMA shadow is XLA device memory that gets
        # donated/freed after execution; attempting ema.state_dict() causes
        # "ToLiteral() called on deleted/donated buffer" crash.
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "val_acc": v_acc,
        }
        latest_path = os.path.join(flags.save_dir, f"csmamba_v6_{flags.dataset}_latest.pt")
        xm.save(ckpt, latest_path, master_only=True, global_master=True)

        if v_acc > best_acc:
            best_acc = v_acc
            best_path = os.path.join(flags.save_dir, f"csmamba_v6_{flags.dataset}_best.pt")
            xm.save(ckpt, best_path, master_only=True, global_master=True)
            xm.master_print(f"New best! Val Acc: {best_acc:.2f}% saved to {best_path}")

    xm.master_print(f"Training complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    _make_process_group()
    _install_parent_signal_handlers()
    flags = parse_args()
    if flags.dataset == "tiny-imagenet":
        from datasets import load_dataset
        print("[Main] Pre-fetching Tiny-ImageNet cache...")
        load_dataset("Maysee/tiny-imagenet")
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method="spawn")

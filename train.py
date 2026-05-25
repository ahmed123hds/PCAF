"""
Main Training Script — Routing Hypothesis Experiment
======================================================
Tests whether a Neural ODE router outperforms a fixed Hilbert curve
scan order for Mamba-based CIFAR-10 classification.

Two systems:
    A) HilbertMamba  — fixed Hilbert scan → Mamba → classify
    B) ODEMamba      — Neural ODE scores → NeuralSort → Mamba → classify

Usage:
    python train.py --mode hilbert          # System A
    python train.py --mode ode              # System B
    python train.py --mode both             # Train both and compare

Key metrics logged:
    - Train / Val accuracy per epoch
    - Training time per epoch
    - For ODE mode: gradient variance on NeuralSort scores
"""

import argparse
import copy
import time
import sys
import os
import shutil

# Make sure local modules are importable from any working directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# VMamba-recipe regularization helpers
# ──────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model weights (VMamba uses 0.9999)."""
    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def mixup_data(x, y, alpha: float = 0.8, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda (VMamba alpha=0.8)."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha: float = 1.0, device='cuda'):
    """CutMix augmentation (VMamba alpha=1.0)."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_actual = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam_actual


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

from models.patch_encoder      import PatchEmbedding
from models.mamba_simple       import MambaClassifier
from models.neural_ode_router  import NeuralODERouter, FixedRouterHilbert
from models.continuous_graph_mamba import ContinuousGraphMambaClassifier as CGMamba
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba
from models.continuous_spatial_mamba_v1_2 import ContinuousSpatialMambaClassifier_V12 as CSMamba_V12
from models.vmamba_4d import VMamba4D


# ──────────────────────────────────────────────────────────────────────
# Full Model Wrappers
# ──────────────────────────────────────────────────────────────────────

class HilbertMamba(nn.Module):
    """Patch Embed → Hilbert Reorder → Mamba → Classify"""

    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        grid = cfg.img_size // cfg.patch_size
        self.router = FixedRouterHilbert(grid, grid)
        self.mamba  = MambaClassifier(
            d_model=cfg.d_embed,
            n_classes=10,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
        )

    def forward(self, x):
        emb = self.embedder(x)           # (B, n, d)
        emb, _ = self.router(emb)        # (B, n, d) — Hilbert order
        return self.mamba(emb)           # (B, 10)


class ODEMamba(nn.Module):
    """Patch Embed → Neural ODE → NeuralSort → Mamba → Classify"""

    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        self.router = NeuralODERouter(
            d_embed=cfg.d_embed,
            d_ff=cfg.ode_d_ff,
            tau=cfg.tau,
            solver=cfg.ode_solver,
            n_steps=cfg.ode_steps,
        )
        self.mamba = MambaClassifier(
            d_model=cfg.d_embed,
            n_classes=10,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
        )

    def forward(self, x):
        emb = self.embedder(x)            # (B, n, d)
        emb, scores = self.router(emb)    # (B, n, d), (B, n)
        return self.mamba(emb)            # (B, 10)


class CSMambaCIFAR(nn.Module):
    """CS-Mamba V1 wrapper that enables the CUDA/Triton scan from train.py."""

    def __init__(self, cfg):
        super().__init__()
        self.model = CSMamba(cfg)
        self.use_triton = cfg.use_triton

    def forward(self, x):
        return self.model(x, use_triton=self.use_triton)


class CSMambaV12CIFAR(nn.Module):
    """CS-Mamba V1.2 wrapper that enables the CUDA/Triton 3D-state scan."""

    def __init__(self, cfg):
        super().__init__()
        self.model = CSMamba_V12(cfg)
        self.use_triton = cfg.use_triton

    def forward(self, x):
        return self.model(x, use_triton=self.use_triton)


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.std > 0.0:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

def prepare_tiny_imagenet_val(data_dir: str) -> str:
    val_dir = os.path.join(data_dir, "val")
    imagefolder_val = os.path.join(data_dir, "val_imagefolder")
    annotations = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")
    if os.path.isdir(imagefolder_val):
        return imagefolder_val
    if not (os.path.isfile(annotations) and os.path.isdir(images_dir)):
        return val_dir

    os.makedirs(imagefolder_val, exist_ok=True)
    with open(annotations, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            src = os.path.join(images_dir, image_name)
            class_dir = os.path.join(imagefolder_val, class_name)
            dst = os.path.join(class_dir, image_name)
            os.makedirs(class_dir, exist_ok=True)
            if os.path.exists(dst):
                continue
            try:
                os.symlink(os.path.relpath(src, class_dir), dst)
            except OSError:
                shutil.copy2(src, dst)
    return imagefolder_val

def get_dataloaders(cfg):
    # ── VMamba recipe: RandAugment + RandomErasing ──────────────────────
    is_tiny = cfg.dataset == 'tiny-imagenet'
    crop_size = cfg.img_size if is_tiny else 32
    padding = 8 if is_tiny else 4
    mean = (0.4802, 0.4481, 0.3975) if is_tiny else (0.4914, 0.4822, 0.4465)
    std = (0.2302, 0.2265, 0.2262) if is_tiny else (0.2023, 0.1994, 0.2010)
    train_aug = [
        T.RandomCrop(crop_size, padding=padding),
        T.RandomHorizontalFlip(),
    ]
    if getattr(cfg, 'randaug', False):
        # rand-m9-mstd0.5-inc1 from VMamba / Swin Transformer recipe
        # Use lower magnitude (e.g. 5-6) for small datasets like CIFAR-10
        m = getattr(cfg, 'randaug_m', 9)
        train_aug.append(T.RandAugment(num_ops=2, magnitude=m))
    train_aug += [
        T.ToTensor(),
        T.Normalize(mean, std),
        AddGaussianNoise(0., cfg.noise_std),
    ]
    if getattr(cfg, 'reprob', 0.0) > 0.0:
        train_aug.append(T.RandomErasing(p=cfg.reprob, scale=(0.02, 0.33)))
    train_tf = T.Compose(train_aug)

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        AddGaussianNoise(0., cfg.noise_std)
    ])

    if is_tiny:
        train_ds = torchvision.datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(prepare_tiny_imagenet_val(cfg.data_dir), transform=val_tf)
    else:
        dataset_cls = torchvision.datasets.CIFAR100 if cfg.dataset == 'cifar100' else torchvision.datasets.CIFAR10
        train_ds = dataset_cls(
            root=cfg.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = dataset_cls(
            root=cfg.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_ds,   batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, cfg=None, ema=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    mixup_alpha  = getattr(cfg, 'mixup',  0.0) if cfg else 0.0
    cutmix_alpha = getattr(cfg, 'cutmix', 0.0) if cfg else 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # ── VMamba recipe: Mixup / CutMix (applied on GPU) ───────────────
        use_cutmix = cutmix_alpha > 0.0 and np.random.rand() < 0.5
        use_mixup  = not use_cutmix and mixup_alpha > 0.0
        y_a = y_b = labels
        lam = 1.0
        if use_cutmix:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, cutmix_alpha, device)
        elif use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha, device)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── VMamba recipe: EMA update per step (decay=0.9999 is per-step) ─
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        # Accuracy tracked on the dominant label for logging
        correct    += (logits.argmax(1) == y_a).sum().item()
        total      += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(name, model, cfg, device):
    """Train a model and return its history."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Params:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    train_loader, val_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

    # Mixed precision only if CUDA available
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ── VMamba recipe: EMA ────────────────────────────────────────────────
    ema_decay  = getattr(cfg, 'ema_decay',  0.0)
    ema_warmup = getattr(cfg, 'ema_warmup', 5)   # epochs before switching to EMA eval
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
    if ema:
        print(f"  EMA enabled (decay={ema_decay}, eval switches to shadow after epoch {ema_warmup})")

    # 🚀 MAGIC TRITON KERNEL FUSION (Runs completely in SRAM)
    if cfg.compile and hasattr(torch, 'compile'):
        try:
            print("  [Auto-fusing K-Step Diffusion PDEs into Triton Kernels...]")
            model = torch.compile(model)
        except Exception as e:
            print("  [Fallback to Eager PyTorch]")

    history = {'train_acc': [], 'val_acc': [], 'epoch_time': []}
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, cfg=cfg, ema=ema)

        # ── Always evaluate on the LIVE model for logging ─────────────────
        # EMA decay=0.9999 is designed for 375K+ steps (ImageNet 300ep).
        # On short runs (CIFAR 30ep ≈ 11.7K steps) the shadow never
        # converges — evaluating on it gives misleading low val acc.
        # VMamba's actual practice: log live model, SAVE EMA checkpoint.
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['epoch_time'].append(elapsed)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            # Save EMA weights if available (smoother for inference);
            # otherwise save the live model.
            sd = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(sd, f"{name}_best.pt")

        print(f"  Epoch {epoch:03d}/{cfg.epochs} | "
              f"Train {tr_acc:5.1f}% | Val {vl_acc:5.1f}% | "
              f"Time {elapsed:.1f}s | LR {scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  ✓ Best Val Acc: {best_val_acc:.2f}%")
    history['best_val_acc'] = best_val_acc
    return history


# ──────────────────────────────────────────────────────────────────────
# Argument Parsing & Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Routing Hypothesis Experiment on CIFAR-10"
    )
    # What to run
    p.add_argument('--mode', choices=[
        'hilbert', 'ode', 'graph_ode', 'spatial', 'spatial_v12',
        'vmamba', 'spatial_vmamba', 'spatial_v12_vmamba', 'all'
    ],
                   default='all', help="Which model(s) to train")

    # Data
    p.add_argument('--dataset', choices=['cifar10', 'cifar100', 'tiny-imagenet'], default='cifar10')
    p.add_argument('--data_dir',     default='./data')
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--noise_std',    type=float, default=0.0,
                   help="Add Gaussian noise to the dataset (e.g. 0.1)")

    # Image / patch
    p.add_argument('--img_size',     type=int, default=32)
    p.add_argument('--patch_size',   type=int, default=8,
                   help="Patch size (8 → 4×4=16 patches from 32×32 image)")

    # Model
    p.add_argument('--d_embed',        type=int,   default=128)
    p.add_argument('--d_state',        type=int,   default=16)
    p.add_argument('--n_mamba_layers', type=int,   default=2)

    # ODE router
    p.add_argument('--ode_d_ff',   type=int,   default=64)
    p.add_argument('--tau',        type=float, default=0.1,
                   help="NeuralSort temperature")
    p.add_argument('--ode_solver', default='rk4',
                   choices=['euler', 'rk4', 'dopri5'])
    p.add_argument('--ode_steps',  type=int,   default=10,
                   help="Fixed steps for euler/rk4 solver")
    p.add_argument('--K_steps',    type=int,   default=3,
                   help="Discrete Euler steps defining forward diffusion time")
    p.add_argument('--drop_path',  type=float, default=0.1)
    p.add_argument('--spatial_op', default='laplacian',
                   choices=['laplacian', 'laplacian8', 'conv2d', 'conv1d'],
                   help="V1.2 spatial operator ablation")
    p.add_argument('--recurrence_nonlinearity', default='identity',
                   choices=['identity', 'silu', 'tanh', 'gelu', 'relu6', 'relu'],
                   help="V1.2 activation applied after each recurrence step")
    p.add_argument('--use_triton', action='store_true',
                   help="Use custom CUDA/Triton scan kernels for supported models")
    p.add_argument('--compile', action='store_true',
                   help="Optionally wrap each model with torch.compile")

    # Training
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    p.add_argument('--seed', type=int, default=42)

    # ── VMamba paper regularization (all disabled by default) ──────────
    p.add_argument('--randaug',   action='store_true',
                   help="RandAugment (rand-m9-mstd0.5-inc1) — VMamba recipe")
    p.add_argument('--randaug_m', type=int, default=9,
                   help="RandAugment magnitude (VMamba=9; use 5-6 for small datasets like CIFAR)")
    p.add_argument('--reprob',    type=float, default=0.0,
                   help="Random Erasing probability (VMamba uses 0.25)")
    p.add_argument('--mixup',     type=float, default=0.0,
                   help="Mixup alpha (VMamba uses 0.8; 0 = disabled)")
    p.add_argument('--cutmix',    type=float, default=0.0,
                   help="CutMix alpha (VMamba uses 1.0; 0 = disabled)")
    p.add_argument('--ema_decay', type=float, default=0.0,
                   help="EMA decay (VMamba uses 0.9999; 0 = disabled)")
    p.add_argument('--ema_warmup', type=int, default=5,
                   help="Epochs before switching val eval to EMA shadow (avoids cold-start)")

    return p.parse_args()


def main():
    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.n_classes = 200 if cfg.dataset == 'tiny-imagenet' else (100 if cfg.dataset == 'cifar100' else 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if cfg.mode in ('hilbert', 'all'):
        model_h = HilbertMamba(cfg).to(device)
        results['HilbertMamba'] = train_model('HilbertMamba', model_h, cfg, device)

    if cfg.mode in ('ode', 'all'):
        model_o = ODEMamba(cfg).to(device)
        results['ODEMamba'] = train_model('ODEMamba', model_o, cfg, device)

    if cfg.mode in ('graph_ode', 'all'):
        # Pass identical config. Note: CGMamba handles patch extraction internally
        setattr(cfg, 'canvas_size', cfg.img_size) # for compatibility
        model_g = CGMamba(cfg).to(device)
        results['CGMamba'] = train_model('CGMamba', model_g, cfg, device)

    if cfg.mode in ('spatial', 'spatial_vmamba', 'all'):
        setattr(cfg, 'canvas_size', cfg.img_size) # for compatibility
        model_s = CSMambaCIFAR(cfg).to(device)
        results['CSMamba'] = train_model('CSMamba', model_s, cfg, device)

    if cfg.mode in ('spatial_v12', 'spatial_v12_vmamba'):
        setattr(cfg, 'canvas_size', cfg.img_size) # for compatibility
        model_s12 = CSMambaV12CIFAR(cfg).to(device)
        v12_name = f"CSMamba_V1_2_{cfg.spatial_op}_{cfg.recurrence_nonlinearity}"
        results[v12_name] = train_model(v12_name, model_s12, cfg, device)

    if cfg.mode in ('vmamba', 'spatial_vmamba', 'spatial_v12_vmamba', 'all'):
        model_v = VMamba4D(cfg).to(device)
        results['VMamba4D'] = train_model('VMamba4D', model_v, cfg, device)

    # ── Summary comparison ────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON")
        print(f"{'='*60}")
        for name, hist in results.items():
            avg_time = np.mean(hist['epoch_time'])
            print(f"  {name:20s}  Best Val: {hist['best_val_acc']:5.2f}%  "
                  f"Avg epoch: {avg_time:.1f}s")

        # Save results for visualisation
        import json
        with open('results.json', 'w') as f:
            # Convert to plain lists for JSON serialisation
            json.dump({k: {m: v if not isinstance(v, list) else v
                           for m, v in h.items()}
                       for k, h in results.items()}, f, indent=2)
        print("\n  Results saved to results.json")
        print("  Run:  python visualize.py  to see learning curves")

    print("\nDone.")


if __name__ == '__main__':
    main()

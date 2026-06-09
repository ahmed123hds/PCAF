from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random

from datasets import load_dataset

from train_pcaf_jax import (
    adamw_update,
    bucket_hash,
    init_adam_state,
    init_linear,
    layer_norm,
    local_block_forward,
    l2_normalize,
    metric_scalar,
    replicate,
    split_key,
)


CIFAR_MEAN = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32)


class ImageBatcher:
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        per_process_batch: int,
        local_device_count: int,
        seed: int,
        train: bool,
        augment: bool,
        crop_padding: int,
        hflip_prob: float,
        cutout_size: int,
        mixup_alpha: float,
        num_classes: int,
    ) -> None:
        if per_process_batch % local_device_count != 0:
            raise ValueError("per-process batch must divide local_device_count")
        self.images = images
        self.labels = labels
        self.per_process_batch = per_process_batch
        self.local_device_count = local_device_count
        self.per_device_batch = per_process_batch // local_device_count
        self.rng = np.random.default_rng(seed)
        self.train = train
        self.augment = augment
        self.crop_padding = crop_padding
        self.hflip_prob = hflip_prob
        self.cutout_size = cutout_size
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.images.shape[0], size=(self.per_process_batch,))
        images = self.images[idx].copy()
        labels = self.labels[idx]
        if self.train and self.augment:
            images = self._augment(images)
        if self.train and self.mixup_alpha > 0.0:
            labels = self._mixup(images, labels)
        shard = (self.local_device_count, self.per_device_batch)
        return images.reshape(*shard, *images.shape[1:]), labels.reshape(*shard, *labels.shape[1:])

    def _augment(self, images: np.ndarray) -> np.ndarray:
        if self.crop_padding > 0:
            pad = self.crop_padding
            padded = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
            crop_y = self.rng.integers(0, 2 * pad + 1, size=(images.shape[0],))
            crop_x = self.rng.integers(0, 2 * pad + 1, size=(images.shape[0],))
            cropped = np.empty_like(images)
            for i, (y, x) in enumerate(zip(crop_y, crop_x)):
                cropped[i] = padded[i, :, y : y + 32, x : x + 32]
            images = cropped
        if self.hflip_prob > 0.0:
            flip = self.rng.random(images.shape[0]) < self.hflip_prob
            images[flip] = images[flip, :, :, ::-1]
        if self.cutout_size > 0:
            cut = min(self.cutout_size, images.shape[-1])
            half = cut // 2
            cy = self.rng.integers(0, images.shape[-2], size=(images.shape[0],))
            cx = self.rng.integers(0, images.shape[-1], size=(images.shape[0],))
            for i, (y, x) in enumerate(zip(cy, cx)):
                y0 = max(y - half, 0)
                y1 = min(y0 + cut, images.shape[-2])
                x0 = max(x - half, 0)
                x1 = min(x0 + cut, images.shape[-1])
                images[i, :, y0:y1, x0:x1] = 0.0
        return images

    def _mixup(self, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
        perm = self.rng.permutation(images.shape[0])
        lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
        images[:] = lam * images + (1.0 - lam) * images[perm]
        target = np.eye(self.num_classes, dtype=np.float32)[labels]
        target_perm = np.eye(self.num_classes, dtype=np.float32)[labels[perm]]
        return lam * target + (1.0 - lam) * target_perm


def preprocess_split(
    split,
    *,
    image_field: str,
    label_field: str,
    class_subset: int,
    max_examples: int,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    fields = [f.strip() for f in label_field.split(",")]
    for row in split:
        label = None
        for field in fields:
            if field in row:
                label = int(row[field])
                break
        if label is None:
            raise KeyError(f"none of label fields {fields} found in dataset row")
        if class_subset > 0 and label >= class_subset:
            continue
        image = row[image_field].convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - CIFAR_MEAN[None, None, :]) / CIFAR_STD[None, None, :]
        images.append(arr.transpose(2, 0, 1))
        labels.append(label)
        if max_examples > 0 and len(labels) >= max_examples:
            break
    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int32)


def patchify(images, patch_size: int):
    batch, channels, height, width = images.shape
    gh = height // patch_size
    gw = width // patch_size
    x = images.reshape(batch, channels, gh, patch_size, gw, patch_size)
    x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))
    return x.reshape(batch, gh * gw, channels * patch_size * patch_size)


def patch_hash(patches, num_buckets: int):
    q = jnp.clip(jnp.rint((patches + 2.5) * 16.0), 0, 127).astype(jnp.int32)
    coeff = jnp.arange(1, patches.shape[-1] + 1, dtype=jnp.int32)
    h = jnp.sum(q * coeff[None, None, :], axis=-1)
    return bucket_hash(h, num_buckets)


def init_vision_params(
    key,
    *,
    patch_dim: int,
    num_patches: int,
    num_classes: int,
    d_model: int,
    d_hidden: int,
    local_layers: int,
    local_kernel_size: int,
    semantic_buckets: int,
):
    key, keys = split_key(key, 10 + 4 * local_layers)
    params: dict[str, Any] = {}
    params["patch_w"], params["patch_b"] = init_linear(keys.pop(), patch_dim, d_model)
    params["pos"] = random.normal(keys.pop(), (num_patches, d_model), dtype=jnp.float32) * 0.02
    blocks = []
    for _ in range(local_layers):
        k_conv = keys.pop()
        k_w1 = keys.pop()
        k_w2 = keys.pop()
        conv_kernel = random.normal(
            k_conv, (local_kernel_size, d_model), dtype=jnp.float32
        ) * (1.0 / max(local_kernel_size, 1) ** 0.5)
        w1, b1 = init_linear(k_w1, d_model, d_hidden)
        w2, b2 = init_linear(k_w2, d_hidden, d_model)
        blocks.append(
            {
                "ln_scale": jnp.ones((d_model,), dtype=jnp.float32),
                "ln_bias": jnp.zeros((d_model,), dtype=jnp.float32),
                "conv_kernel": conv_kernel,
                "conv_bias": jnp.zeros((d_model,), dtype=jnp.float32),
                "w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
            }
        )
    params["blocks"] = blocks
    params["wq"], _ = init_linear(keys.pop(), d_model, d_model)
    params["wk"], _ = init_linear(keys.pop(), d_model, d_model)
    params["wv"], _ = init_linear(keys.pop(), d_model, d_model)
    params["semantic_w"], params["semantic_b"] = init_linear(keys.pop(), d_model, semantic_buckets)
    gate_hidden = max(d_model // 2, 1)
    params["gate_ln_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
    params["gate_ln_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["gate_w1"], params["gate_b1"] = init_linear(keys.pop(), d_model, gate_hidden)
    params["gate_w2"], params["gate_b2"] = init_linear(keys.pop(), gate_hidden, 1)
    params["head_ln_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
    params["head_ln_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["head_w1"], params["head_b1"] = init_linear(keys.pop(), d_model, d_hidden)
    params["head_w2"], params["head_b2"] = init_linear(keys.pop(), d_hidden, num_classes)
    return params


def vision_forward(params, images, cfg):
    patches = patchify(images, cfg["patch_size"])
    x = patches @ params["patch_w"] + params["patch_b"] + params["pos"][None, :, :]
    for block in params["blocks"]:
        x = local_block_forward(block, x)
    if cfg["use_cache"]:
        query = l2_normalize(x @ params["wq"])
        keys = l2_normalize(x @ params["wk"])
        values = x @ params["wv"]
        batch, seq_len, _ = x.shape
        del batch
        not_self = ~jnp.eye(seq_len, dtype=bool)[None, :, :]

        route_scores = None
        if cfg["routing_mode"] in {"semantic_hash", "hybrid_semantic_hash"}:
            semantic_logits = x @ params["semantic_w"] + params["semantic_b"]
            probs = jax.nn.softmax(semantic_logits / cfg["semantic_temperature"], axis=-1)
            route_scores = jnp.einsum("btc,bsc->bts", probs, probs)

        if cfg["routing_mode"] == "semantic_hash":
            scores = jnp.where(not_self, route_scores, -1.0e9)
        else:
            buckets = patch_hash(patches, cfg["num_buckets"])
            mask = (buckets[:, :, None] == buckets[:, None, :]) & not_self
            scores = jnp.einsum("btd,bsd->bts", query, keys) * cfg["scale"]
            scores = jnp.where(mask, scores, -1.0e9)
            if cfg["routing_mode"] == "hybrid_semantic_hash":
                scores = jnp.maximum(scores, jnp.where(not_self, route_scores, -1.0e9))

        top_scores, top_idx = lax.top_k(scores, cfg["top_k"])
        valid = top_scores > -1.0e8
        batch_ids = jnp.arange(images.shape[0])[:, None, None]
        cand_values = values[batch_ids, top_idx, :]
        weights = jax.nn.softmax(top_scores, axis=-1) * valid.astype(jnp.float32)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-6)
        context = jnp.einsum("btk,btkd->btd", weights, cand_values)
        if cfg["use_gate"]:
            g = layer_norm(x, params["gate_ln_scale"], params["gate_ln_bias"])
            g = jax.nn.gelu(g @ params["gate_w1"] + params["gate_b1"])
            gate = jax.nn.sigmoid(g @ params["gate_w2"] + params["gate_b2"])
        else:
            gate = jnp.full_like(context[..., :1], cfg["fixed_cache_weight"])
        x = (1.0 - gate) * x + gate * context

    pooled = jnp.mean(x, axis=1)
    h = layer_norm(pooled, params["head_ln_scale"], params["head_ln_bias"])
    h = jax.nn.gelu(h @ params["head_w1"] + params["head_b1"])
    return h @ params["head_w2"] + params["head_b2"]


def loss_and_metrics(params, images, labels, cfg):
    logits = vision_forward(params, images, cfg)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if labels.ndim == 2:
        targets = labels.astype(jnp.float32)
        hard_labels = jnp.argmax(targets, axis=-1)
    else:
        hard_labels = labels
        targets = jax.nn.one_hot(labels, cfg["num_classes"], dtype=jnp.float32)
    if cfg["label_smoothing"] > 0.0:
        smooth = cfg["label_smoothing"]
        targets = targets * (1.0 - smooth) + smooth / cfg["num_classes"]
    loss = -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
    acc = jnp.mean((jnp.argmax(logits, axis=-1) == hard_labels).astype(jnp.float32))
    return loss, {"loss": loss, "acc": acc}


def make_train_step(cfg, weight_decay: float):
    def train_step(params, opt_state, images, labels, lr):
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            params, images, labels, cfg
        )
        grads = lax.pmean(grads, axis_name="data")
        metrics = lax.pmean(metrics, axis_name="data")
        params, opt_state = adamw_update(
            params, grads, opt_state, lr=lr, weight_decay=weight_decay
        )
        metrics = dict(metrics)
        metrics["lr"] = lax.pmean(lr, axis_name="data")
        return params, opt_state, metrics

    return jax.pmap(train_step, axis_name="data", donate_argnums=(0, 1))


def make_eval_step(cfg):
    def eval_step(params, images, labels):
        _, metrics = loss_and_metrics(params, images, labels, cfg)
        return lax.pmean(metrics, axis_name="data")

    return jax.pmap(eval_step, axis_name="data")


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path or jax.process_index() != 0:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def parse_args():
    parser = argparse.ArgumentParser(description="JAX TPU Vision PCAF on CIFAR")
    parser.add_argument("--jax-distributed", action="store_true")
    parser.add_argument("--model", choices=["local_conv", "pcaf_context", "pcaf_semantic", "pcaf_hybrid", "pcaf_no_gate"], default="pcaf_context")
    parser.add_argument("--dataset", default="uoft-cs/cifar100")
    parser.add_argument("--dataset-config", default="none")
    parser.add_argument("--image-field", default="img")
    parser.add_argument("--label-field", default="fine_label,label")
    parser.add_argument("--class-subset", type=int, default=50)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-sample-seed", type=int, default=10001)
    parser.add_argument("--eval-sample-seed", type=int, default=20001)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--cutout-size", type=int, default=8)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--local-layers", type=int, default=6)
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--num-buckets", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--semantic-buckets", type=int, default=256)
    parser.add_argument("--semantic-temperature", type=float, default=0.2)
    parser.add_argument("--fixed-cache-weight", type=float, default=0.5)
    parser.add_argument("--log-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jax_distributed:
        jax.distributed.initialize()
    local_devices = jax.local_devices()
    local_device_count = len(local_devices)
    process_count = jax.process_count()
    process_index = jax.process_index()
    if args.global_batch_size % process_count != 0:
        raise ValueError("global batch size must divide process_count")
    per_process_batch = args.global_batch_size // process_count

    cfg_name = None if args.dataset_config == "none" else args.dataset_config
    raw = load_dataset(args.dataset, cfg_name)
    train_images, train_labels = preprocess_split(
        raw["train"],
        image_field=args.image_field,
        label_field=args.label_field,
        class_subset=args.class_subset,
        max_examples=args.max_train_examples,
    )
    eval_split = "test" if "test" in raw else "validation"
    eval_images, eval_labels = preprocess_split(
        raw[eval_split],
        image_field=args.image_field,
        label_field=args.label_field,
        class_subset=args.class_subset,
        max_examples=args.max_eval_examples,
    )
    num_classes = (
        args.class_subset
        if args.class_subset > 0
        else int(max(train_labels.max(), eval_labels.max())) + 1
    )

    train_batcher = ImageBatcher(
        train_images,
        train_labels,
        per_process_batch=per_process_batch,
        local_device_count=local_device_count,
        seed=args.train_sample_seed + process_index,
        train=True,
        augment=not args.no_augment,
        crop_padding=args.crop_padding,
        hflip_prob=args.hflip_prob,
        cutout_size=args.cutout_size,
        mixup_alpha=args.mixup_alpha,
        num_classes=num_classes,
    )
    eval_batcher = ImageBatcher(
        eval_images,
        eval_labels,
        per_process_batch=per_process_batch,
        local_device_count=local_device_count,
        seed=args.eval_sample_seed + process_index,
        train=False,
        augment=False,
        crop_padding=0,
        hflip_prob=0.0,
        cutout_size=0,
        mixup_alpha=0.0,
        num_classes=num_classes,
    )

    num_patches = (32 // args.patch_size) ** 2
    patch_dim = 3 * args.patch_size * args.patch_size
    params = init_vision_params(
        random.PRNGKey(args.seed),
        patch_dim=patch_dim,
        num_patches=num_patches,
        num_classes=num_classes,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        local_layers=args.local_layers,
        local_kernel_size=args.local_kernel_size,
        semantic_buckets=args.semantic_buckets,
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    opt_state = init_adam_state(params)
    params = replicate(params, local_devices)
    opt_state = replicate(opt_state, local_devices)

    routing_mode = "token_hash"
    use_cache = True
    use_gate = True
    if args.model == "local_conv":
        use_cache = False
    elif args.model == "pcaf_semantic":
        routing_mode = "semantic_hash"
    elif args.model == "pcaf_hybrid":
        routing_mode = "hybrid_semantic_hash"
    elif args.model == "pcaf_no_gate":
        use_gate = False

    cfg = {
        "patch_size": args.patch_size,
        "num_buckets": args.num_buckets,
        "top_k": args.top_k,
        "routing_mode": routing_mode,
        "semantic_temperature": args.semantic_temperature,
        "use_cache": use_cache,
        "use_gate": use_gate,
        "fixed_cache_weight": args.fixed_cache_weight,
        "scale": args.d_model**-0.5,
        "num_classes": num_classes,
        "label_smoothing": args.label_smoothing,
    }
    eval_cfg = dict(cfg)
    eval_cfg["label_smoothing"] = 0.0
    train_step = make_train_step(cfg, args.weight_decay)
    eval_step = make_eval_step(eval_cfg)

    if process_index == 0:
        print(f"jax_devices={jax.device_count()} local_devices={local_device_count} processes={process_count} process_index={process_index}")
        print(f"dataset={args.dataset} split={eval_split} model={args.model} classes={num_classes} params={n_params:,}")
        print(f"global_batch={args.global_batch_size} per_process_batch={per_process_batch}")
        print(
            f"augment={not args.no_augment} crop_padding={args.crop_padding} "
            f"hflip_prob={args.hflip_prob} cutout_size={args.cutout_size} "
            f"mixup_alpha={args.mixup_alpha} label_smoothing={args.label_smoothing} "
            f"warmup_steps={args.warmup_steps} min_lr_ratio={args.min_lr_ratio}"
        )

    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        images, labels = train_batcher.sample()
        lr = learning_rate_at_step(
            step,
            base_lr=args.lr,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        lr_shard = np.full((local_device_count,), lr, dtype=np.float32)
        params, opt_state, train_metrics = train_step(params, opt_state, images, labels, lr_shard)
        train_step_sec = time.perf_counter() - step_start
        if step == 1 or step % args.eval_every == 0:
            eval_loss = 0.0
            eval_acc = 0.0
            eval_start = time.perf_counter()
            for _ in range(args.eval_batches):
                eval_images_batch, eval_labels_batch = eval_batcher.sample()
                metrics = eval_step(params, eval_images_batch, eval_labels_batch)
                eval_loss += metric_scalar(metrics, "loss")
                eval_acc += metric_scalar(metrics, "acc")
            eval_loss /= args.eval_batches
            eval_acc /= args.eval_batches
            eval_sec = time.perf_counter() - eval_start
            elapsed = time.perf_counter() - start
            if process_index == 0:
                img_per_sec = args.global_batch_size / max(train_step_sec, 1.0e-9)
                print(
                    f"step={step:05d} loss={metric_scalar(train_metrics, 'loss'):.4f} "
                    f"train_acc={metric_scalar(train_metrics, 'acc'):.4f} "
                    f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                    f"lr={metric_scalar(train_metrics, 'lr'):.6g} "
                    f"train_step_sec={train_step_sec:.4f} img_per_sec={img_per_sec:.1f} "
                    f"eval_sec={eval_sec:.2f} elapsed_min={elapsed / 60.0:.2f}"
                )
                append_jsonl(
                    args.log_jsonl,
                    {
                        "step": step,
                        "model": args.model,
                        "params": n_params,
                        "classes": num_classes,
                        "train_loss": metric_scalar(train_metrics, "loss"),
                        "train_acc": metric_scalar(train_metrics, "acc"),
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "lr": metric_scalar(train_metrics, "lr"),
                        "train_step_sec": train_step_sec,
                        "img_per_sec": img_per_sec,
                        "eval_sec": eval_sec,
                        "elapsed_min": elapsed / 60.0,
                        "augment": not args.no_augment,
                        "crop_padding": args.crop_padding,
                        "hflip_prob": args.hflip_prob,
                        "cutout_size": args.cutout_size,
                        "mixup_alpha": args.mixup_alpha,
                        "label_smoothing": args.label_smoothing,
                    },
                )


if __name__ == "__main__":
    main()

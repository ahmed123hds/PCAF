from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random

from datasets import load_dataset


PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HASH_MOD = 2_147_483_647


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


class HostBatcher:
    def __init__(
        self,
        ids: list[int],
        *,
        seq_len: int,
        per_process_batch: int,
        local_device_count: int,
        seed: int,
    ) -> None:
        if per_process_batch % local_device_count != 0:
            raise ValueError("per-process batch must divide local_device_count")
        if len(ids) <= seq_len + 1:
            raise ValueError("not enough tokens for requested seq_len")
        self.data = np.asarray(ids, dtype=np.int32)
        self.seq_len = seq_len
        self.per_process_batch = per_process_batch
        self.local_device_count = local_device_count
        self.per_device_batch = per_process_batch // local_device_count
        self.rng = np.random.default_rng(seed)
        self.offsets = np.arange(seq_len + 1, dtype=np.int64)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(
            0,
            self.data.shape[0] - self.seq_len - 1,
            size=(self.per_process_batch,),
            dtype=np.int64,
        )
        block = self.data[starts[:, None] + self.offsets[None, :]]
        tokens = block[:, :-1]
        targets = block[:, -1]
        shard_shape = (self.local_device_count, self.per_device_batch)
        return (
            tokens.reshape(*shard_shape, self.seq_len),
            targets.reshape(*shard_shape),
        )


def iter_texts(dataset_split, text_field: str = "text") -> list[str]:
    """Extract text strings from a HuggingFace dataset split.

    Supports multiple field names so datasets like PG-19 (field='book_text')
    work without any code changes — just pass --text-field book_text.
    """
    # Allow a comma-separated fallback list, e.g. "book_text,text"
    fields = [f.strip() for f in text_field.split(",")]
    results: list[str] = []
    for row in dataset_split:
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, str) and val.strip():
                results.append(val)
                break
    return results


def build_vocab(texts: list[str], *, max_vocab: int, min_freq: int) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(TOKEN_RE.findall(text))

    itos = [PAD, UNK, EOS]
    for token, freq in counter.most_common(max_vocab - len(itos)):
        if freq < min_freq:
            break
        itos.append(token)
    return Vocab(stoi={token: i for i, token in enumerate(itos)}, itos=itos)


def encode_corpus(texts: list[str], vocab: Vocab, *, limit: int) -> list[int]:
    eos = vocab.stoi[EOS]
    ids: list[int] = []
    for text in texts:
        ids.extend(vocab.encode(text))
        ids.append(eos)
        if 0 < limit <= len(ids):
            return ids[:limit]
    return ids


def init_linear(key, fan_in: int, fan_out: int, scale: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    std = scale / math.sqrt(fan_in)
    w = random.normal(key, (fan_in, fan_out), dtype=jnp.float32) * std
    b = jnp.zeros((fan_out,), dtype=jnp.float32)
    return w, b


def split_key(key, n: int):
    keys = random.split(key, n + 1)
    return keys[0], list(keys[1:])


def init_params(
    key,
    *,
    vocab_size: int,
    d_model: int,
    d_hidden: int,
    local_layers: int,
    local_kernel_size: int,
    semantic_buckets: int,
) -> dict[str, Any]:
    key, keys = split_key(key, 10 + 4 * local_layers)
    params: dict[str, Any] = {}
    params["emb"] = random.normal(keys.pop(), (vocab_size, d_model), dtype=jnp.float32) * 0.02

    blocks = []
    for _ in range(local_layers):
        k_conv = keys.pop()
        k_w1 = keys.pop()
        k_w2 = keys.pop()
        conv_kernel = random.normal(
            k_conv, (local_kernel_size, d_model), dtype=jnp.float32
        ) * (1.0 / math.sqrt(local_kernel_size))
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
    params["semantic_w"], params["semantic_b"] = init_linear(
        keys.pop(), d_model, semantic_buckets
    )

    params["head_ln_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
    params["head_ln_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["head_w1"], params["head_b1"] = init_linear(keys.pop(), d_model, d_hidden)
    params["head_w2"], params["head_b2"] = init_linear(keys.pop(), d_hidden, vocab_size)

    gate_hidden = max(d_model // 2, 1)
    params["gate_ln_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
    params["gate_ln_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["gate_w1"], params["gate_b1"] = init_linear(keys.pop(), d_model, gate_hidden)
    params["gate_w2"], params["gate_b2"] = init_linear(keys.pop(), gate_hidden, 1)
    params["recency_scale"] = jnp.asarray(1.0, dtype=jnp.float32)
    return params


def sinusoidal_positions(max_len: int, d_model: int) -> jnp.ndarray:
    pos = jnp.arange(max_len, dtype=jnp.float32)[:, None]
    half = (d_model + 1) // 2
    freq = jnp.exp(
        -math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
    )
    angles = pos * freq[None, :]
    pe = jnp.zeros((max_len, d_model), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(angles[:, : pe[:, 0::2].shape[1]]))
    pe = pe.at[:, 1::2].set(jnp.cos(angles[:, : pe[:, 1::2].shape[1]]))
    return pe


def init_transformer_params(
    key,
    *,
    vocab_size: int,
    d_model: int,
    d_hidden: int,
    layers: int,
    heads: int,
    max_seq_len: int,
) -> dict[str, Any]:
    if d_model % heads != 0:
        raise ValueError("d_model must divide heads")
    key, keys = split_key(key, 4 + 4 * layers)
    params: dict[str, Any] = {}
    params["emb"] = random.normal(keys.pop(), (vocab_size, d_model), dtype=jnp.float32) * 0.02
    params["pos"] = sinusoidal_positions(max_seq_len, d_model)

    blocks = []
    for _ in range(layers):
        k_qkv = keys.pop()
        k_out = keys.pop()
        k_ff1 = keys.pop()
        k_ff2 = keys.pop()
        qkv_w, qkv_b = init_linear(k_qkv, d_model, 3 * d_model)
        out_w, out_b = init_linear(k_out, d_model, d_model)
        ff1_w, ff1_b = init_linear(k_ff1, d_model, d_hidden)
        ff2_w, ff2_b = init_linear(k_ff2, d_hidden, d_model)
        blocks.append(
            {
                "ln1_scale": jnp.ones((d_model,), dtype=jnp.float32),
                "ln1_bias": jnp.zeros((d_model,), dtype=jnp.float32),
                "qkv_w": qkv_w,
                "qkv_b": qkv_b,
                "out_w": out_w,
                "out_b": out_b,
                "ln2_scale": jnp.ones((d_model,), dtype=jnp.float32),
                "ln2_bias": jnp.zeros((d_model,), dtype=jnp.float32),
                "ff1_w": ff1_w,
                "ff1_b": ff1_b,
                "ff2_w": ff2_w,
                "ff2_b": ff2_b,
            }
        )
    params["blocks"] = blocks
    params["head_ln_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
    params["head_ln_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["head_w1"], params["head_b1"] = init_linear(keys.pop(), d_model, d_hidden)
    params["head_w2"], params["head_b2"] = init_linear(keys.pop(), d_hidden, vocab_size)
    return params


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def init_adam_state(params):
    return {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": jnp.asarray(0, dtype=jnp.int32)}


def adamw_update(params, grads, state, *, lr: float, weight_decay: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1.0e-8):
    t = state["t"] + jnp.asarray(1, dtype=jnp.int32)
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1.0 - b2) * (g * g), state["v"], grads)
    t_f = t.astype(jnp.float32)

    def update_param(p, m_, v_):
        m_hat = m_ / (1.0 - b1**t_f)
        v_hat = v_ / (1.0 - b2**t_f)
        return p - lr * (m_hat / (jnp.sqrt(v_hat) + eps) + weight_decay * p)

    new_params = jax.tree_util.tree_map(update_param, params, m, v)
    return new_params, {"m": m, "v": v, "t": t}


def layer_norm(x, scale, bias, eps: float = 1.0e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * lax.rsqrt(var + eps) * scale + bias


def causal_depthwise_conv(x, kernel, bias):
    # x: [B, T, D], kernel: [K, D], causal lags [0..K-1]
    bsz, seq_len, d_model = x.shape
    del bsz, d_model
    out = jnp.zeros_like(x)
    for lag in range(kernel.shape[0]):
        if lag == 0:
            shifted = x
        else:
            pad = jnp.zeros_like(x[:, :lag, :])
            shifted = jnp.concatenate([pad, x[:, : seq_len - lag, :]], axis=1)
        out = out + shifted * kernel[lag][None, None, :]
    return out + bias[None, None, :]


def local_block_forward(block, x):
    y = layer_norm(x, block["ln_scale"], block["ln_bias"])
    y = causal_depthwise_conv(y, block["conv_kernel"], block["conv_bias"])
    y = jax.nn.gelu(y @ block["w1"] + block["b1"])
    y = y @ block["w2"] + block["b2"]
    return x + y


def l2_normalize(x, eps: float = 1.0e-6):
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_ngram_hash(tokens, order: int):
    h = jnp.zeros_like(tokens, dtype=jnp.int32)
    for shift in range(order - 1, -1, -1):
        if shift == 0:
            shifted = tokens.astype(jnp.int32)
        else:
            shifted = jnp.concatenate(
                [jnp.zeros_like(tokens[:, :shift]), tokens[:, :-shift]], axis=1
            ).astype(jnp.int32)
        h = jnp.mod(h * jnp.asarray(1_000_003, jnp.int32) + shifted + 97, HASH_MOD)
    return h


def bucket_hash(values, num_buckets: int):
    return jnp.mod(values.astype(jnp.int32) * jnp.asarray(1_000_003, jnp.int32) + 97_531, num_buckets)


def pcaf_forward(
    params,
    tokens,
    *,
    vocab_size: int,
    d_model: int,
    num_buckets: int,
    top_k: int,
    context_order: int,
    routing_mode: str,
    semantic_buckets: int,
    semantic_temperature: float,
    semantic_score_scale: float,
    use_cache: bool,
    use_gate: bool,
    fixed_cache_weight: float,
):
    x = params["emb"][tokens]
    for block in params["blocks"]:
        x = local_block_forward(block, x)

    final_state = x[:, -1, :]
    h = layer_norm(final_state, params["head_ln_scale"], params["head_ln_bias"])
    h = jax.nn.gelu(h @ params["head_w1"] + params["head_b1"])
    param_logits = h @ params["head_w2"] + params["head_b2"]
    param_log_probs = jax.nn.log_softmax(param_logits, axis=-1)
    if not use_cache:
        return param_log_probs

    value_tokens = tokens[:, 1:]

    record_keys = l2_normalize(x[:, :-1, :] @ params["wk"])
    query = l2_normalize(final_state @ params["wq"])
    n_records = value_tokens.shape[1]
    recency = jnp.linspace(0.0, 1.0, n_records, dtype=jnp.float32)

    semantic_route_scores = None
    if routing_mode in {"semantic_hash", "hybrid_semantic_hash"}:
        record_sem_logits = x[:, :-1, :] @ params["semantic_w"] + params["semantic_b"]
        query_sem_logits = final_state @ params["semantic_w"] + params["semantic_b"]
        temp = max(float(semantic_temperature), 1.0e-4)
        record_sem = jax.nn.softmax(record_sem_logits / temp, axis=-1)
        query_sem = jax.nn.softmax(query_sem_logits / temp, axis=-1)
        semantic_scores = jnp.einsum("bnc,bc->bn", record_sem, query_sem)
        semantic_rank_scores = semantic_scores + 1.0e-4 * params["recency_scale"] * recency[None, :]
        semantic_top_scores, semantic_top_idx = lax.top_k(semantic_rank_scores, top_k)
        semantic_route_scores = jnp.take_along_axis(
            semantic_scores, semantic_top_idx, axis=1
        )

    if routing_mode in {"token_hash", "hybrid_semantic_hash"}:
        context_hashes = causal_ngram_hash(tokens, context_order)
        record_hashes = context_hashes[:, :-1]
        query_hashes = context_hashes[:, -1]
        scores = jnp.einsum("bnd,bd->bn", record_keys, query) * (d_model**-0.5)
        scores = scores + params["recency_scale"] * recency[None, :]
        record_bucket = bucket_hash(record_hashes, num_buckets)
        query_bucket = bucket_hash(query_hashes, num_buckets)[:, None]
        mask = record_bucket == query_bucket
        masked_scores = jnp.where(mask, scores, -1.0e9)
        token_top_scores, token_top_idx = lax.top_k(masked_scores, top_k)
        token_valid = token_top_scores > -1.0e8
    else:
        token_top_idx = jnp.zeros((tokens.shape[0], top_k), dtype=jnp.int32)
        token_valid = jnp.zeros((tokens.shape[0], top_k), dtype=jnp.bool_)

    if routing_mode == "semantic_hash":
        top_idx = semantic_top_idx
        valid = jnp.ones_like(top_idx, dtype=jnp.bool_)
        candidate_route_scores = semantic_route_scores
    elif routing_mode == "hybrid_semantic_hash":
        top_idx = jnp.concatenate([token_top_idx, semantic_top_idx], axis=1)
        valid = jnp.concatenate(
            [token_valid, jnp.ones_like(semantic_top_idx, dtype=jnp.bool_)], axis=1
        )
        token_route_scores = jnp.take_along_axis(semantic_scores, token_top_idx, axis=1)
        candidate_route_scores = jnp.concatenate(
            [token_route_scores, semantic_route_scores], axis=1
        )
    else:
        top_idx = token_top_idx
        valid = token_valid
        candidate_route_scores = None

    cand_tokens = jnp.take_along_axis(value_tokens, top_idx, axis=1)

    batch_idx = jnp.arange(tokens.shape[0])[:, None]  # [B, 1]
    gathered_keys = record_keys[batch_idx, top_idx]    # [B, K, D]
    cache_scores = jnp.einsum("bkd,bd->bk", gathered_keys, query) * (d_model**-0.5)
    safe_idx = jnp.maximum(top_idx, 0)
    cache_scores = cache_scores + params["recency_scale"] * (
        safe_idx.astype(jnp.float32) / jnp.maximum(float(n_records - 1), 1.0)
    )
    if candidate_route_scores is not None:
        cache_scores = cache_scores + semantic_score_scale * jnp.log(
            jnp.maximum(candidate_route_scores, 1.0e-6)
        )
    cache_scores = jnp.where(valid, cache_scores, -1.0e9)

    weights = jax.nn.softmax(cache_scores, axis=1) * valid.astype(jnp.float32)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0e-6)

    cache_probs = jnp.einsum("bk,bkv->bv", weights,
                             jax.nn.one_hot(cand_tokens, vocab_size))
    cache_log_probs = jnp.log(jnp.maximum(cache_probs, 1.0e-6))

    if use_gate:
        g = layer_norm(final_state, params["gate_ln_scale"], params["gate_ln_bias"])
        g = jax.nn.gelu(g @ params["gate_w1"] + params["gate_b1"])
        gate = jax.nn.sigmoid(g @ params["gate_w2"] + params["gate_b2"])
    else:
        gate = jnp.full((tokens.shape[0], 1), fixed_cache_weight, dtype=jnp.float32)
    has_cache = jnp.any(valid, axis=1, keepdims=True)
    gate = gate * has_cache.astype(jnp.float32)
    gate = jnp.clip(gate, 1.0e-5, 1.0 - 1.0e-5)

    return jnp.logaddexp(jnp.log1p(-gate) + param_log_probs, jnp.log(gate) + cache_log_probs)


def attention_mask(seq_len: int, *, attention_mode: str, window_size: int, global_tokens: int):
    q = jnp.arange(seq_len, dtype=jnp.int32)[:, None]
    k = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    allowed = k <= q
    if attention_mode == "local":
        allowed = allowed & ((q - k) < window_size)
    elif attention_mode == "global_local":
        local = (q - k) < window_size
        global_key = k < global_tokens
        allowed = allowed & (local | global_key)
    elif attention_mode != "dense":
        raise ValueError(f"unknown attention_mode={attention_mode}")
    return allowed


def causal_linear_attention_chunked(q, k, v, *, chunk_size: int):
    """Causal linear attention without materializing [B, H, T, D, D] prefixes."""
    bsz, heads, seq_len, head_dim = q.shape
    chunk_size = min(int(chunk_size), seq_len)
    pad_len = (-seq_len) % chunk_size
    if pad_len:
        pad_spec = ((0, 0), (0, 0), (0, pad_len), (0, 0))
        q = jnp.pad(q, pad_spec)
        k = jnp.pad(k, pad_spec)
        v = jnp.pad(v, pad_spec)

    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size
    q = q.reshape(bsz, heads, num_chunks, chunk_size, head_dim)
    k = k.reshape(bsz, heads, num_chunks, chunk_size, head_dim)
    v = v.reshape(bsz, heads, num_chunks, chunk_size, head_dim)

    chunk_k = jnp.sum(k, axis=3)
    chunk_kv = jnp.einsum("bhncd,bhnce->bhnde", k, v)
    prefix_k = jnp.concatenate(
        [jnp.zeros_like(chunk_k[:, :, :1, :]), jnp.cumsum(chunk_k, axis=2)[:, :, :-1, :]],
        axis=2,
    )
    prefix_kv = jnp.concatenate(
        [
            jnp.zeros_like(chunk_kv[:, :, :1, :, :]),
            jnp.cumsum(chunk_kv, axis=2)[:, :, :-1, :, :],
        ],
        axis=2,
    )

    cross_numer = jnp.einsum("bhncd,bhnde->bhnce", q, prefix_kv)
    cross_denom = jnp.einsum("bhncd,bhnd->bhnc", q, prefix_k)

    local_scores = jnp.einsum("bhncd,bhned->bhnce", q, k)
    local_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    local_scores = jnp.where(local_mask[None, None, None, :, :], local_scores, 0.0)
    local_numer = jnp.einsum("bhnce,bhned->bhncd", local_scores, v)
    local_denom = jnp.sum(local_scores, axis=-1)

    out = (cross_numer + local_numer) / jnp.maximum(
        (cross_denom + local_denom)[..., None], 1.0e-6
    )
    return out.reshape(bsz, heads, padded_len, head_dim)[:, :, :seq_len, :]


def transformer_block_forward(
    block,
    x,
    *,
    heads: int,
    attention_mode: str,
    window_size: int,
    global_tokens: int,
    linear_chunk_size: int,
):
    bsz, seq_len, d_model = x.shape
    head_dim = d_model // heads

    y = layer_norm(x, block["ln1_scale"], block["ln1_bias"])
    qkv = y @ block["qkv_w"] + block["qkv_b"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(bsz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

    if attention_mode == "linear":
        q_phi = jax.nn.elu(q) + 1.0
        k_phi = jax.nn.elu(k) + 1.0
        attn = causal_linear_attention_chunked(
            q_phi, k_phi, v, chunk_size=linear_chunk_size
        )
    else:
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * (head_dim**-0.5)
        allowed = attention_mask(
            seq_len,
            attention_mode=attention_mode,
            window_size=window_size,
            global_tokens=global_tokens,
        )
        scores = jnp.where(allowed[None, None, :, :], scores, -1.0e9)
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    attn = attn.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
    x = x + (attn @ block["out_w"] + block["out_b"])

    z = layer_norm(x, block["ln2_scale"], block["ln2_bias"])
    z = jax.nn.gelu(z @ block["ff1_w"] + block["ff1_b"])
    z = z @ block["ff2_w"] + block["ff2_b"]
    return x + z


def transformer_forward(
    params,
    tokens,
    *,
    heads: int,
    attention_mode: str,
    window_size: int,
    global_tokens: int,
    linear_chunk_size: int,
):
    seq_len = tokens.shape[1]
    x = params["emb"][tokens] + params["pos"][None, :seq_len, :]
    for block in params["blocks"]:
        x = transformer_block_forward(
            block,
            x,
            heads=heads,
            attention_mode=attention_mode,
            window_size=window_size,
            global_tokens=global_tokens,
            linear_chunk_size=linear_chunk_size,
        )
    h = layer_norm(x[:, -1, :], params["head_ln_scale"], params["head_ln_bias"])
    h = jax.nn.gelu(h @ params["head_w1"] + params["head_b1"])
    return jax.nn.log_softmax(h @ params["head_w2"] + params["head_b2"], axis=-1)


def loss_and_metrics(params, tokens, targets, cfg):
    if cfg["model_family"] == "transformer":
        log_probs = transformer_forward(
            params,
            tokens,
            heads=cfg["heads"],
            attention_mode=cfg["attention_mode"],
            window_size=cfg["attention_window"],
            global_tokens=cfg["global_tokens"],
            linear_chunk_size=cfg["linear_chunk_size"],
        )
    else:
        log_probs = pcaf_forward(
            params,
            tokens,
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_buckets=cfg["num_buckets"],
            top_k=cfg["top_k"],
            context_order=cfg["context_order"],
            routing_mode=cfg["routing_mode"],
            semantic_buckets=cfg["semantic_buckets"],
            semantic_temperature=cfg["semantic_temperature"],
            semantic_score_scale=cfg["semantic_score_scale"],
            use_cache=cfg["use_cache"],
            use_gate=cfg["use_gate"],
            fixed_cache_weight=cfg["fixed_cache_weight"],
        )
    target_log_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=1)[:, 0]
    loss = -jnp.mean(target_log_probs)
    pred = jnp.argmax(log_probs, axis=-1)
    acc = jnp.mean((pred == targets).astype(jnp.float32))
    return loss, {"loss": loss, "acc": acc}


def make_train_step(cfg, lr: float, weight_decay: float):
    def train_step(params, opt_state, tokens, targets):
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            params, tokens, targets, cfg
        )
        grads = lax.pmean(grads, axis_name="data")
        metrics = lax.pmean(metrics, axis_name="data")
        params, opt_state = adamw_update(
            params, grads, opt_state, lr=lr, weight_decay=weight_decay
        )
        return params, opt_state, metrics

    return jax.pmap(train_step, axis_name="data", donate_argnums=(0, 1))


def make_eval_step(cfg):
    def eval_step(params, tokens, targets):
        _, metrics = loss_and_metrics(params, tokens, targets, cfg)
        return lax.pmean(metrics, axis_name="data")

    return jax.pmap(eval_step, axis_name="data")


def replicate(tree, devices):
    return jax.device_put_replicated(tree, devices)


def metric_scalar(metrics, name: str) -> float:
    return float(jax.device_get(metrics[name][0]))


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path or jax.process_index() != 0:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure-JAX TPU PCAF and baseline LM")
    parser.add_argument("--jax-distributed", action="store_true")
    parser.add_argument(
        "--model",
        choices=[
            "pcaf_context",
            "local_conv",
            "pcaf_no_gate",
            "pcaf_semantic",
            "pcaf_hybrid",
            "transformer_dense",
            "linear_attention",
            "local_transformer",
            "global_local_transformer",
        ],
        default="pcaf_context",
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument(
        "--text-field",
        default="text",
        help="Dataset column name(s) containing raw text. Comma-separated for "
             "fallback order. PG-19 uses 'book_text'; WikiText uses 'text'.",
    )
    parser.add_argument("--cache-dir", default="/tmp/hf_cache")
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-train-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-eval-tokens", type=int, default=200_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-sample-seed", type=int, default=10001)
    parser.add_argument("--eval-sample-seed", type=int, default=20001)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=1000)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=128)
    parser.add_argument("--global-tokens", type=int, default=16)
    parser.add_argument("--linear-chunk-size", type=int, default=64)
    parser.add_argument("--num-buckets", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--context-order", type=int, default=1)
    parser.add_argument(
        "--routing-mode",
        choices=["token_hash", "semantic_hash", "hybrid_semantic_hash"],
        default="token_hash",
    )
    parser.add_argument("--semantic-buckets", type=int, default=256)
    parser.add_argument("--semantic-temperature", type=float, default=0.2)
    parser.add_argument("--semantic-score-scale", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
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
        raise ValueError("global_batch_size must divide jax.process_count()")
    per_process_batch = args.global_batch_size // process_count
    if per_process_batch % local_device_count != 0:
        raise ValueError("per-process batch must divide local_device_count")

    dataset_config = None if args.dataset_config.lower() in {"", "none", "null"} else args.dataset_config
    raw = load_dataset(args.dataset, dataset_config, cache_dir=args.cache_dir)
    train_texts = iter_texts(raw["train"], text_field=args.text_field)
    eval_split = "validation" if "validation" in raw else "test"
    eval_texts = iter_texts(raw[eval_split], text_field=args.text_field)
    vocab = build_vocab(train_texts, max_vocab=args.max_vocab, min_freq=args.min_freq)
    train_ids = encode_corpus(train_texts, vocab, limit=args.max_train_tokens)
    eval_ids = encode_corpus(eval_texts, vocab, limit=args.max_eval_tokens)

    train_batcher = HostBatcher(
        train_ids,
        seq_len=args.seq_len,
        per_process_batch=per_process_batch,
        local_device_count=local_device_count,
        seed=args.train_sample_seed + process_index,
    )
    eval_batcher = HostBatcher(
        eval_ids,
        seq_len=args.seq_len,
        per_process_batch=per_process_batch,
        local_device_count=local_device_count,
        seed=args.eval_sample_seed + process_index,
    )

    key = random.PRNGKey(args.seed)
    transformer_models = {
        "transformer_dense",
        "linear_attention",
        "local_transformer",
        "global_local_transformer",
    }
    if args.model in transformer_models:
        params = init_transformer_params(
            key,
            vocab_size=vocab.size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            layers=args.layers,
            heads=args.heads,
            max_seq_len=args.seq_len,
        )
    else:
        params = init_params(
            key,
            vocab_size=vocab.size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            local_layers=args.local_layers,
            local_kernel_size=args.local_kernel_size,
            semantic_buckets=args.semantic_buckets,
        )
    opt_state = init_adam_state(params)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    params = replicate(params, local_devices)
    opt_state = replicate(opt_state, local_devices)

    routing_mode = args.routing_mode
    use_cache = not args.no_cache
    use_gate = not args.no_gate
    if args.model == "local_conv":
        use_cache = False
    elif args.model == "pcaf_no_gate":
        use_gate = False
    elif args.model == "pcaf_semantic":
        routing_mode = "semantic_hash"
    elif args.model == "pcaf_hybrid":
        routing_mode = "hybrid_semantic_hash"

    attention_mode = "dense"
    if args.model == "linear_attention":
        attention_mode = "linear"
    elif args.model == "local_transformer":
        attention_mode = "local"
    elif args.model == "global_local_transformer":
        attention_mode = "global_local"

    cfg = {
        "model_family": "transformer" if args.model in transformer_models else "pcaf",
        "vocab_size": vocab.size,
        "d_model": args.d_model,
        "num_buckets": args.num_buckets,
        "top_k": args.top_k,
        "context_order": args.context_order,
        "routing_mode": routing_mode,
        "semantic_buckets": args.semantic_buckets,
        "semantic_temperature": args.semantic_temperature,
        "semantic_score_scale": args.semantic_score_scale,
        "use_cache": use_cache,
        "use_gate": use_gate,
        "fixed_cache_weight": args.fixed_cache_weight,
        "heads": args.heads,
        "attention_mode": attention_mode,
        "attention_window": args.attention_window,
        "global_tokens": args.global_tokens,
        "linear_chunk_size": args.linear_chunk_size,
    }
    train_step = make_train_step(cfg, args.lr, args.weight_decay)
    eval_step = make_eval_step(cfg)

    if process_index == 0:
        print(
            f"jax_devices={jax.device_count()} local_devices={local_device_count} "
            f"processes={process_count} process_index={process_index}"
        )
        print(
            f"dataset={args.dataset}/{dataset_config or 'default'} split={eval_split} "
            f"model={args.model} "
            f"vocab={vocab.size} train_tokens={len(train_ids):,} "
            f"eval_tokens={len(eval_ids):,} params={n_params:,}"
        )
        print(
            f"seq_len={args.seq_len} global_batch={args.global_batch_size} "
            f"per_process_batch={per_process_batch} "
            f"per_device_batch={train_batcher.per_device_batch}"
        )

    run_start = time.perf_counter()
    last_log_time = run_start
    last_metrics = None
    for step in range(1, args.steps + 1):
        tokens_np, targets_np = train_batcher.sample()
        params, opt_state, last_metrics = train_step(params, opt_state, tokens_np, targets_np)

        if step == 1 or step % args.eval_every == 0:
            # Force completion before timing.
            _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), last_metrics)
            eval_start = time.perf_counter()
            eval_loss = 0.0
            eval_acc = 0.0
            for _ in range(args.eval_batches):
                eval_tokens_np, eval_targets_np = eval_batcher.sample()
                metrics = eval_step(params, eval_tokens_np, eval_targets_np)
                _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
                eval_loss += metric_scalar(metrics, "loss")
                eval_acc += metric_scalar(metrics, "acc")
            eval_loss /= args.eval_batches
            eval_acc /= args.eval_batches
            now = time.perf_counter()
            eval_sec = now - eval_start
            interval_steps = 1 if step == 1 else args.eval_every
            interval_elapsed = now - last_log_time
            train_step_sec = max((interval_elapsed - eval_sec) / interval_steps, 0.0)
            tok_per_sec = (
                args.global_batch_size * args.seq_len / train_step_sec
                if train_step_sec > 0
                else 0.0
            )
            train_loss = metric_scalar(last_metrics, "loss") if last_metrics is not None else 0.0
            train_acc = metric_scalar(last_metrics, "acc") if last_metrics is not None else 0.0
            ppl = math.exp(min(20.0, eval_loss))

            if process_index == 0:
                print(
                    f"step={step:05d} loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"eval_loss={eval_loss:.4f} eval_ppl={ppl:.2f} eval_acc={eval_acc:.4f} "
                    f"train_step_sec={train_step_sec:.4f} tok_per_sec={tok_per_sec:.1f} "
                    f"eval_sec={eval_sec:.2f} elapsed_min={(now - run_start) / 60.0:.2f}"
                )
                append_jsonl(
                    args.log_jsonl,
                    {
                        "step": step,
                        "model": args.model,
                        "dataset": args.dataset,
                        "dataset_config": dataset_config or "",
                        "eval_split": eval_split,
                        "params": n_params,
                        "seq_len": args.seq_len,
                        "global_batch_size": args.global_batch_size,
                        "batch_size": args.global_batch_size,
                        "per_process_batch": per_process_batch,
                        "per_device_batch": train_batcher.per_device_batch,
                        "max_vocab": args.max_vocab,
                        "train_tokens": len(train_ids),
                        "eval_tokens": len(eval_ids),
                        "steps_total": args.steps,
                        "eval_every": args.eval_every,
                        "eval_batches": args.eval_batches,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "seed": args.seed,
                        "train_sample_seed": args.train_sample_seed,
                        "eval_sample_seed": args.eval_sample_seed,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "eval_loss": eval_loss,
                        "eval_ppl": ppl,
                        "eval_acc": eval_acc,
                        "train_step_sec": train_step_sec,
                        "train_tokens_per_sec": tok_per_sec,
                        "eval_sec": eval_sec,
                        "elapsed_sec": now - run_start,
                        "d_model": args.d_model,
                        "d_hidden": args.d_hidden,
                        "local_layers": args.local_layers,
                        "local_kernel_size": args.local_kernel_size,
                        "layers": args.layers,
                        "heads": args.heads,
                        "attention_mode": attention_mode,
                        "attention_window": args.attention_window,
                        "global_tokens": args.global_tokens,
                        "linear_chunk_size": args.linear_chunk_size,
                        "num_buckets": args.num_buckets,
                        "top_k": args.top_k,
                        "context_order": args.context_order,
                        "routing_mode": routing_mode,
                        "semantic_buckets": args.semantic_buckets,
                        "semantic_temperature": args.semantic_temperature,
                        "semantic_score_scale": args.semantic_score_scale,
                        "use_cache": use_cache,
                        "use_gate": use_gate,
                    },
                )
            last_log_time = now


if __name__ == "__main__":
    main()

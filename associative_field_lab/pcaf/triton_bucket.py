from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _build_bucket_indices_kernel(
        tokens_ptr,
        counts_ptr,
        indices_ptr,
        n_records: tl.constexpr,
        num_buckets: tl.constexpr,
        max_bucket_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_records

        tokens = tl.load(
            tokens_ptr + batch_id * n_records + offsets,
            mask=mask,
            other=0,
        ).to(tl.int64)
        buckets = ((tokens * 1000003 + 97531) % num_buckets).to(tl.int64)
        count_offsets = batch_id * num_buckets + buckets
        slots = tl.atomic_add(counts_ptr + count_offsets, 1, sem="relaxed", mask=mask)

        keep = mask & (slots < max_bucket_size)
        out_offsets = (
            (batch_id * num_buckets + buckets) * max_bucket_size + slots
        )
        tl.store(indices_ptr + out_offsets, offsets.to(tl.int32), mask=keep)


    @triton.jit
    def _sparse_read_forward_kernel(
        query_ptr,
        keys_ptr,
        values_ptr,
        indices_ptr,
        out_ptr,
        n_records: tl.constexpr,
        d_model: tl.constexpr,
        max_bucket_size: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        k_offsets = tl.arange(0, BLOCK_K)
        d_offsets = tl.arange(0, BLOCK_D)

        idx = tl.load(
            indices_ptr + batch_id * max_bucket_size + k_offsets,
            mask=k_offsets < max_bucket_size,
            other=-1,
        ).to(tl.int64)
        valid_k = idx >= 0
        safe_idx = tl.maximum(idx, 0)

        query = tl.load(
            query_ptr + batch_id * d_model + d_offsets,
            mask=d_offsets < d_model,
            other=0.0,
        )
        key_offsets = (
            (batch_id * n_records + safe_idx[:, None]) * d_model
            + d_offsets[None, :]
        )
        valid = valid_k[:, None] & (d_offsets[None, :] < d_model)
        keys = tl.load(keys_ptr + key_offsets, mask=valid, other=0.0)

        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(valid_k, scores, -1.0e20)
        scores = scores - tl.max(scores, axis=0)
        weights = tl.exp(scores)
        weights = weights / (tl.sum(weights, axis=0) + 1.0e-6)
        weights = tl.where(valid_k, weights, 0.0)

        values = tl.load(values_ptr + key_offsets, mask=valid, other=0.0)
        out = tl.sum(values * weights[:, None], axis=0)
        tl.store(
            out_ptr + batch_id * d_model + d_offsets,
            out,
            mask=d_offsets < d_model,
        )


def triton_available() -> bool:
    return triton is not None


def token_hash(tokens: torch.Tensor, num_buckets: int) -> torch.Tensor:
    return torch.remainder(tokens.long() * 1_000_003 + 97_531, num_buckets)


def build_bucket_indices(
    tokens: torch.Tensor,
    *,
    num_buckets: int,
    max_bucket_size: int,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-size hash buckets for one batch of token records.

    Args:
        tokens: int tensor with shape [batch, n_records].
        num_buckets: number of hash buckets.
        max_bucket_size: retained record indices per bucket.
        block_size: Triton program block size.

    Returns:
        indices: int32 tensor [batch, num_buckets, max_bucket_size], filled with -1.
        counts: int32 tensor [batch, num_buckets], full counts before truncation.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not tokens.is_cuda:
        raise RuntimeError("build_bucket_indices requires a CUDA tensor")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, n_records]")
    if max_bucket_size <= 0:
        raise ValueError("max_bucket_size must be positive")

    tokens = tokens.contiguous()
    batch, n_records = tokens.shape
    counts = torch.zeros(
        (batch, num_buckets), device=tokens.device, dtype=torch.int32
    )
    indices = torch.full(
        (batch, num_buckets, max_bucket_size),
        -1,
        device=tokens.device,
        dtype=torch.int32,
    )

    grid = (batch, triton.cdiv(n_records, block_size))
    _build_bucket_indices_kernel[grid](
        tokens,
        counts,
        indices,
        n_records,
        num_buckets,
        max_bucket_size,
        BLOCK=block_size,
    )
    return indices, counts


def query_bucket_indices(
    record_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    *,
    num_buckets: int,
    max_bucket_size: int,
    block_size: int = 256,
) -> torch.Tensor:
    bucket_indices, _ = build_bucket_indices(
        record_tokens,
        num_buckets=num_buckets,
        max_bucket_size=max_bucket_size,
        block_size=block_size,
    )
    query_buckets = token_hash(query_tokens, num_buckets).long()
    batch_ids = torch.arange(record_tokens.size(0), device=record_tokens.device)
    return bucket_indices[batch_ids, query_buckets].long()


def sparse_read_forward(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    candidate_idx: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Fused forward-only sparse read.

    This is intended for inference/benchmarking. During training, use ordinary
    PyTorch gather and softmax so autograd computes the resolver gradients.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (query.is_cuda and keys.is_cuda and values.is_cuda and candidate_idx.is_cuda):
        raise RuntimeError("sparse_read_forward requires CUDA tensors")
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape")
    if query.ndim != 2 or keys.ndim != 3 or candidate_idx.ndim != 2:
        raise ValueError("expected query [B,D], keys/values [B,N,D], indices [B,K]")
    if query.size(0) != keys.size(0) or query.size(1) != keys.size(2):
        raise ValueError("query shape must match keys batch and channel dimensions")

    query = query.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    candidate_idx = candidate_idx.contiguous()

    batch, n_records, d_model = keys.shape
    max_bucket_size = candidate_idx.size(1)
    block_d = triton.next_power_of_2(d_model)
    block_k = triton.next_power_of_2(max_bucket_size)
    out = torch.empty_like(query)
    _sparse_read_forward_kernel[(batch,)](
        query,
        keys,
        values,
        candidate_idx,
        out,
        n_records,
        d_model,
        max_bucket_size,
        scale,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
    )
    return out

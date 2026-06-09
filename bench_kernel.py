from __future__ import annotations

import argparse
import time

import torch

from pcaf.triton_bucket import query_bucket_indices, sparse_read_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PCAF Triton hash buckets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--records", type=int, default=4096)
    parser.add_argument("--num-buckets", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def time_cpu(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def dense_hash_candidates(
    tokens: torch.Tensor,
    queries: torch.Tensor,
    *,
    num_buckets: int,
    top_k: int,
) -> torch.Tensor:
    record_buckets = torch.remainder(tokens.long() * 1_000_003 + 97_531, num_buckets)
    query_buckets = torch.remainder(queries.long() * 1_000_003 + 97_531, num_buckets)
    mask = record_buckets == query_buckets.unsqueeze(1)
    scores = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
    scores = scores.masked_fill(~mask, -1)
    return scores.topk(top_k, dim=1).indices


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokens = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.records),
        device=device,
        dtype=torch.long,
    )
    query_pos = torch.randint(0, args.records, (args.batch_size,), device=device)
    queries = tokens[torch.arange(args.batch_size, device=device), query_pos]

    triton_fn = lambda: query_bucket_indices(
        tokens,
        queries,
        num_buckets=args.num_buckets,
        max_bucket_size=args.top_k,
    )
    dense_fn = lambda: dense_hash_candidates(
        tokens,
        queries,
        num_buckets=args.num_buckets,
        top_k=args.top_k,
    )

    if device.type == "cuda":
        triton_ms = time_cuda(triton_fn, warmup=args.warmup, iters=args.iters)
        dense_ms = time_cuda(dense_fn, warmup=args.warmup, iters=args.iters)
        candidates = triton_fn()
        query = torch.randn(args.batch_size, args.d_model, device=device)
        keys = torch.randn(args.batch_size, args.records, args.d_model, device=device)
        values = torch.randn_like(keys)
        read_fn = lambda: sparse_read_forward(
            query,
            keys,
            values,
            candidates,
            scale=args.d_model**-0.5,
        )
        read_ms = time_cuda(read_fn, warmup=args.warmup, iters=args.iters)
        torch.cuda.synchronize()
    else:
        triton_ms = time_cpu(triton_fn, warmup=args.warmup, iters=args.iters)
        dense_ms = time_cpu(dense_fn, warmup=args.warmup, iters=args.iters)
        candidates = triton_fn()
        read_ms = float("nan")

    covered = (
        candidates
        == query_pos.unsqueeze(1).to(candidates.device)
    ).any(dim=1).float().mean().item()

    print(f"batch={args.batch_size} records={args.records} top_k={args.top_k}")
    print(f"num_buckets={args.num_buckets} vocab={args.vocab_size}")
    print(f"triton_bucket_ms={triton_ms:.4f}")
    print(f"triton_sparse_read_ms={read_ms:.4f}")
    print(f"dense_hash_topk_ms={dense_ms:.4f}")
    print(f"query_record_coverage={covered:.4f}")


if __name__ == "__main__":
    main()

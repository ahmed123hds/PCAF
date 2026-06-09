from __future__ import annotations

import argparse
import random


def token_hash(token: int, num_buckets: int) -> int:
    return ((token * 1_000_003) + 97_531) % num_buckets


def run_trial(n_pairs: int, n_keys: int, n_values: int, num_buckets: int, top_k: int) -> tuple[bool, bool]:
    keys = random.sample(range(n_keys), n_pairs)
    values = [random.randrange(n_values) for _ in range(n_pairs)]
    query_index = random.randrange(n_pairs)
    query_key = keys[query_index]
    target_value = values[query_index]

    records = list(zip(keys, values))
    oracle = next(value for key, value in records if key == query_key)

    bucket = token_hash(query_key, num_buckets)
    bucket_records = [
        (key, value)
        for key, value in records
        if token_hash(key, num_buckets) == bucket
    ]
    limited_records = bucket_records[-top_k:] if top_k > 0 else bucket_records
    hash_has_target = any(
        key == query_key and value == target_value for key, value in limited_records
    )

    return oracle == target_value, hash_has_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--n-pairs", type=int, default=128)
    parser.add_argument("--n-keys", type=int, default=4096)
    parser.add_argument("--n-values", type=int, default=4096)
    parser.add_argument("--num-buckets", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.n_pairs > args.n_keys:
        raise ValueError("n_pairs must be <= n_keys")

    random.seed(args.seed)
    oracle_ok = 0
    hash_covered = 0
    for _ in range(args.trials):
        oracle, covered = run_trial(
            args.n_pairs,
            args.n_keys,
            args.n_values,
            args.num_buckets,
            args.top_k,
        )
        oracle_ok += int(oracle)
        hash_covered += int(covered)

    print(f"trials={args.trials}")
    print(f"oracle_retrieval_accuracy={oracle_ok / args.trials:.4f}")
    print(f"hash_candidate_coverage={hash_covered / args.trials:.4f}")
    print(
        "hash_candidate_coverage is the probability that the correct record is "
        "inside the bounded hash bucket read."
    )


if __name__ == "__main__":
    main()

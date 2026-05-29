from __future__ import annotations

from dataclasses import dataclass

import torch


PAD_TOKEN = 0
QUERY_TOKEN = 1
FILL_TOKEN = 2


@dataclass(frozen=True)
class Batch:
    tokens: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class TaskInfo:
    vocab_size: int
    num_classes: int
    chance_accuracy: float


def task_info(task: str, n_keys: int, n_values: int, symbol_vocab: int) -> TaskInfo:
    if task == "kv_recall":
        return TaskInfo(
            vocab_size=3 + n_keys + n_values,
            num_classes=n_values,
            chance_accuracy=1.0 / float(n_values),
        )
    if task == "induction":
        return TaskInfo(
            vocab_size=2 + symbol_vocab,
            num_classes=symbol_vocab,
            chance_accuracy=1.0 / float(symbol_vocab),
        )
    raise ValueError(f"unknown task: {task}")


def make_batch(
    task: str,
    batch_size: int,
    device: torch.device,
    *,
    n_pairs: int,
    n_keys: int,
    n_values: int,
    seq_len: int,
    symbol_vocab: int,
) -> Batch:
    if task == "kv_recall":
        return make_kv_recall_batch(
            batch_size=batch_size,
            n_pairs=n_pairs,
            n_keys=n_keys,
            n_values=n_values,
            device=device,
        )
    if task == "induction":
        return make_induction_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            symbol_vocab=symbol_vocab,
            device=device,
        )
    raise ValueError(f"unknown task: {task}")


def make_kv_recall_batch(
    *,
    batch_size: int,
    n_pairs: int,
    n_keys: int,
    n_values: int,
    device: torch.device,
) -> Batch:
    """Random key-value lookup.

    Sequence format:
        k_1, v_1, k_2, v_2, ..., QUERY, query_key

    The target is the value paired with query_key. The model can solve this by
    writing adjacent records k_i -> v_i and retrieving by query_key.
    """

    if n_pairs > n_keys:
        raise ValueError("n_pairs must be <= n_keys so keys are unique per sample")

    key_offset = 3
    value_offset = 3 + n_keys
    seq_len = 2 * n_pairs + 2

    tokens = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.empty(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(n_keys, device=device)[:n_pairs]
        values = torch.randint(0, n_values, (n_pairs,), device=device)
        query_index = int(torch.randint(0, n_pairs, ()).item())

        tokens[b, 0 : 2 * n_pairs : 2] = key_offset + keys
        tokens[b, 1 : 2 * n_pairs : 2] = value_offset + values
        tokens[b, 2 * n_pairs] = QUERY_TOKEN
        tokens[b, 2 * n_pairs + 1] = key_offset + keys[query_index]
        targets[b] = values[query_index]

    return Batch(tokens=tokens, targets=targets)


def make_induction_batch(
    *,
    batch_size: int,
    seq_len: int,
    symbol_vocab: int,
    device: torch.device,
) -> Batch:
    """Single-bigram induction recall.

    A random bigram a,b is inserted once into noise. The final query repeats a,
    and the target is b. Filler tokens avoid a, so the target association is
    unambiguous.
    """

    if seq_len < 6:
        raise ValueError("seq_len must be at least 6")
    if symbol_vocab < 3:
        raise ValueError("symbol_vocab must be at least 3")

    symbol_offset = 2
    context_len = seq_len - 2

    tokens = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.empty(batch_size, dtype=torch.long, device=device)

    for b_ix in range(batch_size):
        a = int(torch.randint(0, symbol_vocab, ()).item())
        b_raw = int(torch.randint(0, symbol_vocab - 1, ()).item())
        b = b_raw + int(b_raw >= a)

        filler = torch.randint(0, symbol_vocab - 1, (context_len,), device=device)
        filler = filler + (filler >= a).long()
        tokens[b_ix, :context_len] = symbol_offset + filler

        insert_at = int(torch.randint(0, context_len - 1, ()).item())
        tokens[b_ix, insert_at] = symbol_offset + a
        tokens[b_ix, insert_at + 1] = symbol_offset + b
        tokens[b_ix, context_len] = QUERY_TOKEN
        tokens[b_ix, context_len + 1] = symbol_offset + a
        targets[b_ix] = b

    return Batch(tokens=tokens, targets=targets)

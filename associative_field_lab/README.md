# Associative Field Lab

This folder is a self-contained prototype for the idea discussed in the chat:
replace a single recurrent state with a sparse, content-addressed memory field.

The model here is called `pcaf`: Parallel Causal Associative Field. It writes
local key-value records from adjacent tokens, then lets the final query token
retrieve a small candidate set by content hash and resolve it with learned
dot-product scoring.

## Why These Datasets

The included datasets are synthetic because they isolate the exact thing that
Mamba-style fixed-state recurrence struggles with: preserving token identity
over long context.

- `kv_recall`: random key-value pairs appear in the context, then the final
  query gives a key and the model must output its value.
- `induction`: a random bigram appears once in a noisy sequence, then the final
  query repeats the first token and the model must output the remembered next
  token.

Both are generated online, so there is no download step. Chance accuracy is
roughly `1 / num_classes`.

## Install

This repo checkout's local `pytorch_env` does not have PyTorch installed, but
the CVPR-level environment does:

```bash
source ~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
```

Or call it directly:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python
```

If you want a fresh environment instead, install PyTorch with:

```bash
python3 -m pip install -r associative_field_lab/requirements.txt
```

## No-Torch Smoke Test

This only checks the associative-memory data primitive and candidate coverage.

```bash
python3 associative_field_lab/smoke_no_torch.py --n-pairs 128 --n-keys 4096 --num-buckets 4096 --top-k 8
```

## Triton Kernel

The sparse GPU path is implemented in [pcaf/triton_bucket.py](pcaf/triton_bucket.py).
It builds bounded content-hash buckets:

```text
record token -> hash bucket -> up to K record indices
query token  -> same hash bucket -> sparse candidate read
```

Benchmark it:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/bench_kernel.py \
  --batch-size 128 \
  --records 4096 \
  --num-buckets 8192 \
  --top-k 8 \
  --device cuda
```

Checked result on this machine:

```text
triton_bucket_ms=0.4753
triton_sparse_read_ms=0.0204
dense_hash_topk_ms=0.5045
query_record_coverage=1.0000
```

Use `--candidate-mode triton_hash` to train through the sparse bucket path. The
fused sparse-read kernel is forward-only for inference/benchmarking; training
uses PyTorch gather/softmax after Triton candidate selection so autograd remains
correct.

## Train The Proposed Model

Key-value recall with hashed sparse retrieval:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train.py \
  --model pcaf \
  --task kv_recall \
  --n-pairs 64 \
  --n-keys 2048 \
  --n-values 2048 \
  --candidate-mode triton_hash \
  --num-buckets 4096 \
  --top-k 8 \
  --steps 2000 \
  --batch-size 128 \
  --eval-every 100 \
  --device auto
```

Induction-style associative recall:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train.py \
  --model pcaf \
  --task induction \
  --seq-len 256 \
  --symbol-vocab 2048 \
  --candidate-mode triton_hash \
  --num-buckets 4096 \
  --top-k 8 \
  --steps 2000 \
  --batch-size 128 \
  --eval-every 100 \
  --device auto
```

## Transformer Baseline

This is deliberately included because the real question is whether the memory
field can solve the same retrieval task without dense all-pairs attention.

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train.py \
  --model transformer \
  --task kv_recall \
  --n-pairs 64 \
  --n-keys 2048 \
  --n-values 2048 \
  --steps 2000 \
  --batch-size 128 \
  --eval-every 100 \
  --device auto
```

## Long-Context Generalization Check

Train on 64 pairs but evaluate on 256 pairs:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train.py \
  --model pcaf \
  --task kv_recall \
  --n-pairs 64 \
  --eval-n-pairs 256 \
  --n-keys 4096 \
  --n-values 2048 \
  --candidate-mode triton_hash \
  --num-buckets 8192 \
  --top-k 8 \
  --steps 2000 \
  --batch-size 128 \
  --eval-every 100 \
  --device auto
```

## Real Dataset: WikiText LM

The real benchmark script is [train_lm.py](train_lm.py). It downloads WikiText
through Hugging Face `datasets`, builds the same word vocabulary for every
model, and evaluates next-token prediction with perplexity and accuracy.

This is the right first real test because PCAF becomes a learned neural cache:
the current context queries previous memory records and reads their successors.
That directly tests whether content-addressed memory can compete with dense
causal attention and Mamba-style recurrence on real text.

Use `--model pcaf_context` for the stronger version. The first PCAF prototype
was mostly a one-token cache. `pcaf_context` adds the missing language-modeling
pieces:

- local causal convolution blocks for short-range composition
- sparse cache probabilities over retrieved successor tokens
- a learned gate between parametric LM logits and cache probabilities
- optional context hashes through `--context-order`

Best current PCAF variant:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model pcaf_context \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 256 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --candidate-mode triton_hash \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --d-model 256 \
  --d-hidden 1000 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/pcaf_context_order1_5k.jsonl \
  --device cuda
```

Learned semantic routing variant:

The default `triton_hash` route uses exact token/context hashes. This is fast,
but it is semantically silent: contexts such as `red car` and `scarlet
automobile` cannot retrieve each other unless their token hashes collide. Use
`semantic_hash` to route by learned hidden-state semantic buckets, or
`hybrid_semantic_hash` to combine exact-cache hits with learned semantic hits.

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model pcaf_context \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 2048 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --candidate-mode hybrid_semantic_hash \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --semantic-buckets 256 \
  --semantic-temperature 0.2 \
  --semantic-score-scale 1.0 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --d-model 256 \
  --d-hidden 1000 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/pcaf_hybrid_semantic_2048_5k.jsonl \
  --device cuda
```

Use `semantic_hash` alone for a stricter ablation:

```bash
--candidate-mode semantic_hash
```

Equal-parameter Transformer competitor:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model transformer \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 256 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --d-model 192 \
  --d-hidden 1024 \
  --layers 4 \
  --heads 4 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/transformer_equal_5k.jsonl \
  --device cuda
```

Sliding-window sparse attention competitor:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model local_transformer \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 1024 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --d-model 192 \
  --d-hidden 1024 \
  --layers 4 \
  --heads 4 \
  --attention-window 128 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/local_transformer_w128_5k.jsonl \
  --device cuda
```

Global+local sparse attention competitor:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model global_local_transformer \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 1024 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --d-model 192 \
  --d-hidden 1024 \
  --layers 4 \
  --heads 4 \
  --attention-window 128 \
  --global-tokens 16 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/global_local_transformer_w128_g16_5k.jsonl \
  --device cuda
```

Local-conv-only ablation:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model local_conv \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 1024 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --d-model 256 \
  --d-hidden 1000 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/local_conv_5k.jsonl \
  --device cuda
```

PCAF without cache ablation:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model pcaf_context \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 1024 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --candidate-mode triton_hash \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --d-model 256 \
  --d-hidden 1000 \
  --no-cache \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/pcaf_context_no_cache_5k.jsonl \
  --device cuda
```

PCAF without learned gate ablation:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model pcaf_context \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 1024 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --candidate-mode triton_hash \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --d-model 256 \
  --d-hidden 1000 \
  --no-gate \
  --fixed-cache-weight 0.5 \
  --seed 1234 \
  --train-sample-seed 10001 \
  --eval-sample-seed 20001 \
  --log-jsonl associative_field_lab/logs/pcaf_context_no_gate_5k.jsonl \
  --device cuda
```

## RTX 3060 Runner

For the full 1024/2048 long-context table with timing and ablations, run:

```bash
./associative_field_lab/scripts/run_3060_ablation.sh
```

The runner logs each model to:

```text
associative_field_lab/logs/rtx3060_<timestamp>/
```

It writes:

```text
*.stdout.log   raw training output
*.jsonl        metrics per eval point
*.time.log     /usr/bin/time -v output
summary.tsv    final/best PPL, tokens/sec, elapsed time, peak CUDA memory
status.tsv     OK/FAIL status for each run
```

Default batch sizes are chosen for a 12GB RTX 3060:

```text
BATCH_PCAF=64
BATCH_MAMBA=64
BATCH_ATTENTION=32
BATCH_DENSE_2048=8
```

Override them if needed:

```bash
STEPS=5000 BATCH_ATTENTION=16 BATCH_DENSE_2048=4 \
  ./associative_field_lab/scripts/run_3060_ablation.sh
```

Dense Transformer at `seq_len=2048` may still OOM on some RTX 3060 cards. The
script records the failure and continues to the next model.

Mamba competitor, using the installed `mamba_ssm` package:

```bash
~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python associative_field_lab/train_lm.py \
  --model mamba \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 256 \
  --batch-size 64 \
  --steps 5000 \
  --eval-every 250 \
  --eval-batches 50 \
  --d-model 256 \
  --d-hidden 1024 \
  --layers 4 \
  --d-state 16 \
  --device cuda
```

A verified fixed-seed 5000-step equal-parameter WikiText comparison produced:

```text
pcaf_context params=26,592,730 best_eval_ppl=228.71 final_eval_ppl=229.97
transformer  params=26,711,712 best_eval_ppl=259.06 final_eval_ppl=259.06
```

The old one-token PCAF reached `432.05` PPL at 1000 fixed-seed steps; the
context/gated-cache version reached `334.98` at the same point and `229.97`
after 5000 steps.

## Important Interpretation

This prototype does not prove exact `O(1)` parallel depth for full sequence
modeling. The Triton kernel gives the core sparse memory primitive: parallel
hash-bucket construction plus `K`-candidate reads. The research claim being
tested is narrower and practical:

> If the useful attention pattern is sparse or retrievable, a content-addressed
> memory field may match attention-like retrieval with `O(TK)` candidate work,
> where `K` is small, instead of dense `O(T^2)` attention.

Use `--candidate-mode full` to test the upper bound where every previous record
is available. Use `--candidate-mode triton_hash` to test the scalable GPU
approximation.

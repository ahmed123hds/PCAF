# JAX TPU PCAF

This is a pure-JAX/XLA implementation of `pcaf_context` and TPU-native
baselines. It does not use PyTorch, Triton, Flax, or Optax. The optimizer is a
small manual AdamW implementation, and the model is distributed with `jax.pmap`.

The TPU path differs from the CUDA path:

- use this script on TPU: `jax_tpu/train_pcaf_jax.py`
- do not use `--candidate-mode triton_hash`
- candidate selection is implemented with static JAX tensor ops and `lax.top_k`
- semantic routing is available through `--routing-mode semantic_hash` or
  `--routing-mode hybrid_semantic_hash`
- `--model` supports `local_conv`, `pcaf_no_gate`, `pcaf_semantic`,
  `pcaf_hybrid`, `pcaf_context`, `transformer_dense`, `linear_attention`,
  `local_transformer`, and `global_local_transformer`
- `linear_attention` uses chunked causal kernel attention via
  `--linear-chunk-size` instead of materializing full prefix tensors
- the model trains a single next-token target after each sampled context window,
  matching the current PyTorch experiment

## Install On TPU VM

Official JAX TPU install command:

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install datasets
```

## Single-Host TPU / One Process

```bash
python associative_field_lab/jax_tpu/train_pcaf_jax.py \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 2048 \
  --global-batch-size 256 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --d-model 256 \
  --d-hidden 1000 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --routing-mode token_hash \
  --log-jsonl associative_field_lab/logs/jax_tpu_pcaf_2048.jsonl
```

Learned semantic routing:

```bash
python associative_field_lab/jax_tpu/train_pcaf_jax.py \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len 2048 \
  --global-batch-size 256 \
  --steps 5000 \
  --eval-every 500 \
  --eval-batches 50 \
  --d-model 256 \
  --d-hidden 1000 \
  --local-layers 2 \
  --local-kernel-size 5 \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --routing-mode hybrid_semantic_hash \
  --semantic-buckets 256 \
  --semantic-temperature 0.2 \
  --semantic-score-scale 1.0 \
  --log-jsonl associative_field_lab/logs/jax_tpu_pcaf_hybrid_semantic_2048.jsonl
```

## Multi-Host TPU v4-32

On Cloud TPU multi-host VMs, run the same command on every worker and add
`--jax-distributed`. JAX documentation says `jax.distributed.initialize()` can
automatically detect Cloud TPU environments when called with no arguments.

Example using `gcloud` from your local machine:

```bash
TPU_NAME=your-tpu-name
ZONE=your-zone
PROJECT=your-project

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone "$ZONE" \
  --project "$PROJECT" \
  --worker=all \
  --command='cd ~/CG_Mamba && \
    python associative_field_lab/jax_tpu/train_pcaf_jax.py \
      --jax-distributed \
      --dataset Salesforce/wikitext \
      --dataset-config wikitext-2-raw-v1 \
      --max-vocab 20000 \
      --max-train-tokens 2000000 \
      --max-eval-tokens 200000 \
      --seq-len 2048 \
      --global-batch-size 256 \
      --steps 5000 \
      --eval-every 500 \
      --eval-batches 50 \
      --d-model 256 \
      --d-hidden 1000 \
      --local-layers 2 \
      --local-kernel-size 5 \
      --num-buckets 32768 \
      --top-k 16 \
      --context-order 1 \
      --routing-mode hybrid_semantic_hash \
      --semantic-buckets 256 \
      --semantic-temperature 0.2 \
      --log-jsonl associative_field_lab/logs/jax_tpu_pcaf_2048.jsonl'
```

For longer context:

```bash
--seq-len 4096 --global-batch-size 128
```

Semantic route through the helper script:

```bash
ROUTING_MODE=hybrid_semantic_hash SEQ_LEN=2048 GLOBAL_BATCH_SIZE=256 \
  bash associative_field_lab/scripts/run_tpu_v4_32_pcaf.sh
```

## ICLR Baseline Sweep

This is the main TPU execution script for the paper table. It runs PCAF
ablations and TPU-native attention baselines, including a causal linear
attention baseline, on WikiText-103 at 1024 and 2048 context length by default.

```bash
bash associative_field_lab/scripts/run_tpu_iclr_baselines.sh
```

Full ICLR package:

```bash
bash associative_field_lab/scripts/run_tpu_iclr_full_package.sh
```

This runs the 50k-step WikiText-103 comparison, a PG-19 transfer check, the
PCAF top-k/bucket/context/sequence-length ablation grid, and writes
`iclr_audit.md`. The PG-19 stage defaults to `emozilla/pg19`, a parquet-backed
mirror compatible with current Hugging Face `datasets`.

From your local machine, launch it on every v4-32 worker:

```bash
TPU_NAME=your-tpu-name
ZONE=your-zone
PROJECT=your-project

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone "$ZONE" \
  --project "$PROJECT" \
  --worker=all \
  --command='cd ~/PCAF && bash associative_field_lab/scripts/run_tpu_iclr_baselines.sh'
```

Quick preflight before the full 20k-step run:

```bash
DATASET_CONFIG=wikitext-2-raw-v1 \
MAX_VOCAB=20000 \
MAX_TRAIN_TOKENS=2000000 \
MAX_EVAL_TOKENS=200000 \
STEPS=100 \
EVAL_EVERY=50 \
EVAL_BATCHES=5 \
SEQ_LENS="1024" \
bash associative_field_lab/scripts/run_tpu_iclr_baselines.sh
```

For a quick compile/smoke run:

```bash
python associative_field_lab/jax_tpu/train_pcaf_jax.py \
  --seq-len 512 \
  --global-batch-size 32 \
  --steps 10 \
  --eval-every 5 \
  --eval-batches 2 \
  --d-model 128 \
  --d-hidden 256 \
  --max-vocab 5000 \
  --max-train-tokens 100000 \
  --max-eval-tokens 20000
```

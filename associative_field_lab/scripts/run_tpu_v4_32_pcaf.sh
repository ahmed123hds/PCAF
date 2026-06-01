#!/usr/bin/env bash
set -euo pipefail

# Run this script on every TPU VM worker. For Cloud TPU v4-32, launch it with:
#   gcloud compute tpus tpu-vm ssh TPU_NAME --worker=all --command='cd ~/CG_Mamba && bash associative_field_lab/scripts/run_tpu_v4_32_pcaf.sh'

PYTHON="${PYTHON:-python}"
SEQ_LEN="${SEQ_LEN:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
STEPS="${STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-50}"
LOG_JSONL="${LOG_JSONL:-associative_field_lab/logs/jax_tpu_pcaf_seq${SEQ_LEN}.jsonl}"
ROUTING_MODE="${ROUTING_MODE:-token_hash}"

"$PYTHON" associative_field_lab/jax_tpu/train_pcaf_jax.py \
  --jax-distributed \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --max-vocab 20000 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 200000 \
  --seq-len "$SEQ_LEN" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --steps "$STEPS" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --d-model 224 \
  --d-hidden 896 \
  --local-layers 14 \
  --local-kernel-size 5 \
  --num-buckets 32768 \
  --top-k 16 \
  --context-order 1 \
  --routing-mode "$ROUTING_MODE" \
  --semantic-buckets 256 \
  --semantic-temperature 0.2 \
  --semantic-score-scale 1.0 \
  --log-jsonl "$LOG_JSONL"

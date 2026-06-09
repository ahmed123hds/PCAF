#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN="$ROOT_DIR/jax_tpu/train_pcaf_jax.py"
PYTHON="${PYTHON:-python}"

LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/full_ar_42m_seq2048_semantic_buckets_k32_resume_$(date +%Y%m%d_%H%M%S)}"
SEMANTIC_BUCKETS_SWEEP="${SEMANTIC_BUCKETS_SWEEP:-512 1024 2048 4096}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

mkdir -p "$LOG_ROOT"

COMMON_ARGS=(
  --jax-distributed
  --loss-mode full_ar
  --dataset "${DATASET:-Salesforce/wikitext}"
  --dataset-config "${DATASET_CONFIG:-wikitext-103-raw-v1}"
  --text-field "${TEXT_FIELD:-text}"
  --max-vocab "${MAX_VOCAB:-32000}"
  --max-train-tokens "${MAX_TRAIN_TOKENS:-50000000}"
  --max-eval-tokens "${MAX_EVAL_TOKENS:-2000000}"
  --seq-len "${SEQ_LEN:-2048}"
  --steps "${STEPS:-25000}"
  --eval-every "${EVAL_EVERY:-5000}"
  --eval-batches "${EVAL_BATCHES:-20}"
  --lr "${LR:-0.000075}"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --seed "${SEED:-1234}"
  --train-sample-seed "${TRAIN_SAMPLE_SEED:-10001}"
  --eval-sample-seed "${EVAL_SAMPLE_SEED:-20001}"
  --d-model "${D_MODEL:-224}"
  --d-hidden "${D_HIDDEN:-896}"
  --local-layers "${LOCAL_LAYERS:-14}"
  --local-kernel-size "${LOCAL_KERNEL_SIZE:-5}"
  --context-order "${CONTEXT_ORDER:-1}"
  --num-buckets "${NUM_BUCKETS:-32768}"
  --top-k "${TOP_K:-32}"
  --model pcaf_semantic
  --global-batch-size "$GLOBAL_BATCH_SIZE"
)

for semantic_buckets in $SEMANTIC_BUCKETS_SWEEP; do
  echo "=== RUN pcaf_semantic seq=${SEQ_LEN:-2048} semantic_buckets=${semantic_buckets} batch=${GLOBAL_BATCH_SIZE} ==="
  "$PYTHON" "$TRAIN" \
    "${COMMON_ARGS[@]}" \
    --semantic-buckets "$semantic_buckets" \
    --log-jsonl "$LOG_ROOT/pcaf_semantic_seq${SEQ_LEN:-2048}_sem${semantic_buckets}_k${TOP_K:-32}_b${GLOBAL_BATCH_SIZE}_${STEPS:-25000}.jsonl"
  rc=$?
  echo "=== DONE pcaf_semantic seq=${SEQ_LEN:-2048} semantic_buckets=${semantic_buckets} rc=${rc} ==="
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
done

if [[ "${JAX_PROCESS_INDEX:-0}" = "0" ]]; then
  python "$ROOT_DIR/scripts/summarize_logs.py" "$LOG_ROOT" | tee "$LOG_ROOT/summary.tsv"
fi

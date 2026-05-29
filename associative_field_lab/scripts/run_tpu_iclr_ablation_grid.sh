#!/usr/bin/env bash
set -u

# Focused PCAF ablation grid for reviewer questions.
# Runs on every TPU worker. Defaults target WikiText-103 at 20k steps.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAIN="$ROOT_DIR/associative_field_lab/jax_tpu/train_pcaf_jax.py"
SUMMARY="$ROOT_DIR/associative_field_lab/scripts/summarize_logs.py"

DATASET="${DATASET:-Salesforce/wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-103-raw-v1}"
CACHE_DIR="${CACHE_DIR:-/tmp/hf_cache}"
MAX_VOCAB="${MAX_VOCAB:-32000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-50000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2000000}"

STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SEED="${SEED:-1234}"
TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-10001}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-20001}"
JAX_DISTRIBUTED="${JAX_DISTRIBUTED:-1}"

SEQ_LEN="${SEQ_LEN:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
PCAF_D_MODEL="${PCAF_D_MODEL:-256}"
PCAF_D_HIDDEN="${PCAF_D_HIDDEN:-1000}"
PCAF_LOCAL_LAYERS="${PCAF_LOCAL_LAYERS:-2}"
PCAF_LOCAL_KERNEL_SIZE="${PCAF_LOCAL_KERNEL_SIZE:-5}"
SEMANTIC_BUCKETS="${SEMANTIC_BUCKETS:-256}"
SEMANTIC_TEMPERATURE="${SEMANTIC_TEMPERATURE:-0.2}"
SEMANTIC_SCORE_SCALE="${SEMANTIC_SCORE_SCALE:-0.5}"

TOPK_VALUES="${TOPK_VALUES:-4 8 16 32}"
BUCKET_VALUES="${BUCKET_VALUES:-8192 32768 131072}"
CONTEXT_VALUES="${CONTEXT_VALUES:-1 2 3}"
SCALE_SEQ_LENS="${SCALE_SEQ_LENS:-512 1024 2048 4096}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/associative_field_lab/logs/tpu_iclr_ablation_${STAMP}}"
mkdir -p "$LOG_DIR"

DIST_FLAG=()
if [[ "$JAX_DISTRIBUTED" == "1" ]]; then
  DIST_FLAG=(--jax-distributed)
fi

batch_for_seq() {
  local seq_len="$1"
  if [[ "$seq_len" -ge 4096 ]]; then
    echo "${GLOBAL_BATCH_4096:-128}"
  else
    echo "$GLOBAL_BATCH_SIZE"
  fi
}

run_one() {
  local name="$1"
  local seq_len="$2"
  local batch_size="$3"
  shift 3

  local stdout_log="$LOG_DIR/${name}_seq${seq_len}.stdout.log"
  local jsonl_log="$LOG_DIR/${name}_seq${seq_len}.jsonl"
  local time_log="$LOG_DIR/${name}_seq${seq_len}.time.log"
  local status_log="$LOG_DIR/status.tsv"

  echo "=== RUN ${name} seq=${seq_len} batch=${batch_size} ==="
  echo -e "${name}\t${seq_len}\t${batch_size}\tSTART\t$(date --iso-8601=seconds)" >> "$status_log"

  /usr/bin/time -v -o "$time_log" \
    "$PYTHON" "$TRAIN" \
      "${DIST_FLAG[@]}" \
      --dataset "$DATASET" \
      --dataset-config "$DATASET_CONFIG" \
      --cache-dir "$CACHE_DIR" \
      --max-vocab "$MAX_VOCAB" \
      --max-train-tokens "$MAX_TRAIN_TOKENS" \
      --max-eval-tokens "$MAX_EVAL_TOKENS" \
      --seq-len "$seq_len" \
      --global-batch-size "$batch_size" \
      --steps "$STEPS" \
      --eval-every "$EVAL_EVERY" \
      --eval-batches "$EVAL_BATCHES" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --seed "$SEED" \
      --train-sample-seed "$TRAIN_SAMPLE_SEED" \
      --eval-sample-seed "$EVAL_SAMPLE_SEED" \
      --d-model "$PCAF_D_MODEL" \
      --d-hidden "$PCAF_D_HIDDEN" \
      --local-layers "$PCAF_LOCAL_LAYERS" \
      --local-kernel-size "$PCAF_LOCAL_KERNEL_SIZE" \
      --semantic-buckets "$SEMANTIC_BUCKETS" \
      --semantic-temperature "$SEMANTIC_TEMPERATURE" \
      --semantic-score-scale "$SEMANTIC_SCORE_SCALE" \
      --log-jsonl "$jsonl_log" \
      "$@" 2>&1 | tee "$stdout_log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    echo -e "${name}\t${seq_len}\t${batch_size}\tOK\t$(date --iso-8601=seconds)" >> "$status_log"
  else
    echo -e "${name}\t${seq_len}\t${batch_size}\tFAIL_${rc}\t$(date --iso-8601=seconds)" >> "$status_log"
  fi
  echo "=== DONE ${name} seq=${seq_len} rc=${rc} ==="
}

for top_k in $TOPK_VALUES; do
  run_one "pcaf_topk${top_k}" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" \
    --model pcaf_context \
    --routing-mode token_hash \
    --num-buckets 32768 \
    --top-k "$top_k" \
    --context-order 1
done

for buckets in $BUCKET_VALUES; do
  run_one "pcaf_buckets${buckets}" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" \
    --model pcaf_context \
    --routing-mode token_hash \
    --num-buckets "$buckets" \
    --top-k 16 \
    --context-order 1
done

for context_order in $CONTEXT_VALUES; do
  run_one "pcaf_context_order${context_order}" "$SEQ_LEN" "$GLOBAL_BATCH_SIZE" \
    --model pcaf_context \
    --routing-mode token_hash \
    --num-buckets 32768 \
    --top-k 16 \
    --context-order "$context_order"
done

for seq_len in $SCALE_SEQ_LENS; do
  run_one "pcaf_scale" "$seq_len" "$(batch_for_seq "$seq_len")" \
    --model pcaf_context \
    --routing-mode token_hash \
    --num-buckets 32768 \
    --top-k 16 \
    --context-order 1
done

if compgen -G "$LOG_DIR/*.jsonl" > /dev/null; then
  "$PYTHON" "$SUMMARY" "$LOG_DIR" | tee "$LOG_DIR/summary.tsv"
  echo "Logs written to: $LOG_DIR"
  echo "Summary: $LOG_DIR/summary.tsv"
else
  echo "No JSONL logs on this worker; summary is produced on process 0."
fi

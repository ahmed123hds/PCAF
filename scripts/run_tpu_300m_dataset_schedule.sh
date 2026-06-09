#!/usr/bin/env bash
set -u

# Dataset-specific 300M full autoregressive schedule for TPU v4-32.
#
# This script intentionally keeps WikiText-103 and PG-19 on different
# effective epoch budgets. WikiText-103 overfits quickly at 300M scale under
# full AR; PG-19 can safely run longer because the corpus is larger.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/run_tpu_iclr_baselines.sh"
TABLES="$ROOT_DIR/scripts/make_paper_ready_tables.py"
PYTHON="${PYTHON:-python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/logs/scale300m_dataset_scheduled_${STAMP}}"
FAILED_LOG_DIR="${FAILED_LOG_DIR:-${BASE_LOG_DIR}_failed}"
mkdir -p "$BASE_LOG_DIR" "$FAILED_LOG_DIR"

export LOSS_MODE="${LOSS_MODE:-full_ar}"
export SEQ_LENS="${SEQ_LENS:-2048}"
export SEED="${SEED:-1234}"
export TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-10001}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-20001}"
export MAX_VOCAB="${MAX_VOCAB:-32000}"
export MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2000000}"
export NUM_BUCKETS="${NUM_BUCKETS:-32768}"
export TOP_K="${TOP_K:-32}"
export CONTEXT_ORDER="${CONTEXT_ORDER:-1}"

# 300M balanced model shape.
export PCAF_D_MODEL="${PCAF_D_MODEL:-896}"
export PCAF_D_HIDDEN="${PCAF_D_HIDDEN:-3584}"
export PCAF_LOCAL_LAYERS="${PCAF_LOCAL_LAYERS:-24}"
export ATTN_D_MODEL="${ATTN_D_MODEL:-896}"
export ATTN_D_HIDDEN="${ATTN_D_HIDDEN:-3584}"
export ATTN_LAYERS="${ATTN_LAYERS:-16}"
export ATTN_HEADS="${ATTN_HEADS:-14}"
export ATTN_WINDOW="${ATTN_WINDOW:-128}"
export GLOBAL_TOKENS="${GLOBAL_TOKENS:-16}"

PCAF_MODELS="${PCAF_MODELS:-local_conv pcaf_semantic pcaf_context}"
ATTN_MODELS="${ATTN_MODELS:-transformer_dense local_transformer_w128}"

batch_candidates_from_start() {
  local start_batch="$1"
  case "$start_batch" in
    256) echo "256 128 64 32 16" ;;
    128) echo "128 64 32 16" ;;
    64) echo "64 32 16" ;;
    32) echo "32 16" ;;
    16) echo "16" ;;
    *) echo "$start_batch 16" ;;
  esac
}

run_with_batch_fallback() {
  local kind="$1"
  local name="$2"
  local dataset_name="$3"
  local dataset_config="$4"
  local text_field="$5"
  local train_tokens="$6"
  local steps="$7"
  local eval_every="$8"
  local lr="$9"
  local warmup_steps="${10}"
  local min_lr_ratio="${11}"
  local weight_decay="${12}"
  local models="${13}"
  local batch_var="${14}"
  local batches="${15}"

  local first=1
  local batch
  for batch in $batches; do
    local log_dir="$BASE_LOG_DIR/${name}_${kind}"
    if [ "$first" -ne 1 ]; then
      echo "=== Retrying $kind for $name at global batch $batch ==="
    fi
    first=0

    env \
    LOG_DIR="$log_dir" \
    DATASET="$dataset_name" \
    DATASET_CONFIG="$dataset_config" \
    TEXT_FIELD="$text_field" \
    MAX_TRAIN_TOKENS="$train_tokens" \
    STEPS="$steps" \
    EVAL_EVERY="$eval_every" \
    EVAL_BATCHES="${EVAL_BATCHES:-100}" \
    LR="$lr" \
    WARMUP_STEPS="$warmup_steps" \
    MIN_LR_RATIO="$min_lr_ratio" \
    WEIGHT_DECAY="$weight_decay" \
    RUN_MODELS="$models" \
    "$batch_var=$batch" \
    bash "$RUNNER"

    if [ ! -f "$log_dir/status.tsv" ] || ! grep -q "FAIL_" "$log_dir/status.tsv"; then
      echo "=== $kind for $name succeeded at global batch $batch ==="
      return 0
    fi

    echo "=== $kind for $name failed at global batch $batch ==="
    mv "$log_dir" "$FAILED_LOG_DIR/${name}_${kind}_b${batch}"
  done

  echo "=== $kind for $name failed for all candidate batches: $batches ==="
  return 1
}

run_dataset() {
  local name="$1"
  local dataset_name="$2"
  local dataset_config="$3"
  local text_field="$4"
  local train_tokens="$5"
  local steps="$6"
  local eval_every="$7"
  local lr="$8"
  local warmup_steps="$9"
  local min_lr_ratio="${10}"
  local weight_decay="${11}"

  echo "=== 300M DATASET: $name tokens=$train_tokens steps=$steps lr=$lr warmup=$warmup_steps min_lr_ratio=$min_lr_ratio wd=$weight_decay ==="

  local pcaf_start="${GLOBAL_BATCH_PCAF:-32}"
  local pcaf_batches="${PCAF_BATCH_CANDIDATES:-$(batch_candidates_from_start "$pcaf_start")}"
  run_with_batch_fallback \
    pcaf "$name" "$dataset_name" "$dataset_config" "$text_field" \
    "$train_tokens" "$steps" "$eval_every" "$lr" "$warmup_steps" \
    "$min_lr_ratio" "$weight_decay" "$PCAF_MODELS" GLOBAL_BATCH_PCAF "$pcaf_batches"

  local attention_start="${GLOBAL_BATCH_ATTENTION_2048:-32}"
  local attention_batches="${ATTN_BATCH_CANDIDATES:-$(batch_candidates_from_start "$attention_start")}"
  run_with_batch_fallback \
    attention "$name" "$dataset_name" "$dataset_config" "$text_field" \
    "$train_tokens" "$steps" "$eval_every" "$lr" "$warmup_steps" \
    "$min_lr_ratio" "$weight_decay" "$ATTN_MODELS" GLOBAL_BATCH_ATTENTION_2048 "$attention_batches"
}

# WikiText-103: 100M tokens, batch 32, seq 2048 => ~1,526 steps/epoch.
# 12k steps is ~7.9 effective epochs, matching the observed 10k-12k peak.
run_dataset \
  wikitext103 \
  Salesforce/wikitext \
  wikitext-103-raw-v1 \
  text \
  100000000 \
  12000 \
  1000 \
  0.000075 \
  1500 \
  0.02 \
  0.05

# PG-19: 200M tokens, batch 32, seq 2048 => ~3,052 steps/epoch.
# 30k steps is ~9.8 effective epochs; long enough to converge but not cycle
# through the corpus as aggressively as the failed 100k WikiText-103 run.
run_dataset \
  pg19 \
  emozilla/pg19 \
  none \
  book_text,text \
  200000000 \
  30000 \
  2500 \
  0.000075 \
  3000 \
  0.02 \
  0.03

if [ "${JAX_PROCESS_INDEX:-0}" = "0" ]; then
  "$PYTHON" "$TABLES" "$BASE_LOG_DIR"
  echo "300M dataset-scheduled results:"
  echo "$BASE_LOG_DIR/paper_ready_results.md"
  echo "$BASE_LOG_DIR/paper_ready_tables.tex"
  echo "Failed/fallback logs, if any:"
  echo "$FAILED_LOG_DIR"
else
  echo "Worker ${JAX_PROCESS_INDEX:-unknown}: paper tables are produced on process 0."
fi

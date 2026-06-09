#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/run_tpu_iclr_baselines.sh"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/attn_300m_tuned_wt103_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_ROOT"

RECIPES=(
  "lr5e-5_wd005_b095:0.00005:1000:0.02:0.05:0.95"
  "lr7p5e-5_wd01_b095:0.000075:1500:0.02:0.10:0.95"
  "lr1e-4_wd01_b095:0.0001:2000:0.05:0.10:0.95"
  "lr1p5e-4_wd01_b095:0.00015:2000:0.05:0.10:0.95"
)

for recipe in "${RECIPES[@]}"; do
  IFS=: read -r tag lr_value warmup min_lr weight_decay beta2 <<< "$recipe"
  echo "=== ATTENTION TUNE $tag ==="

  LOG_DIR="$LOG_ROOT/$tag" \
  LOSS_MODE=full_ar \
  DATASET=Salesforce/wikitext \
  DATASET_CONFIG=wikitext-103-raw-v1 \
  TEXT_FIELD=text \
  MAX_VOCAB=32000 \
  MAX_TRAIN_TOKENS=100000000 \
  MAX_EVAL_TOKENS=2000000 \
  SEQ_LENS=1024 \
  STEPS=12000 \
  EVAL_EVERY=1000 \
  EVAL_BATCHES=100 \
  LR="$lr_value" \
  WARMUP_STEPS="$warmup" \
  MIN_LR_RATIO="$min_lr" \
  WEIGHT_DECAY="$weight_decay" \
  ADAM_BETA1=0.9 \
  ADAM_BETA2="$beta2" \
  SEED=1234 \
  TRAIN_SAMPLE_SEED=10001 \
  EVAL_SAMPLE_SEED=20001 \
  ATTN_D_MODEL=896 \
  ATTN_D_HIDDEN=3584 \
  ATTN_LAYERS=16 \
  ATTN_HEADS=14 \
  ATTN_WINDOW=128 \
  GLOBAL_TOKENS=16 \
  GLOBAL_BATCH_ATTENTION_1024=32 \
  RUN_MODELS="transformer_dense local_transformer_w128" \
  bash "$RUNNER"
done

if [[ "${JAX_PROCESS_INDEX:-0}" = "0" ]]; then
  python "$ROOT_DIR/scripts/make_paper_ready_tables.py" "$LOG_ROOT"
  echo "Tuned attention baseline results:"
  echo "$LOG_ROOT/paper_ready_results.md"
  find "$LOG_ROOT" -name summary.tsv -print
fi

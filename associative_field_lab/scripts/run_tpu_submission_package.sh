#!/usr/bin/env bash
set -u

# Submission evidence package.
#
# Produces:
#   paper_ready_results.md
#   paper_ready_tables.tex
#   iclr_audit.md
#
# Run on every TPU worker via gcloud tpu-vm ssh --worker=all.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT_DIR/associative_field_lab/scripts/run_tpu_iclr_baselines.sh"
ABLATIONS="$ROOT_DIR/associative_field_lab/scripts/run_tpu_iclr_ablation_grid.sh"
AUDIT="$ROOT_DIR/associative_field_lab/scripts/audit_iclr_results.py"
TABLES="$ROOT_DIR/associative_field_lab/scripts/make_paper_ready_tables.py"
PYTHON="${PYTHON:-python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/associative_field_lab/logs/submission_package_${STAMP}}"
mkdir -p "$LOG_ROOT"

SEEDS="${SEEDS:-1234 2345 3456}"
HEADLINE_STEPS="${HEADLINE_STEPS:-50000}"
HEADLINE_EVAL_EVERY="${HEADLINE_EVAL_EVERY:-2500}"
HEADLINE_EVAL_BATCHES="${HEADLINE_EVAL_BATCHES:-100}"
HEADLINE_SEQ_LENS="${HEADLINE_SEQ_LENS:-2048}"
HEADLINE_MODELS="${HEADLINE_MODELS:-pcaf_context pcaf_semantic local_conv transformer_dense linear_attention global_local_transformer_w128_g16}"

ABLATION_STEPS="${ABLATION_STEPS:-20000}"
ABLATION_SEQ_LEN="${ABLATION_SEQ_LEN:-2048}"
ABLATION_TOPK_VALUES="${ABLATION_TOPK_VALUES:-4 8 16 32}"
ABLATION_BUCKET_VALUES="${ABLATION_BUCKET_VALUES:-8192 32768 131072}"
ABLATION_CONTEXT_VALUES="${ABLATION_CONTEXT_VALUES:-1 2 3}"

echo "=== SUBMISSION PACKAGE LOG ROOT: $LOG_ROOT ==="
echo "=== Headline seeds: $SEEDS ==="

seed_index=0
for seed in $SEEDS; do
  seed_index=$((seed_index + 1))
  train_seed=$((10001 + 1000 * seed_index))
  eval_seed=$((20001 + 1000 * seed_index))

  echo "=== WIKITEXT-103 HEADLINE seed=$seed ==="
  LOG_DIR="$LOG_ROOT/wikitext103_seed${seed}" \
  DATASET="Salesforce/wikitext" \
  DATASET_CONFIG="wikitext-103-raw-v1" \
  TEXT_FIELD="text" \
  MAX_VOCAB="${MAX_VOCAB:-32000}" \
  MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-50000000}" \
  MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2000000}" \
  SEQ_LENS="$HEADLINE_SEQ_LENS" \
  STEPS="$HEADLINE_STEPS" \
  EVAL_EVERY="$HEADLINE_EVAL_EVERY" \
  EVAL_BATCHES="$HEADLINE_EVAL_BATCHES" \
  RUN_MODELS="$HEADLINE_MODELS" \
  SEED="$seed" \
  TRAIN_SAMPLE_SEED="$train_seed" \
  EVAL_SAMPLE_SEED="$eval_seed" \
  bash "$RUNNER"

  echo "=== PG-19 HEADLINE seed=$seed ==="
  LOG_DIR="$LOG_ROOT/pg19_seed${seed}" \
  DATASET="${PG19_DATASET:-emozilla/pg19}" \
  DATASET_CONFIG="${PG19_DATASET_CONFIG:-none}" \
  TEXT_FIELD="${PG19_TEXT_FIELD:-text}" \
  MAX_VOCAB="${PG19_MAX_VOCAB:-32000}" \
  MAX_TRAIN_TOKENS="${PG19_MAX_TRAIN_TOKENS:-50000000}" \
  MAX_EVAL_TOKENS="${PG19_MAX_EVAL_TOKENS:-2000000}" \
  SEQ_LENS="$HEADLINE_SEQ_LENS" \
  STEPS="${PG19_STEPS:-$HEADLINE_STEPS}" \
  EVAL_EVERY="${PG19_EVAL_EVERY:-$HEADLINE_EVAL_EVERY}" \
  EVAL_BATCHES="${PG19_EVAL_BATCHES:-$HEADLINE_EVAL_BATCHES}" \
  RUN_MODELS="$HEADLINE_MODELS" \
  SEED="$seed" \
  TRAIN_SAMPLE_SEED="$train_seed" \
  EVAL_SAMPLE_SEED="$eval_seed" \
  bash "$RUNNER"
done

echo "=== WIKITEXT-103 TOP-K / BUCKET / CONTEXT ABLATIONS ==="
LOG_DIR="$LOG_ROOT/wikitext103_ablation_grid" \
SEQ_LEN="$ABLATION_SEQ_LEN" \
STEPS="$ABLATION_STEPS" \
EVAL_EVERY="${ABLATION_EVAL_EVERY:-1000}" \
EVAL_BATCHES="${ABLATION_EVAL_BATCHES:-100}" \
TOPK_VALUES="$ABLATION_TOPK_VALUES" \
BUCKET_VALUES="$ABLATION_BUCKET_VALUES" \
CONTEXT_VALUES="$ABLATION_CONTEXT_VALUES" \
bash "$ABLATIONS"

echo "=== AUDIT AND PAPER TABLES ==="
if [ "${JAX_PROCESS_INDEX:-0}" = "0" ]; then
  "$PYTHON" "$AUDIT" "$LOG_ROOT" | tee "$LOG_ROOT/iclr_audit.md"
  "$PYTHON" "$TABLES" "$LOG_ROOT"
  echo "Paper-ready Markdown: $LOG_ROOT/paper_ready_results.md"
  echo "Paper-ready LaTeX:    $LOG_ROOT/paper_ready_tables.tex"
else
  echo "Worker ${JAX_PROCESS_INDEX:-unknown}: summary files are produced on process 0."
fi

echo "Submission package logs: $LOG_ROOT"

#!/usr/bin/env bash
set -u

# Full ICLR evidence package.
#
# Run on every TPU VM worker:
#   gcloud compute tpus tpu-vm ssh TPU_NAME --zone ZONE --project PROJECT --worker=all \
#     --command='cd ~/models/PCAF && git pull && bash associative_field_lab/scripts/run_tpu_iclr_full_package.sh'
#
# This script intentionally separates the decisive long run from wider ablation
# runs. It produces log subdirectories under one root so the audit script can
# summarize the full package.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT_DIR/associative_field_lab/scripts/run_tpu_iclr_baselines.sh"
ABLATIONS="$ROOT_DIR/associative_field_lab/scripts/run_tpu_iclr_ablation_grid.sh"
AUDIT="$ROOT_DIR/associative_field_lab/scripts/audit_iclr_results.py"
PYTHON="${PYTHON:-python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/associative_field_lab/logs/tpu_iclr_full_${STAMP}}"
mkdir -p "$LOG_ROOT"

MAIN_STEPS="${MAIN_STEPS:-50000}"
MAIN_EVAL_EVERY="${MAIN_EVAL_EVERY:-2500}"
MAIN_EVAL_BATCHES="${MAIN_EVAL_BATCHES:-100}"
ABLATION_STEPS="${ABLATION_STEPS:-20000}"
PG19_STEPS="${PG19_STEPS:-20000}"

DECISIVE_MODELS="${DECISIVE_MODELS:-pcaf_context local_conv pcaf_semantic transformer_dense linear_attention global_local_transformer_w128_g16}"
PG19_MODELS="${PG19_MODELS:-pcaf_context local_conv pcaf_semantic linear_attention global_local_transformer_w128_g16}"

echo "=== ICLR FULL PACKAGE LOG ROOT: $LOG_ROOT ==="

echo "=== STAGE 1: WikiText-103 50k decisive comparison ==="
LOG_DIR="$LOG_ROOT/wikitext103_50k" \
DATASET="Salesforce/wikitext" \
DATASET_CONFIG="wikitext-103-raw-v1" \
MAX_VOCAB="${MAX_VOCAB:-32000}" \
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-50000000}" \
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2000000}" \
SEQ_LENS="${SEQ_LENS_MAIN:-1024 2048}" \
STEPS="$MAIN_STEPS" \
EVAL_EVERY="$MAIN_EVAL_EVERY" \
EVAL_BATCHES="$MAIN_EVAL_BATCHES" \
RUN_MODELS="$DECISIVE_MODELS" \
bash "$RUNNER"

echo "=== STAGE 2: PG-19 long-document transfer check ==="
LOG_DIR="$LOG_ROOT/pg19_20k" \
DATASET="${PG19_DATASET:-emozilla/pg19}" \
DATASET_CONFIG="${PG19_DATASET_CONFIG:-none}" \
TEXT_FIELD="${PG19_TEXT_FIELD:-text}" \
MAX_VOCAB="${PG19_MAX_VOCAB:-32000}" \
MAX_TRAIN_TOKENS="${PG19_MAX_TRAIN_TOKENS:-50000000}" \
MAX_EVAL_TOKENS="${PG19_MAX_EVAL_TOKENS:-2000000}" \
SEQ_LENS="${PG19_SEQ_LENS:-2048}" \
STEPS="$PG19_STEPS" \
EVAL_EVERY="${PG19_EVAL_EVERY:-1000}" \
EVAL_BATCHES="${PG19_EVAL_BATCHES:-100}" \
RUN_MODELS="$PG19_MODELS" \
bash "$RUNNER"

echo "=== STAGE 3: WikiText-103 PCAF routing and scale ablation grid ==="
LOG_DIR="$LOG_ROOT/wikitext103_ablation_grid" \
STEPS="$ABLATION_STEPS" \
EVAL_EVERY="${ABLATION_EVAL_EVERY:-1000}" \
EVAL_BATCHES="${ABLATION_EVAL_BATCHES:-100}" \
bash "$ABLATIONS"

echo "=== STAGE 4: Audit summary ==="
"$PYTHON" "$AUDIT" "$LOG_ROOT" | tee "$LOG_ROOT/iclr_audit.md"

echo "Full package logs: $LOG_ROOT"
echo "Audit: $LOG_ROOT/iclr_audit.md"

#!/usr/bin/env bash
set -u

# Controlled mechanism tests for the paper: key-value recall and induction.
# These isolate sparse associative retrieval from language-modeling shortcuts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAIN="$ROOT_DIR/train.py"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/mechanism_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  shift
  local stdout_log="$LOG_DIR/${name}.stdout.log"
  local time_log="$LOG_DIR/${name}.time.log"

  echo "=== RUN ${name} ==="
  /usr/bin/time -v -o "$time_log" \
    "$PYTHON" "$TRAIN" \
      --steps "$STEPS" \
      --eval-every "$EVAL_EVERY" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      "$@" 2>&1 | tee "$stdout_log"
  echo "=== DONE ${name} rc=${PIPESTATUS[0]} ==="
}

run_one "kv_pcaf_train64_eval256" \
  --model pcaf \
  --task kv_recall \
  --n-pairs 64 \
  --eval-n-pairs 256 \
  --n-keys 4096 \
  --n-values 2048 \
  --candidate-mode triton_hash \
  --num-buckets 8192 \
  --top-k 8

run_one "kv_transformer_train64_eval256" \
  --model transformer \
  --task kv_recall \
  --n-pairs 64 \
  --eval-n-pairs 256 \
  --n-keys 4096 \
  --n-values 2048

run_one "induction_pcaf_seq512" \
  --model pcaf \
  --task induction \
  --seq-len 512 \
  --symbol-vocab 4096 \
  --candidate-mode triton_hash \
  --num-buckets 8192 \
  --top-k 8

run_one "induction_transformer_seq512" \
  --model transformer \
  --task induction \
  --seq-len 512 \
  --symbol-vocab 4096

echo "Mechanism logs written to: $LOG_DIR"

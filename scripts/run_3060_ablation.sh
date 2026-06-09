#!/usr/bin/env bash
set -u

# RTX 3060 friendly benchmark runner.
#
# Defaults target a 12GB RTX 3060 with 64GB system RAM. Dense attention at
# seq_len=2048 may still OOM; this script keeps going and records failures.
#
# Override examples:
#   STEPS=1000 ./scripts/run_3060_ablation.sh
#   PYTHON=/path/to/python BATCH_PCAF=64 BATCH_ATTENTION=16 bash ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/python}"
TRAIN="$ROOT_DIR/train_lm.py"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/rtx3060_$(date +%Y%m%d_%H%M%S)}"

STEPS="${STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-50}"
MAX_VOCAB="${MAX_VOCAB:-20000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-2000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-200000}"

BATCH_PCAF="${BATCH_PCAF:-64}"
BATCH_MAMBA="${BATCH_MAMBA:-64}"
BATCH_ATTENTION="${BATCH_ATTENTION:-32}"
BATCH_DENSE_2048="${BATCH_DENSE_2048:-8}"

mkdir -p "$LOG_DIR"

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
  echo -e "${name}\t${seq_len}\tSTART\t$(date --iso-8601=seconds)" >> "$status_log"

  /usr/bin/time -v -o "$time_log" \
    "$PYTHON" "$TRAIN" \
      --dataset Salesforce/wikitext \
      --dataset-config wikitext-2-raw-v1 \
      --max-vocab "$MAX_VOCAB" \
      --max-train-tokens "$MAX_TRAIN_TOKENS" \
      --max-eval-tokens "$MAX_EVAL_TOKENS" \
      --seq-len "$seq_len" \
      --batch-size "$batch_size" \
      --steps "$STEPS" \
      --eval-every "$EVAL_EVERY" \
      --eval-batches "$EVAL_BATCHES" \
      --seed 1234 \
      --train-sample-seed 10001 \
      --eval-sample-seed 20001 \
      --log-jsonl "$jsonl_log" \
      --device cuda \
      "$@" 2>&1 | tee "$stdout_log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    echo -e "${name}\t${seq_len}\tOK\t$(date --iso-8601=seconds)" >> "$status_log"
  else
    echo -e "${name}\t${seq_len}\tFAIL_${rc}\t$(date --iso-8601=seconds)" >> "$status_log"
  fi
  echo "=== DONE ${name} seq=${seq_len} rc=${rc} ==="
}

for seq_len in 1024 2048; do
  run_one "pcaf_context" "$seq_len" "$BATCH_PCAF" \
    --model pcaf_context \
    --candidate-mode triton_hash \
    --num-buckets 32768 \
    --top-k 16 \
    --context-order 1 \
    --local-layers 2 \
    --local-kernel-size 5 \
    --d-model 256 \
    --d-hidden 1000

  run_one "mamba" "$seq_len" "$BATCH_MAMBA" \
    --model mamba \
    --d-model 384 \
    --d-hidden 512 \
    --layers 8 \
    --d-state 16

  dense_batch="$BATCH_ATTENTION"
  if [[ "$seq_len" -ge 2048 ]]; then
    dense_batch="$BATCH_DENSE_2048"
  fi

  run_one "transformer_dense" "$seq_len" "$dense_batch" \
    --model transformer \
    --d-model 192 \
    --d-hidden 1024 \
    --layers 4 \
    --heads 4

  run_one "local_transformer_w128" "$seq_len" "$BATCH_ATTENTION" \
    --model local_transformer \
    --d-model 192 \
    --d-hidden 1024 \
    --layers 4 \
    --heads 4 \
    --attention-window 128

  run_one "global_local_transformer_w128_g16" "$seq_len" "$BATCH_ATTENTION" \
    --model global_local_transformer \
    --d-model 192 \
    --d-hidden 1024 \
    --layers 4 \
    --heads 4 \
    --attention-window 128 \
    --global-tokens 16

  run_one "local_conv" "$seq_len" "$BATCH_PCAF" \
    --model local_conv \
    --d-model 256 \
    --d-hidden 1000 \
    --local-layers 2 \
    --local-kernel-size 5

  run_one "pcaf_no_cache" "$seq_len" "$BATCH_PCAF" \
    --model pcaf_context \
    --candidate-mode triton_hash \
    --num-buckets 32768 \
    --top-k 16 \
    --context-order 1 \
    --local-layers 2 \
    --local-kernel-size 5 \
    --d-model 256 \
    --d-hidden 1000 \
    --no-cache

  run_one "pcaf_no_gate" "$seq_len" "$BATCH_PCAF" \
    --model pcaf_context \
    --candidate-mode triton_hash \
    --num-buckets 32768 \
    --top-k 16 \
    --context-order 1 \
    --local-layers 2 \
    --local-kernel-size 5 \
    --d-model 256 \
    --d-hidden 1000 \
    --no-gate \
    --fixed-cache-weight 0.5
done

"$PYTHON" "$ROOT_DIR/scripts/summarize_logs.py" "$LOG_DIR" \
  | tee "$LOG_DIR/summary.tsv"

echo "Logs written to: $LOG_DIR"
echo "Summary: $LOG_DIR/summary.tsv"

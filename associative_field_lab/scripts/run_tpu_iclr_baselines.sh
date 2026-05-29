#!/usr/bin/env bash
set -u

# ICLR-grade TPU runner.
#
# Run this on every TPU VM worker. For Cloud TPU v4-32:
#   gcloud compute tpus tpu-vm ssh TPU_NAME --worker=all \
#     --command='cd ~/PCAF && bash associative_field_lab/scripts/run_tpu_iclr_baselines.sh'
#
# Defaults are intentionally stronger than the quick WikiText-2 smoke runs:
# WikiText-103, 20k steps, 1024/2048 contexts, PCAF ablations, and attention
# baselines. Override STEPS=5000 DATASET_CONFIG=wikitext-2-raw-v1 for a quick
# preflight before spending a full TPU run.
# Limit the matrix with RUN_MODELS, e.g.
#   RUN_MODELS="pcaf_context local_conv transformer_dense linear_attention"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAIN="$ROOT_DIR/associative_field_lab/jax_tpu/train_pcaf_jax.py"
SUMMARY="$ROOT_DIR/associative_field_lab/scripts/summarize_logs.py"

DATASET="${DATASET:-Salesforce/wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-103-raw-v1}"
TEXT_FIELD="${TEXT_FIELD:-text}"  # PG-19 uses 'book_text'; WikiText uses 'text'
CACHE_DIR="${CACHE_DIR:-/tmp/hf_cache}"
MAX_VOCAB="${MAX_VOCAB:-32000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-50000000}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-2000000}"

SEQ_LENS="${SEQ_LENS:-1024 2048}"
STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

GLOBAL_BATCH_PCAF="${GLOBAL_BATCH_PCAF:-256}"
GLOBAL_BATCH_ATTENTION_1024="${GLOBAL_BATCH_ATTENTION_1024:-128}"
GLOBAL_BATCH_ATTENTION_2048="${GLOBAL_BATCH_ATTENTION_2048:-64}"
GLOBAL_BATCH_ATTENTION_4096="${GLOBAL_BATCH_ATTENTION_4096:-16}"
GLOBAL_BATCH_LINEAR_1024="${GLOBAL_BATCH_LINEAR_1024:-128}"
GLOBAL_BATCH_LINEAR_2048="${GLOBAL_BATCH_LINEAR_2048:-64}"
GLOBAL_BATCH_LINEAR_4096="${GLOBAL_BATCH_LINEAR_4096:-16}"

PCAF_D_MODEL="${PCAF_D_MODEL:-256}"
PCAF_D_HIDDEN="${PCAF_D_HIDDEN:-1000}"
PCAF_LOCAL_LAYERS="${PCAF_LOCAL_LAYERS:-2}"
PCAF_LOCAL_KERNEL_SIZE="${PCAF_LOCAL_KERNEL_SIZE:-5}"
NUM_BUCKETS="${NUM_BUCKETS:-32768}"
TOP_K="${TOP_K:-16}"
CONTEXT_ORDER="${CONTEXT_ORDER:-1}"
SEMANTIC_BUCKETS="${SEMANTIC_BUCKETS:-256}"
SEMANTIC_TEMPERATURE="${SEMANTIC_TEMPERATURE:-0.2}"
SEMANTIC_SCORE_SCALE="${SEMANTIC_SCORE_SCALE:-0.5}"

ATTN_D_MODEL="${ATTN_D_MODEL:-192}"
ATTN_D_HIDDEN="${ATTN_D_HIDDEN:-1024}"
ATTN_LAYERS="${ATTN_LAYERS:-4}"
ATTN_HEADS="${ATTN_HEADS:-4}"
ATTN_WINDOW="${ATTN_WINDOW:-128}"
GLOBAL_TOKENS="${GLOBAL_TOKENS:-16}"

TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-10001}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-20001}"
SEED="${SEED:-1234}"
JAX_DISTRIBUTED="${JAX_DISTRIBUTED:-1}"
RUN_MODELS="${RUN_MODELS:-all}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/associative_field_lab/logs/tpu_iclr_${STAMP}}"
mkdir -p "$LOG_DIR"

DIST_FLAG=()
if [[ "$JAX_DISTRIBUTED" == "1" ]]; then
  DIST_FLAG=(--jax-distributed)
fi

common_args=(
  "${DIST_FLAG[@]}"
  --dataset "$DATASET"
  --dataset-config "$DATASET_CONFIG"
  --text-field "$TEXT_FIELD"
  --cache-dir "$CACHE_DIR"
  --max-vocab "$MAX_VOCAB"
  --max-train-tokens "$MAX_TRAIN_TOKENS"
  --max-eval-tokens "$MAX_EVAL_TOKENS"
  --steps "$STEPS"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --train-sample-seed "$TRAIN_SAMPLE_SEED"
  --eval-sample-seed "$EVAL_SAMPLE_SEED"
)

attention_batch_for_seq() {
  local seq_len="$1"
  if [[ "$seq_len" -ge 4096 ]]; then
    echo "$GLOBAL_BATCH_ATTENTION_4096"
  elif [[ "$seq_len" -ge 2048 ]]; then
    echo "$GLOBAL_BATCH_ATTENTION_2048"
  else
    echo "$GLOBAL_BATCH_ATTENTION_1024"
  fi
}

linear_batch_for_seq() {
  local seq_len="$1"
  if [[ "$seq_len" -ge 4096 ]]; then
    echo "$GLOBAL_BATCH_LINEAR_4096"
  elif [[ "$seq_len" -ge 2048 ]]; then
    echo "$GLOBAL_BATCH_LINEAR_2048"
  else
    echo "$GLOBAL_BATCH_LINEAR_1024"
  fi
}

want_model() {
  local name="$1"
  if [[ "$RUN_MODELS" == "all" ]]; then
    return 0
  fi
  for candidate in $RUN_MODELS; do
    if [[ "$candidate" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

run_one() {
  local name="$1"
  local seq_len="$2"
  local global_batch="$3"
  shift 3

  local stdout_log="$LOG_DIR/${name}_seq${seq_len}.stdout.log"
  local jsonl_log="$LOG_DIR/${name}_seq${seq_len}.jsonl"
  local time_log="$LOG_DIR/${name}_seq${seq_len}.time.log"
  local status_log="$LOG_DIR/status.tsv"

  echo "=== RUN ${name} seq=${seq_len} global_batch=${global_batch} ==="
  echo -e "${name}\t${seq_len}\t${global_batch}\tSTART\t$(date --iso-8601=seconds)" >> "$status_log"

  /usr/bin/time -v -o "$time_log" \
    "$PYTHON" "$TRAIN" \
      "${common_args[@]}" \
      --seq-len "$seq_len" \
      --global-batch-size "$global_batch" \
      --log-jsonl "$jsonl_log" \
      "$@" 2>&1 | tee "$stdout_log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    echo -e "${name}\t${seq_len}\t${global_batch}\tOK\t$(date --iso-8601=seconds)" >> "$status_log"
  else
    echo -e "${name}\t${seq_len}\t${global_batch}\tFAIL_${rc}\t$(date --iso-8601=seconds)" >> "$status_log"
  fi
  echo "=== DONE ${name} seq=${seq_len} rc=${rc} ==="
}

run_pcaf_family() {
  local seq_len="$1"

  if want_model "local_conv"; then
    run_one "local_conv" "$seq_len" "$GLOBAL_BATCH_PCAF" \
    --model local_conv \
    --d-model "$PCAF_D_MODEL" \
    --d-hidden "$PCAF_D_HIDDEN" \
    --local-layers "$PCAF_LOCAL_LAYERS" \
    --local-kernel-size "$PCAF_LOCAL_KERNEL_SIZE" \
    --num-buckets "$NUM_BUCKETS" \
    --top-k "$TOP_K" \
    --context-order "$CONTEXT_ORDER"
  fi

  if want_model "pcaf_no_gate"; then
    run_one "pcaf_no_gate" "$seq_len" "$GLOBAL_BATCH_PCAF" \
    --model pcaf_no_gate \
    --d-model "$PCAF_D_MODEL" \
    --d-hidden "$PCAF_D_HIDDEN" \
    --local-layers "$PCAF_LOCAL_LAYERS" \
    --local-kernel-size "$PCAF_LOCAL_KERNEL_SIZE" \
    --num-buckets "$NUM_BUCKETS" \
    --top-k "$TOP_K" \
    --context-order "$CONTEXT_ORDER" \
    --fixed-cache-weight 0.5
  fi

  if want_model "pcaf_semantic"; then
    run_one "pcaf_semantic" "$seq_len" "$GLOBAL_BATCH_PCAF" \
    --model pcaf_semantic \
    --d-model "$PCAF_D_MODEL" \
    --d-hidden "$PCAF_D_HIDDEN" \
    --local-layers "$PCAF_LOCAL_LAYERS" \
    --local-kernel-size "$PCAF_LOCAL_KERNEL_SIZE" \
    --num-buckets "$NUM_BUCKETS" \
    --top-k "$TOP_K" \
    --context-order "$CONTEXT_ORDER" \
    --semantic-buckets "$SEMANTIC_BUCKETS" \
    --semantic-temperature "$SEMANTIC_TEMPERATURE" \
    --semantic-score-scale "$SEMANTIC_SCORE_SCALE"
  fi

  if want_model "pcaf_hybrid"; then
    run_one "pcaf_hybrid" "$seq_len" "$GLOBAL_BATCH_PCAF" \
    --model pcaf_hybrid \
    --d-model "$PCAF_D_MODEL" \
    --d-hidden "$PCAF_D_HIDDEN" \
    --local-layers "$PCAF_LOCAL_LAYERS" \
    --local-kernel-size "$PCAF_LOCAL_KERNEL_SIZE" \
    --num-buckets "$NUM_BUCKETS" \
    --top-k "$TOP_K" \
    --context-order "$CONTEXT_ORDER" \
    --semantic-buckets "$SEMANTIC_BUCKETS" \
    --semantic-temperature "$SEMANTIC_TEMPERATURE" \
    --semantic-score-scale "$SEMANTIC_SCORE_SCALE"
  fi

  if want_model "pcaf_context"; then
    run_one "pcaf_context" "$seq_len" "$GLOBAL_BATCH_PCAF" \
    --model pcaf_context \
    --routing-mode token_hash \
    --d-model "$PCAF_D_MODEL" \
    --d-hidden "$PCAF_D_HIDDEN" \
    --local-layers "$PCAF_LOCAL_LAYERS" \
    --local-kernel-size "$PCAF_LOCAL_KERNEL_SIZE" \
    --num-buckets "$NUM_BUCKETS" \
    --top-k "$TOP_K" \
    --context-order "$CONTEXT_ORDER"
  fi
}

run_attention_family() {
  local seq_len="$1"
  local attention_batch
  local linear_batch
  attention_batch="$(attention_batch_for_seq "$seq_len")"
  linear_batch="$(linear_batch_for_seq "$seq_len")"

  if want_model "transformer_dense"; then
    run_one "transformer_dense" "$seq_len" "$attention_batch" \
    --model transformer_dense \
    --d-model "$ATTN_D_MODEL" \
    --d-hidden "$ATTN_D_HIDDEN" \
    --layers "$ATTN_LAYERS" \
    --heads "$ATTN_HEADS" \
    --attention-window "$ATTN_WINDOW" \
    --global-tokens "$GLOBAL_TOKENS"
  fi

  if want_model "linear_attention"; then
    run_one "linear_attention" "$seq_len" "$linear_batch" \
      --model linear_attention \
      --d-model "$ATTN_D_MODEL" \
      --d-hidden "$ATTN_D_HIDDEN" \
      --layers "$ATTN_LAYERS" \
      --heads "$ATTN_HEADS" \
      --attention-window "$ATTN_WINDOW" \
      --global-tokens "$GLOBAL_TOKENS"
  fi

  if want_model "local_transformer_w${ATTN_WINDOW}"; then
    run_one "local_transformer_w${ATTN_WINDOW}" "$seq_len" "$attention_batch" \
    --model local_transformer \
    --d-model "$ATTN_D_MODEL" \
    --d-hidden "$ATTN_D_HIDDEN" \
    --layers "$ATTN_LAYERS" \
    --heads "$ATTN_HEADS" \
    --attention-window "$ATTN_WINDOW" \
    --global-tokens "$GLOBAL_TOKENS"
  fi

  if want_model "global_local_transformer_w${ATTN_WINDOW}_g${GLOBAL_TOKENS}"; then
    run_one "global_local_transformer_w${ATTN_WINDOW}_g${GLOBAL_TOKENS}" "$seq_len" "$attention_batch" \
    --model global_local_transformer \
    --d-model "$ATTN_D_MODEL" \
    --d-hidden "$ATTN_D_HIDDEN" \
    --layers "$ATTN_LAYERS" \
    --heads "$ATTN_HEADS" \
    --attention-window "$ATTN_WINDOW" \
    --global-tokens "$GLOBAL_TOKENS"
  fi
}

for seq_len in $SEQ_LENS; do
  run_pcaf_family "$seq_len"
  run_attention_family "$seq_len"
done

if compgen -G "$LOG_DIR/*.jsonl" > /dev/null; then
  "$PYTHON" "$SUMMARY" "$LOG_DIR" | tee "$LOG_DIR/summary.tsv"
  echo "Logs written to: $LOG_DIR"
  echo "Summary: $LOG_DIR/summary.tsv"
else
  echo "No JSONL logs on this worker; summary is produced on process 0."
fi

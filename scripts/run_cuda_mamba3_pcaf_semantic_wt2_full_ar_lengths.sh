#!/usr/bin/env bash
set -u

# CUDA full-autoregressive WikiText-2 sweep on a 12GB RTX 3060.
#
# Models:
#   - mamba3: latest Mamba family baseline available in mamba_ssm
#   - pcaf_semantic: semantic associative cache routing
#
# Parameter budgets with max_vocab=20000:
#   mamba3        d=384, hidden=1536, layers=12, d_state=64  -> ~41.04M params
#   pcaf_semantic d=320, hidden=1280, local_layers=10        -> ~41.00M params
#
# Memory note for T=8192, batch=4:
#   A naive full-AR vocabulary head would allocate
#   4 * 8192 * 20000 = 655,360,000 fp32 logits ~= 2.44 GiB
#   before backward/workspace overhead. The long-context runs therefore use a
#   chunked vocabulary head. PCAF routing uses --routing-chunk-size 0, which
#   means "full routing chunk" rather than small Python-side routing chunks.

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-logs/cuda_wt2_full_ar_mamba3_pcaf_semantic_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_ROOT"

COMMON=(
  train_lm.py
  --loss-mode full_ar
  --dataset Salesforce/wikitext
  --dataset-config wikitext-2-raw-v1
  --max-vocab 20000
  --max-train-tokens 2000000
  --max-eval-tokens 200000
  --batch-size 4
  --steps 5000
  --eval-every 500
  --eval-batches 20
  --seed 1234
  --train-sample-seed 10001
  --eval-sample-seed 20001
  --weight-decay 0.01
  --device cuda
)

run_one() {
  local model="$1"
  local seq_len="$2"
  local lr="$3"
  local top_k="$4"
  local buckets="$5"
  local head_chunk="$6"

  local extra=()
  local log_name

  if [[ "$model" == "mamba3" ]]; then
    log_name="mamba3_seq${seq_len}_b4.jsonl"
    extra=(
      --model mamba3
      --d-model 384
      --d-hidden 1536
      --layers 12
      --d-state 64
    )
  elif [[ "$model" == "pcaf_semantic" ]]; then
    log_name="pcaf_semantic_seq${seq_len}_topk${top_k}_buckets${buckets}_b4.jsonl"
    extra=(
      --model pcaf_semantic
      --d-model 320
      --d-hidden 1280
      --local-layers 10
      --local-kernel-size 5
      --semantic-buckets 256
      --semantic-temperature 0.2
      --semantic-score-scale 1.0
      --num-buckets "$buckets"
      --top-k "$top_k"
      --context-order 1
      --routing-chunk-size 0
    )
  else
    echo "unknown model: $model" >&2
    return 2
  fi

  if [[ "$head_chunk" != "0" ]]; then
    extra+=(--full-ar-param-chunk-size "$head_chunk")
  fi

  echo "=== RUN $model seq=$seq_len batch=4 lr=$lr top_k=$top_k buckets=$buckets head_chunk=$head_chunk ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/time -v \
    "$PYTHON_BIN" "${COMMON[@]}" \
      --seq-len "$seq_len" \
      --lr "$lr" \
      "${extra[@]}" \
      --log-jsonl "$LOG_ROOT/$log_name"
  local rc=$?
  echo -e "${model}\t${seq_len}\t${rc}" >> "$LOG_ROOT/status.tsv"
  echo "=== DONE $model seq=$seq_len rc=$rc ==="
  return "$rc"
}

# LR schedule: keep the 1024-token setting at 3e-4 and scale by
# sqrt(1024 / T) as the number of tokens per step grows.
#
# Retrieval schedule:
#   T=1024:  K=16, buckets=32768
#   T=2048:  K=32, buckets=32768
#   T=4096:  K=32, buckets=65536
#   T=8192:  K=64, buckets=131072
#
# Vocabulary-head chunking:
#   Direct head for 1024/2048.
#   Chunked head for 4096/8192 to keep batch=4 practical on 12GB VRAM.
SEQS=(1024 2048 4096 8192)
LRS=(0.000300 0.000212 0.000150 0.000106)
TOPKS=(16 32 32 64)
BUCKETS=(32768 32768 65536 131072)
HEAD_CHUNKS=(0 0 256 256)

echo "Logs: $LOG_ROOT"
echo -e "model\tseq_len\trc" > "$LOG_ROOT/status.tsv"

for i in "${!SEQS[@]}"; do
  seq_len="${SEQS[$i]}"
  lr="${LRS[$i]}"
  top_k="${TOPKS[$i]}"
  buckets="${BUCKETS[$i]}"
  head_chunk="${HEAD_CHUNKS[$i]}"

  run_one mamba3 "$seq_len" "$lr" "$top_k" "$buckets" "$head_chunk" || true
  run_one pcaf_semantic "$seq_len" "$lr" "$top_k" "$buckets" "$head_chunk" || true
done

if [[ -f scripts/summarize_logs.py ]]; then
  "$PYTHON_BIN" scripts/summarize_logs.py "$LOG_ROOT" | tee "$LOG_ROOT/summary.tsv" || true
fi

echo "Logs written to: $LOG_ROOT"

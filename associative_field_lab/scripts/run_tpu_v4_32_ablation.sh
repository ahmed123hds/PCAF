#!/usr/bin/env bash
set -uo pipefail

# Pure-JAX TPU Ablation Sweep for PCAF.
# Run this on every TPU VM worker in the cluster:
#   gcloud compute tpus tpu-vm ssh node-v4-32 --worker=all --command="cd ~/models/PCAF && bash associative_field_lab/scripts/run_tpu_v4_32_ablation.sh"

PYTHON="${PYTHON:-python}"
LOG_DIR="associative_field_lab/logs/tpu_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local seq_len="$2"
  shift 2

  echo "=========================================================="
  echo "=== RUNNING ABLATION: ${name} (seq_len=${seq_len}) ==="
  echo "=========================================================="

  "$PYTHON" associative_field_lab/jax_tpu/train_pcaf_jax.py \
    --jax-distributed \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --cache-dir /tmp/hf_cache \
    --max-vocab 20000 \
    --max-train-tokens 2000000 \
    --max-eval-tokens 200000 \
    --seq-len "$seq_len" \
    --global-batch-size 256 \
    --steps 5000 \
    --eval-every 500 \
    --eval-batches 50 \
    --d-model 224 \
    --d-hidden 896 \
    --local-layers 14 \
    --local-kernel-size 5 \
    --num-buckets 32768 \
    --top-k 16 \
    --context-order 1 \
    --log-jsonl "${LOG_DIR}/${name}_seq${seq_len}.jsonl" \
    "$@" || echo "!!! ABLATION FAILED: ${name} seq=${seq_len} (exit $?) — continuing..."
}

# Sweep over sequence lengths and model ablations
for seq_len in 1024 2048; do
  # 1. Full PCAF-Context (Default Token Hash Routing)
  run_one "pcaf_context" "$seq_len" \
    --routing-mode token_hash

  # 2. PCAF Semantic-routing (using learned semantic codebook keys instead of discrete n-grams)
  run_one "pcaf_semantic" "$seq_len" \
    --routing-mode semantic_hash

  # 3. Local Conv Only (No associative cache; equivalent to no-cache baseline)
  run_one "local_conv" "$seq_len" \
    --no-cache

  # 4. PCAF No-Gate (Fixed 0.5/0.5 blend; isolates the gate's contribution)
  run_one "pcaf_no_gate" "$seq_len" \
    --no-gate \
    --fixed-cache-weight 0.5
done

echo "=== All JAX TPU ablations complete! ==="
echo "Logs saved in: ${LOG_DIR}"

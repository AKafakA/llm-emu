#!/bin/bash
# Pull the 4 minimal-paper cells from the GPU hosts to local VPS for paper
# writing. Cross-model + cross-hardware coverage:
#   M2  : Qwen3-8B  / RTX 8000  (apr27-m2-main-postfix)
#   Q14B: Qwen3-14B / RTX 8000  (apr27-m5-qwen3-14b-detkv5882, KV-override fix)
#   B1  : Qwen3-8B  / A40       (apr27-a40-qwen3-8b-igeos, --ignore-eos)
#   B2  : Qwen3-4B  / A40       (apr27-a40-qwen3-4b)
#
# Pulls per_rate_deltas.csv + bench JSONs (real_r*.json + emu_r*.json) +
# server logs + run.log + per-rate bench.log files. Skips the multi-GB
# profile pack (only needed for emu reruns, not paper text).
#
# Local destination: ~/Code/llm/vllm-emulator/results/paper_pack_apr27/
set -uo pipefail
DEST="$HOME/Code/llm/vllm-emulator/results/paper_pack_apr27"
mkdir -p "$DEST"
echo "[$(date -u +%T)] === Paper data-pack collection ===" | tee "$DEST/_collection.log"

pull_cell() {
    local SRC_HOST="$1"
    local SRC_PATH="$2"
    local LABEL="$3"
    echo "[$(date -u +%T)] Pulling $LABEL from $SRC_HOST:$SRC_PATH" | tee -a "$DEST/_collection.log"
    mkdir -p "$DEST/$LABEL"
    rsync -avz --no-perms --no-owner --no-group \
        --include='per_rate_deltas.csv' \
        --include='run.log' \
        --include='real_r*.json' --include='emu_r*.json' \
        --include='server_*.log' --include='bench_*.log' \
        --exclude='*' \
        "$SRC_HOST:$SRC_PATH/" "$DEST/$LABEL/" 2>&1 | tail -5 | tee -a "$DEST/_collection.log"
}

pull_cell personal_gpu_vm \
    "$HOME/Code/llm/vllm-emulator/results/apr27-m2-main-postfix" \
    "M2_qwen3-8b_rtx8000"

pull_cell personal_gpu_vm \
    "$HOME/Code/llm/vllm-emulator/results/apr27-m5-qwen3-14b-detkv5882" \
    "Q14B_qwen3-14b_rtx8000_kvoverride"

pull_cell vast48 \
    "/workspace/vllm-emulator/results/apr27-a40-qwen3-8b-igeos" \
    "B1_qwen3-8b_a40_igeos"

pull_cell vast48 \
    "/workspace/vllm-emulator/results/apr27-a40-qwen3-4b" \
    "B2_qwen3-4b_a40"

echo "" | tee -a "$DEST/_collection.log"
echo "[$(date -u +%T)] === DONE ===" | tee -a "$DEST/_collection.log"
echo "" | tee -a "$DEST/_collection.log"
echo "Summary table (per-rate deltas):" | tee -a "$DEST/_collection.log"
for d in "$DEST"/*/; do
    name=$(basename "$d")
    csv="$d/per_rate_deltas.csv"
    if [ -f "$csv" ]; then
        echo "" | tee -a "$DEST/_collection.log"
        echo "=== $name ===" | tee -a "$DEST/_collection.log"
        cat "$csv" | tee -a "$DEST/_collection.log"
    fi
done

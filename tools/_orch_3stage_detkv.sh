#!/bin/bash
# Deterministic-KV-pool 3-phase orchestrator.
# Phase A — Stage 2 only (profile capture, no override): yields server boots
#           whose `GPU KV cache size: N tokens` lines record vllm's profile_run
#           variance.
# Phase B — calibrate: tools/_calibrate_kv_min.py picks LOWEST observed token
#           count and emits num_gpu_blocks (= tokens / 16).
# Phase C — Stage 1 + Stage 3: real bench AND emu validate run with
#           --num-gpu-blocks-override=N. Identical, deterministic KV pool
#           across both sides; eliminates profile_run variance.
#
# Result: real_r* and emu_r* are bit-comparable on KV behavior.
#
# Usage (env-driven):
#   CELL_TAG, BENCH_MODEL                   — required
#   HW_PREFIX                               — RTX-8000 / A40 / A10
#   STUB_DIR                                — emu CUDA stubs path
#   EXTRA_SERVER_ARGS, BENCH_BURSTINESS,
#   BENCH_TEMPERATURE, BENCH_IGNORE_EOS,
#   BENCH_TIMEOUT, REPO_ROOT                — passed through unchanged
set -uo pipefail
ulimit -n 65536 2>/dev/null || true

REPO_ROOT="${REPO_ROOT:-$HOME/Code/llm/vllm-emulator}"
cd "$REPO_ROOT"
export REPO_ROOT
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

: "${CELL_TAG:?required}"
: "${BENCH_MODEL:?required}"
: "${HW_PREFIX:?required}"

LOG="/tmp/${CELL_TAG}_orch_detkv.log"
echo "[$(date -u +%T)] === detkv orchestrator: cell=$CELL_TAG ===" > "$LOG"

CELL_DIR="$REPO_ROOT/results/${CELL_TAG}"
PROFILE_DIR="$REPO_ROOT/results/${HW_PREFIX}-adaptive-${CELL_TAG}"
PROFILE_PACK="${PROFILE_DIR}/serving-full.json"

if pgrep -af "vllm.entrypoints|EngineCore|bench serve|run_one_full" 2>/dev/null \
    | grep -v "$$" | grep -v "_orch_3stage_detkv" | head -1 | grep -q .; then
    echo "FATAL: existing process detected" >> "$LOG"
    exit 1
fi
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 5

#### Phase A: Stage 2 only (profile capture, no override) ####
echo "[$(date -u +%T)] PHASE A: profile capture (no KV override)" >> "$LOG"
env CELL_TAG="$CELL_TAG" \
    BENCH_MODEL="$BENCH_MODEL" \
    EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}" \
    BENCH_BURSTINESS="${BENCH_BURSTINESS:-}" \
    BENCH_TEMPERATURE="${BENCH_TEMPERATURE:-}" \
    BENCH_IGNORE_EOS="${BENCH_IGNORE_EOS:-}" \
    BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}" \
    REPO_ROOT="$REPO_ROOT" \
    STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}" \
    SKIP_STAGE1=1 SKIP_STAGE3=1 \
    bash "$REPO_ROOT/tools/run_one_full_sharegpt_cell.sh" \
    >> "$LOG" 2>&1

if [ ! -f "$PROFILE_PACK" ]; then
    echo "FATAL: PHASE A no profile pack at $PROFILE_PACK" >> "$LOG"
    touch "/tmp/${CELL_TAG}_orch_detkv.done"
    exit 2
fi

#### Phase B: calibrate min KV pool ####
echo "[$(date -u +%T)] PHASE B: calibrate min KV pool" >> "$LOG"
KV_NUM_BLOCKS=$("$REPO_ROOT/.venv/bin/python3" \
    "$REPO_ROOT/tools/_calibrate_kv_min.py" "$PROFILE_DIR" 2>>"$LOG")
if [ -z "$KV_NUM_BLOCKS" ] || [ "$KV_NUM_BLOCKS" -lt 100 ]; then
    echo "FATAL: PHASE B calibrate produced invalid value: $KV_NUM_BLOCKS" >> "$LOG"
    touch "/tmp/${CELL_TAG}_orch_detkv.done"
    exit 3
fi
echo "[$(date -u +%T)] KV_NUM_GPU_BLOCKS_OVERRIDE=$KV_NUM_BLOCKS" >> "$LOG"

#### Phase C: Stage 1 (real) + Stage 3 (emu) with override ####
echo "[$(date -u +%T)] PHASE C: Stage 1 real + Stage 3 emu, pinned to $KV_NUM_BLOCKS blocks" >> "$LOG"
env CELL_TAG="$CELL_TAG" \
    BENCH_MODEL="$BENCH_MODEL" \
    EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}" \
    BENCH_BURSTINESS="${BENCH_BURSTINESS:-}" \
    BENCH_TEMPERATURE="${BENCH_TEMPERATURE:-}" \
    BENCH_IGNORE_EOS="${BENCH_IGNORE_EOS:-}" \
    BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}" \
    REPO_ROOT="$REPO_ROOT" \
    STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}" \
    KV_NUM_GPU_BLOCKS_OVERRIDE="$KV_NUM_BLOCKS" \
    REUSE_PROFILE="$PROFILE_PACK" \
    SKIP_STAGE2=1 \
    bash "$REPO_ROOT/tools/run_one_full_sharegpt_cell.sh" \
    >> "$LOG" 2>&1

# Patch the profile pack so emu's cuda_mock returns matching available_kv_cache_bytes
# (cosmetic — override flag is what vllm actually uses)
REAL_LOG="$CELL_DIR/server_real_r2.log"
[ -f "$PROFILE_PACK" ] && [ -f "$REAL_LOG" ] && \
    "$REPO_ROOT/.venv/bin/python3" \
    "$REPO_ROOT/tools/_patch_profile_pack_kv.py" \
    "$PROFILE_PACK" "$REAL_LOG" >> "$LOG" 2>&1 || true

echo "[$(date -u +%T)] === DONE ===" >> "$LOG"
echo "" >> "$LOG"
echo "=== per_rate_deltas.csv ===" >> "$LOG"
cat "$CELL_DIR/per_rate_deltas.csv" 2>/dev/null >> "$LOG" || echo "(no deltas)" >> "$LOG"
touch "/tmp/${CELL_TAG}_orch_detkv.done"

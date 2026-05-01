#!/bin/bash
# Adaptive multi-rate profile capture — Stage 2 of run_one_full_sharegpt_cell.
#
# Sweeps a range of request rates, captures step-cycle traces with
# VLLM_EMULATOR_TRACE_STEP_CYCLE=1, and builds the 2D (tt, conc) profile
# pack via build_serving_profile_filtered.py.
#
# Coverage targets (per round, density chosen to populate both low-conc
# and saturation buckets evenly):
#   r=2,4   : 3000 prompts each — fixes Apr 25 r=4 sparse-bucket regression
#   r=8,12,16,24 : 2500 prompts each — mid-conc
#   r=32,48 : 3000 prompts each — saturation
#   r=inf   : 4000 prompts — deep saturation
# Total ~30k prompts/round × ROUNDS=2 = ~61k samples.
#
# Server config (parity-critical): DEFAULT max-num-seqs everywhere.
# EXTRA_SERVER_ARGS flows through (e.g. --no-prefix-caching, --attention-backend).
# DO NOT add --max-num-seqs here (run_one_full_sharegpt_cell.sh asserts).
#
# Required env (from caller):
#   TAG               — cell tag, used to pick output dir
#   BENCH_MODEL, BENCH_PORT, BENCH_MAX_MODEL_LEN  (set by _bench_common.sh)
# Optional env:
#   ROUNDS           — default 2
#   SHAREGPT         — default ./results/sharegpt_filtered_256_128.json
#   EXTRA_SERVER_ARGS — appended to api_server (default "")

set -uo pipefail
ulimit -n 65536 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="${REPO_ROOT:-$HOME/Code/llm/vllm-emulator}"
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"
source "$(dirname "$0")/_bench_common.sh"

TAG="${TAG:?required}"
ROUNDS="${ROUNDS:-2}"
SHAREGPT="${SHAREGPT:-./results/sharegpt_filtered_256_128.json}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

if echo "$EXTRA_SERVER_ARGS" | grep -qE '\-\-max-num-seqs|\-\-max_num_seqs'; then
    echo "FATAL: EXTRA_SERVER_ARGS contains --max-num-seqs. Profile must be" >&2
    echo "       captured at DEFAULT max-num-seqs to match bench config." >&2
    exit 1
fi

# Pick HW prefix from first available GPU model (best-effort).
HW="UNKNOWN"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "$GPU_NAME" in
    *RTX*8000*) HW="RTX-8000" ;;
    *A100*)     HW="A100" ;;
    *A800*)     HW="A800" ;;
    *A40*)      HW="A40" ;;
    *A10*)      HW="A10" ;;
    *L40S*)     HW="L40S" ;;
    *H200*)     HW="H200" ;;
    *H100*)     HW="H100" ;;
    *)          HW="UNKNOWN" ;;
esac

OUT_DIR="./results/${HW}-adaptive-${TAG}"
TRACE_DIR="$OUT_DIR/per_rate_traces"
FINAL_TRACE="$OUT_DIR/step_cycle_trace.jsonl"
PROFILE="$OUT_DIR/serving-full.json"
LOG="$OUT_DIR/run.log"
mkdir -p "$TRACE_DIR" "$OUT_DIR/logs"
rm -f "$FINAL_TRACE"

if [ ! -f "$SHAREGPT" ]; then
    echo "FATAL: ShareGPT dataset missing at $SHAREGPT" >&2
    exit 1
fi

# Balanced rate-and-prompts: dense at low conc AND saturation.
RATES_AND_PROMPTS="1:2000 2:2000 4:2000 8:2000 12:2000 16:2000 20:2000 24:2500 28:3000 32:3000 40:3000 48:3000 inf:4000"

echo "=== adaptive_profile_capture start $(date -u) ===" > "$LOG"
echo "TAG=$TAG  HW=$HW  ROUNDS=$ROUNDS  EXTRA=$EXTRA_SERVER_ARGS" >> "$LOG"
echo "Rate list: $RATES_AND_PROMPTS" >> "$LOG"

# StepCycleTracer hardcodes /tmp/emulator_step_trace.jsonl (env var override
# not implemented). Before this capture starts:
#   1. If /tmp has data, archive it to a tagged backup (recovery path in case
#      the previous cell's result dir corrupted)
#   2. Then clear /tmp so this capture's trace file contains ONLY this cell's
#      data — no contamination from prior orchestrator runs.
if [ -s /tmp/emulator_step_trace.jsonl ]; then
    BACKUP="/tmp/emulator_step_trace.pre-${TAG}.$(date -u +%Y%m%d-%H%M%S).jsonl"
    mv /tmp/emulator_step_trace.jsonl "$BACKUP" 2>/dev/null
    echo "[$(date -u +%T)] archived prior /tmp trace to $BACKUP ($(stat -c%s "$BACKUP" 2>/dev/null) bytes)" >> "$LOG"
fi
rm -f /tmp/emulator_step_trace.jsonl
echo "[$(date -u +%T)] cleared /tmp/emulator_step_trace.jsonl (StepCycleTracer hardcoded path)" >> "$LOG"

for ROUND in $(seq 1 "$ROUNDS"); do
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND / $ROUNDS at $(date -u +%T) ===" >> "$LOG"
    common_cleanup
    sleep 2

    env VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
        VLLM_EMULATOR_STEP_CYCLE_TRACE_PATH="$TRACE_DIR/round${ROUND}.jsonl" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "${BENCH_MODEL:-Qwen/Qwen3-8B}" \
            --max-model-len "${BENCH_MAX_MODEL_LEN:-4096}" \
            --port "$BENCH_PORT" --trust-remote-code \
            $EXTRA_SERVER_ARGS \
            > "$OUT_DIR/logs/server_round${ROUND}.log" 2>&1 &
    if ! common_wait_server; then
        echo "    server FAIL round $ROUND" >> "$LOG"
        continue
    fi
    common_warmup

    # BENCH_TEMPERATURE=0 enables greedy sampling for deterministic outputs;
    # required when emu validate also runs greedy so the profile-pack step
    # times reflect greedy-regime KV occupancy and per-step latencies.
    TEMP_FLAG=""
    if [ -n "${BENCH_TEMPERATURE:-}" ]; then
        TEMP_FLAG="--temperature ${BENCH_TEMPERATURE}"
    fi
    IGNORE_EOS_FLAG=""
    if [ "${BENCH_IGNORE_EOS:-}" = "1" ]; then
        IGNORE_EOS_FLAG="--ignore-eos"
    fi
    # BENCH_BURSTINESS: pass to bench's --burstiness so Stage 2 captures the
    # same arrival-distribution regime that Stage 3 emu validates against.
    # Default unset (Poisson γ=1.0). For burstiness ablation cells, set 0.25
    # or whatever the validate stage uses.
    BURST_FLAG=""
    if [ -n "${BENCH_BURSTINESS:-}" ]; then
        BURST_FLAG="--burstiness ${BENCH_BURSTINESS}"
    fi
    # BENCH_TIMEOUT: per-rate bench wall-clock cap. Default 1800s. Bump to
    # 3600s for slow attention backends (TRITON_ATTN ~25-30% slower per step
    # than FA → 1800s timeout caused 28% sample loss on apr26-triton).
    BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
    for entry in $RATES_AND_PROMPTS; do
        RATE="${entry%%:*}"
        PROMPTS="${entry##*:}"
        echo "[$(date -u +%T)] round=$ROUND rate=$RATE prompts=$PROMPTS temp=${BENCH_TEMPERATURE:-default} ignore_eos=${BENCH_IGNORE_EOS:-0} burstiness=${BENCH_BURSTINESS:-1.0} timeout=${BENCH_TIMEOUT}s" >> "$LOG"
        if [ "$RATE" = "inf" ]; then
            timeout "$BENCH_TIMEOUT" python3 -m vllm.entrypoints.cli.main bench serve \
                --model "${BENCH_MODEL:-Qwen/Qwen3-8B}" \
                --base-url "http://localhost:$BENCH_PORT" \
                --dataset-name sharegpt --dataset-path "$SHAREGPT" \
                --num-prompts "$PROMPTS" --seed "$ROUND" $TEMP_FLAG $IGNORE_EOS_FLAG $BURST_FLAG \
                > "$TRACE_DIR/round${ROUND}_r${RATE}.log" 2>&1 || true
        else
            timeout "$BENCH_TIMEOUT" python3 -m vllm.entrypoints.cli.main bench serve \
                --model "${BENCH_MODEL:-Qwen/Qwen3-8B}" \
                --base-url "http://localhost:$BENCH_PORT" \
                --dataset-name sharegpt --dataset-path "$SHAREGPT" \
                --num-prompts "$PROMPTS" --request-rate "$RATE" --seed "$ROUND" $TEMP_FLAG $IGNORE_EOS_FLAG $BURST_FLAG \
                > "$TRACE_DIR/round${ROUND}_r${RATE}.log" 2>&1 || true
        fi
    done
    common_cleanup
    sleep 3
done

# Concatenate per-round traces and build the profile pack.
# StepCycleTracer writes to hardcoded /tmp/emulator_step_trace.jsonl
# (overrides via VLLM_EMULATOR_STEP_CYCLE_TRACE_PATH are not honored —
# bug in trace_profiler.py StepCycleTracer.__init__). Until that's fixed,
# read /tmp directly. The pre-capture rm above ensures freshness.
if [ -s /tmp/emulator_step_trace.jsonl ]; then
    cp /tmp/emulator_step_trace.jsonl "$FINAL_TRACE"
fi
# Fallback to legacy concat if /tmp file missing.
if [ ! -s "$FINAL_TRACE" ]; then
    cat "$TRACE_DIR"/round*.jsonl > "$FINAL_TRACE" 2>/dev/null
fi

# Sanity guard: FINAL_TRACE must be non-empty AND contain at least one
# valid header line. If absent or empty, abort the build step — operator
# can recover from /tmp manually if it still exists.
if [ ! -s "$FINAL_TRACE" ]; then
    echo "[$(date -u +%T)] FATAL: $FINAL_TRACE is empty — capture failed to produce trace data" >> "$LOG"
    echo "  /tmp/emulator_step_trace.jsonl size: $(stat -c%s /tmp/emulator_step_trace.jsonl 2>/dev/null || echo 'absent')" >> "$LOG"
    exit 2
fi
HEADER_COUNT=$(grep -c '"_header"' "$FINAL_TRACE" 2>/dev/null || echo 0)
if [ "$HEADER_COUNT" -eq 0 ]; then
    echo "[$(date -u +%T)] FATAL: $FINAL_TRACE has no _header line — trace is malformed" >> "$LOG"
    exit 2
fi
echo "[$(date -u +%T)] FINAL_TRACE OK: $(wc -l < "$FINAL_TRACE") lines, $HEADER_COUNT headers" >> "$LOG"
echo "" >> "$LOG"
echo "[$(date -u +%T)] building profile pack from $FINAL_TRACE" >> "$LOG"

python3 vllm_emulator/profile/build_serving_profile_filtered.py \
    "$FINAL_TRACE" "$PROFILE" \
    --gpu-model "$HW" \
    --model-name "${BENCH_MODEL:-Qwen/Qwen3-8B}" \
    >> "$LOG" 2>&1 || true

echo "=== adaptive_profile_capture DONE $(date -u) ===" >> "$LOG"

#!/bin/bash
# Shared bench/profile helpers. Source this from every chain script so
# the warmup command is byte-identical everywhere; drift between profile
# capture and bench validation creates systematic bias in emu predictions.
#
# Usage:
#   source "$(dirname "$0")/_bench_common.sh"
#   Relies on env vars BENCH_PORT, BENCH_MODEL being set by the caller
#   (defaults provided here if unset).

# CRITICAL: bench client opens 1 socket per concurrent request. At r>=8
# with 2000 prompts under saturation, default 1024 fd limit triggers
# `aiohttp.ClientConnectorError: Too many open files` and silently marks
# tail requests as Failed → biased mean_ttft_ms / mean_e2el_ms. Always
# raise this BEFORE any bench command. See feedback_bench_ulimit.md.
ulimit -n 65536 2>/dev/null || true

BENCH_PORT="${BENCH_PORT:-8100}"
BENCH_MODEL="${BENCH_MODEL:-Qwen/Qwen3-8B}"
BENCH_MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN:-4096}"

# The ONE warmup used by both profiler and validator. Keep this identical;
# do NOT branch per-caller. Changes here invalidate profile↔bench state-matching.
BENCH_WARMUP_INPUT="${BENCH_WARMUP_INPUT:-256}"
BENCH_WARMUP_OUTPUT="${BENCH_WARMUP_OUTPUT:-128}"
BENCH_WARMUP_PROMPTS="${BENCH_WARMUP_PROMPTS:-100}"
BENCH_WARMUP_RATE="${BENCH_WARMUP_RATE:-4}"
BENCH_WARMUP_SEED="${BENCH_WARMUP_SEED:-0}"

common_cleanup() {
    pkill -TERM -f "vllm.entrypoints" 2>/dev/null
    sleep 3
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "bench serve" 2>/dev/null
    fuser ${BENCH_PORT}/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
}

common_preflight() {
    # Abort if any vllm proc survived cleanup — a parallel run would
    # poison measurements (see feedback_bench_preflight memory).
    if pgrep -f "VLLM::EngineCore|bench serve|vllm.entrypoints" > /dev/null; then
        echo "FATAL: vllm procs survived cleanup" >&2
        pgrep -af "VLLM::EngineCore|bench serve|vllm.entrypoints" >&2
        return 1
    fi
    return 0
}

common_wait_server() {
    for i in $(seq 1 300); do
        curl -s "http://localhost:${BENCH_PORT}/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# Matched warmup for profile capture AND bench validation.
# Byte-identical args in both contexts is a hard invariant.
common_warmup() {
    python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" \
        --base-url "http://localhost:${BENCH_PORT}" \
        --dataset-name random \
        --random-input-len "$BENCH_WARMUP_INPUT" \
        --random-output-len "$BENCH_WARMUP_OUTPUT" \
        --num-prompts "$BENCH_WARMUP_PROMPTS" \
        --request-rate "$BENCH_WARMUP_RATE" \
        --seed "$BENCH_WARMUP_SEED" > /dev/null 2>&1 || true
    sleep 2
}

# Print the warmup config — profile outputs should embed this so the
# profile pack is self-documenting vs. which bench setting it matches.
common_warmup_config_json() {
    cat << EOF
{
  "input_len": $BENCH_WARMUP_INPUT,
  "output_len": $BENCH_WARMUP_OUTPUT,
  "num_prompts": $BENCH_WARMUP_PROMPTS,
  "request_rate": $BENCH_WARMUP_RATE,
  "seed": $BENCH_WARMUP_SEED
}
EOF
}

# Start a real (non-emulator) vLLM server. Returns once /health responds.
common_start_real_server() {
    local LOGFILE="$1"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" \
        --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" \
        --trust-remote-code \
        > "$LOGFILE" 2>&1 &
    common_wait_server
}

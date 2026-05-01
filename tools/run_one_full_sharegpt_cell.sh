#!/bin/bash
# Generic 3-stage full-ShareGPT cell runner (v4 + DEFAULT max-num-seqs).
#
# v4 invariants (parity-asserted by Apr 25 quick test):
#   - All 3 stages (real bench, profile capture, emu validate) run with
#     DEFAULT max-num-seqs (no --max-num-seqs flag passed to api_server).
#     Mismatch in this flag was the root cause of the Apr 22-24 fake
#     "A10 saturation gap" — see MEMORY.md / feedback_config_parity.
#   - Emu stage uses CUDA-invisible v4 path (CUDA_VISIBLE_DEVICES=""
#     + LD_LIBRARY_PATH stub + NCCL→gloo). Emu sees no GPU; oracle
#     samples from profile pack and the executor hook returns fake
#     output before any kernel dispatches.
#   - parity_audit.sh runs after each stage; cell aborts on violation.
#
# Required env:
#   CELL_TAG         - unique tag for this cell, e.g. "apr26-r3-prefix-off"
#   BENCH_MODEL      - HF model id, e.g. "Qwen/Qwen3-8B"
# Optional:
#   EXTRA_SERVER_ARGS - appended to api_server (default ""). Use this for
#                       cell-specific server config like --no-prefix-caching
#                       or --attention-backend TRITON_ATTN. NEVER include
#                       --max-num-seqs here — it must remain DEFAULT.
#   REUSE_PROFILE    - if set, path to existing profile pack; skip stage 2
#   VALIDATE_ONLY    - if set, skip stage 1 real and use existing real baseline
#   REAL_BASELINE_DIR - required if VALIDATE_ONLY; dir with real_r*.json
#   STUB_DIR         - cuda stub dir (default ~/cuda_stubs); must contain
#                       libcuda.so.1 + libcudart.so.* for v4 emu mode
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
# REPO_ROOT may be passed by wrapper (e.g., vast hosts where repo is at
# /workspace/vllm-emulator). Default to the personal_gpu_vm location.
REPO_ROOT="${REPO_ROOT:-$HOME/Code/llm/vllm-emulator}"
# Source venv only if not already active (wrapper may have done it).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi
cd "$REPO_ROOT"
source tools/_bench_common.sh

: "${CELL_TAG:?required}"
: "${BENCH_MODEL:?required}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
REUSE_PROFILE="${REUSE_PROFILE:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-}"
REAL_BASELINE_DIR="${REAL_BASELINE_DIR:-}"
STUB_DIR="${STUB_DIR:-$HOME/cuda_stubs}"

# Refuse known-bad parity violation: --max-num-seqs in EXTRA_SERVER_ARGS.
if echo "$EXTRA_SERVER_ARGS" | grep -qE '\-\-max-num-seqs|\-\-max_num_seqs'; then
    echo "FATAL: EXTRA_SERVER_ARGS contains --max-num-seqs. The v4 path requires" >&2
    echo "       DEFAULT max-num-seqs across all 3 stages (see Apr 25 parity bug" >&2
    echo "       in MEMORY.md / feedback_config_parity)." >&2
    exit 1
fi

# v4 mandatory parity: ALL 3 stages run at DEFAULT max-num-seqs. Cell-specific
# server args (TRITON, prefix-caching off, etc.) flow through EXTRA_SERVER_ARGS.
FULL_SERVER_ARGS="$EXTRA_SERVER_ARGS"

# Optional KV pool override: deterministic KV-block parity across Stage 1
# (real) and Stage 3 (emu). Eliminates vllm `profile_run` run-to-run variance
# (~6% spread observed on Q14B/RTX-8000) by pinning both servers to a fixed
# block count. Use tools/_calibrate_kv_min.py on Stage 2's server logs to
# pick a value that's <= every observed pool (so vllm can always allocate it).
# When set, the value gets appended to FULL_SERVER_ARGS for every server boot.
KV_NUM_GPU_BLOCKS_OVERRIDE="${KV_NUM_GPU_BLOCKS_OVERRIDE:-}"
if [ -n "$KV_NUM_GPU_BLOCKS_OVERRIDE" ]; then
    FULL_SERVER_ARGS="$FULL_SERVER_ARGS --num-gpu-blocks-override=$KV_NUM_GPU_BLOCKS_OVERRIDE"
fi

export BENCH_PORT=8100
export BENCH_MAX_MODEL_LEN=4096
export SHAREGPT="./results/sharegpt_full_4096.json"

OUT="./results/${CELL_TAG}"
mkdir -p "$OUT"
MASTER_LOG="$OUT/run.log"
MARKER="/tmp/vllm_${CELL_TAG}"
touch "${MARKER}.started"
echo "=== ${CELL_TAG} start $(date -u) ===" > "$MASTER_LOG"
echo "MODEL=$BENCH_MODEL" >> "$MASTER_LOG"
echo "EXTRA_SERVER_ARGS=$FULL_SERVER_ARGS  (max-num-seqs=DEFAULT enforced)" >> "$MASTER_LOG"
echo "REUSE_PROFILE=$REUSE_PROFILE" >> "$MASTER_LOG"
echo "VALIDATE_ONLY=$VALIDATE_ONLY" >> "$MASTER_LOG"
echo "STUB_DIR=$STUB_DIR" >> "$MASTER_LOG"

# Shared profile-time CSV (single row per cell).
SUMMARY_CSV="./results/profile_time_summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "cell_tag,model,stage1_real_s,stage2_profile_s,stage3_validate_s,total_s,date_utc" > "$SUMMARY_CSV"
fi

# Per-cell per-rate delta CSV (parsed by correctness cron).
DELTA_CSV="$OUT/per_rate_deltas.csv"
echo "rate,ttft_mean_pct,tpot_mean_pct,itl_mean_pct,e2e_mean_pct,tput_pct" > "$DELTA_CSV"

if [ -n "${FORCED_RATES:-}" ]; then
    # Space-separated list, e.g. FORCED_RATES="32" or FORCED_RATES="4 32"
    read -ra RATES <<< "$FORCED_RATES"
else
    RATES=(2 4 8 16 32)
fi
PROMPTS=2000
PARITY_REAL=""
PARITY_EMU=""

# ---- BENCH_IGNORE_EOS preflight: every stage that runs `bench serve` MUST
# honor the env var. If BENCH_IGNORE_EOS=1 is set but adaptive_profile_capture.sh
# does not consume it, Stage 2 will silently use stochastic-EOS bench while
# Stage 1+3 use --ignore-eos, producing a profile-vs-bench sampling-mode
# mismatch (silent multi-hour data corruption).
if [ "${BENCH_IGNORE_EOS:-}" = "1" ]; then
    PROFILE_CAPTURE_SCRIPT="$(dirname "$0")/adaptive_profile_capture.sh"
    if [ -f "$PROFILE_CAPTURE_SCRIPT" ]; then
        if ! grep -q "IGNORE_EOS_FLAG" "$PROFILE_CAPTURE_SCRIPT" \
                || ! grep -q "BENCH_IGNORE_EOS" "$PROFILE_CAPTURE_SCRIPT"; then
            echo "FATAL: BENCH_IGNORE_EOS=1 but $PROFILE_CAPTURE_SCRIPT" >> "$MASTER_LOG"
            echo "FATAL:   does not honor it (stage 2 would run without --ignore-eos)" >> "$MASTER_LOG"
            echo "FATAL:   sync the IGNORE_EOS-aware version before launching." >> "$MASTER_LOG"
            echo "FATAL: BENCH_IGNORE_EOS preflight failed — see $MASTER_LOG"
            exit 7
        fi
    fi
fi
PARITY_PROFILE=""
S1_SECS=0
S2_SECS=0
S3_SECS=0

start_real_server() {
    local SRV="$1"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code $FULL_SERVER_ARGS \
        > "$SRV" 2>&1 &
}

# v4 emu server: CUDA invisible, NCCL→gloo, no kernels dispatched.
# Profile pack drives oracle sampling via the executor hook.
start_emu_server() {
    local SRV="$1" PROFILE="$2"
    env LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}" \
        CUDA_VISIBLE_DEVICES="" \
        VLLM_EMULATOR_ENABLE_ORACLE=1 \
        VLLM_EMULATOR_PROFILE_PACK="$PROFILE" \
        VLLM_EMULATOR_MODE=realtime \
        VLLM_EMULATOR_EXECUTOR_HOOK=1 \
        VLLM_EMULATOR_SCHEDULER_HOOK=0 \
        VLLM_EMULATOR_IPC_POSITION=disabled \
        VLLM_EMULATOR_PREP_SURROGATE=0 \
        VLLM_EMULATOR_ORACLE_AGG=sample \
        VLLM_EMULATOR_ORACLE_K=auto \
        VLLM_EMULATOR_ORACLE_MIN_SAMPLES=30 \
        VLLM_EMULATOR_BW_SLOPE_SOURCE=disabled \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$BENCH_MODEL" --max-model-len "$BENCH_MAX_MODEL_LEN" \
        --port "$BENCH_PORT" --trust-remote-code $FULL_SERVER_ARGS \
        > "$SRV" 2>&1 &
}

run_bench_r() {
    local TAG="$1" R="$2"
    local BURST_FLAG=""
    if [ -n "${BURSTINESS:-}" ]; then
        BURST_FLAG="--burstiness ${BURSTINESS}"
    fi
    # BENCH_TEMPERATURE: when set, propagates --temperature to bench. Use
    # 0 for greedy (deterministic outputs). When unset, vllm 0.18 server
    # default applies (Qwen3-8B → stochastic).
    local TEMP_FLAG=""
    if [ -n "${BENCH_TEMPERATURE:-}" ]; then
        TEMP_FLAG="--temperature ${BENCH_TEMPERATURE}"
    fi
    # BENCH_IGNORE_EOS: when set to 1, adds --ignore-eos. Forces real bench
    # to run to max_tokens (matching emu's filler-token behavior). Required
    # for models that EOS earlier than max_tokens (e.g. Llama-3.1-8B).
    # Must be set consistently across Stage 1 / Stage 2 / Stage 3.
    local IGNORE_EOS_FLAG=""
    if [ "${BENCH_IGNORE_EOS:-}" = "1" ]; then
        IGNORE_EOS_FLAG="--ignore-eos"
    fi
    # BENCH_TIMEOUT: per-rate bench wall-clock cap. Default 1800s. Bump for
    # slow attention backends (TRITON_ATTN ~25-30% slower → 1800s caused
    # 28% Stage 2 sample loss on apr26-triton). MUST be applied to all 3
    # stages so per-rate completion conditions match across them.
    local TIMEOUT_S="${BENCH_TIMEOUT:-1800}"
    timeout "$TIMEOUT_S" python3 -m vllm.entrypoints.cli.main bench serve \
        --model "$BENCH_MODEL" --base-url "http://localhost:${BENCH_PORT}" \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts "$PROMPTS" --request-rate "$R" --seed 0 $BURST_FLAG $TEMP_FLAG $IGNORE_EOS_FLAG \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir "$OUT" --result-filename "${TAG}_r${R}.json" \
        > "$OUT/bench_${TAG}_r${R}.log" 2>&1 || true
}

# ---------------- STAGE 1: real baseline ----------------
if [ -n "${SKIP_STAGE1:-}" ]; then
    echo "[$(date -u +%T)] STAGE 1: skipped (SKIP_STAGE1=$SKIP_STAGE1)" >> "$MASTER_LOG"
    S1_SECS=0
elif [ -n "$VALIDATE_ONLY" ] && [ -n "$REAL_BASELINE_DIR" ]; then
    echo "[$(date -u +%T)] STAGE 1: skipped (VALIDATE_ONLY, using $REAL_BASELINE_DIR)" >> "$MASTER_LOG"
    S1_SECS=0
    for R in "${RATES[@]}"; do
        [ -f "$REAL_BASELINE_DIR/real_r${R}.json" ] && cp "$REAL_BASELINE_DIR/real_r${R}.json" "$OUT/real_r${R}.json"
    done
else
    echo "" >> "$MASTER_LOG"
    T0=$(date +%s)
    echo "[$(date -u +%T)] STAGE 1: real 5-rate baseline (DEFAULT max-num-seqs)" >> "$MASTER_LOG"
    for R in "${RATES[@]}"; do
        SRV="$OUT/server_real_r${R}.log"
        common_cleanup
        common_preflight || { echo "preflight FAIL real r=$R" >> "$MASTER_LOG"; continue; }
        start_real_server "$SRV"
        common_wait_server || { echo "server FAIL real r=$R" >> "$MASTER_LOG"; common_cleanup; continue; }
        common_warmup
        if [ -z "$PARITY_REAL" ]; then
            PARITY_REAL=$(grep -m1 'non-default args' "$SRV" 2>/dev/null || echo NONE)
            echo "PARITY REAL: $PARITY_REAL" >> "$MASTER_LOG"
        fi
        echo "  [$(date +%T)] bench real r=$R" >> "$MASTER_LOG"
        run_bench_r "real" "$R"
        common_cleanup
    done
    T1=$(date +%s)
    S1_SECS=$((T1-T0))
    echo "[$(date -u +%T)] STAGE 1 duration: ${S1_SECS}s ($(awk "BEGIN{printf \"%.2f\",$S1_SECS/3600}")h)" >> "$MASTER_LOG"
fi

# ---------------- STAGE 2: dense profile ----------------
if [ -n "${SKIP_STAGE2:-}" ]; then
    echo "" >> "$MASTER_LOG"
    echo "[$(date -u +%T)] STAGE 2: skipped (SKIP_STAGE2=$SKIP_STAGE2)" >> "$MASTER_LOG"
    S2_SECS=0
    PROFILE="${REUSE_PROFILE:-}"
elif [ -n "$REUSE_PROFILE" ]; then
    echo "" >> "$MASTER_LOG"
    echo "[$(date -u +%T)] STAGE 2: skipped (REUSE_PROFILE=$REUSE_PROFILE)" >> "$MASTER_LOG"
    PROFILE="$REUSE_PROFILE"
    S2_SECS=0
    # Capture parity from profile pack's first server log (if present)
    PROFILE_DIR="$(dirname "$PROFILE")"
    PROFILE_LOGS=$(ls "$PROFILE_DIR"/logs/server_*.log 2>/dev/null | head -1)
    if [ -n "$PROFILE_LOGS" ]; then
        PARITY_PROFILE=$(grep -m1 'non-default args' "$PROFILE_LOGS" 2>/dev/null || echo NONE)
        echo "PARITY PROFILE: $PARITY_PROFILE" >> "$MASTER_LOG"
    fi
else
    echo "" >> "$MASTER_LOG"
    T0=$(date +%s)
    echo "[$(date -u +%T)] STAGE 2: dense profile capture (DEFAULT max-num-seqs)" >> "$MASTER_LOG"
    # Profile capture script — adaptive 5-rate sweep to populate the 2D table.
    # Reads BENCH_MODEL, BENCH_MAX_MODEL_LEN, BENCH_PORT, EXTRA_SERVER_ARGS from env.
    ROUNDS=2 TAG="${CELL_TAG}" EXTRA_SERVER_ARGS="$FULL_SERVER_ARGS" \
        bash tools/adaptive_profile_capture.sh >> "$MASTER_LOG" 2>&1
    T1=$(date +%s)
    S2_SECS=$((T1-T0))
    echo "[$(date -u +%T)] STAGE 2 duration: ${S2_SECS}s ($(awk "BEGIN{printf \"%.2f\",$S2_SECS/3600}")h)" >> "$MASTER_LOG"

    # ---- Post-Stage-2 sampling-mode audit. Verify Stage 2 bench cmdlines
    # actually carried --ignore-eos when BENCH_IGNORE_EOS=1 (defense in depth:
    # script greps verify the source has the flag, this verifies the bench
    # process ran with the flag). Fail-closed if the audit fails.
    EXPECT_IGEOS="${BENCH_IGNORE_EOS:-0}"
    PROFILE_LOG_DIR=""
    for PREFIX in RTX-8000 A100 A800 A40 A10 L40S H200 H100 UNKNOWN; do
        CAND="./results/${PREFIX}-adaptive-${CELL_TAG}/per_rate_traces"
        [ -d "$CAND" ] && PROFILE_LOG_DIR="$CAND" && break
    done
    if [ -n "$PROFILE_LOG_DIR" ]; then
        STAGE2_HAS_IGEOS=0
        for tlog in "$PROFILE_LOG_DIR"/round*_r*.log; do
            [ -f "$tlog" ] || continue
            if grep -q "ignore_eos=True" "$tlog" 2>/dev/null \
                    || grep -q '\-\-ignore-eos' "$tlog" 2>/dev/null; then
                STAGE2_HAS_IGEOS=1
                break
            fi
        done
        if [ "$EXPECT_IGEOS" = "1" ] && [ "$STAGE2_HAS_IGEOS" = "0" ]; then
            echo "FATAL: BENCH_IGNORE_EOS=1 but Stage 2 bench logs lack --ignore-eos." >> "$MASTER_LOG"
            echo "FATAL: profile pack would be calibrated against a different sampling" >> "$MASTER_LOG"
            echo "FATAL: mode than Stage 1+3 — silent multi-hour data corruption." >> "$MASTER_LOG"
            echo "FATAL: aborting before Stage 3 to prevent wasted compute." >> "$MASTER_LOG"
            exit 8
        fi
        if [ "$EXPECT_IGEOS" = "0" ] && [ "$STAGE2_HAS_IGEOS" = "1" ]; then
            echo "FATAL: BENCH_IGNORE_EOS unset but Stage 2 USED --ignore-eos." >> "$MASTER_LOG"
            echo "FATAL: same sampling-mode mismatch in the other direction." >> "$MASTER_LOG"
            exit 8
        fi
        echo "PASS: Stage 2 sampling-mode audit (igeos=$STAGE2_HAS_IGEOS, expect=$EXPECT_IGEOS)" >> "$MASTER_LOG"
    fi

    # Find the produced profile (HW prefix depends on platform).
    PROFILE=""
    for PREFIX in RTX-8000 A100 A800 A40 A10 L40S H200 H100 UNKNOWN; do
        CAND="./results/${PREFIX}-adaptive-${CELL_TAG}/serving-full.json"
        [ -f "$CAND" ] && PROFILE="$CAND" && break
    done
    if [ -z "$PROFILE" ]; then
        echo "STAGE 2 FAIL — profile missing" >> "$MASTER_LOG"
        touch "${MARKER}.done"
        exit 1
    fi
    echo "PROFILE=$PROFILE ($(stat -c%s "$PROFILE") bytes)" >> "$MASTER_LOG"
    PROFILE_DIR="$(dirname "$PROFILE")"
    PROFILE_LOGS=$(ls "$PROFILE_DIR"/logs/server_*.log 2>/dev/null | head -1)
    if [ -n "$PROFILE_LOGS" ]; then
        PARITY_PROFILE=$(grep -m1 'non-default args' "$PROFILE_LOGS" 2>/dev/null || echo NONE)
        echo "PARITY PROFILE: $PARITY_PROFILE" >> "$MASTER_LOG"
    fi
fi

# Auto-patch KV pool from real_r2 server log into the profile pack.
# Without this, vllm V1 in emu falls back to a heuristic that overestimates
# KV pool, breaking saturation-regime parity. Skipped when REUSE_PROFILE
# is set (assume caller has already patched).
if [ -z "$REUSE_PROFILE" ] && [ -n "$PROFILE" ]; then
    SRV_LOG_FOR_PATCH=""
    for R in "${RATES[@]}"; do
        CAND="$OUT/server_real_r${R}.log"
        if [ -f "$CAND" ] && grep -q "GPU KV cache size" "$CAND" 2>/dev/null; then
            SRV_LOG_FOR_PATCH="$CAND"
            break
        fi
    done
    if [ -n "$SRV_LOG_FOR_PATCH" ]; then
        echo "[$(date -u +%T)] auto-patching KV pool from $SRV_LOG_FOR_PATCH" >> "$MASTER_LOG"
        python3 "$REPO_ROOT/tools/_patch_profile_pack_kv.py" \
            "$PROFILE" "$SRV_LOG_FOR_PATCH" >> "$MASTER_LOG" 2>&1 || \
            echo "  WARN: KV pool auto-patch failed; emu may use fallback heuristic" >> "$MASTER_LOG"
    else
        echo "[$(date -u +%T)] WARN: no real server log with KV size; skipping auto-patch" >> "$MASTER_LOG"
    fi
fi

# Parity gate after Stage 2: real-bench args MUST match profile-capture args.
# Extract ONLY the dict {...} (not the (APIServer pid=NNN) INFO TIMESTAMP prefix
# which always differs between invocations) and strip 'port': pair which is
# allowed to differ. Then compare the canonical arg sets.
extract_args() {
    # input line ends with "non-default args: {'port': N, 'model': '...', ...}"
    # we want the dict body without 'port' k:v
    echo "$1" | grep -oE "\{[^}]*\}" | sed -E "s/'port'[[:space:]]*:[[:space:]]*[0-9]+[[:space:]]*,?[[:space:]]*//"
}
if [ -n "$PARITY_REAL" ] && [ -n "$PARITY_PROFILE" ]; then
    NORM_REAL=$(extract_args "$PARITY_REAL")
    NORM_PROFILE=$(extract_args "$PARITY_PROFILE")
    if [ "$NORM_REAL" != "$NORM_PROFILE" ]; then
        echo "PARITY VIOLATION (real vs profile, ignoring port):" >> "$MASTER_LOG"
        echo "  real:    $NORM_REAL" >> "$MASTER_LOG"
        echo "  profile: $NORM_PROFILE" >> "$MASTER_LOG"
        echo "ABORT — fix scripts before continuing" >> "$MASTER_LOG"
        touch "${MARKER}.done"
        exit 2
    fi
fi

# ---------------- STAGE 3: emu validate ----------------
echo "" >> "$MASTER_LOG"
if [ -n "${SKIP_STAGE3:-}" ]; then
    echo "[$(date -u +%T)] STAGE 3: skipped (SKIP_STAGE3=$SKIP_STAGE3)" >> "$MASTER_LOG"
    S3_SECS=0
    touch "$OUT/.stage3_skipped"
    echo "[$(date -u +%T)] === Stages 1+2 done; SKIP_STAGE3 set ===" >> "$MASTER_LOG"
    exit 0
fi
T0=$(date +%s)
echo "[$(date -u +%T)] STAGE 3: emu 5-rate validate (v4: CUDA invisible, NCCL→gloo)" >> "$MASTER_LOG"
for R in "${RATES[@]}"; do
    SRV="$OUT/server_emu_r${R}.log"
    common_cleanup
    common_preflight || { echo "preflight FAIL emu r=$R" >> "$MASTER_LOG"; continue; }
    start_emu_server "$SRV" "$PROFILE"
    common_wait_server || { echo "server FAIL emu r=$R" >> "$MASTER_LOG"; common_cleanup; continue; }
    common_warmup
    if [ -z "$PARITY_EMU" ]; then
        PARITY_EMU=$(grep -m1 'non-default args' "$SRV" 2>/dev/null || echo NONE)
        echo "PARITY EMU : $PARITY_EMU" >> "$MASTER_LOG"
    fi
    echo "  [$(date +%T)] bench emu r=$R" >> "$MASTER_LOG"
    run_bench_r "emu" "$R"
    common_cleanup
done
T1=$(date +%s)
S3_SECS=$((T1-T0))
echo "[$(date -u +%T)] STAGE 3 duration: ${S3_SECS}s" >> "$MASTER_LOG"

# Parity gate after Stage 3: emu-bench args MUST match real-bench args.
if [ -n "$PARITY_REAL" ] && [ -n "$PARITY_EMU" ] && [ "$PARITY_REAL" != "$PARITY_EMU" ]; then
    NORM_REAL=$(echo "$PARITY_REAL" | sed -E "s/'port':[^,]*, ?//")
    NORM_EMU=$(echo "$PARITY_EMU" | sed -E "s/'port':[^,]*, ?//")
    if [ "$NORM_REAL" != "$NORM_EMU" ]; then
        echo "PARITY VIOLATION (real vs emu, ignoring port):" >> "$MASTER_LOG"
        echo "  real: $NORM_REAL" >> "$MASTER_LOG"
        echo "  emu : $NORM_EMU" >> "$MASTER_LOG"
        echo "WARNING — Stage 3 results may be unusable" >> "$MASTER_LOG"
    fi
fi

# ---------------- Per-rate MEAN deltas (paper-grade) ----------------
echo "" >> "$MASTER_LOG"
echo "=== PER-RATE DELTAS (mean) ===" >> "$MASTER_LOG"
for R in "${RATES[@]}"; do
    python3 - "$R" "$OUT" "$DELTA_CSV" <<'PY' >> "$MASTER_LOG" 2>&1
import json, sys
R = int(sys.argv[1]); out = sys.argv[2]; delta_csv = sys.argv[3]
try:
    r = json.load(open(f"{out}/real_r{R}.json"))
    e = json.load(open(f"{out}/emu_r{R}.json"))
    pct = lambda a, b: (b/a-1)*100 if a > 0 else 0.0
    d_ttft = pct(r["mean_ttft_ms"], e["mean_ttft_ms"])
    d_tpot = pct(r["mean_tpot_ms"], e["mean_tpot_ms"])
    d_itl  = pct(r["mean_itl_ms"],  e["mean_itl_ms"])
    d_e2e  = pct(r.get("mean_e2el_ms", 0), e.get("mean_e2el_ms", 0))
    d_tput = pct(r["output_throughput"], e["output_throughput"])
    print(f"r={R:>2} | TTFT_mean {d_ttft:+7.2f}% | TPOT_mean {d_tpot:+6.2f}% | "
          f"ITL_mean {d_itl:+6.2f}% | E2E_mean {d_e2e:+6.2f}% | tput {d_tput:+5.2f}%")
    with open(delta_csv, "a") as f:
        f.write(f"{R},{d_ttft:.4f},{d_tpot:.4f},{d_itl:.4f},{d_e2e:.4f},{d_tput:.4f}\n")
except Exception as ex:
    print(f"r={R}: ERROR {ex}")
PY
done

# ---------------- Profile-time summary row ----------------
TOTAL=$((S1_SECS + S2_SECS + S3_SECS))
echo "${CELL_TAG},${BENCH_MODEL},${S1_SECS},${S2_SECS},${S3_SECS},${TOTAL},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY_CSV"

echo "" >> "$MASTER_LOG"
echo "[$(date -u +%T)] DONE (total=${TOTAL}s)" >> "$MASTER_LOG"
touch "${MARKER}.done"

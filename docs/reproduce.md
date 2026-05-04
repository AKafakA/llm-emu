# Reproduce paper Table 1

Each cell runs the same 3-stage orchestrator with cell-specific config:

| Cell | Hardware | Model | Extra config |
| --- | --- | --- | --- |
| **M-Q8** (main)              | RTX 8000 | Qwen/Qwen3-8B  | (default) |
| **M-Q14** (model-scale up)   | RTX 8000 | Qwen/Qwen3-14B | (default) |
| **A40-Q8** (hardware swap)   | A40      | Qwen/Qwen3-8B  | (default) |
| **M-Q8-Burst** (workload)    | RTX 8000 | Qwen/Qwen3-8B  | `BENCH_BURSTINESS=0.25` |
| **A40-L8** (model-family)    | A40      | meta-llama/Llama-3.1-8B | `BENCH_IGNORE_EOS=1` |
| **A40-Q4** (model-scale down)| A40      | Qwen/Qwen3-4B  | (default) |

## Prerequisites

```bash
# 1. Install vLLM 0.18.1 + the LLM-Emu patches (see README.md)
# 2. ShareGPT prompts (one-time, ~5 min)
python tools/download_filter_sharegpt.py
# 3. CUDA stubs for emu mode (one-time)
mkdir -p $HOME/cuda_stubs && cd $HOME/cuda_stubs
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so libcuda.so.1
ln -sf /usr/local/cuda/lib64/libcudart.so.12 .
cd -
```

To skip Phase A on a first try, point `VLLM_EMULATOR_PROFILE_PACK` at
the bundled `example_profiling_data/A40-Q8-Qwen3-8B.json` (Qwen3-8B
on A40 from the paper). For other models / hardware / flags you'll
still need a fresh capture below.

## Single-cell drive

```bash
# Example: M-Q8 cell on RTX 8000
CELL_TAG=mq8-main \
BENCH_MODEL=Qwen/Qwen3-8B \
HW_PREFIX=RTX-8000 \
STUB_DIR=$HOME/cuda_stubs \
bash tools/_orch_3stage_detkv.sh
```

The orchestrator runs three phases:
1. **Phase A** — `tools/adaptive_profile_capture.sh` captures the profile pack on real GPU (no KV override, ~3 hours).
2. **Phase B** — `tools/_calibrate_kv_min.py` reads server logs from Phase A and emits `--num-gpu-blocks-override=N` (instant).
3. **Phase C** — `tools/run_one_full_sharegpt_cell.sh` runs real bench AND emu validate at the SAME pinned KV pool (~50 min real + ~30 min emu).

Total wall time ≈ 4-5 h on RTX 8000.

## Per-cell flags

```bash
# M-Q14
CELL_TAG=mq14-modelscale-up BENCH_MODEL=Qwen/Qwen3-14B \
HW_PREFIX=RTX-8000 \
bash tools/_orch_3stage_detkv.sh

# A40-Q8
CELL_TAG=a40q8-hardware-swap BENCH_MODEL=Qwen/Qwen3-8B \
HW_PREFIX=A40 \
bash tools/_orch_3stage_detkv.sh

# M-Q8-Burst
CELL_TAG=mq8-burst-gamma025 BENCH_MODEL=Qwen/Qwen3-8B \
HW_PREFIX=RTX-8000 BENCH_BURSTINESS=0.25 \
bash tools/_orch_3stage_detkv.sh

# A40-L8
CELL_TAG=a40l8-llama-igeos BENCH_MODEL=meta-llama/Llama-3.1-8B \
HW_PREFIX=A40 BENCH_IGNORE_EOS=1 \
bash tools/_orch_3stage_detkv.sh

# A40-Q4
CELL_TAG=a40q4-modelscale-down BENCH_MODEL=Qwen/Qwen3-4B HW_PREFIX=A40 \
bash tools/_orch_3stage_detkv.sh
```

## Outputs

```
results/<CELL_TAG>/real_r{2,4,8,16,32}.json   # vLLM bench output (real GPU)
                  emu_r{2,4,8,16,32}.json    # LLM-Emu bench output (emulator)
                  per_rate_deltas.csv         # %-error per metric per rate
results/<HW_PREFIX>-adaptive-<CELL_TAG>/
                  serving-full.json           # the profile pack
                  per_rate_traces/            # raw step-cycle traces
```

`per_rate_deltas.csv` columns: `rate, ttft_mean_pct, tpot_mean_pct,
itl_mean_pct, e2e_mean_pct, tps_pct` — TTFT, TPOT, ITL, E2E latency,
TPS (tokens per second).

## Reference deltas

Published v1 Table 1 max-abs deltas on the same hardware/model
(your numbers will move within capture-to-capture variance):

| Cell | TTFT | TPOT | ITL | E2E | TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| **M-Q8**       | 9.81%  | 0.85% | 0.80% | 2.02% | 0.78% |
| **M-Q14**      | 9.10%  | 1.37% | 1.33% | 1.51% | 1.10% |
| **A40-Q8**     | 9.22%  | 1.52% | 1.26% | 3.72% | 1.25% |
| **M-Q8-Burst** | 10.41% | 4.75% | 4.69% | 4.77% | 1.88% |
| **A40-L8**     | 9.32%  | 1.98% | 1.92% | 3.48% | 1.76% |
| **A40-Q4**     | 7.78%  | 3.10% | 3.01% | 5.22% | 1.50% |

Single-rate spikes at the prefill→saturation transition (typically
r=8 on Qwen3-8B/RTX 8000) can occur from capture-to-capture variance
and are characteristic of profile-driven emulation; rerunning Phase A
on a quiet host usually pulls them back in.

## Audit script

```bash
python tools/_paper_pack_audit.py results/
```

Computes the per-cell max-abs deltas across all rates and prints a
Table-1-style summary.

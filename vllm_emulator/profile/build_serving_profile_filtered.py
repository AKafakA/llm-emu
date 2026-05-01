#!/usr/bin/env python3
"""Build serving profile pack from a step-cycle JSONL trace.

Buckets records by (total_tokens, concurrency), splits into prefill /
decode by num_new_reqs, stores raw samples per bucket. No outlier
filtering, no synthetic constants.

Usage:
    python build_serving_profile_filtered.py <step_cycle_file> <output> \\
        [--model-name NAME] [--gpu-model GPU] \\
        [--tt-bucket-width W] [--conc-bucket-width W]
"""
import argparse
import json
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build serving profile from step-cycle trace records.")
    parser.add_argument("step_cycle_file", help="Path to step_cycle JSONL trace")
    parser.add_argument("output", help="Path to output JSON profile")
    parser.add_argument("--model-name", default=None,
                        help="Model name (overrides trace header)")
    parser.add_argument("--gpu-model", default=None,
                        help="GPU model (overrides trace header)")
    parser.add_argument("--tt-bucket-width", type=int, default=1,
                        help="Total-tokens bucket width (default: 1, no bucketing)")
    parser.add_argument("--conc-bucket-width", type=int, default=5,
                        help="Concurrency bucket width (default: 5)")
    return parser.parse_args()


def load_trace(path):
    """Load trace, keeping records between profiling_start/stop markers
    (or all records if no markers are present)."""
    trace_header = None
    records = []
    all_raw = []
    in_profiling = False

    for line in open(path):
        r = json.loads(line)
        if r.get("_header"):
            trace_header = r
            continue
        if r.get("__marker__") == "profiling_start":
            in_profiling = True
            continue
        if r.get("__marker__") == "profiling_stop":
            in_profiling = False
            continue
        if r.get("__marker__"):
            continue
        if "total_tokens" in r:
            all_raw.append(r)
            if in_profiling:
                records.append(r)

    if not records and all_raw:
        print("  No profiling_start markers found; using all records")
        records = all_raw

    print(f"Records: {len(records)} (from {len(all_raw)} total)")
    return trace_header, records


def extract_metadata(trace_header, cli_model_name, cli_gpu_model):
    model_name = cli_model_name or (trace_header or {}).get("model_name") or "unknown"
    gpu_model = cli_gpu_model or (trace_header or {}).get("gpu_name") or "unknown"

    model_config = None
    if trace_header:
        _MC_KEYS = ("num_hidden_layers", "hidden_size", "num_attention_heads",
                     "num_key_value_heads", "vocab_size", "intermediate_size",
                     "head_dim", "max_model_len", "block_size",
                     "eos_token_id")
        model_config = {k: trace_header[k] for k in _MC_KEYS if k in trace_header}
        _GPU_KEYS = ("gpu_name", "gpu_memory_bytes", "gpu_sm_count",
                      "gpu_compute_capability", "gpu_count")
        gpu_config = {k: trace_header[k] for k in _GPU_KEYS if k in trace_header}
        if gpu_config:
            model_config["gpu"] = gpu_config
        print(f"Trace header: gpu={gpu_model}, model={model_name}, "
              f"layers={model_config.get('num_hidden_layers', '?')}, "
              f"vocab={model_config.get('vocab_size', '?')}")
    else:
        print("WARNING: No _header in trace file. GPU/model metadata not available.")

    return model_name, gpu_model, model_config


def build_2d_distribution(data, label):
    """Build {(tt_b, conc_b) -> [latency_us]} into a list of bucket dicts."""
    distribution = []
    for (ttb, cb), lats in sorted(data.items()):
        samples = [round(v, 1) for v in lats]
        distribution.append({
            "tt": ttb,
            "conc": cb,
            "num_samples": len(samples),
            "samples": samples,
        })
    print(f"  {label}: {len(distribution)} cells")
    return distribution


def main():
    args = parse_args()

    trace_header, records = load_trace(args.step_cycle_file)
    model_name, gpu_model, model_config = extract_metadata(
        trace_header, args.model_name, args.gpu_model)

    if not records:
        print("ERROR: No records found in trace file", file=sys.stderr)
        sys.exit(1)

    tt_w = args.tt_bucket_width
    conc_w = args.conc_bucket_width

    def tt_bucket(tt):
        return int(tt // tt_w) * tt_w + tt_w // 2

    def conc_bucket(n):
        return (n // conc_w) * conc_w + conc_w // 2

    # Prefill = steps with new_reqs > 0 (eager mode in vLLM V1).
    # Decode  = steps with no new_reqs (CUDA graph mode).
    prefill_2d_data = defaultdict(list)
    decode_2d_data = defaultdict(list)
    step_cycle_2d_data = defaultdict(list)

    for r in records:
        tt = r["total_tokens"]
        nnr = r.get("num_new_reqs", 0)
        conc = nnr + r.get("num_decode_seqs", 0)
        if conc < 1:
            conc = 1
        ttb = tt_bucket(tt)
        cb = conc_bucket(conc)
        step_cycle_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        if nnr > 0:
            prefill_2d_data[(ttb, cb)].append(r["step_cycle_us"])
        else:
            decode_2d_data[(ttb, cb)].append(r["step_cycle_us"])

    print(f"\n2D distributions (tt_bucket_width={tt_w}, conc_bucket_width={conc_w}):")
    step_cycle_dist = build_2d_distribution(step_cycle_2d_data, "step_cycle")
    prefill_dist = build_2d_distribution(prefill_2d_data, "prefill")
    decode_dist = build_2d_distribution(decode_2d_data, "decode")

    if step_cycle_dist:
        tt_set = sorted({e["tt"] for e in step_cycle_dist})
        conc_set = sorted({e["conc"] for e in step_cycle_dist})
        total_samples = sum(e["num_samples"] for e in step_cycle_dist)
        print(f"  {len(step_cycle_dist)} cells, tt range={tt_set[0]}-{tt_set[-1]}, "
              f"conc buckets={conc_set}, total samples={total_samples}")

    profile = {
        "version": "2.0",
        "gpu_model": gpu_model,
        "model_name": model_name,
        "profile_type": "serving_step_cycle_2d",
        "bucketing": {
            "tt_bucket_width": tt_w,
            "conc_bucket_width": conc_w,
        },
        "prefill_2d_distribution": sorted(prefill_dist, key=lambda e: (e["tt"], e["conc"])),
        "decode_2d_distribution": sorted(decode_dist, key=lambda e: (e["tt"], e["conc"])),
        "step_cycle_2d_distribution": sorted(step_cycle_dist, key=lambda e: (e["tt"], e["conc"])),
    }
    if model_config:
        profile["model_config"] = model_config

    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

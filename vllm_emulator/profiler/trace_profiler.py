"""StepCycleTracer — records vLLM step-cycle latency for profile capture.

Triggered when VLLM_EMULATOR_TRACE_STEP_CYCLE=1 is set. The tracer is
attached inside the engine core and writes one JSONL record per step to
/tmp/emulator_step_trace.jsonl (or the path in
VLLM_EMULATOR_STEP_TRACE_OUTPUT). The output file is then consumed by
vllm_emulator.profile.build_serving_profile_filtered to produce a
profile pack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class StepCycleTracer:
    """Records full step-cycle time (schedule + execute + output processing).

    Measures the complete _process_engine_step() duration including
    scheduling, execute_model, output processing, and detokenization.
    This captures the real per-step overhead that the GPU-only profile
    misses, making the emulator rate-independent.

    Enabled via VLLM_EMULATOR_TRACE_STEP_CYCLE=1.
    """

    def __init__(self, output_path: str = "/tmp/emulator_step_trace.jsonl"):
        self._output_path = output_path
        self._records: list[dict[str, Any]] = []
        self._step_count = 0
        self._header_written = False
        # Stash scheduler output info set before step_fn()
        self._pending_batch_info: dict[str, Any] | None = None

    def write_header(self, vllm_config: Any) -> None:
        """Write a _header record with GPU and model metadata.

        Called once at startup from EngineCore after the tracer is created.
        Auto-collects GPU properties (via torch.cuda) and model architecture
        (via HuggingFace config), making the trace file self-describing.
        The profile builder reads this header to produce a self-contained
        profile pack — no manual GPU/model configuration needed.
        """
        if self._header_written:
            return

        header: dict[str, Any] = {"_header": True}

        # --- GPU properties ---
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                header["gpu_name"] = props.name
                header["gpu_memory_bytes"] = props.total_memory
                header["gpu_sm_count"] = props.multi_processor_count
                cap = torch.cuda.get_device_capability(0)
                header["gpu_compute_capability"] = list(cap)
                header["gpu_count"] = torch.cuda.device_count()
        except Exception as e:
            print(f"[StepCycleTracer] GPU metadata collection failed: {e}")

        # --- Model properties (from HuggingFace config) ---
        try:
            model_config = vllm_config.model_config
            header["model_name"] = model_config.model
            header["max_model_len"] = model_config.max_model_len

            hf_cfg = model_config.hf_text_config
            for attr in ("num_hidden_layers", "hidden_size",
                         "num_attention_heads", "num_key_value_heads",
                         "vocab_size", "intermediate_size"):
                val = getattr(hf_cfg, attr, None)
                if val is not None:
                    header[attr] = val

            # head_dim: explicit or derived
            head_dim = getattr(hf_cfg, "head_dim", None)
            if head_dim is None and hasattr(hf_cfg, "hidden_size") and hasattr(hf_cfg, "num_attention_heads"):
                head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads
            if head_dim is not None:
                header["head_dim"] = head_dim
        except Exception as e:
            print(f"[StepCycleTracer] Model metadata collection failed: {e}")

        # --- Scheduler/cache config ---
        try:
            header["block_size"] = vllm_config.cache_config.block_size
            header["enable_chunked_prefill"] = (
                vllm_config.scheduler_config.enable_chunked_prefill)
            header["max_num_seqs"] = (
                vllm_config.scheduler_config.max_num_seqs)
        except Exception:
            pass

        # Write header as first line of trace file
        from pathlib import Path
        path = Path(self._output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
        self._header_written = True
        print(f"[StepCycleTracer] Header written: gpu={header.get('gpu_name', '?')}, "
              f"model={header.get('model_name', '?')}, "
              f"layers={header.get('num_hidden_layers', '?')}, "
              f"vocab={header.get('vocab_size', '?')}")

    def set_batch_info(
        self,
        total_tokens: int,
        num_new_reqs: int,
        num_decode_seqs: int,
        sum_kv: int = 0,
    ) -> None:
        """Called before step_fn() with the current batch info.

        sum_kv: sum of num_computed_tokens across scheduled requests (the
        effective KV-cache depth for attention-work cost accounting).
        Used by the KV-adjustment (α) in the profile builder to bucket
        records by attention-equivalent token count.
        """
        self._pending_batch_info = {
            "total_tokens": total_tokens,
            "num_new_reqs": num_new_reqs,
            "num_decode_seqs": num_decode_seqs,
            "sum_kv": sum_kv,
        }

    def set_extra_fields(self, fields: dict[str, Any]) -> None:
        """Set extra fields to be included in the next record_step() call."""
        self._pending_extra = fields

    def record_step(self, step_latency_us: float) -> None:
        record: dict[str, Any] = {"step_cycle_us": round(step_latency_us, 1)}
        if self._pending_batch_info:
            record.update(self._pending_batch_info)
            self._pending_batch_info = None
        if hasattr(self, '_pending_extra') and self._pending_extra:
            record.update(self._pending_extra)
            self._pending_extra = None
        self._records.append(record)
        self._step_count += 1
        if len(self._records) >= 200:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        from pathlib import Path
        path = Path(self._output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
        count = len(self._records)
        self._records.clear()
        print(f"[StepCycleTracer] Flushed {count} step records to {path}")

    def get_stats(self) -> dict:
        """Return statistics for calibration."""
        import statistics
        if not self._records:
            return {}
        return {
            "num_steps": len(self._records),
            "median_us": statistics.median(self._records),
            "mean_us": statistics.mean(self._records),
        }


# ---------------------------------------------------------------------------
# Trace → Profile Pack conversion
# ---------------------------------------------------------------------------

def load_trace(trace_path: str) -> list[dict[str, Any]]:
    """Load a JSONL trace file."""
    records = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def trace_to_profile_pack(
    records: list[dict[str, Any]],
    gpu_model: str = "unknown",
    model_name: str = "unknown",
    bucket_size: int = 16,
) -> dict[str, Any]:
    """Convert trace records into a profile pack with forward_pass section.

    Groups records by total_tokens (bucketed) and computes median latency
    per bucket. Also generates prefill and decode sections for backward
    compatibility.

    Args:
        records: List of trace records from ExecuteModelTracer.
        gpu_model: GPU model name for the profile pack.
        model_name: Model name for the profile pack.
        bucket_size: Bucket size for total_tokens grouping.

    Returns:
        A profile pack dict compatible with the emulator oracle.
    """
    import statistics

    if not records:
        raise ValueError("No trace records to convert")

    # --- Forward pass: bucket by total_tokens ---
    fwd_buckets: dict[int, list[float]] = {}
    for rec in records:
        total = rec["total_tokens"]
        if total <= 0:
            continue
        # Use per-token granularity for small total_tokens (online serving
        # operating range), coarser buckets above to keep profile compact.
        if total <= 32:
            bucket = total
        else:
            bucket = max(1, (total + bucket_size // 2) // bucket_size * bucket_size)
        fwd_buckets.setdefault(bucket, []).append(rec["latency_us"])

    forward_pass = []
    for bucket in sorted(fwd_buckets):
        latencies = fwd_buckets[bucket]
        forward_pass.append({
            "total_tokens": bucket,
            "latency_us": round(statistics.median(latencies), 1),
            "num_samples": len(latencies),
        })

    # --- Prefill: bucket by (avg_seq_len, num_new_reqs) ---
    prefill_buckets: dict[tuple[int, int], list[float]] = {}
    for rec in records:
        if rec["num_prefill_tokens"] > 0 and rec["num_decode_seqs"] == 0:
            # Pure prefill step
            seq_len_bucket = max(1, rec["avg_seq_len"] // 64 * 64) or 64
            bs = rec["num_new_reqs"]
            prefill_buckets.setdefault((seq_len_bucket, bs), []).append(
                rec["latency_us"]
            )

    prefill = []
    for (seq_len, bs) in sorted(prefill_buckets):
        latencies = prefill_buckets[(seq_len, bs)]
        prefill.append({
            "seq_len": seq_len,
            "batch_size": bs,
            "latency_us": round(statistics.median(latencies), 1),
        })

    # --- Decode: bucket by num_decode_seqs ---
    decode_buckets: dict[int, list[float]] = {}
    for rec in records:
        if rec["num_prefill_tokens"] == 0 and rec["num_decode_seqs"] > 0:
            # Pure decode step
            decode_buckets.setdefault(rec["num_decode_seqs"], []).append(
                rec["latency_us"]
            )

    decode = []
    for n_seqs in sorted(decode_buckets):
        latencies = decode_buckets[n_seqs]
        decode.append({
            "active_seqs": n_seqs,
            "latency_us_per_token": round(statistics.median(latencies), 1),
        })

    # Ensure at least one entry in each section
    if not prefill:
        prefill = [{"seq_len": 128, "batch_size": 1, "latency_us": 1.0}]
    if not decode:
        decode = [{"active_seqs": 1, "latency_us_per_token": 1.0}]

    return {
        "version": "2.0",
        "gpu_model": gpu_model,
        "model_name": model_name,
        "profile_method": "trace",
        "num_trace_records": len(records),
        "prefill": prefill,
        "decode": decode,
        "forward_pass": forward_pass,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert execute_model() trace to profile pack.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to JSONL trace file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="trace_profile_pack.json",
        help="Output JSON profile pack path.",
    )
    parser.add_argument(
        "--gpu-model",
        default="unknown",
        help="GPU model name for the profile pack.",
    )
    parser.add_argument(
        "--model-name",
        default="unknown",
        help="Model name for the profile pack.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=16,
        help="Bucket size for total_tokens grouping.",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print trace statistics.",
    )
    args = parser.parse_args()

    records = load_trace(args.trace)
    print(f"Loaded {len(records)} trace records from {args.trace}")

    if args.print_stats:
        import statistics
        latencies = [r["latency_us"] for r in records]
        tokens = [r["total_tokens"] for r in records]
        print(f"  Latency: median={statistics.median(latencies):.0f}us, "
              f"mean={statistics.mean(latencies):.0f}us, "
              f"p99={sorted(latencies)[int(len(latencies)*0.99)]:.0f}us")
        print(f"  Tokens/step: median={statistics.median(tokens):.0f}, "
              f"max={max(tokens)}")
        prefill_steps = sum(1 for r in records if r["num_prefill_tokens"] > 0)
        decode_steps = sum(1 for r in records if r["num_decode_seqs"] > 0 and r["num_prefill_tokens"] == 0)
        mixed_steps = sum(1 for r in records if r["num_prefill_tokens"] > 0 and r["num_decode_seqs"] > 0)
        print(f"  Steps: {len(records)} total, {prefill_steps} prefill, "
              f"{decode_steps} decode, {mixed_steps} mixed")

    pack = trace_to_profile_pack(
        records,
        gpu_model=args.gpu_model,
        model_name=args.model_name,
        bucket_size=args.bucket_size,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"Profile pack written to {out}")
    print(f"  forward_pass: {len(pack['forward_pass'])} buckets")
    print(f"  prefill: {len(pack['prefill'])} entries")
    print(f"  decode: {len(pack['decode'])} entries")


if __name__ == "__main__":
    _cli_main()

"""Extract min `GPU KV cache size: N tokens` across all server logs in given
dirs (or files). Prints both the min token count and the corresponding
num_gpu_blocks (using vllm's default block_size=16).

Use this AFTER Stage 2 profile capture finishes — it scans every per-rate /
per-round server startup, finds the lowest reported KV pool, and emits a value
safe to pass as `--num-gpu-blocks-override=N` to BOTH Stage 1 (real) and Stage
3 (emu) servers. Lowest-observed is the correct floor: we know vllm CAN
allocate at least that many blocks (because it did, on this hardware, in some
profile-capture run). Setting the override to that value:

  - eliminates run-to-run profile_run variance (deterministic pool every boot)
  - guarantees both real and emu use exactly the same KV pool size
  - never asks vllm for more than it could naturally allocate (safe)

Usage:
    python3 _calibrate_kv_min.py [--block-size 16] <log_or_dir> [...]

Reports: min_tokens, max_tokens, spread_pct, num_blocks (= min_tokens // block).
Exit 0 on success; 1 if no log lines matched.
"""
import argparse, glob, os, re, sys

PAT = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")

def collect(targets):
    pools = []
    for t in targets:
        if os.path.isdir(t):
            files = glob.glob(f"{t}/**/server_*.log", recursive=True)
            files += glob.glob(f"{t}/**/round*_r*.log", recursive=True)
        elif os.path.isfile(t):
            files = [t]
        else:
            files = []
        for f in files:
            try:
                with open(f) as fh:
                    for line in fh:
                        m = PAT.search(line)
                        if m:
                            pools.append((int(m.group(1).replace(",", "")), f))
                            break  # one per file
            except Exception:
                pass
    return pools

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=16,
                    help="vllm KV block size in tokens (default 16)")
    ap.add_argument("targets", nargs="+", help="server log files or dirs")
    args = ap.parse_args()
    pools = collect(args.targets)
    if not pools:
        print("FATAL: no 'GPU KV cache size: N tokens' lines found", file=sys.stderr)
        sys.exit(1)
    sizes = [p[0] for p in pools]
    mn, mx = min(sizes), max(sizes)
    spread_pct = (mx - mn) / mn * 100 if mn else 0
    nblocks = mn // args.block_size
    print(f"min_tokens={mn:,}", file=sys.stderr)
    print(f"max_tokens={mx:,}", file=sys.stderr)
    print(f"spread_pct={spread_pct:.2f}%", file=sys.stderr)
    print(f"observations={len(pools)}", file=sys.stderr)
    print(f"block_size={args.block_size}", file=sys.stderr)
    print(f"num_gpu_blocks_override={nblocks}", file=sys.stderr)
    # Emit the override value on stdout so callers can capture with $(...)
    print(nblocks)

if __name__ == "__main__":
    main()

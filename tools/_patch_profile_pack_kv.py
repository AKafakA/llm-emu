"""Backfill profile pack with available_kv_cache_bytes from real bench server log.

Usage:
    python3 _patch_profile_pack_kv.py <profile_pack.json> <real_server_log>

Extracts the `GPU KV cache size: X tokens` line from a real-bench vllm server
log, computes per-token KV bytes from the model config in the profile pack,
and writes available_kv_cache_bytes into model_config.gpu so the emulator's
cuda_mock stub returns this exact value to vllm's V1 worker.
"""
import json, re, sys, shutil

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

pack_path, log_path = sys.argv[1], sys.argv[2]

# Parse "GPU KV cache size: 28,576 tokens"
kv_tokens = None
with open(log_path) as f:
    for line in f:
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
        if m:
            kv_tokens = int(m.group(1).replace(",", ""))
            break
if kv_tokens is None:
    print(f"FATAL: no 'GPU KV cache size' line in {log_path}")
    sys.exit(2)
print(f"Real KV pool tokens (from log): {kv_tokens:,}")

with open(pack_path) as f:
    pack = json.load(f)
mc = pack.get("model_config", {})

# Per-token KV bytes for full attention (vllm's standard layout):
#   per_token = num_layers × num_kv_heads × head_dim × 2 (K+V) × bytes_per_elem
n_layers = int(mc.get("num_hidden_layers", 0))
n_kv_heads = int(mc.get("num_key_value_heads", 0))
head_dim = int(mc.get("head_dim", 0))
if not (n_layers and n_kv_heads and head_dim):
    print(f"FATAL: profile pack model_config missing arch fields: "
          f"layers={n_layers} kv_heads={n_kv_heads} head_dim={head_dim}")
    sys.exit(3)
# Assume bf16 KV cache (vllm default for bf16 model)
bytes_per_elem = 2
per_token_bytes = n_layers * n_kv_heads * head_dim * 2 * bytes_per_elem
print(f"per-token KV bytes (bf16): {n_layers}×{n_kv_heads}×{head_dim}×2×{bytes_per_elem} = {per_token_bytes:,}")

available_kv_cache_bytes = kv_tokens * per_token_bytes
print(f"available_kv_cache_bytes = {kv_tokens:,} × {per_token_bytes:,} = {available_kv_cache_bytes:,} bytes ({available_kv_cache_bytes/1e9:.2f} GB)")

# Backup, then write
shutil.copy2(pack_path, pack_path + ".bak")
mc.setdefault("gpu", {})
mc["gpu"]["available_kv_cache_bytes"] = available_kv_cache_bytes
mc["gpu"]["available_kv_cache_tokens"] = kv_tokens
mc["gpu"]["per_token_kv_bytes"] = per_token_bytes
mc["gpu"]["kv_source_log"] = log_path
pack["model_config"] = mc

with open(pack_path, "w") as f:
    json.dump(pack, f)
print(f"Patched {pack_path}")
print(f"Backup at {pack_path}.bak")

#!/usr/bin/env python3
"""Download ShareGPT dataset and filter to fit archive's 256/128 profile envelope.

Output: ./results/sharegpt_filtered_256_128.json — sharegpt-compatible JSON
with entries where first human prompt tokens ≤256 AND first gpt response
tokens ≤128 (using AutoTokenizer for Qwen/Qwen3-8B).
"""
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: needs huggingface_hub + transformers", file=sys.stderr)
    sys.exit(1)

OUT = "./results/sharegpt_filtered_256_128.json"
MODEL = "Qwen/Qwen3-8B"
MAX_INPUT = 256
MAX_OUTPUT = 128
TARGET_N = 5000  # filter count

print(f"Loading tokenizer {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("Downloading ShareGPT_V3_unfiltered_cleaned_split.json (may take a few min)...")
path = hf_hub_download(
    repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
    filename="ShareGPT_V3_unfiltered_cleaned_split.json",
    repo_type="dataset",
)
print(f"Downloaded to {path}")

with open(path) as f:
    data = json.load(f)
print(f"Loaded {len(data)} conversations")

filtered = []
for conv in data:
    msgs = conv.get("conversations", [])
    if len(msgs) < 2:
        continue
    human_msg = None
    gpt_msg = None
    for m in msgs:
        if m.get("from") == "human" and human_msg is None:
            human_msg = m.get("value", "")
        elif m.get("from") == "gpt" and human_msg is not None:
            gpt_msg = m.get("value", "")
            break
    if not human_msg or not gpt_msg:
        continue
    try:
        in_len = len(tok.encode(human_msg))
        out_len = len(tok.encode(gpt_msg))
    except Exception:
        continue
    if in_len <= MAX_INPUT and out_len <= MAX_OUTPUT and in_len >= 10 and out_len >= 10:
        filtered.append({
            "id": conv.get("id", f"idx_{len(filtered)}"),
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": gpt_msg},
            ],
        })
    if len(filtered) >= TARGET_N:
        break

print(f"Filtered to {len(filtered)} entries with input≤{MAX_INPUT}, output≤{MAX_OUTPUT}")
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(filtered, f)
print(f"Saved to {OUT}")

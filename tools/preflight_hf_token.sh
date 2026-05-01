#!/bin/bash
# Verify HF token + access to a target gated model before launching a
# long-running profile capture or bench. Used before the Llama-3.1-8B
# cell (gated repo).
#
# Usage:
#   bash tools/preflight_hf_token.sh <hf_repo_id>
#   e.g. bash tools/preflight_hf_token.sh meta-llama/Llama-3.1-8B
#
# Exit 0 on access OK; exit 1 with reason on failure.

set -u

REPO="${1:-}"
if [ -z "$REPO" ]; then
    echo "Usage: $0 <hf_repo_id>" >&2
    exit 2
fi

# Whoami
WHOAMI="$(huggingface-cli whoami 2>&1 || true)"
if echo "$WHOAMI" | grep -qE 'Not logged in|401|403'; then
    echo "FATAL: HF token missing or invalid." >&2
    echo "       Run: huggingface-cli login" >&2
    exit 1
fi
echo "HF user: $(echo "$WHOAMI" | head -1)"

# Lightweight access probe — fetch config.json (small, gated repos still
# return 401/403 if the user lacks access).
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if ! huggingface-cli download "$REPO" --include "config.json" --local-dir "$TMP" >/dev/null 2>&1; then
    echo "FATAL: cannot fetch $REPO/config.json. Likely missing license/agreement." >&2
    echo "       Visit https://huggingface.co/$REPO and accept terms." >&2
    exit 1
fi

if [ ! -f "$TMP/config.json" ]; then
    echo "FATAL: download succeeded but config.json missing — repo may be empty." >&2
    exit 1
fi

echo "PREFLIGHT OK $REPO ($(wc -c < "$TMP/config.json") bytes config.json)"

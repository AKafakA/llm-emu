#!/bin/bash
# Clean reinstall with VLLM_USE_PRECOMPILED.
# Kills any running builds, recreates venv from scratch.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd ~/Code/llm/vllm-emulator

LOG=/tmp/vllm_reinstall.log
echo "=== Clean precompiled reinstall at $(date) ===" > "$LOG"

# Kill any running builds
pkill -f "uv pip install" 2>/dev/null || true
pkill -f ninja 2>/dev/null || true
pkill -f cmake 2>/dev/null || true
sleep 3

# Remove old venv and build artifacts
rm -rf .venv
rm -f vllm/_C*.so
rm -rf .deps build
echo "Cleaned old venv and artifacts" >> "$LOG"

# Fresh venv
uv venv --python 3.12
source .venv/bin/activate
echo "Python: $(python3 --version)" >> "$LOG"

# Force precompiled wheel variant to match torch cu128
# System CUDA 13.1 causes auto-detect to pick cu130, but torch resolves to cu128
echo "Installing with VLLM_USE_PRECOMPILED (variant=cu128)..." >> "$LOG"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu128 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0 uv pip install -e . >> "$LOG" 2>&1

echo "PyTorch version:" >> "$LOG"
python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')" >> "$LOG" 2>&1

echo "Testing vllm._C import:" >> "$LOG"
python3 -c "import vllm._C; print('vllm._C OK')" >> "$LOG" 2>&1
RC=$?

if [ $RC -eq 0 ]; then
    echo "=== PRECOMPILED INSTALL SUCCEEDED ===" >> "$LOG"
    # Install test deps
    uv pip install pytest pytest-asyncio >> "$LOG" 2>&1
else
    echo "=== PRECOMPILED INSTALL FAILED ===" >> "$LOG"
    echo "Falling back to source build..." >> "$LOG"
    SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0 MAX_JOBS=4 uv pip install -e . >> "$LOG" 2>&1
    python3 -c "import vllm._C; print('vllm._C OK (source)')" >> "$LOG" 2>&1
fi

echo "=== Done at $(date) ===" >> "$LOG"
cat "$LOG"

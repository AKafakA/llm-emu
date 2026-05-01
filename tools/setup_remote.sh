#!/bin/bash
# Setup script for vllm-emulator on a fresh GPU host.
# Creates venv, installs dependencies, verifies GPU access.
#
# Usage: bash setup_remote.sh
set -euo pipefail

REPO_DIR="$HOME/Code/llm/vllm-emulator"
VENV_DIR="$REPO_DIR/.venv"

echo "=== vllm-emulator setup ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(python3 --version)"
echo ""

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi
echo "Creating venv (Python 3.12)..."
cd "$REPO_DIR"
uv venv --python 3.12
source "$VENV_DIR/bin/activate"

# Install vllm-emulator
# SETUPTOOLS_SCM_PRETEND_VERSION needed because we rsync without .git
# Try precompiled first (fast), fall back to source build if ABI mismatch
echo "Installing vllm-emulator..."
if VLLM_USE_PRECOMPILED=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0 uv pip install -e . 2>&1 | tail -5; then
    # Verify the C extension actually loads
    if python3 -c "import vllm._C" 2>/dev/null; then
        echo "Precompiled install OK"
    else
        echo "Precompiled .so ABI mismatch — rebuilding from source (this takes ~20-30 min)..."
        SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0 MAX_JOBS=4 uv pip install -e . 2>&1 | tail -5
    fi
else
    echo "Precompiled install failed — building from source (~20-30 min)..."
    SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0 MAX_JOBS=4 uv pip install -e . 2>&1 | tail -5
fi

# Install test/bench dependencies
echo "Installing test dependencies..."
uv pip install pytest pytest-asyncio 2>&1 | tail -3

# Download model weights (Qwen3-8B)
echo "Pre-downloading Qwen/Qwen3-8B model..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', ignore_patterns=['*.gguf', '*.bin'])
print('Model downloaded OK')
" 2>&1 | tail -3

# Verify installation
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'SM count: {torch.cuda.get_device_properties(0).multi_processor_count}')
    print(f'CC: {torch.cuda.get_device_capability(0)}')
"

python3 -c "
from vllm_emulator.hooks.executor_hook import ExecutorEmulatorHook
from vllm_emulator.profiler.trace_profiler import StepCycleTracer
print('Emulator imports: OK')
print('StepCycleTracer.write_header:', hasattr(StepCycleTracer, 'write_header'))
"

echo ""
echo "=== Setup complete ==="
echo "Activate: source $VENV_DIR/bin/activate"
echo "Run eval: bash tools/e2e/profile_and_eval.sh --model Qwen/Qwen3-8B --result-dir ./results/RTX-8000"

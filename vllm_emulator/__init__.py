# SPDX-License-Identifier: Apache-2.0
"""LLM-Emu — profile-driven online emulator for vLLM."""

__version__ = "0.1.0"

# cuda_mock auto-installs on import; it self-gates on
# VLLM_EMULATOR_ENABLE_ORACLE / VLLM_EMULATOR_MOCK_CUDA so non-emulator
# processes are unaffected.
import os as _os
if (_os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true", "yes")
        or _os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() in ("1", "true", "yes")):
    try:
        from . import cuda_mock as _cuda_mock  # noqa: F401
    except Exception:
        pass

try:
    from .platform import EmulatorPlatform, emulator_platform_plugin
    __all__ = ["EmulatorPlatform", "emulator_platform_plugin"]
except Exception:
    __all__ = []

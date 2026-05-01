# SPDX-License-Identifier: Apache-2.0
"""GhostServe platform plugin for vLLM."""

from typing import TYPE_CHECKING

import torch

from vllm.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

DEFAULT_EMULATOR_MEMORY = 80 * 1024**3


def _load_gpu_config_from_profile() -> dict:
    import os
    profile_path = os.environ.get("VLLM_EMULATOR_PROFILE_PACK", "")
    if not profile_path or not os.path.exists(profile_path):
        return {}
    try:
        import json
        with open(profile_path) as f:
            pack = json.load(f)
        return pack.get("model_config", {}).get("gpu", {})
    except Exception:
        return {}


_GPU_CONFIG_CACHE: dict | None = None


def _get_gpu_config() -> dict:
    global _GPU_CONFIG_CACHE
    if _GPU_CONFIG_CACHE is None:
        _GPU_CONFIG_CACHE = _load_gpu_config_from_profile()
    return _GPU_CONFIG_CACHE


class EmulatorPlatform(Platform):
    """vLLM platform plugin that simulates GPU behavior from a profile pack."""

    _enum = PlatformEnum.OOT
    device_name = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CPU"
    ray_device_key: str = ""

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        from vllm.platforms.interface import DeviceCapability
        gpu = _get_gpu_config()
        cc = gpu.get("gpu_compute_capability", [8, 0])
        return DeviceCapability(major=cc[0], minor=cc[1])

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        gpu = _get_gpu_config()
        return gpu.get("gpu_name", "Emulator Device")

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"emulator-{device_id}-0000-000000000000"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        import os
        env_mem = os.environ.get("VLLM_EMULATOR_MEMORY")
        if env_mem:
            return int(env_mem)
        gpu = _get_gpu_config()
        return gpu.get("gpu_memory_bytes", DEFAULT_EMULATOR_MEMORY)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        return 0.0

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        vllm_config.compilation_config.enable = False
        # Keep device="cuda" so vllm sizes num_gpu_blocks from the
        # cuda_mock'd target memory and applies its built-in KV admission.
        # device="cpu" routes to CPUWorker which sizes from system RAM
        # and oversizes the pool by ~30% on memory-constrained targets.
        import os
        _emu_active = (
            os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower()
                in ("1", "true", "yes")
            or os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower()
                in ("1", "true", "yes")
        )
        if _emu_active:
            vllm_config.device_config.device = "cuda"
        else:
            vllm_config.device_config.device = "cpu"
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = (
                "vllm.v1.worker.gpu_worker.Worker"
            )

    @classmethod
    def check_if_supports_dtype(cls, dtype: "torch.dtype") -> None:
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False

    @classmethod
    def get_attn_backend_cls(cls, backend, attn_selector_config=None,
                             num_heads=None, **kwargs):
        # The hook intercepts execute_model before any attention kernel
        # runs, so the backend is never actually used. TRITON_ATTN has the
        # lightest init footprint and bypasses the real-GPU capability
        # check that would reject FLASHINFER/FLEX_ATTENTION on cpu_host.
        return "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        gpu = _get_gpu_config()
        return gpu.get("gpu_sm_count", 108)

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return False


def emulator_platform_plugin() -> str | None:
    """Entry point for `vllm.platform_plugins`."""
    import os
    if (os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() in ("1", "true", "yes")
            or os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() in ("1", "true", "yes")):
        return "vllm_emulator.platform.EmulatorPlatform"
    return None

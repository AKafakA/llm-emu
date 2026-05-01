"""Base interface for the GPU step-latency oracle."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGpuCostOracle(ABC):
    """Estimate per-step forward-pass latency in microseconds."""

    @abstractmethod
    def estimate_step_latency_us(
        self, total_tokens: int,
        has_prefill: bool = False,
        num_requests: int = 0,
        **kwargs,
    ) -> float:
        """Estimate latency for one scheduler step."""

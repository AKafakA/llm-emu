"""Profile-driven GPU step-latency oracle.

Given (total_tokens, num_requests, has_prefill), looks up the nearest
populated bucket in the 2D (tt, conc) distribution and samples uniformly
from its raw latency samples. No synthetic fallbacks, no heuristic
constants. Set VLLM_EMULATOR_DEBUG=1 to log every call to stderr.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

from .base import BaseGpuCostOracle


_DEBUG = os.environ.get("VLLM_EMULATOR_DEBUG", "").lower() in ("1", "true", "yes")


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[OracleDebug] {msg}", file=sys.stderr, flush=True)


class ProfileGpuCostOracle(BaseGpuCostOracle):
    """Samples per-step latency from profiled 2D (tt, conc) distributions.

    Lookup uses nearest-neighbor (K=1) by default; setting
    VLLM_EMULATOR_ORACLE_K=auto enables adaptive-K Shepard inverse-distance
    pooling that grows the neighbor set until cumulative bucket size
    reaches VLLM_EMULATOR_ORACLE_MIN_SAMPLES (default 30).
    """

    def __init__(self, profile_pack: dict[str, Any]):
        self._profile = profile_pack
        self._gpu_model = profile_pack["gpu_model"]
        self._rng = random.Random(42)

        # Adaptive-K kNN. K=1 (default) keeps the nearest-neighbor path.
        # K="auto" expands the neighbor set per-query until cumulative
        # sample count reaches the reliability floor M.
        k_env = os.environ.get("VLLM_EMULATOR_ORACLE_K", "1")
        self._oracle_k_mode = "fixed"
        self._oracle_k_min_samples = 0
        if k_env.lower() in ("auto", "adaptive"):
            self._oracle_k_mode = "auto"
            self._oracle_k = 1
            m_env = os.environ.get("VLLM_EMULATOR_ORACLE_MIN_SAMPLES", "30")
            try:
                self._oracle_k_min_samples = int(m_env)
            except (TypeError, ValueError):
                raise ValueError(
                    f"VLLM_EMULATOR_ORACLE_MIN_SAMPLES must be an integer, "
                    f"got {m_env!r}")
            if self._oracle_k_min_samples < 1:
                raise ValueError(
                    f"VLLM_EMULATOR_ORACLE_MIN_SAMPLES must be >= 1, "
                    f"got {self._oracle_k_min_samples}")
        else:
            try:
                self._oracle_k = int(k_env)
            except (TypeError, ValueError):
                raise ValueError(
                    f"VLLM_EMULATOR_ORACLE_K must be an integer or "
                    f"'auto'/'adaptive', got {k_env!r}")
            if self._oracle_k < 1:
                raise ValueError(
                    f"VLLM_EMULATOR_ORACLE_K must be >= 1, got {self._oracle_k}")

        # Optional percentile trim on raw samples (default 0,100 = no trim).
        # Trimming dense saturation profiles biases emu fast — opt in only
        # for explicit diagnostics.
        trim_env = os.environ.get("VLLM_EMULATOR_SAMPLE_TRIM", "")
        if trim_env:
            parts = trim_env.split(",")
            self._trim_lo = int(parts[0])
            self._trim_hi = int(parts[1])
        else:
            self._trim_lo = 0
            self._trim_hi = 100

        # 2D (tt, conc) distributions — decode-only, prefill-or-mixed,
        # and a combined fallback for older profile packs.
        self._decode_2d_distribution: dict[tuple[int, int], dict] = {
            (e["tt"], e["conc"]): e
            for e in profile_pack.get("decode_2d_distribution", [])
        }
        self._prefill_2d_distribution: dict[tuple[int, int], dict] = {
            (e["tt"], e["conc"]): e
            for e in profile_pack.get("prefill_2d_distribution", [])
        }
        self._combined_2d_distribution: dict[tuple[int, int], dict] = {
            (e["tt"], e["conc"]): e
            for e in profile_pack.get("step_cycle_2d_distribution", [])
        }

        # Pre-trim samples in each bucket once at load time.
        for table in (self._decode_2d_distribution,
                      self._prefill_2d_distribution,
                      self._combined_2d_distribution):
            for bucket in table.values():
                raw = bucket.get("samples")
                if raw:
                    bucket["samples"] = self._trim_samples(raw)

    def _trim_samples(self, samples: list[float]) -> list[float]:
        if self._trim_lo == 0 and self._trim_hi == 100:
            return samples
        n = len(samples)
        if n == 0:
            return samples
        sorted_s = sorted(samples)
        lo_idx = int(n * self._trim_lo / 100)
        hi_idx = int(n * self._trim_hi / 100)
        if hi_idx <= lo_idx:
            return samples
        return sorted_s[lo_idx:hi_idx]

    @property
    def gpu_model(self) -> str:
        return self._gpu_model

    def _sample(self, samples):
        return float(self._rng.choice(samples)) if samples else None

    def _sample_2d_distribution(
        self, total_tokens: int, num_requests: int,
        has_prefill: bool = False,
    ) -> float | None:
        """Sample latency from the (tt, conc) bucket nearest the query.

        Picks the prefill-or-mixed table if has_prefill (else decode), with
        the combined table as fallback for older profile packs. K=1 uses
        nearest-neighbor; K="auto" pools via Shepard p=2 inverse-distance
        weighting until the cumulative sample floor is met.
        """
        if has_prefill and self._prefill_2d_distribution:
            table = self._prefill_2d_distribution
        elif not has_prefill and self._decode_2d_distribution:
            table = self._decode_2d_distribution
        elif self._combined_2d_distribution:
            table = self._combined_2d_distribution
        else:
            return None

        if self._oracle_k_mode == "auto":
            return self._sample_knn_adaptive_2d(
                table, total_tokens, num_requests,
                min_samples=self._oracle_k_min_samples,
            )

        if self._oracle_k > 1:
            return self._sample_knn_2d(table, total_tokens, num_requests)

        # K=1 nearest-neighbor.
        tts_in_table = sorted({k[0] for k in table.keys()})
        if not tts_in_table:
            return None
        if total_tokens <= tts_in_table[0]:
            tt_near = tts_in_table[0]
        elif total_tokens >= tts_in_table[-1]:
            tt_near = tts_in_table[-1]
        else:
            tt_near = min(tts_in_table, key=lambda t: abs(t - total_tokens))

        available_concs = [c for (tt, c) in table.keys() if tt == tt_near]
        if not available_concs:
            return None
        conc_near = min(available_concs, key=lambda c: abs(c - num_requests))

        bucket = table.get((tt_near, conc_near))
        if not bucket:
            return None
        return self._sample(bucket.get("samples"))

    def _sample_knn_adaptive_2d(self, table, total_tokens, num_requests,
                                 min_samples: int):
        """Adaptive-K: grow K until cumulative bucket samples >= min_samples,
        then Shepard p=2 inverse-distance-weighted draw."""
        keys = list(table.keys())
        if not keys:
            return None

        tts = [k[0] for k in keys]
        concs = [k[1] for k in keys]
        tt_range = (max(tts) - min(tts)) or 1
        conc_range = (max(concs) - min(concs)) or 1

        def _dist(k):
            dt = (k[0] - total_tokens) / tt_range
            dc = (k[1] - num_requests) / conc_range
            return (dt * dt + dc * dc) ** 0.5

        ranked = sorted(((_dist(k), k) for k in keys), key=lambda x: x[0])

        chosen = []
        cum_samples = 0
        for d, k in ranked:
            n = len(table[k].get("samples") or [])
            if n == 0:
                continue
            chosen.append((d, k))
            cum_samples += n
            if cum_samples >= min_samples:
                break

        if not chosen:
            return None

        # Exact-match hit — return that bucket directly.
        if chosen[0][0] == 0.0:
            return self._sample(table[chosen[0][1]].get("samples"))

        # Single-bucket case: nearest already meets the floor.
        if len(chosen) == 1:
            return self._sample(table[chosen[0][1]].get("samples"))

        weights = [1.0 / (d * d) for d, _ in chosen]
        total_w = sum(weights)
        if total_w == 0:
            return None
        r = self._rng.random() * total_w
        cum_w = 0.0
        chosen_k = chosen[-1][1]
        for (d, k), w in zip(chosen, weights):
            cum_w += w
            if r <= cum_w:
                chosen_k = k
                break
        return self._sample(table[chosen_k].get("samples"))

    def _sample_knn_2d(self, table, total_tokens, num_requests):
        """Fixed-K Shepard p=2 kNN over range-normalised (tt, conc)."""
        keys = list(table.keys())
        if not keys:
            return None

        tts = [k[0] for k in keys]
        concs = [k[1] for k in keys]
        tt_range = (max(tts) - min(tts)) or 1
        conc_range = (max(concs) - min(concs)) or 1

        def _dist(k):
            dt = (k[0] - total_tokens) / tt_range
            dc = (k[1] - num_requests) / conc_range
            return (dt * dt + dc * dc) ** 0.5

        ranked = sorted(((_dist(k), k) for k in keys), key=lambda x: x[0])
        top_k = ranked[: self._oracle_k]

        for d, k in top_k:
            if d == 0.0:
                return self._sample(table[k].get("samples"))

        weights = [1.0 / (d * d) for d, _ in top_k]
        total_w = sum(weights)
        if total_w == 0:
            return None
        r = self._rng.random() * total_w
        cum = 0.0
        chosen_k = top_k[-1][1]
        for (d, k), w in zip(top_k, weights):
            cum += w
            if r <= cum:
                chosen_k = k
                break
        return self._sample(table[chosen_k].get("samples"))

    def estimate_step_latency_us(
        self, total_tokens: int,
        has_prefill: bool = False,
        num_requests: int = 0,
        num_new_reqs: int = 0,
        sum_kv: int = 0,
        **kwargs,
    ) -> float:
        """Sample a per-step latency for the given batch shape."""
        if total_tokens <= 0:
            return 0.0

        result = self._sample_2d_distribution(
            total_tokens, max(num_requests, 1),
            has_prefill=has_prefill,
        )
        if result is None:
            result = 0.0
        _dbg(
            f"call tt={total_tokens} conc={num_requests} sum_kv={sum_kv} "
            f"has_prefill={has_prefill} -> sample_us={result:.1f}"
        )
        return result


def create_oracle_from_profile_pack(
    profile_pack: dict[str, Any],
) -> ProfileGpuCostOracle:
    return ProfileGpuCostOracle(profile_pack)

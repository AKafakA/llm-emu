"""Executor-level hook for emulator mode.

Replaces vLLM's GPU forward pass with a profile-driven latency draw.
The oracle predicts step_cycle_us; the hook returns a pending Future
that resolves after that delay (async engine) or sleeps then returns
the output directly (sync engine).
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING

from vllm_emulator.oracle import BaseGpuCostOracle, create_oracle_from_profile_pack
from vllm_emulator.profile.loader import load_profile_pack

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

ORACLE_ENABLED_ENV = "VLLM_EMULATOR_ENABLE_ORACLE"
ORACLE_PROFILE_PATH_ENV = "VLLM_EMULATOR_PROFILE_PACK"
ORACLE_MODE_ENV = "VLLM_EMULATOR_MODE"

EMULATOR_MODE_REALTIME = "realtime"
EMULATOR_MODE_ACCELERATED = "accelerated"
_MODE_ALIASES = {"online": EMULATOR_MODE_REALTIME}


class ExecutorEmulatorHook:
    """Executor-level emulator hook.

    realtime mode: time.sleep() / threading.Timer for predicted latency.
    accelerated mode: resolve immediately (fast simulation).

    Async engine returns a pending Future via threading.Timer — the
    scheduler requires num_output_placeholders > 0 when the next batch
    is scheduled, so a resolved Future deadlocks. Sync engine blocks
    on time.sleep() and returns the output directly.
    """

    def __init__(self):
        self._oracle: BaseGpuCostOracle | None = None
        self._enabled = False
        self._emulator_mode = EMULATOR_MODE_REALTIME
        self._sample_future: Future | None = None
        self._gpu_free_time = 0.0  # virtual GPU timeline for chain timer

        # Fake-output sampling. _filler_token_id is a fixed non-stop id;
        # _eos_token_ids are populated from the profile pack so we never
        # accidentally emit a stop and end the request early.
        self._vocab_size = 32000
        self._eos_token_ids: set[int] = {2}
        self._filler_token_id = 100
        self._debug_count = 0

        self._initialize()

    def _initialize(self) -> None:
        if os.environ.get(ORACLE_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
            return

        profile_path = os.environ.get(ORACLE_PROFILE_PATH_ENV)
        if not profile_path:
            return

        mode = os.environ.get(ORACLE_MODE_ENV, EMULATOR_MODE_REALTIME).lower()
        mode = _MODE_ALIASES.get(mode, mode)
        if mode not in (EMULATOR_MODE_REALTIME, EMULATOR_MODE_ACCELERATED):
            print(f"[ExecutorEmulatorHook] Unknown mode '{mode}', using realtime")
            mode = EMULATOR_MODE_REALTIME
        self._emulator_mode = mode

        try:
            profile_pack = load_profile_pack(profile_path)
            self._oracle = create_oracle_from_profile_pack(profile_pack)
            self._enabled = True

            model_cfg = profile_pack.get("model_config", {})
            if model_cfg.get("vocab_size"):
                self._vocab_size = model_cfg["vocab_size"]
            # eos_token_id may be int or list (e.g., Qwen3 ships
            # <|endoftext|>=151643 and <|im_end|>=151645).
            eos = model_cfg.get("eos_token_id")
            if isinstance(eos, int):
                self._eos_token_ids = {eos}
            elif isinstance(eos, (list, tuple)):
                self._eos_token_ids = {int(t) for t in eos if isinstance(t, int)}
            for _t in range(100, max(101, self._vocab_size)):
                if _t not in self._eos_token_ids:
                    self._filler_token_id = _t
                    break

            print(f"[ExecutorEmulatorHook] Enabled: mode={self._emulator_mode}, "
                  f"filler_id={self._filler_token_id}, "
                  f"stops={sorted(self._eos_token_ids)}")
        except Exception as e:
            print(f"[ExecutorEmulatorHook] Failed to initialize: {e}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def should_use_oracle(self, scheduler_output: "SchedulerOutput") -> bool:
        return self._enabled and scheduler_output.total_num_scheduled_tokens > 0

    def has_pending_future(self) -> bool:
        return self._sample_future is not None

    def get_sample_future(self) -> "Future":
        fut = self._sample_future
        self._sample_future = None
        return fut

    def create_delayed_future(
        self,
        scheduler_output: "SchedulerOutput",
        non_block: bool = False,
    ) -> "Future | ModelRunnerOutput | None":
        """Create output with predicted GPU latency.

        non_block=True (async engine) returns a pending Future via
        threading.Timer; non_block=False (sync engine) blocks on
        time.sleep and stores the output for get_pending_output().
        """
        total_tokens = scheduler_output.total_num_scheduled_tokens
        has_prefill = len(scheduler_output.scheduled_new_reqs) > 0

        new_req_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        num_decode = sum(
            1 for rid in scheduler_output.num_scheduled_tokens
            if rid not in new_req_ids
        )
        num_new = len(scheduler_output.scheduled_new_reqs)
        num_total_reqs = len(scheduler_output.num_scheduled_tokens)

        cached = scheduler_output.scheduled_cached_reqs
        sum_kv = sum(cached.num_computed_tokens) if cached.num_reqs > 0 else 0

        latency_us = self._oracle.estimate_step_latency_us(
            total_tokens,
            has_prefill=has_prefill,
            num_requests=num_total_reqs,
            num_new_reqs=num_new,
            sum_kv=sum_kv,
        )
        latency_s = latency_us / 1e6

        fake_output = self._create_fake_output(scheduler_output)
        if fake_output is None:
            if non_block:
                fut: Future = Future()
                fut.set_result(None)
                return fut
            return None

        if not non_block:
            return self._handle_sync(latency_s, fake_output)
        return self._handle_async(latency_s, fake_output,
                                  total_tokens, scheduler_output)

    def _handle_sync(self, latency_s: float, fake_output) -> None:
        if self._emulator_mode == EMULATOR_MODE_REALTIME and latency_s >= 0.001:
            time.sleep(latency_s)
        self._pending_output = fake_output
        return None

    def _handle_async(self, latency_s: float, fake_output,
                      total_tokens: int,
                      scheduler_output: "SchedulerOutput") -> "Future":
        exec_fut: Future = Future()
        exec_fut.set_result(None)

        sample_fut: Future = Future()
        if self._emulator_mode == EMULATOR_MODE_REALTIME:
            now = time.perf_counter()
            start_time = max(now, self._gpu_free_time)
            end_time = start_time + latency_s
            self._gpu_free_time = end_time
            delay = end_time - now

            if delay >= 0.001:
                timer = threading.Timer(delay,
                    lambda: sample_fut.set_result(fake_output))
                timer.daemon = True
                timer.start()
            else:
                sample_fut.set_result(fake_output)
        else:
            sample_fut.set_result(fake_output)

        self._sample_future = sample_fut

        self._debug_count += 1
        if self._debug_count <= 10 or self._debug_count % 100 == 0:
            n_reqs = len(scheduler_output.num_scheduled_tokens)
            n_new = len(scheduler_output.scheduled_new_reqs)
            print(f"[ExecutorHook] step={self._debug_count} tt={total_tokens} "
                  f"reqs={n_reqs} new={n_new} "
                  f"latency={latency_s*1000:.1f}ms")

        return exec_fut

    def has_pending_output(self) -> bool:
        return getattr(self, '_pending_output', None) is not None

    def get_pending_output(self):
        output = self._pending_output
        self._pending_output = None
        return output

    def _create_fake_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | None":
        from vllm.v1.outputs import ModelRunnerOutput
        import numpy as np

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        if not req_ids:
            return None

        # A request is mid-prefill iff cached + scheduled_this_step <
        # total_prompt_len. Without the cached check, prefix-cache hits
        # that finish prefill in one chunk are misclassified and miss
        # their first decode token.
        prefill_chunk_ids = set()
        for req in scheduler_output.scheduled_new_reqs:
            if req.prompt_token_ids:
                scheduled = scheduler_output.num_scheduled_tokens.get(req.req_id, 0)
                computed = getattr(req, "num_computed_tokens", 0) or 0
                if computed + scheduled < len(req.prompt_token_ids):
                    prefill_chunk_ids.add(req.req_id)

        sampled_token_ids = []
        for req_id in req_ids:
            if req_id in prefill_chunk_ids:
                sampled_token_ids.append([])
            else:
                sampled_token_ids.append([self._filler_token_id])

        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}

        num_with_tokens = sum(1 for toks in sampled_token_ids if toks)
        if num_with_tokens > 0:
            from vllm.v1.outputs import LogprobsLists
            token_vals = [toks[0] for toks in sampled_token_ids if toks]
            logprobs = LogprobsLists(
                logprob_token_ids=np.array([[t] for t in token_vals], dtype=np.int32),
                logprobs=np.full((num_with_tokens, 1), -0.1, dtype=np.float32),
                sampled_token_ranks=np.zeros(num_with_tokens, dtype=np.int32),
            )
        else:
            logprobs = None

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )

    def shutdown(self):
        pass


def get_executor_hook() -> ExecutorEmulatorHook | None:
    if os.environ.get(ORACLE_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
        return None
    hook = ExecutorEmulatorHook()
    return hook if hook.is_enabled else None

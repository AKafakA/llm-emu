# vLLM patches for GhostServe

GhostServe needs three small wirings inside vLLM (~170 lines total)
so that the plugin's executor hook receives `execute_model()` calls
and the profile-capture tracer (StepCycleTracer) records per-step
latency during real-GPU runs.

## Files

| | path | role | LoC added |
| --- | --- | --- | --- |
| 1 | `vllm/v1/engine/core.py` | StepCycleTracer init + per-step record points | +135 |
| 2 | `vllm/v1/executor/uniproc_executor.py` | hand off `execute_model` to `vllm_emulator.hooks.executor_hook.get_executor_hook()` when `VLLM_EMULATOR_EXECUTOR_HOOK=1` | +35 |
| 3 | `vllm/v1/core/sched/scheduler.py` | three-line shim | +3 |

All three change vLLM's call graph only when the corresponding env vars
are set; with them unset, vLLM behaves identically to upstream.

## How to apply

### Option 1 — patch

```bash
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
cd "$VLLM_DIR/.." && patch -p1 < ghostserve-vllm-0.18.1.patch
```

The patch is a unified diff against vllm at commit
`a26e8dc7f0` (≈ release v0.18.1). See `BASE_COMMIT`.

### Option 2 — file override

If you can't apply the patch (e.g., a binary wheel install), drop the
three pre-patched files from `overrides/vllm/...` over the matching
files in your installed `vllm/` package:

```bash
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
cp overrides/vllm/v1/engine/core.py             "$VLLM_DIR/v1/engine/core.py"
cp overrides/vllm/v1/executor/uniproc_executor.py "$VLLM_DIR/v1/executor/uniproc_executor.py"
cp overrides/vllm/v1/core/sched/scheduler.py    "$VLLM_DIR/v1/core/sched/scheduler.py"
```

The override files are byte-equivalent to running the patch on a
clean upstream checkout — they're emitted from the same source tree
the patch was generated from.

## Verifying the install

After applying either way:

```bash
python -c "
import vllm.v1.engine.core as c
assert 'VLLM_EMULATOR_TRACE_STEP_CYCLE' in open(c.__file__).read()
print('GhostServe vLLM patches: applied')
"
```

## Compatibility

The patches target vllm v0.18.1 (commit `a26e8dc7f0`). Upgrading vLLM
to a newer release likely requires regenerating the patch — the three
target functions in `core.py` and `uniproc_executor.py` may have
moved, but the wiring is small enough (~170 lines) to port by hand
once the relevant call sites are located.

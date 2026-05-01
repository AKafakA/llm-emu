"""Minimal CUDA mock for running GPU vLLM on CPU-only hosts.

When VLLM_EMULATOR_MOCK_CUDA=1, this module patches torch.cuda to return
fake values, allowing GPU vLLM code paths to execute on CPU-only hosts.
The emulator hook intercepts execute_model() before any real GPU work,
so the mock only needs to fool startup, scheduling, and memory management.

Usage:
    export VLLM_EMULATOR_MOCK_CUDA=1
    # Then import before vllm:
    import vllm_emulator.cuda_mock  # patches torch.cuda

Or call vllm_emulator.cuda_mock.install() explicitly.
"""

import os
import sys
from unittest.mock import MagicMock

# Default fake GPU memory (12GB, configurable)
_FAKE_GPU_MEMORY = int(os.environ.get("VLLM_EMULATOR_MEMORY",
                                        12 * 1024**3))


def _load_gpu_config():
    """Load GPU metadata from profile pack for accurate CUDA mocking."""
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


def _install_module_patcher(target, patch_fn, label):
    """Install a meta-path import hook that calls patch_fn(module) AFTER
    `target` is imported. Removes itself after one fire. If `target`
    is already in sys.modules, patch immediately."""
    import importlib.abc
    import importlib.util

    if target in sys.modules:
        try:
            patch_fn(sys.modules[target])
            print(f"[EmulatorCudaMock] (eager) patched {label}")
        except Exception as e:
            print(f"[EmulatorCudaMock] eager patch {label} failed: {e}")
        return

    class _Patcher(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, _target=None):
            if fullname != target:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            outer = self
            real_loader = spec.loader

            class _WrappedLoader:
                def create_module(self, _spec):
                    return real_loader.create_module(_spec)

                def exec_module(self, module):
                    real_loader.exec_module(module)
                    try:
                        patch_fn(module)
                        print(f"[EmulatorCudaMock] (deferred) patched {label}")
                    except Exception as e:
                        print(f"[EmulatorCudaMock] deferred patch {label} failed: {e}")
                    try:
                        sys.meta_path.remove(outer)
                    except ValueError:
                        pass

            spec.loader = _WrappedLoader()
            return spec

    sys.meta_path.insert(0, _Patcher())
    print(f"[EmulatorCudaMock] meta-path patcher armed for {label}")


def _patch_gpu_model_runner(module):
    """Stub out model-related GPUModelRunner methods so the worker
    NEVER materializes weights or runs forward passes during init.
    The emulator hook drives all latency from the profile pack — we
    are emulating GPU behavior, not running vllm-cpu.

    Patches:
    - synchronize_input_prep: no-op contextmanager (avoids
      prepare_inputs_event.record() which hits C-locked CUDA event API).
    - load_model: no-op (no weight materialization on CPU).
    - profile_run: no-op (no forward-pass-for-memory-sizing).
    - _dummy_run: no-op stub returning empty hidden states.
    """
    from contextlib import contextmanager
    @contextmanager
    def _noop_ctx(self):
        yield
    if hasattr(module, 'GPUModelRunner'):
        cls = module.GPUModelRunner
        cls.synchronize_input_prep = _noop_ctx

        def _noop_load_model(self, *a, **kw):
            print("[EmulatorCudaMock] GPUModelRunner.load_model SKIPPED")
        cls.load_model = _noop_load_model

        def _noop_profile_run(self, *a, **kw):
            print("[EmulatorCudaMock] GPUModelRunner.profile_run SKIPPED")
            return None
        cls.profile_run = _noop_profile_run

        def _noop_dummy_run(self, num_tokens=0, *a, **kw):
            import torch as _t
            empty = _t.zeros((0, 0))
            return empty, empty
        cls._dummy_run = _noop_dummy_run

        # Some vllm code paths (e.g. get_supported_generation_tasks)
        # call get_model() to introspect the loaded model. With load
        # skipped we have no real model — return a benign mock so any
        # introspection .__class__.__mro__ etc. doesn't crash.
        def _stub_get_model(self):
            from unittest.mock import MagicMock
            return MagicMock()
        cls.get_model = _stub_get_model


def _patch_gpu_worker(module):
    """Stub out model-touching gpu_worker.Worker methods. Together
    with _patch_gpu_model_runner, this prevents any model weight
    instantiation, any forward pass, and any CUDA-init device call
    on a host with no real GPU. The bench will boot fast and the
    emulator hook + oracle drive all per-step latency."""
    if not hasattr(module, 'Worker'):
        return
    import torch as _t

    cls = module.Worker
    _orig_init_device = cls.init_device

    def _patched_init_device(self):
        # Skip torch.accelerator.set_device_index → torch._C._cuda_init.
        # Provide the minimal state that downstream code expects.
        self.device = _t.device('cuda', getattr(self, 'local_rank', 0))
        try:
            self.init_snapshot = type('S', (), {'free_gpu_mem': 0,
                                                 'total_gpu_mem': 0})()
        except Exception:
            self.init_snapshot = None
        # Try the original (set_device_index is now a no-op via
        # torch.accelerator patch). If it fails, build a minimal model
        # runner stand-in so downstream gpu_worker methods that touch
        # self.model_runner (reset_mm_cache, get_supported_tasks etc.)
        # don't crash with AttributeError on None.
        try:
            _orig_init_device(self)
        except Exception as e:
            print(f"[EmulatorCudaMock] gpu_worker.init_device fallback: {e}")
            try:
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                self.model_runner = GPUModelRunner(
                    vllm_config=self.vllm_config, device=self.device)
                print("[EmulatorCudaMock] built model_runner manually post-fallback")
            except Exception as e2:
                print(f"[EmulatorCudaMock] model_runner manual build failed: {e2}; "
                      f"installing stub")
                # Stub: provide every method the engine might call on
                # model_runner as a no-op. Skip-everything proxy.
                class _StubRunner:
                    kv_cache_config = None
                    def __getattr__(self, name):
                        return lambda *a, **kw: None
                self.model_runner = _StubRunner()
    cls.init_device = _patched_init_device

    def _noop_load_model(self, *a, **kw):
        print("[EmulatorCudaMock] gpu_worker.Worker.load_model SKIPPED "
              "(emulator drives latency from profile pack)")
    cls.load_model = _noop_load_model

    def _estimate_weights_bytes(model_config) -> int:
        """Approximate model weights memory from HF config + dtype.
        On real GPU, vllm measures this exactly via load_model. With
        load_model stubbed, we estimate from architecture so the
        emulator can subtract it from the KV budget — matching real
        vllm's available_kv = total × util − weights − activations.
        """
        try:
            hf = model_config.hf_config
            n_layers = getattr(hf, "num_hidden_layers", 32)
            h_dim = getattr(hf, "hidden_size", 4096)
            inter = getattr(hf, "intermediate_size", h_dim * 4)
            n_heads = getattr(hf, "num_attention_heads", 32)
            n_kv = getattr(hf, "num_key_value_heads", n_heads)
            head_dim = getattr(hf, "head_dim", h_dim // n_heads)
            vocab = getattr(hf, "vocab_size", 32000)
            # Per-layer params (transformer block):
            #   QKV proj : h × (n_heads + 2*n_kv) * head_dim
            #   O proj   : (n_heads * head_dim) × h
            #   MLP gate+up+down : 3 × h × inter
            #   layernorms small (~2h, ignore)
            qkv = h_dim * (n_heads + 2 * n_kv) * head_dim
            o   = n_heads * head_dim * h_dim
            mlp = 3 * h_dim * inter
            per_layer = qkv + o + mlp
            embed = vocab * h_dim       # token embedding
            head  = vocab * h_dim       # lm_head (often tied; double-count is small over-estimate, safe)
            n_params = n_layers * per_layer + embed + head
        except Exception:
            n_params = 8_000_000_000  # conservative default
        # Bytes per param from model dtype
        try:
            dtype_str = str(model_config.dtype).lower()
        except Exception:
            dtype_str = "bf16"
        if "32" in dtype_str:
            bpp = 4
        elif "8" in dtype_str and ("fp8" in dtype_str or "int8" in dtype_str):
            bpp = 1
        else:
            bpp = 2  # fp16 / bf16
        return int(n_params * bpp)

    def _stub_determine_available_memory(self) -> int:
        # available_kv_cache_memory_bytes priority:
        #   1. VLLM_EMULATOR_MEMORY env override
        #   2. profile pack's available_kv_cache_bytes (the value vllm
        #      computed on the real target during capture — this is what
        #      makes the emulator reproduce vllm's KV admission cliff)
        #   3. heuristic estimate from total - weights - activations
        env_mem = os.environ.get("VLLM_EMULATOR_MEMORY")
        if env_mem:
            available = int(env_mem)
            print(f"[EmulatorCudaMock] determine_available_memory → "
                  f"{available // (1024**3)} GiB (src=env)")
            return available
        gpu_cfg = _load_gpu_config()
        # Preferred: real-hardware-measured KV pool from profile capture.
        if gpu_cfg.get("available_kv_cache_bytes"):
            available = int(gpu_cfg["available_kv_cache_bytes"])
            print(f"[EmulatorCudaMock] determine_available_memory → "
                  f"{available // (1024**3)} GiB "
                  f"(src=profile_pack.available_kv_cache_bytes — exact match "
                  f"to real {gpu_cfg.get('gpu_name','?')} measurement)")
            return available
        # Heuristic fallback: subtract estimated weights + activations
        # from requested = total × util.
        if gpu_cfg.get("gpu_memory_bytes"):
            total = int(gpu_cfg["gpu_memory_bytes"])
            src = "profile_pack.gpu_memory_bytes (heuristic non-kv subtraction)"
        else:
            total = 80 * 1024**3
            src = "default-80GB (no profile loaded)"
        util = float(getattr(self.cache_config, "gpu_memory_utilization", 0.9))
        requested = int(total * util)
        weights = _estimate_weights_bytes(self.model_config)
        activations = 1 * 1024**3
        available = max(0, requested - weights - activations)
        print(f"[EmulatorCudaMock] determine_available_memory → "
              f"{available // (1024**3)} GiB "
              f"(total={total // (1024**3)} GiB × util={util:.2f} = "
              f"req={requested // (1024**3)} GiB; "
              f"− weights={weights // (1024**3)} GiB "
              f"− activations={activations // (1024**3)} GiB; "
              f"src={src})")
        return available
    cls.determine_available_memory = _stub_determine_available_memory

    def _stub_get_kv_cache_spec(self):
        """Synthesize KV cache spec from HF model config without loading
        weights. Mirrors what EmulatorWorker.get_kv_cache_spec does."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(
                self.model_config.model,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            num_layers = hf_config.num_hidden_layers
            num_kv_heads = getattr(hf_config, 'num_key_value_heads',
                                   getattr(hf_config, 'num_attention_heads', 32))
            head_dim = getattr(hf_config, 'head_dim',
                               hf_config.hidden_size // hf_config.num_attention_heads)
        except Exception as e:
            print(f"[EmulatorCudaMock] HF config fallback: {e}")
            num_layers, num_kv_heads, head_dim = 32, 8, 128
        tp_size = self.parallel_config.tensor_parallel_size
        num_kv_heads_per_tp = max(1, num_kv_heads // tp_size)
        block_size = self.cache_config.block_size
        specs = {}
        for i in range(num_layers):
            specs[f"model.layers.{i}.self_attn.attn"] = FullAttentionSpec(
                num_kv_heads=num_kv_heads_per_tp,
                head_size=head_dim,
                dtype=self.model_config.dtype,
                sliding_window=self.model_config.get_sliding_window(),
                block_size=block_size,
            )
        print(f"[EmulatorCudaMock] gpu_worker.Worker.get_kv_cache_spec"
              f" → {num_layers} layers, kv_heads/tp={num_kv_heads_per_tp},"
              f" head_dim={head_dim}, block_size={block_size}")
        return specs
    cls.get_kv_cache_spec = _stub_get_kv_cache_spec

    def _noop_compile_or_warmup(self, *a, **kw):
        return 0.0
    cls.compile_or_warm_up_model = _noop_compile_or_warmup

    def _stub_initialize_from_config(self, kv_cache_config=None, *a, **kw):
        # The original allocates real GPU KV-cache buffers. We don't
        # need them — the hook fakes every step's output. But the
        # model_runner.kv_cache_config attribute IS read by
        # _may_reorder_batch and other downstream code, so it must be
        # set on both the worker and the runner.
        if kv_cache_config is None and a:
            kv_cache_config = a[0]
        self.kv_cache_config = kv_cache_config
        if hasattr(self, 'model_runner') and self.model_runner is not None:
            self.model_runner.kv_cache_config = kv_cache_config
        print("[EmulatorCudaMock] gpu_worker.Worker.initialize_from_config "
              "→ stored kv_cache_config (no GPU alloc)")
    cls.initialize_from_config = _stub_initialize_from_config

    def _stub_get_supported_tasks(self):
        # Without a real loaded model, vllm cannot introspect the model
        # for supported tasks. We're a generate-only emulator.
        return ("generate",)
    cls.get_supported_tasks = _stub_get_supported_tasks


def install():
    """Install CUDA mock if enabled via environment variable.

    GPU properties (name, memory, SM count, compute capability) are read
    from the profile pack's model_config.gpu section when available,
    falling back to env vars or A100-like defaults.
    """
    # Single-path emulator design: when ENABLE_ORACLE=1, install CUDA
    # shims regardless of whether real CUDA is available. This makes
    # the emulator behave identically on cpu_host and on real-GPU hosts:
    # zero kernels dispatched, zero VRAM allocated, zero CUDA context.
    # MOCK_CUDA is kept as a deprecated alias so older scripts still work.
    if (os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() not in ("1", "true", "yes")
            and os.environ.get("VLLM_EMULATOR_MOCK_CUDA", "").lower() not in ("1", "true")):
        return False

    import torch

    gpu_cfg = _load_gpu_config()

    # Resolve GPU properties: env var > profile pack > defaults
    fake_memory = int(os.environ.get("VLLM_EMULATOR_MEMORY",
                                     gpu_cfg.get("gpu_memory_bytes",
                                                 _FAKE_GPU_MEMORY)))
    fake_name = os.environ.get("VLLM_EMULATOR_GPU_NAME",
                               gpu_cfg.get("gpu_name", "Emulator Device"))
    fake_cc = gpu_cfg.get("gpu_compute_capability", [8, 0])
    fake_sm = gpu_cfg.get("gpu_sm_count", 108)
    fake_gpu_count = int(os.environ.get("VLLM_EMULATOR_NUM_GPUS",
                                        gpu_cfg.get("gpu_count", 1)))

    _real_cuda = False
    try:
        # Detect using the underlying torch._C, before our shim takes effect.
        _real_cuda = torch._C._cuda_getDeviceCount() > 0  # type: ignore[attr-defined]
    except Exception:
        _real_cuda = False
    print(f"[EmulatorCudaMock] Installing CUDA mock "
          f"({'real GPU host — bypassing' if _real_cuda else 'no-GPU host'} "
          f"→ all CUDA calls will be shimmed; emulator runs on CPU only)")

    # --- Core CUDA functions ---
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: fake_gpu_count
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda *a, **kw: None
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **kw: None
    torch.cuda.reset_max_memory_allocated = lambda *a, **kw: None
    torch.cuda.reset_max_memory_cached = lambda *a, **kw: None
    torch.cuda.mem_get_info = lambda device=None: (fake_memory, fake_memory)
    torch.cuda.memory_allocated = lambda device=None: 0
    torch.cuda.max_memory_allocated = lambda device=None: 0
    torch.cuda.memory_reserved = lambda device=None: 0
    torch.cuda.manual_seed = lambda seed: None
    torch.cuda.manual_seed_all = lambda seed: None
    torch.cuda.seed = lambda: None
    torch.cuda.seed_all = lambda: None
    torch.cuda.initial_seed = lambda: 0

    # torch.cuda init/lazy-init guards — gpu_worker may force CUDA
    # initialization at startup which trips the driver check before our
    # event/stream patches matter.
    if hasattr(torch.cuda, '_lazy_init'):
        torch.cuda._lazy_init = lambda: None
    if hasattr(torch.cuda, 'init'):
        torch.cuda.init = lambda: None
    if hasattr(torch.cuda, 'is_initialized'):
        torch.cuda.is_initialized = lambda: True
    if hasattr(torch.cuda, '_initialized'):
        try:
            torch.cuda._initialized = True
        except Exception:
            pass
    if hasattr(torch.cuda, '_is_in_bad_fork'):
        torch.cuda._is_in_bad_fork = lambda: False
    if hasattr(torch.cuda, '_check_capability'):
        torch.cuda._check_capability = lambda: None
    if hasattr(torch.cuda, '_check_cubins'):
        torch.cuda._check_cubins = lambda: None

    # --- Mock device properties ---
    class FakeDeviceProps:
        name = fake_name
        major = fake_cc[0]
        minor = fake_cc[1]
        total_memory = fake_memory
        multi_processor_count = fake_sm
        max_threads_per_multi_processor = 2048

    torch.cuda.get_device_properties = lambda device=None: FakeDeviceProps()
    torch.cuda.get_device_name = lambda device=None: FakeDeviceProps.name
    torch.cuda.get_device_capability = lambda device=None: (
        fake_cc[0], fake_cc[1])

    # --- Stream / Event ---
    class FakeStream:
        def __init__(self, *a, **kw): pass
        def synchronize(self): pass
        def wait_event(self, *a): pass
        def wait_stream(self, *a): pass
        def record_event(self, *a): return FakeEvent()
        def query(self): return True
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeEvent:
        def __init__(self, *a, **kw): pass
        def record(self, *a, **kw): pass
        def synchronize(self): pass
        def wait(self, *a, **kw): pass
        def elapsed_time(self, other): return 0.0
        def query(self): return True

    _orig_Stream = torch.cuda.Stream
    _orig_Event = torch.cuda.Event

    torch.cuda.Stream = FakeStream
    torch.cuda.Event = FakeEvent
    # Cache singletons — vllm's hot path calls current_stream() many times
    # per step. Returning a fresh FakeStream() each time costs ~1 µs of
    # Python alloc + GC overhead per call. The singleton is identical in
    # behavior (all methods are pass).
    _CACHED_STREAM = FakeStream()
    _CACHED_EVENT = FakeEvent()
    torch.cuda.current_stream = lambda device=None: _CACHED_STREAM
    torch.cuda.default_stream = lambda device=None: _CACHED_STREAM

    # Walk sys.modules and replace any cached references to the original
    # Stream/Event classes (some vllm modules do `from torch.cuda import
    # Event, Stream` at import-time, before our shim runs).
    for _mod_name, _mod in list(sys.modules.items()):
        if _mod is None:
            continue
        try:
            _attrs = list(vars(_mod).items())
        except Exception:
            continue
        for _attr_name, _attr_val in _attrs:
            try:
                if _attr_val is _orig_Event:
                    setattr(_mod, _attr_name, FakeEvent)
                elif _attr_val is _orig_Stream:
                    setattr(_mod, _attr_name, FakeStream)
            except Exception:
                pass

    # Best-effort: try to override methods on the original C-level Event
    # / Stream base classes too. torch._C extension types may reject
    # setattr — we ignore failures and rely on the module-import patcher
    # for synchronize_input_prep instead.
    for _base_attr, _methods in (
        ('_CudaEventBase', (
            ('record', lambda self, *a, **kw: None),
            ('wait', lambda self, *a, **kw: None),
            ('synchronize', lambda self, *a, **kw: None),
            ('query', lambda self, *a, **kw: True),
            ('elapsed_time', lambda self, other, *a, **kw: 0.0),
        )),
        ('_CudaStreamBase', (
            ('synchronize', lambda self, *a, **kw: None),
            ('wait_event', lambda self, *a, **kw: None),
            ('wait_stream', lambda self, *a, **kw: None),
            ('record_event', lambda self, *a, **kw: FakeEvent()),
            ('query', lambda self, *a, **kw: True),
        )),
    ):
        _base = getattr(torch._C, _base_attr, None)
        if _base is None:
            continue
        for _method_name, _replacement in _methods:
            try:
                setattr(_base, _method_name, _replacement)
            except Exception:
                pass

    # --- torch.distributed: short-circuit collectives at world_size=1 ---
    # vllm calls barrier() / all_reduce() on tp/dp groups during init and
    # at various points during stepping. With tp=dp=1 there is literally
    # nothing to coordinate — but the underlying gloo backend still does
    # a TCP socket round-trip (~50-200 µs each). At ~10k steps in a r=8
    # bench, those add up to a measurable fraction of the +5% overhead
    # vs v3. No-op them at the world_size=1 case.
    try:
        import torch.distributed as _td_short
        _orig_barrier = getattr(_td_short, 'barrier', None)
        _orig_all_reduce = getattr(_td_short, 'all_reduce', None)
        _orig_all_gather = getattr(_td_short, 'all_gather', None)
        _orig_broadcast = getattr(_td_short, 'broadcast', None)
        def _ws1(group=None):
            try:
                return _td_short.get_world_size(group=group) <= 1 if group is not None else _td_short.get_world_size() <= 1
            except Exception:
                return True  # not initialized yet → safe to no-op
        if _orig_barrier is not None:
            def _patched_barrier(group=None, *a, **kw):
                if _ws1(group):
                    return None
                return _orig_barrier(group=group, *a, **kw)
            _td_short.barrier = _patched_barrier
        if _orig_all_reduce is not None:
            def _patched_all_reduce(tensor, *a, group=None, **kw):
                if _ws1(group):
                    return None
                return _orig_all_reduce(tensor, *a, group=group, **kw)
            _td_short.all_reduce = _patched_all_reduce
        print("[EmulatorCudaMock] short-circuited torch.distributed.barrier/all_reduce at world_size=1")
    except Exception as _e:
        print(f"[EmulatorCudaMock] could not patch torch.distributed collectives: {_e}")

    # --- torch.distributed: force gloo when CUDA_VISIBLE_DEVICES="" ---
    # On a real-driver host with CUDA hidden, libnccl IS still loadable so
    # vllm's `is_backend_available("nccl")` returns True; vllm picks NCCL
    # and crashes at runtime ("ProcessGroupNCCL is only supported with
    # GPUs"). Force NCCL to appear unavailable so vllm's auto-fallback at
    # parallel_state.py:1409 kicks in and selects gloo. This makes
    # init_device succeed → real GPUModelRunner instance → no stub
    # overhead → request scheduling matches real-GPU code path.
    try:
        import torch.distributed as _td
        _td.is_nccl_available = lambda: False
        if hasattr(_td, 'distributed_c10d'):
            _td.distributed_c10d.is_nccl_available = lambda: False
        # is_backend_available defers to per-backend probes; override
        # to explicitly return True only for gloo.
        _orig_is_backend_avail = getattr(_td, 'is_backend_available', None)
        if _orig_is_backend_avail is not None:
            def _patched_is_backend_avail(name):
                if str(name).lower() in ('nccl',):
                    return False
                if str(name).lower() in ('gloo',):
                    return True
                try:
                    return _orig_is_backend_avail(name)
                except Exception:
                    return False
            _td.is_backend_available = _patched_is_backend_avail
        print("[EmulatorCudaMock] forced torch.distributed: NCCL→unavailable, GLOO→available")
    except Exception as _e:
        print(f"[EmulatorCudaMock] could not patch torch.distributed: {_e}")

    # --- C-level CUDA init: prevent "no NVIDIA driver" error ---
    if hasattr(torch._C, '_cuda_init'):
        torch._C._cuda_init = lambda: None
    if hasattr(torch._C, '_cuda_getDeviceCount'):
        torch._C._cuda_getDeviceCount = lambda: int(os.environ.get(
            "VLLM_EMULATOR_NUM_GPUS", "1"))
    if hasattr(torch._C, '_cuda_getDevice'):
        torch._C._cuda_getDevice = lambda: 0
    if hasattr(torch._C, '_cuda_setDevice'):
        torch._C._cuda_setDevice = lambda x: None

    # --- torch.tensor(..., device='cuda') redirected to 'cpu' ---
    _original_tensor = torch.tensor
    def _patched_tensor(*args, **kwargs):
        dev = kwargs.get('device')
        if dev is not None and 'cuda' in str(dev):
            kwargs['device'] = 'cpu'
        return _original_tensor(*args, **kwargs)
    torch.tensor = _patched_tensor

    # Patch Tensor.cuda() and Tensor.to('cuda') to be no-ops.
    def _patched_cuda(self, *args, **kwargs):
        return self
    torch.Tensor.cuda = _patched_cuda

    _original_tensor_to = torch.Tensor.to
    def _patched_to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)):
            arg0 = str(args[0])
            if 'cuda' in arg0:
                args = ('cpu',) + args[1:]
        if 'device' in kwargs and 'cuda' in str(kwargs['device']):
            kwargs['device'] = 'cpu'
        return _original_tensor_to(self, *args, **kwargs)
    torch.Tensor.to = _patched_to

    # Tensor factory functions that take a device kwarg.
    _factories = (
        'empty', 'zeros', 'ones', 'arange', 'full', 'randn', 'rand',
        'eye', 'linspace', 'logspace', 'normal',
        'empty_like', 'zeros_like', 'ones_like', 'full_like',
        'randn_like', 'rand_like', 'randint_like',
    )
    for _factory_name in _factories:
        _original = getattr(torch, _factory_name, None)
        if _original is None:
            continue
        def _make_patched(_orig):
            def _patched(*args, **kwargs):
                dev = kwargs.get('device')
                if dev is not None and 'cuda' in str(dev):
                    kwargs['device'] = 'cpu'
                return _orig(*args, **kwargs)
            return _patched
        setattr(torch, _factory_name, _make_patched(_original))

    # --- torch.accelerator (newer vLLM uses this Python wrapper) ---
    if hasattr(torch, 'accelerator'):
        torch.accelerator.device_count = torch.cuda.device_count
        torch.accelerator.current_device_index = lambda *a, **kw: 0
        torch.accelerator.is_available = lambda: True
        torch.accelerator.synchronize = lambda *a, **kw: None
        torch.accelerator.set_device_index = lambda *a, **kw: None
        torch.accelerator.set_device = lambda *a, **kw: None
        torch.accelerator.current_accelerator = lambda *a, **kw: torch.device('cuda', 0)
        torch.accelerator.device_index = lambda *a, **kw: 0

        # Sweep: anything else callable on torch.accelerator becomes a
        # type-aware no-op so internal stats/reset/sync calls don't trip
        # the C-level _cuda_init.
        _accel = torch.accelerator
        def _noop_factory(name, _mem=fake_memory):
            lname = name.lower()
            def _noop(*a, **kw):
                if 'memory_stats' in lname or 'stats' in lname:
                    return {}
                if 'memory' in lname and any(s in lname for s in
                        ('max', 'alloc', 'reserved', 'cached', 'usage')):
                    return 0
                if 'mem_get_info' in lname or 'meminfo' in lname:
                    return (_mem, _mem)
                return None
            return _noop
        _already = {'device_count', 'current_device_index', 'is_available',
                    'synchronize', 'set_device_index', 'set_device',
                    'current_accelerator', 'device_index'}
        for _name in dir(_accel):
            if _name.startswith('_') or _name in _already:
                continue
            try:
                if callable(getattr(_accel, _name)):
                    setattr(_accel, _name, _noop_factory(_name))
            except Exception:
                pass

    # --- Sweep torch._C._accelerator_* and torch._C._cuda_* ---
    for _prefix in ('_accelerator_', '_cuda_'):
        for _name in list(dir(torch._C)):
            if not _name.startswith(_prefix):
                continue
            try:
                _attr = getattr(torch._C, _name)
                if callable(_attr):
                    if 'getDeviceCount' in _name or 'count' in _name.lower():
                        setattr(torch._C, _name, lambda *a, **kw: 1)
                    elif 'getDevice' in _name or 'currentDevice' in _name:
                        setattr(torch._C, _name, lambda *a, **kw: 0)
                    elif 'memory' in _name.lower() and 'reset' not in _name.lower():
                        setattr(torch._C, _name, lambda *a, **kw: 0)
                    else:
                        setattr(torch._C, _name, lambda *a, **kw: None)
            except Exception:
                pass

    print(f"[EmulatorCudaMock] Fake GPU: {FakeDeviceProps.name}, "
          f"{fake_memory // (1024**3)}GB, SM={fake_sm}, "
          f"CC={fake_cc[0]}.{fake_cc[1]}, "
          f"device_count={torch.cuda.device_count()}")

    return True


def install_model_stubs():
    """Install the model-stub meta-path patchers on top of an existing
    vllm install. Fires regardless of GPU presence — same code path on
    cpu_host (no GPU) and on real-GPU hosts.

    Effect: gpu_worker skips load_model, profile_run,
    determine_available_memory, get_kv_cache_spec (synthesized),
    get_supported_tasks, initialize_from_config. Model weights are NEVER
    materialized in GPU OR CPU RAM. The emulator hook intercepts every
    execute_model() call and drives latency from the profile pack.

    On a real-GPU host this means the emulator does NOT lock the GPU
    with weights — it only consumes the small CUDA context.

    Gated by VLLM_EMULATOR_ENABLE_ORACLE=1, so non-emulator runs are
    unaffected."""
    if os.environ.get("VLLM_EMULATOR_ENABLE_ORACLE", "").lower() not in (
            "1", "true", "yes"):
        return False
    print("[EmulatorModelStubs] Installing model-load stubs "
          "(no weights, no profile_run; hook+oracle drive latency)")
    _install_module_patcher(
        'vllm.v1.worker.gpu_model_runner',
        _patch_gpu_model_runner,
        'GPUModelRunner (synchronize_input_prep+load_model+profile_run)',
    )
    _install_module_patcher(
        'vllm.v1.worker.gpu_worker',
        _patch_gpu_worker,
        'gpu_worker.Worker (load_model+determine_avail_mem+kv_spec+'
        'supported_tasks+init_device+init_from_config)',
    )
    return True


# Auto-install if env vars are set at import time.
# Single-path emulator design: ENABLE_ORACLE=1 alone is sufficient to
# fire CUDA shims AND model stubs. MOCK_CUDA kept as deprecated alias.
# install() itself self-gates (no-op if neither var set), so it's safe
# to call unconditionally.
install()
install_model_stubs()

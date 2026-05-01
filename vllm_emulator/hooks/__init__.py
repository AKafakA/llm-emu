"""Executor-level hook — the only emulator hook on the runtime path."""

from .executor_hook import ExecutorEmulatorHook, get_executor_hook

__all__ = ["ExecutorEmulatorHook", "get_executor_hook"]

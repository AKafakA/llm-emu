"""Validation helpers for emulator profile packs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProfileValidationError(ValueError):
    """Raised when a profile pack does not match the expected schema."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileValidationError(f"{path} must be an object")
    return value


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{path} must be a number")
    if value < 0:
        raise ProfileValidationError(f"{path} must be >= 0")
    return float(value)


def _require_int(value: Any, path: str) -> int:
    if not isinstance(value, int):
        raise ProfileValidationError(f"{path} must be an integer")
    if value < 0:
        raise ProfileValidationError(f"{path} must be >= 0")
    return value


def _validate_prefill_rows(prefill: Sequence[Any]) -> None:
    for i, row in enumerate(prefill):
        obj = _require_mapping(row, f"prefill[{i}]")
        for key in ("seq_len", "batch_size", "latency_us"):
            if key not in obj:
                raise ProfileValidationError(f"prefill[{i}].{key} is required")
        _require_int(obj["seq_len"], f"prefill[{i}].seq_len")
        _require_int(obj["batch_size"], f"prefill[{i}].batch_size")
        _require_number(obj["latency_us"], f"prefill[{i}].latency_us")

    # Check sort order per batch_size group (oracle interpolates within
    # same batch_size). Profile packs may contain multiple batch_size
    # groups sorted by (batch_size, seq_len).
    by_bs: dict[int, list[int]] = {}
    for i, row in enumerate(prefill):
        bs = row["batch_size"]
        by_bs.setdefault(bs, []).append((i, row["seq_len"]))
    for bs, entries in by_bs.items():
        for j in range(1, len(entries)):
            prev_idx, prev_sl = entries[j - 1]
            cur_idx, cur_sl = entries[j]
            if cur_sl <= prev_sl:
                raise ProfileValidationError(
                    f"prefill samples within batch_size={bs} must be sorted "
                    f"by ascending seq_len (prefill[{cur_idx}].seq_len="
                    f"{cur_sl} <= previous {prev_sl})"
                )


def _validate_decode_rows(decode: Sequence[Any]) -> None:
    prev_active = -1
    for i, row in enumerate(decode):
        obj = _require_mapping(row, f"decode[{i}]")
        for key in ("active_seqs", "latency_us_per_token"):
            if key not in obj:
                raise ProfileValidationError(f"decode[{i}].{key} is required")
        active = _require_int(obj["active_seqs"], f"decode[{i}].active_seqs")
        _require_number(
            obj["latency_us_per_token"],
            f"decode[{i}].latency_us_per_token",
        )
        if active <= prev_active:
            raise ProfileValidationError(
                f"decode samples must be sorted by ascending active_seqs "
                f"(decode[{i}].active_seqs={active} <= previous {prev_active})"
            )
        prev_active = active


def validate_profile_pack(profile_pack: Mapping[str, Any]) -> None:
    """Validate a profile pack and raise :class:`ProfileValidationError` on error."""

    obj = _require_mapping(profile_pack, "profile_pack")

    required_top_level = ("version", "gpu_model")
    for key in required_top_level:
        if key not in obj:
            raise ProfileValidationError(f"{key} is required")

    if not isinstance(obj["version"], str) or not obj["version"].strip():
        raise ProfileValidationError("version must be a non-empty string")

    if not isinstance(obj["gpu_model"], str) or not obj["gpu_model"].strip():
        raise ProfileValidationError("gpu_model must be a non-empty string")

    # model_name is optional but recommended — warns if missing
    if "model_name" in obj:
        if not isinstance(obj["model_name"], str) or not obj["model_name"].strip():
            raise ProfileValidationError("model_name must be a non-empty string if provided")

    prefill = obj.get("prefill", [])
    decode = obj.get("decode", [])
    if not isinstance(prefill, Sequence) or isinstance(prefill, (str, bytes)):
        raise ProfileValidationError("prefill must be an array")
    if not isinstance(decode, Sequence) or isinstance(decode, (str, bytes)):
        raise ProfileValidationError("decode must be an array")

    # Serving profiles use forward_pass or 2D distributions instead of prefill/decode
    has_forward_pass = "forward_pass" in obj and len(obj["forward_pass"]) > 0
    has_2d_distribution = any(
        key in obj and len(obj[key]) > 0
        for key in ("step_cycle_2d_distribution", "prefill_2d_distribution",
                     "decode_2d_distribution")
    )

    if len(prefill) == 0 and not has_forward_pass and not has_2d_distribution:
        raise ProfileValidationError("prefill must include at least one sample")
    if len(decode) == 0 and not has_forward_pass and not has_2d_distribution:
        raise ProfileValidationError("decode must include at least one sample")

    if prefill:
        _validate_prefill_rows(prefill)
    if decode:
        _validate_decode_rows(decode)

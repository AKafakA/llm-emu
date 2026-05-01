"""Profile pack loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .validator import ProfileValidationError, validate_profile_pack


def load_profile_pack(path: str | Path) -> dict[str, Any]:
    """Load and validate a profile pack from a JSON file path."""

    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile pack not found: {profile_path}")

    try:
        content = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to read profile pack: {profile_path}") from exc

    return load_profile_pack_from_str(content)


def load_profile_pack_from_str(content: str) -> dict[str, Any]:
    """Load and validate a profile pack from a JSON string."""

    try:
        profile_pack = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProfileValidationError(f"Invalid JSON profile pack: {exc.msg}") from exc

    validate_profile_pack(profile_pack)
    return profile_pack

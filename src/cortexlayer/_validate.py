"""Argument validation shared by ``CortexClient`` and ``Memory`` so both
surfaces accept and reject exactly the same inputs."""

from __future__ import annotations

from typing import Any, Optional

from .errors import InvalidRequestError


def need_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequestError(f"{name} must be a non-empty string")
    return value


def need_int(value: Any, name: str, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequestError(f"{name} must be an integer")
    if value < lo or (hi is not None and value > hi):
        rng = f">= {lo}" if hi is None else f"{lo}..{hi}"
        raise InvalidRequestError(f"{name} must be {rng}")
    return value

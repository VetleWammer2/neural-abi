"""Canonical, bounded JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuralabi.status import PlanValidationError

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 96


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def canonical_dump_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def pretty_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def _validate_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise PlanValidationError(f"JSON nesting exceeds {MAX_JSON_DEPTH}")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PlanValidationError("JSON object keys must be strings")
            _validate_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_depth(child, depth + 1)


def load_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    size = path.stat().st_size
    if size > max_bytes:
        raise PlanValidationError(f"JSON file is too large: {size} bytes (limit {max_bytes})")
    try:

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, child in pairs:
                if key in result:
                    raise PlanValidationError(f"duplicate JSON object key {key!r}")
                result[key] = child
            return result

        value: Any = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanValidationError(f"cannot parse JSON {path}: {exc}") from exc
    _validate_depth(value)
    return value

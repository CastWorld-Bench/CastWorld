from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class AgentError(Exception):
    def __init__(self, message: str, code: str = "INVALID_ARGUMENT"):
        super().__init__(message)
        self.code = code


class InfrastructureError(Exception):
    """Pause this run; never charge the Agent for deployment/transport failures."""


class TransientInfrastructureError(InfrastructureError):
    """A retryable deployment failure isolated to the current request or Episode."""


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def chronological_items(records):
    """JSON object order is not event order; IDs break ties within one committed turn."""
    return sorted(
        records.items(),
        key=lambda item: (item[1]["created_turn"], int(item[0].rsplit("_", 1)[1])),
    )


def load_json(path: str | Path) -> Any:
    def reject(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def write_json(path: str | Path, value: Any) -> None:
    """Atomic output for generated runtime artifacts, not a mutable source file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import uuid

    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json_text(value), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def finite_array(values, length=None) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating))
        for x in values
    ):
        raise AgentError("Expected a numeric array")
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AgentError("Expected finite numeric values") from exc
    if arr.ndim != 1:
        raise AgentError("Expected a one-dimensional numeric array")
    if not np.isfinite(arr).all():
        raise AgentError("Expected finite numeric values; received NaN or infinity")
    if length is not None and len(arr) != length:
        raise AgentError(f"Expected exactly {length} finite values; received {len(arr)} values")
    return arr


def clean_numbers(value):
    """Unavailable diagnostic values become null, never NaN in JSON."""
    if isinstance(value, dict):
        return {str(k): clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_numbers(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def runtime_versions():
    import importlib.metadata
    import platform
    import sys

    return {
        "python": sys.version,
        "platform": platform.system(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "pydantic", "httpx", "matplotlib", "statsmodels")
        },
    }

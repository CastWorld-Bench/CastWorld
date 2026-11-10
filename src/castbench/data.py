"""Offline package construction. This module is never imported by the online runner."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, write_json
from .contracts import PublicBundle, time_axis
from .evaluator import normalization


def build_packages(frame, config, public_root, private_root):
    """Explicit origins and frozen train boundaries, never outcome-based Episode selection."""
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if not isinstance(config["dataset_id"], str) or not re.fullmatch(
        r"\w[\w.-]{0,127}", config["dataset_id"]
    ):
        raise ValueError("dataset_id must be a safe filename component, not a path")
    if (
        public_root == private_root
        or public_root in private_root.parents
        or private_root in public_root.parents
    ):
        raise ValueError("Public and evaluator bundles must be disjoint directories")
    target, time_col = config["target_name"], config["time_column"]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate data columns")
    axis = pd.DatetimeIndex(pd.to_datetime(frame[time_col], errors="raise"))
    if axis.tz is None:
        axis = axis.tz_localize(config["timezone"], ambiguous="raise", nonexistent="raise")
    axis = axis.tz_convert("UTC")
    if axis.has_duplicates or not axis.is_monotonic_increasing:
        raise ValueError("Input must be ordered with unique timestamps; no silent sort/drop")
    expected = pd.date_range(axis[0], axis[-1], freq=config["frequency"])
    if not axis.equals(expected):
        raise ValueError("Input has time gaps; explicitly construct a masked regular grid first")
    target_values = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=float)
    if np.isinf(target_values).any():
        raise ValueError("Infinite target values")
    train_end = pd.Timestamp(config["train_end"])
    if train_end.tzinfo is None:
        raise ValueError("train_end must include timezone")
    train_mask = axis <= train_end
    if not train_mask.any():
        raise ValueError("Empty training partition")

    def to_values(a):
        return [None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v) for v in a]

    train_y = to_values(target_values[train_mask])
    train_ts = [t.isoformat() for t in axis[train_mask]]
    norm = normalization(
        train_ts,
        train_y,
        config["frequency"],
        config.get("seasonal_period", 1),
        config["unit_floor"],
        config.get("minimum_scale_pairs", 3),
        config.get("scale_tolerance", 0),
    )
    h, g, f = config["lookback_length"], config["forecast_gap"], config["horizon_length"]
    if min(h, f) <= 0 or g < 0:
        raise ValueError("Invalid window sizes")
    origins = config["episode_origins"]
    if not origins:
        raise ValueError("At least one evaluation origin is required")
    origin_ts = time_axis([o["timestamp"] for o in origins])
    if origin_ts.has_duplicates or any(train_end > t for t in origin_ts):
        raise ValueError("Dev/test origins must be distinct and after frozen training cutoff")
    ontology = config["variables"]
    if any(v["name"] == target or v.get("target_derived") is not False for v in ontology):
        raise ValueError("Future variables must explicitly exclude target-derived information")
    used_scored = set()
    episodes = []
    samples = []
    stride = config.get("sample_stride", f)
    if stride <= 0:
        raise ValueError("sample_stride must be positive")
    train_n = int(train_mask.sum())
    for start in range(0, train_n - h - g - f + 1, stride):
        stop = start + h + g + f
        if not np.isfinite(target_values[start:stop]).all():
            continue
        samples.append(
            {
                "sample_id": f"train_{start}",
                "partition": "train",
                "history_timestamps": [t.isoformat() for t in axis[start : start + h]],
                "history_target": target_values[start : start + h].tolist(),
                "future_timestamps": [t.isoformat() for t in axis[start + h : stop]],
                "future_target": target_values[start + h : stop].tolist(),
                "context": {
                    v["name"]: to_values(frame[v["name"]].iloc[start:stop].tolist()) for v in ontology
                },
            }
        )
    for index, origin in enumerate(origins):
        if origin["split"] not in {"dev", "test"}:
            raise ValueError("Episode split must be dev or test")
        pos = axis.get_indexer([origin_ts[index]])[0]
        start, end = pos - h + 1, pos + 1 + g + f
        if pos < 0 or start < 0 or end > len(axis):
            raise ValueError("Episode window is outside the source data")
        scored_positions = set(range(pos + 1 + g, end))
        if used_scored & scored_positions:
            raise ValueError("Scored windows overlap; use explicit separate scenarios instead")
        used_scored |= scored_positions
        truth = target_values[pos + 1 + g : end]
        if not np.isfinite(truth).all():
            raise ValueError("Scored truth is incomplete")
        eid = f"{config['dataset_id']}_{origin['split']}_{index:04d}"
        resources = {
            k: eid + ":" + k
            for k in [
                "variable_catalog_id",
                "covariate_resource_id",
                "event_resource_id",
                "knowledge_pack_id",
                "sample_index_id",
                "revision_scale_profile_id",
                "tool_catalog_snapshot_id",
            ]
        }
        episode = {
            "episode_id": eid,
            "dataset_id": config["dataset_id"],
            "domain_id": config["domain_id"],
            "forecast_origin": axis[pos].isoformat(),
            "task": {
                k: config[k]
                for k in [
                    "target_name",
                    "target_unit",
                    "frequency",
                    "timezone",
                    "lookback_length",
                    "forecast_gap",
                    "horizon_length",
                ]
            },
            "history": {
                "timestamps": [t.isoformat() for t in axis[start : pos + 1]],
                "target_values": to_values(target_values[start : pos + 1]),
                "quality_mask_ref": digest(np.isnan(target_values[start : pos + 1]).tolist()),
            },
            "forecast_timeline": {
                "gap_timestamps": [t.isoformat() for t in axis[pos + 1 : pos + 1 + g]],
                "scored_timestamps": [t.isoformat() for t in axis[pos + 1 + g : end]],
            },
            "resources": resources,
            "tool_profile_id": config["dataset_id"] + ":tools-v1",
            "domain_pack_id": config["dataset_id"] + ":domain-v1",
            "budget_profile_id": config["dataset_id"] + ":budget-v1",
        }
        scale = {
            "profile_id": resources["revision_scale_profile_id"],
            "target_unit": config["target_unit"],
            "S": norm["scale_abs"],
            "absolute_tolerance": config["absolute_tolerance"],
            "source_train_hash": norm["source_train_hash"],
        }
        bundle = PublicBundle.model_validate(
            {
                "episode": episode,
                "budget": config.get("budget", {}),
                "revision_scale": scale,
                "task_description": config["task_description"],
                "variables": [
                    {
                        **v,
                        "visibility": v.get("visibility", "benchmark_known_future"),
                        "values": (
                            to_values(frame[v["name"]].iloc[start : pos + 1].tolist()) + [None] * (g + f)
                            if v.get("visibility") == "history_only"
                            else to_values(frame[v["name"]].iloc[start:end].tolist())
                        ),
                    }
                    for v in ontology
                ],
                "knowledge": config.get("knowledge", []),
                "events": [
                    ev
                    for ev in config.get("events", [])
                    if pd.Timestamp(ev["end"]) >= axis[start] and pd.Timestamp(ev["start"]) <= axis[end - 1]
                ],
                "samples": samples,
                "sample_library_cutoff": train_end.isoformat(),
                "constraints": config.get("constraints", []),
                **(
                    {"default_check_profile": config["default_check_profile"]}
                    if "default_check_profile" in config
                    else {}
                ),
            }
        )
        episodes.append((origin["split"], eid, bundle, truth.tolist()))
    # Validate everything before writing any package. Originals are never modified.
    for split, eid, bundle, truth in episodes:
        public_path = public_root / split / (eid + ".json")
        private_path = private_root / (eid + ".json")
        if public_path.exists() or private_path.exists():
            raise ValueError("Refusing to overwrite a frozen Episode; use a new build directory")
    for split, eid, bundle, truth in episodes:
        write_json(public_root / split / (eid + ".json"), bundle.model_dump())
        write_json(
            private_root / (eid + ".json"),
            {
                "episode_id": eid,
                "timestamps": bundle.episode.forecast_timeline.scored_timestamps,
                "values": truth,
                "normalization": norm,
            },
        )
    report = {
        "dataset_id": config["dataset_id"],
        "config_hash": digest(config),
        "episodes": len(episodes),
        "training_cases": len(samples),
        "lineage_audit": "Requires data curator review: declarations cannot detect disguised target-derived fields",
        "duplicate_audit": "Timestamp overlap blocked. Entity/event and near-duplicate review remains a dataset curation gate.",
        "visibility": "benchmark_known_future",
        "private_storage": "Separate directory; production must additionally use separate OS identities/mounts",
    }
    write_json(public_root / "build_report.json", report)
    return report


def synthetic_fixture():
    axis = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    y = 20 + 3 * np.sin(np.arange(200) * 2 * np.pi / 24) + np.arange(200) * 0.01
    frame = pd.DataFrame(
        {
            "date": axis,
            "load": y,
            "temperature": 15 + 4 * np.cos(np.arange(200) * 2 * np.pi / 24),
            "event": ["normal"] * 180 + ["event"] * 20,
        }
    )
    config = {
        "dataset_id": "synthetic_demo",
        "domain_id": "synthetic",
        "time_column": "date",
        "target_name": "load",
        "target_unit": "unit",
        "frequency": "h",
        "timezone": "UTC",
        "lookback_length": 48,
        "forecast_gap": 2,
        "horizon_length": 12,
        "train_end": axis[149].isoformat(),
        "seasonal_period": 24,
        "unit_floor": 0.01,
        "absolute_tolerance": 1e-8,
        "task_description": "Synthetic engineering fixture, not benchmark evidence.",
        "episode_origins": [{"timestamp": axis[180].isoformat(), "split": "dev"}],
        "variables": [
            {
                "name": "temperature",
                "dtype": "numeric",
                "unit": "C",
                "description": "Synthetic temperature",
                "target_derived": False,
                "lineage": "Independent cosine generator",
            },
            {
                "name": "event",
                "dtype": "categorical",
                "unit": "category",
                "description": "Synthetic event category",
                "target_derived": False,
                "lineage": "Predeclared timeline",
            },
        ],
        "knowledge": [
            {
                "entry_id": "general-season",
                "topic": "seasonality",
                "text": "Seasonal patterns can repeat; examine sufficient cycles before inferring periodicity.",
                "source": "CastBench demonstration note",
                "version": "1",
                "scope": "general",
            }
        ],
        "budget": {"max_rounds": 12},
        "constraints": [{"constraint_id": "nonnegative", "kind": "minimum", "value": 0, "hard": False}],
    }
    return frame, config

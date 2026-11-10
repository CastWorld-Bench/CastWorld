"""Frozen dataset-specific, read-only diagnostics over the authorized Episode view."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
import hashlib
from pathlib import Path
import re
from typing import Literal

import numpy as np
from pydantic import Field, create_model

from .catalog import Registry
from .common import AgentError, digest
from .contracts import Contract


class DiagnosticArgs(Contract):
    timeline: Literal["history", "scored"] = "scored"
    groups: list[str] = Field(default_factory=list, max_length=6)
    max_fields: int = Field(default=12, strict=True, ge=1, le=32)


class HistoryDiagnosticArgs(Contract):
    groups: list[str] = Field(default_factory=list, max_length=6)
    max_fields: int = Field(default=12, strict=True, ge=1, le=32)


def numeric(values):
    return np.asarray([float(x) if isinstance(x, (int, float)) and not isinstance(x, bool)
                       and np.isfinite(x) else np.nan for x in values], dtype=float)


def stats(values):
    x = numeric(values)
    finite = x[np.isfinite(x)]
    result = {"points": len(x), "observed": len(finite), "missing": int((~np.isfinite(x)).sum())}
    if len(finite):
        result.update(min=float(finite.min()), max=float(finite.max()),
                      mean=float(finite.mean()), median=float(np.median(finite)),
                      last=float(finite[-1]), std=float(finite.std()),
                      zero_fraction=float(np.mean(finite == 0)))
    return result


def correlation(a, b, min_pairs=8):
    a, b = numeric(a), numeric(b)
    keep = np.isfinite(a) & np.isfinite(b)
    n = int(keep.sum())
    return {"pairs": n, "pearson": float(np.corrcoef(a[keep], b[keep])[0, 1])
            if n >= min_pairs and np.std(a[keep]) > 0 and np.std(b[keep]) > 0 else None,
            "minimum_pairs": min_pairs}


def circular(values):
    x = numeric(values)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"observed": 0, "circular_mean_deg": None, "resultant": None}
    radians = np.deg2rad(x)
    s, c = float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))
    r = float(np.hypot(s, c))
    return {"observed": len(x), "circular_mean_deg": float(np.rad2deg(np.arctan2(s, c)) % 360)
            if r > 1e-10 else None, "resultant": r}


def transfer_support(history, future, target, angular=False, categorical=False):
    """Public support checks, not a fitted effect size or a prediction rule."""
    if categorical:
        known = {str(v) for v in history if v is not None}
        seen = [str(v) for v in future if v is not None]
        return {"historical_distinct_states": len(known), "future_observed_points": len(seen),
                "future_unseen_states": sorted(set(seen) - known), "quantitative_effect_estimate": None}
    h, f, y = numeric(history), numeric(future), numeric(target)
    finite_h, finite_f = h[np.isfinite(h)], f[np.isfinite(f)]
    middle = len(h) // 2
    result = {"history": circular(h) if angular else stats(h),
              "known_future": circular(f) if angular else stats(f),
              "historical_pairs_are_not_independent_samples": True,
              "quantitative_effect_estimate": None}
    if angular:
        result["linear_correlation_or_range_extrapolation"] = "not_applicable_to_circular_angles"
        return result
    result["historical_association"] = {"full": correlation(h, y),
        "early": correlation(h[:middle], y[:middle]), "recent": correlation(h[middle:], y[middle:]),
        "first_difference": correlation(np.diff(h), np.diff(y))}
    result["future_outside_observed_history_range"] = None
    if len(finite_h) and len(finite_f):
        outside = (finite_f < finite_h.min()) | (finite_f > finite_h.max())
        result["future_outside_observed_history_range"] = {"points": int(outside.sum()),
            "fraction_of_observed_future": float(outside.mean()), "future_observed_points": len(finite_f),
            "history_observed_points": len(finite_h)}
    return result


def _handler(profile, operation, public, args):
    e = public["episode"]
    if e["dataset_id"] != profile["dataset_id"]:
        raise AgentError("Dataset diagnostic profile does not match this Episode", "PERMISSION_DENIED")
    length, horizon = e["task"]["lookback_length"], e["task"]["horizon_length"]
    gap = e["task"]["forecast_gap"]
    history_only = operation in {"history", "association", "lag_support", "sensor", "rotation", "signed", "rate", "calendar"}
    timeline = "history" if history_only else args.get("timeline", "scored")
    if operation == "transfer":
        timeline = "all"
    a, b = (0, length) if timeline == "history" else (0, length + gap + horizon) if timeline == "all" else (length + gap, length + gap + horizon)
    variables = {v["name"]: v for v in public["variables"]}
    for name, v in variables.items():
        if "#" in name or name not in profile["fields"]:
            raise AgentError("Undeclared field in domain view", "PERMISSION_DENIED")
        expected_lag = profile["fields"][name].get("historical_target_lag_steps")
        declared_lag = (v.get("historical_target_lag") or {}).get("lag_steps")
        if bool(v.get("target_derived")) != (expected_lag is not None) or declared_lag != expected_lag:
            raise AgentError("Historical target derivation differs from the frozen profile", "PERMISSION_DENIED")
        if v["visibility"] == "history_only" or name.endswith("*"):
            if any(x is not None for x in v["values"][length:]):
                raise AgentError("Hidden future values reached a domain diagnostic", "PERMISSION_DENIED")
    warnings = ["Diagnostic evidence only: no forecast is created, revised or selected.",
                "Association and exposure do not identify a causal effect or a correction coefficient."]
    used = set()
    units = {}
    blocked = set()

    def values(name, history=False):
        if name not in variables:
            return None
        v = variables[name]
        if v["unit"] != profile["fields"][name]["unit"]:
            raise AgentError("Variable unit differs from the frozen domain profile", "PERMISSION_DENIED")
        if not history and timeline != "history" and (v["visibility"] == "history_only" or name.endswith("*")):
            blocked.add(name)
            return None
        used.add(name)
        units[name] = v["unit"]
        return v["values"][:length] if history else v["values"][a:b]

    groups = profile["groups"]
    requested = args.get("groups") or list(groups)
    if not set(requested) <= set(groups):
        raise AgentError("Unknown domain variable group; allowed groups: " + ", ".join(groups)
                         + ". Use groups=[] for the default balanced view.", "PERMISSION_DENIED")
    names = []
    for index in range(max((len(groups[g]) for g in requested), default=0)):
        for group in requested:
            if index < len(groups[group]) and groups[group][index] not in names:
                names.append(groups[group][index])
    cap = args.get("max_fields", 12)
    selected = names[:cap]

    def summarize(names_to_use=selected):
        out = {}
        for name in names_to_use:
            v = values(name)
            if v is None:
                continue
            if variables[name]["dtype"] == "numeric":
                out[name] = circular(v) if "direction" in name.lower() and units[name] == "degree" else stats(v)
            else:
                present = [str(x) for x in v if x is not None]
                out[name] = {"points": len(v), "present": len(present),
                             "counts": dict(Counter(present).most_common(12))}
            out[name]["source"] = variables[name]["lineage"]
        return out

    y = numeric(e["history"]["target_values"])
    result = {}
    if operation == "history":
        middle = max(1, length // 2)
        adjacent = np.isfinite(y[1:]) & np.isfinite(y[:-1])
        result = {"target": stats(y), "early": stats(y[:middle]), "recent": stats(y[middle:]),
                  "adjacent_changes": stats(np.diff(y)[adjacent]),
                  "negative_observations": int(np.sum(y < 0)),
                  "lag_support": {str(lag): correlation(y[:-lag], y[lag:])
                                  for lag in profile["history_lags"] if 0 < lag < length}}
        units["target"] = e["task"]["target_unit"]
    elif operation in {"association", "lag_support"}:
        rows = {}
        lags = [0] if operation == "association" else list(range(min(7, length // 4) + 1))
        for name in selected:
            v = values(name, history=True)
            if v is None or variables[name]["dtype"] != "numeric":
                continue
            x = numeric(v)
            # Original positions are kept; missing observations are never
            # compressed into adjacent samples before lag/difference tests.
            rows[name] = {str(lag): correlation(x[:-lag], y[lag:]) if lag else correlation(x, y)
                          for lag in lags}
            rows[name]["first_difference"] = correlation(np.diff(x), np.diff(y))
        result = {"historical_associations": rows, "lag_unit": "dataset sampling steps",
                  "selection": "specified groups, never ranked on test forecast error"}
        warnings.append("Small samples, common trends and multiple lag checks can create misleading correlations.")
    elif operation == "transfer":
        rows = {}
        for name in selected:
            variable = variables[name]
            if variable["visibility"] == "history_only" or name.endswith("*"):
                blocked.add(name)
                continue
            observed = values(name, history=True)
            future = variable["values"][length + gap:length + gap + horizon]
            if not any(v is not None for v in future):
                continue
            rows[name] = transfer_support(observed, future, y,
                angular="direction" in name.lower() and units[name] == "degree",
                categorical=variable["dtype"] != "numeric")
        result = {"fields": rows, "historical_split": "chronological halves; no shuffling or fitted coefficients",
                  "future_target_read": False}
        warnings.extend(["Different products, heights and proxy definitions are not interchangeable even when correlated.",
                         "Compare full, early, recent and differenced support; no single correlation validates a causal correction.",
                         "Outside-range context flags extrapolation, not an error or a reason to clip valid extreme forecasts."])
    elif operation == "calendar":
        times = e["history"]["timestamps"]
        import pandas as pd

        axis = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
        clock = profile.get("calendar_clock", "source_wallclock_encoded_UTC")
        if clock.startswith("IANA:"):
            axis = axis.tz_convert(clock[5:])
        grouped = {}
        hour, weekday = axis.hour, axis.dayofweek
        for field in ("LOCAL_HOUR", "market_hour"):
            x = values(field, history=True)
            if x is not None:
                hour = numeric(x)
                clock = "declared_source_field:" + field
                break
        for field in ("LOCAL_WEEKDAY", "market_weekday"):
            x = values(field, history=True)
            if x is not None:
                weekday = numeric(x)
                break
        for tag, labels in (("hour", hour), ("weekday", weekday)):
            grouped[tag] = {str(label): stats(y[np.asarray(labels) == label]) for label in sorted(set(labels)) if np.isfinite(label)}
        result = {"historical_target_by_clock": grouped, "clock": clock,
                  "context": summarize([n for n in selected if any(s in n.lower() for s in
                      ("holiday", "workday", "weekend", "calendar", "snap", "issue", "event", "scheduled", "hour"))])}
        warnings.append("Clock groups with few observations are descriptive, not estimated holiday effects.")
    elif operation == "bike_units":
        for name, mul, add, unit in (("temp", 47, -8, "degC"), ("atemp", 66, -16, "degC"),
                                     ("hum", 100, 0, "percent")):
            x = values(name)
            if x is not None:
                result[name] = {"converted": stats(numeric(x) * mul + add), "unit": unit,
                                "formula": f"{mul} * source + {add}"}
        wind = values("windspeed")
        if wind is not None:
            result["windspeed"] = {"source_scale": stats(wind), "rescaled_source": stats(numeric(wind) * 67),
                                    "unit": "source_windspeed_unit_unverified"}
        warnings.append("Daily weather replicated or allocated hourly cannot be summed as independent hourly observations.")
    elif operation == "radiation":
        required = ("shortwave_radiation_fused", "direct_radiation_fused", "diffuse_radiation_fused")
        arrays = [values(n) for n in required]
        if all(x is not None for x in arrays):
            g, horizontal, diffuse = map(numeric, arrays)
            result = {"horizontal_balance_residual_GHI_minus_direct_minus_diffuse": stats(g - horizontal - diffuse),
                      "components": {n: stats(x) for n, x in zip(required, arrays)}}
        warnings.append("DNI is excluded from the horizontal sum. No zenith, panel angle, capacity or loss fraction is inferred.")
    elif operation == "wind":
        result["direction"] = {}
        for n in ("station_wind_direction_80m_fused",):
            x = values(n)
            if x is not None:
                result["direction"][n] = circular(x)
        low, high = values("hf_wind_speed_80m_fused"), values("hf_wind_speed_120m_fused")
        if low is not None and high is not None:
            low, high = numeric(low), numeric(high)
            keep = np.isfinite(low) & np.isfinite(high) & (low > 0) & (high > 0)
            result["80m_120m_shear"] = {"paired": int(keep.sum()), "alpha": stats(np.log(high[keep] / low[keep]) / np.log(120 / 80)),
                                        "formula": "log(v120/v80) / log(120/80)"}
        result["context"] = summarize()
        warnings.append("Height ratios cancel a common wind-speed unit; they do not imply measured hub-height power or stability.")
    elif operation == "regional":
        comparable = [n for n in names if re.search(r"^station_\d+$|temperature_2m$|temp_(north|south)_popw_mean", n)]
        arrays = [(n, values(n)) for n in comparable]
        arrays = [(n, numeric(x)) for n, x in arrays if x is not None]
        confirmed_units = {units[n] for n, _ in arrays}
        if len(arrays) >= 2 and len(confirmed_units) == 1 and "source_unit_unverified" not in confirmed_units:
            m = np.vstack([x for _, x in arrays])
            available = np.isfinite(m).sum(axis=0)
            spreads = [float(np.ptp(m[np.isfinite(m[:, i]), i])) if available[i] >= 2 else None for i in range(m.shape[1])]
            result["same_unit_unweighted_spread"] = {"fields": [n for n, _ in arrays], "spread": stats(spreads),
                                                       "points_with_two_stations": int(np.sum(available >= 2))}
        result["field_summaries"] = summarize()
        if len(arrays) >= 2 and (len(confirmed_units) != 1 or "source_unit_unverified" in confirmed_units):
            warnings.append("Regional fields with different or unverified units are reported separately, without a pooled spread.")
        warnings.append("No load, population, basin-area or sample-size weights are invented. Different quantities are not pooled.")
    elif operation == "hydro":
        for name in names:
            if "daymet_prcp" not in name:
                continue
            x = values(name)
            if x is None:
                continue
            v = numeric(variables[name]["values"][:b])
            result[name] = {"daily": stats(x), "trailing_sums": {str(w): stats([
                float(v[i-w+1:i+1].sum()) if i+1 >= w and np.isfinite(v[i-w+1:i+1]).all() else None
                for i in range(a, b)]) for w in (3, 7)},
                "antecedent_support": "authorized earlier history plus known future precipitation, up to each requested point"}
        warnings.append("Trailing sums preserve the boundary history and never look after each requested point; stations are kept separate.")
    elif operation == "yield":
        for high, low in (("actual_dgs10", "actual_dgs2"), ("actual_dgs10_mean", "actual_dgs2_mean"),
                          ("GS10", "GS1"), ("GS10", "GS5")):
            hi, lo = values(high), values(low)
            if hi is not None and lo is not None:
                result[f"{high}_minus_{low}"] = {"spread": stats(numeric(hi) - numeric(lo)), "unit": "percentage_points"}
        warnings.append("Term spreads are descriptive, not guaranteed future policy-rate or income changes.")
    elif operation in {"signed", "rate", "sensor", "rotation"}:
        pair = np.isfinite(y[1:]) & np.isfinite(y[:-1])
        d = np.diff(y)[pair]
        result = {"target": stats(y), "negative_points": int(np.sum(y < 0)),
                  "unchanged_adjacent_pairs": int(np.sum(d == 0)), "adjacent_pairs": len(d),
                  "observed_changes": stats(d), "context": summarize()}
        if operation == "rotation":
            result["lag_27"] = correlation(y[:-27], y[27:]) if length > 27 else {"pairs": 0, "pearson": None}
            warnings.append("One rotation and fewer than eight valid pairs cannot establish a stable 27-day periodicity.")
        if operation == "sensor":
            warnings.append("No fault, maintenance, measurement location or undocumented sensor meaning is inferred.")
    elif operation in {"exposure", "climate", "price", "trade", "epidemic", "ozone"}:
        result = {"field_summaries": summarize()}
        if operation == "epidemic":
            result["target_interpretation"] = "percentage, not case count or prevalence"
        if operation == "ozone":
            result["missing_mechanism_inputs"] = ["NOx", "VOC", "chemical transport state"]
        if operation == "price":
            result["negative_price_allowed"] = True
        if operation in {"trade", "climate"}:
            warnings.append("Low-frequency values broadcast onto this grid do not constitute new independent observations.")
    else:
        raise AgentError("Unknown frozen diagnostic operation")

    if blocked:
        warnings.append("Future history-only fields were not used: " + ", ".join(sorted(blocked)))
    if len(names) > cap:
        warnings.append("Group list was capped; select a group or raise max_fields to inspect remaining fields.")
    target_operation = history_only or operation in {"calendar", "transfer"}
    if target_operation:
        units["target"] = e["task"]["target_unit"]
    if not result or (operation == "transfer" and not result["fields"]) or (not used and not target_operation) or (used and not target_operation and not any(any(v is not None for v in values(n, history=history_only) or []) for n in used)):
        status = "not_applicable"
    else:
        status = "partial" if blocked or len(names) > cap else "ok"
    return {"status": status, "structured_result": result, "warnings": warnings,
            "observed_scope": {"timeline": timeline, "start": 0, "end": b-a, "variables": sorted(used)},
            "units": units, "coverage": {"requested_points": b-a, "blocked_history_only": sorted(blocked),
                "selected_fields": len(selected), "omitted_by_cap": names[cap:],
                "no_imputation": True, "reads_target_history": target_operation}}


def dataset_registry(domain, profile, registry=None):
    profile = deepcopy(profile)
    if domain != profile["domain_id"] or not re.fullmatch(r"dataset_[a-z0-9_]+", domain):
        raise ValueError("Unregistered dataset domain profile")
    if any("#" in name for name in profile["fields"]):
        raise ValueError("Audit fields cannot enter domain tools")
    registry = Registry() if registry is None else registry
    implementation = digest({"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "profile": profile})
    for tool in profile["tools"]:
        op = tool["operation"]
        history = op in {"history", "association", "lag_support", "sensor", "rotation", "signed", "rate", "calendar"}
        tool_id = f"domain.{domain}.{tool['name']}"
        base = HistoryDiagnosticArgs if history or op == "transfer" else DiagnosticArgs
        group_type = Literal[tuple(profile["groups"])]
        schema = create_model(domain + "_" + tool["name"] + "_Args", __base__=base,
            groups=(list[group_type], Field(default_factory=list, max_length=6,
                description="Allowed groups: " + ", ".join(profile["groups"])
                + "; [] selects the default balanced view.")))
        description = tool["description"] + "。groups 可选：" + ", ".join(profile["groups"]) + "；[] 为默认均衡视图。"
        registry.register_extension(tool_id, schema,
            partial(_handler, profile, op), description, implementation,
            category="feature" if history else "context")
        registry.manifests[tool_id].update(input_variables=list(profile["fields"]),
            variable_groups=deepcopy(profile["groups"]),
            reads_target_history=history or op in {"calendar", "transfer"}, knowledge_profile=profile["profile_id"],
            limitations=["Read-only evidence; no automatic prediction changes", "Missing values retained",
                         "Correlations are descriptive, not causal", "Deployment isolation required"])
    return registry

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .common import AgentError, InfrastructureError, clean_numbers, digest, json_text
from .contracts import PublicBundle, Scope, ToolResult
from .features import analyze, associations, summary


class ToolView:
    """Episode-scoped resource resolver; no path evaluation, eval, SQL or arbitrary Python."""

    def __init__(self, bundle: PublicBundle, state, store, run_id):
        self.bundle, self.state, self.store, self.run_id = bundle, state, store, run_id
        e = bundle.episode
        self.timestamps = (
            e.history.timestamps + e.forecast_timeline.gap_timestamps + e.forecast_timeline.scored_timestamps
        )
        self.variables = {v.name: v for v in bundle.variables}

    def variable(self, name):
        if name not in self.variables:
            raise AgentError("Variable is not in this Episode's authorized catalog", "PERMISSION_DENIED")
        return self.variables[name]

    def indices(self, scope: Scope, n=None):
        t = self.bundle.episode.task
        spans = {
            "history": (0, t.lookback_length),
            "gap": (t.lookback_length, t.lookback_length + t.forecast_gap),
            "scored": (t.lookback_length + t.forecast_gap, len(self.timestamps)),
            "all": (0, len(self.timestamps)),
        }
        a, b = spans[scope.timeline]
        if n is not None:
            b = min(b, n)
        end = b - a if scope.end is None else scope.end
        if not 0 <= scope.start <= end <= b - a:
            raise AgentError("Requested scope exceeds authorized timeline", "PERMISSION_DENIED")
        return a + scope.start, a + end

    def series(self, ref, scope):
        e = self.bundle.episode
        if ref == "episode.history.target":
            if scope.timeline not in {"history", "all"}:
                raise AgentError("Future target is private", "PERMISSION_DENIED")
            values, times, unit = e.history.target_values, e.history.timestamps, e.task.target_unit
            start, end = self.indices(Scope(timeline="history", start=scope.start, end=scope.end))
        elif ref.startswith("covariate:"):
            v = self.variable(ref.split(":", 1)[1])
            if v.dtype != "numeric":
                raise AgentError("Numeric feature tools require a numeric variable")
            values, times, unit = v.values, self.timestamps, v.unit
            start, end = self.indices(scope)
        else:
            raise AgentError("Unknown or unauthorized series reference", "PERMISSION_DENIED")
        if start == end:
            raise AgentError("Empty series scope")
        return (
            np.asarray([np.nan if x is None else x for x in values[start:end]], dtype=float),
            times[start:end],
            unit,
        )


def _lexical(query, text):
    # Unicode characters + Latin tokens: deterministic and usable without embedding services.
    def tokenize(value):
        return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", value.lower())

    q, t = Counter(tokenize(query)), Counter(tokenize(text))
    denom = np.sqrt(sum(x * x for x in q.values()) * sum(x * x for x in t.values()))
    return sum(v * t[k] for k, v in q.items()) / denom if denom else 0.0


def _shape_distance(current, past, scale):
    n = min(len(current), len(past))
    if n == 0:
        return None
    a, b = np.asarray(current[-n:], dtype=float), np.asarray(past[-n:], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    adjacent = mask[:-1] & mask[1:]
    if mask.sum() < 3 or not adjacent.any():
        return None
    ramp_distance = float(np.mean(np.abs(np.diff(a)[adjacent] - np.diff(b)[adjacent])) / scale)
    a, b = a[mask], b[mask]

    def normalize(v):
        return (v - np.mean(v)) / np.std(v) if np.std(v) > 0 else v - v.mean()

    return [
        float(abs(a.mean() - b.mean()) / scale),
        float(np.sqrt(np.mean((normalize(a) - normalize(b)) ** 2))),
        ramp_distance,
    ]


def execute_readonly(tool_id, p, view: ToolView, registry, model_service=None):
    category, name = tool_id.split(".", 1)
    b, e = view.bundle, view.bundle.episode
    raw = {"status": "ok", "structured_result": {}, "warnings": []}
    units, scope = {}, {"timeline": "all", "variables": []}
    if category == "registry":
        if name == "describe_tool":
            if p["tool_id"] not in registry.manifests:
                raise AgentError("Unknown tool", "UNKNOWN_TOOL")
            raw["structured_result"] = registry.manifests[p["tool_id"]]
        elif name == "list_tools":
            raw["structured_result"] = {
                "tools": [
                    x for x in registry.cards() if p["category"] is None or x["category"] == p["category"]
                ]
            }
        else:
            raw["structured_result"] = {"tools": list(registry.extensions)}
    elif category == "model":
        if model_service is None:
            raise InfrastructureError(
                "Model workers are not configured. Configure all four before a live Core run."
            )
        if (p["frequency"], p["forecast_gap"], p["horizon_length"]) != (
            e.task.frequency,
            e.task.forecast_gap,
            e.task.horizon_length,
        ):
            raise AgentError("Model time metadata must exactly match this Episode")
        raw = model_service.predict(name.removesuffix("_forecast"), e)
        scope = {"timeline": "history", "variables": [e.task.target_name]}
    elif category == "feature":
        s = Scope.model_validate(p["scope"])
        if name in {"covariate_association", "lead_lag_scan"}:
            if p["series_ref"] != "episode.history.target" or s.timeline not in {"all", "history"}:
                raise AgentError("Association only uses the observed target history")
            start, end = view.indices(Scope(timeline="history", start=s.start, end=s.end))
            y = [np.nan if x is None else x for x in e.history.target_values[start:end]]
            variables = {}
            for vname in p["variables"]:
                var = view.variable(vname)
                if var.dtype != "numeric":
                    raise AgentError("Association currently supports numeric covariates")
                variables[vname] = [np.nan if x is None else x for x in var.values[start:end]]
            lag = p.get("max_lag", 0)
            if lag >= len(y):
                raise AgentError("Lag exceeds history support")
            raw["structured_result"] = associations(y, variables, lag)
            scope = {
                "timeline": "history",
                "start": start,
                "end": end,
                "variables": [e.task.target_name] + p["variables"],
            }
        else:
            x, times, unit = view.series(p["series_ref"], s)
            raw = analyze(name, x, times, p)
            units = {"values": unit, "slope_per_step": unit + "/step"}
            scope = {**s.model_dump(), "series_ref": p["series_ref"], "timestamps": times}
            if p["series_ref"] == "episode.history.target":
                scope.update(timeline="history", variables=[e.task.target_name], end=s.start + len(x))
            else:
                scope.update(variables=[p["series_ref"].split(":", 1)[1]], end=s.start + len(x))
    elif category == "knowledge":
        if name == "list_topics":
            raw["structured_result"] = {
                "topics": [
                    {"topic": k, "count": v} for k, v in sorted(Counter(x.topic for x in b.knowledge).items())
                ]
            }
        elif name in {"search_general", "search_domain"}:
            entries = [x for x in b.knowledge if x.scope == name.removeprefix("search_")]
            ranked = sorted(
                [
                    (_lexical(p["query"], " ".join([x.topic, x.text, *x.search_aliases])), x.entry_id, x)
                    for x in entries
                ],
                key=lambda r: (-r[0], r[1]),
            )
            raw["structured_result"] = {
                "entries": [
                    {"relevance": score, **x.model_dump()} for score, _, x in ranked[: p["k"]] if score > 0
                ],
                "conflict_policy": "Return competing entries, do not choose a mechanism",
                "search_method": "deterministic lexical match over text, topics and pack-supplied aliases",
                "empty_result_hint": "Try knowledge.list_topics or the entry language/aliases; no match does not establish absence of domain knowledge.",
            }
        elif name == "get_entry":
            matches = [x for x in b.knowledge if x.entry_id == p["entry_id"]]
            if not matches:
                raise AgentError("Unknown knowledge entry", "UNKNOWN_REFERENCE")
            raw["structured_result"] = matches[0].model_dump()
        else:
            if p["variable"] == e.task.target_name:
                raw["structured_result"] = {
                    "name": e.task.target_name,
                    "unit": e.task.target_unit,
                    "role": "prediction_target",
                    "frequency": e.task.frequency,
                    "history_ref": "episode.history.target",
                    "visibility": "History is public; future target values are private and never returned.",
                }
            else:
                raw["structured_result"] = view.variable(p["variable"]).model_dump(exclude={"values"})
        scope = {"timeline": "all", "variables": [], "source_type": "frozen_knowledge"}
    elif category == "context":
        if name == "list_variables":
            raw["structured_result"] = {"variables": [v.model_dump(exclude={"values"}) for v in b.variables]}
        elif name == "get_variable_slice":
            s = Scope.model_validate(p["scope"])
            a, z = view.indices(s)
            width = p["aggregate_steps"]
            if width > 1 and (z - a) % width:
                raise AgentError("Aggregation must use complete bins; request a divisible range")
            rows = {}
            for vname in p["variables"]:
                v = view.variable(vname)
                values = v.values[a:z]
                if width > 1:
                    if v.dtype != "numeric" and p["aggregation"] != "last":
                        raise AgentError("Categorical/boolean aggregation supports last only")
                    aggregated = []
                    for i in range(0, len(values), width):
                        chunk = values[i : i + width]
                        valid = [x for x in chunk if x is not None]
                        if p["aggregation"] == "last":
                            aggregated.append(chunk[-1])
                        elif not valid:
                            aggregated.append(None)
                        else:
                            fn = {"mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max}[
                                p["aggregation"]
                            ]
                            aggregated.append(float(fn(valid)))
                    values = aggregated
                rows[vname] = {"values": values, "unit": v.unit, "missing_mask": [x is None for x in values]}
            raw["structured_result"] = {
                "timestamps": view.timestamps[a:z:width],
                "variables": rows,
                "aggregation": p["aggregation"],
                "steps_per_bin": width,
            }
            scope = {**s.model_dump(), "variables": p["variables"]}
        elif name == "get_event_timeline":
            raw["structured_result"] = {
                "events": [
                    x.model_dump() for x in b.events if not p["event_types"] or x.type in p["event_types"]
                ]
            }
        elif name == "calendar_profile":
            index = pd.to_datetime(
                e.forecast_timeline.gap_timestamps + e.forecast_timeline.scored_timestamps, utc=True
            ).tz_convert(e.task.timezone)
            raw["structured_result"] = {
                "calendar": [
                    {
                        "timestamp": t.isoformat(),
                        "hour": t.hour,
                        "weekday": t.dayofweek,
                        "month": t.month,
                        "is_weekend": t.dayofweek >= 5,
                    }
                    for t in index
                ],
                "note": "No inferred public holidays or operational weekdays",
            }
        elif name == "future_shift_profile":
            shift = {}
            for vname in p["variables"]:
                v = view.variable(vname)
                h, f = v.values[: e.task.lookback_length], v.values[e.task.lookback_length :]
                if v.dtype == "numeric":
                    hv, fv = (
                        np.array([x for x in h if x is not None]),
                        np.array([x for x in f if x is not None]),
                    )
                    shift[vname] = {
                        "history": summary(h),
                        "future": summary(f),
                        "outside_history_fraction": float(np.mean((fv < hv.min()) | (fv > hv.max())))
                        if len(hv) and len(fv)
                        else None,
                        "standardized_mean_shift": float((fv.mean() - hv.mean()) / hv.std())
                        if len(hv) > 1 and len(fv) and hv.std() > 0
                        else None,
                    }
                else:
                    hc, fc = Counter(x for x in h if x is not None), Counter(x for x in f if x is not None)
                    shift[vname] = {
                        "history_counts": dict(hc),
                        "future_counts": dict(fc),
                        "novel_values": [x for x in fc if x not in hc],
                    }
            raw["structured_result"] = {
                "shifts": shift,
                "warning": "Distribution shift alone does not determine target response",
            }
            scope["variables"] = p["variables"]
        else:
            facts = []
            sections = {
                "time_context": [],
                "target_state": [],
                "future_covariate_state": [],
                "event_context": [],
                "historical_support": [],
                "knowledge_refs": [],
                "other_observations": [],
            }
            for ref in p["refs"]:
                if ref in view.state["observations"]:
                    item = view.state["observations"][ref]
                    facts.append({"ref": ref, "scope": item["scope"], "result_ref": item["artifact_ref"]})
                    observed = view.store.read_blob(view.run_id, item["artifact_ref"])
                    content = observed["structured_result"]
                    tool = item["tool_id"]
                    if tool == "context.calendar_profile":
                        section = "time_context"
                    elif tool == "context.get_event_timeline":
                        section = "event_context"
                    elif tool in {"context.get_variable_slice", "context.future_shift_profile"}:
                        section = "future_covariate_state"
                    elif tool.startswith("sample."):
                        section = "historical_support"
                    elif tool.startswith("knowledge."):
                        section = "knowledge_refs"
                    elif (
                        tool.startswith("feature.")
                        and item["scope"].get("series_ref") == "episode.history.target"
                    ):
                        section = "target_state"
                    else:
                        section = "other_observations"
                    if "statistics" in content:
                        excerpt = {"statistics": content["statistics"]}
                        for key in ("recent", "scales", "periods", "change_indices", "current_regime_length"):
                            if key in content:
                                excerpt[key] = content[key]
                    elif tool == "context.get_variable_slice":
                        excerpt = {
                            "variables": {
                                name: {
                                    "unit": v["unit"],
                                    "point_count": len(v["values"]),
                                    "missing_count": sum(v["missing_mask"]),
                                }
                                for name, v in content["variables"].items()
                            },
                            "value_access": "Exact values remain in the cited result artifact",
                        }
                    elif "cases" in content:
                        excerpt = {
                            "sample_ids": [case["sample_id"] for case in content["cases"]],
                            "comparisons": content.get("comparisons", []),
                        }
                    else:
                        excerpt = {
                            k: v
                            for k, v in content.items()
                            if k not in {"image_png_base64", "plot_data", "distribution"}
                        }
                    sections[section].append({"source_ref": ref, "scope": item["scope"], "facts": excerpt})
                elif ref in view.state["evidence"]:
                    facts.append({"ref": ref, "agent_interpretation": view.state["evidence"][ref]})
                else:
                    raise AgentError(
                        "Context card may only use acquired public references", "UNKNOWN_REFERENCE"
                    )
            raw["structured_result"] = {
                **sections,
                "facts": facts,
                "inferences": [],
                "cross_factor_interactions": [],
                "interpretation_notice": "Only supplied observations are assembled. No new causal or forecast claim is generated.",
                "unresolved_items": [
                    k for k, v in view.state["needs"].items() if v["status"] in {"open", "observed"}
                ],
            }
    elif category == "sample":
        cases = {s.sample_id: s for s in b.samples}
        if name.startswith("search_"):
            weights = np.asarray(p["weights"])
            if len(weights) != 3 or (weights < 0).any() or abs(weights.sum() - 1) > 1e-8:
                raise AgentError("Shape weights require three nonnegative values summing to one")
            variables = p.get("variables", [])
            if name == "search_context_conditioned" and not variables:
                raise AgentError("Specify at least one context variable")
            for var in variables:
                view.variable(var)
            ranks = []
            target = [np.nan if x is None else x for x in e.history.target_values]
            for sample in b.samples:
                parts = _shape_distance(target, sample.history_target, b.revision_scale.S)
                if parts is None:
                    continue
                dist = float(weights @ parts)
                context_distances = []
                for vname in variables:
                    if vname not in sample.context:
                        continue
                    var = view.variable(vname)
                    current, past = var.values, sample.context[vname]
                    valid_current = [x for x in current if x is not None]
                    valid_past = [x for x in past if x is not None]
                    if not valid_current or not valid_past:
                        continue
                    if var.dtype == "numeric":
                        history = [
                            x
                            for case in b.samples
                            for x in case.context.get(vname, [])[: len(case.history_target)]
                            if x is not None
                        ]
                        scale = np.std(history) if history else 0
                        if scale == 0:
                            context_distances.append(float(np.mean(valid_current) != np.mean(valid_past)))
                        else:
                            context_distances.append(
                                float(abs(np.mean(valid_current) - np.mean(valid_past)) / scale)
                            )
                    else:
                        a, bcounts = Counter(valid_current), Counter(valid_past)
                        context_distances.append(
                            sum(
                                abs(a[k] / len(valid_current) - bcounts[k] / len(valid_past))
                                for k in set(a) | set(bcounts)
                            )
                            / 2
                        )
                coverage = len(context_distances) / len(variables) if variables else 1.0
                if variables:
                    cw = p["context_weight"]
                    # Missing context is penalized explicitly, not mistaken for similarity.
                    context_distance = float(np.mean(context_distances)) if context_distances else 0
                    dist = (1 - cw) * dist + cw * (context_distance + 1 - coverage)
                ranks.append(
                    {
                        "sample_id": sample.sample_id,
                        "distance": dist,
                        "shape_components": parts,
                        "shape_weights": weights.tolist(),
                        "context_distances": context_distances,
                        "feature_coverage": coverage,
                    }
                )
            raw["structured_result"] = {
                "matches": sorted(ranks, key=lambda x: (x["distance"], x["sample_id"]))[: p["k"]],
                "requested_context_variables": variables,
                "ranking_reads_future_target": False,
                "context_scaling_source": "frozen_training_context_history",
            }
        else:
            ids = [p["sample_id"]] if name == "get_case" else p["sample_ids"]
            if any(i not in cases for i in ids):
                raise AgentError("Sample is not in frozen training library", "PERMISSION_DENIED")
            raw["structured_result"] = {
                "cases": [cases[i].model_dump() for i in ids],
                "training_cutoff": b.sample_library_cutoff,
            }
            if name == "compare_case":
                raw["structured_result"]["comparisons"] = [
                    {
                        "sample_id": i,
                        "shape_components": _shape_distance(
                            e.history.target_values, cases[i].history_target, b.revision_scale.S
                        ),
                    }
                    for i in ids
                ]
        scope = {"timeline": "history", "source_type": "frozen_training", "variables": [e.task.target_name]}
    elif category == "domain":
        # Extensions receive a value copy, never the filesystem, store, or evaluator handle.
        public = b.model_dump(exclude={"samples"})
        public["samples"] = []
        raw = registry.extensions[tool_id](deepcopy(public), deepcopy(p))
        scope = raw.get("observed_scope", scope)
        units = raw.get("units", units)
    else:
        raise AgentError("This tool is not a read-only information tool")
    raw = clean_numbers(raw)
    raw.setdefault("status", "ok")
    raw.setdefault("structured_result", {})
    raw.setdefault("warnings", [])
    raw.setdefault("coverage", {})
    raw["observed_scope"] = scope
    raw["units"] = units
    raw["factual_summary"] = f"{tool_id}: {raw['status']}; see structured_result for factual values."
    raw["provenance"] = {
        "tool_id": tool_id,
        "tool_version": "1.0.0",
        "input_hash": digest({"episode": e.model_dump(), "arguments": p}),
        "implementation_digest": registry.manifests[tool_id]["implementation_digest"],
        "access_scope": "current_episode_and_frozen_train",
        "arguments": p,
    }
    json_text(raw)
    try:
        return ToolResult.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise InfrastructureError("Tool output violates its declared result contract") from exc

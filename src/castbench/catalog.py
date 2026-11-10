from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, create_model

from .action_schema import compact_contract, forecast_schema
from .common import AgentError, digest
from .contracts import Contract, ForecastArgs, NonnegativeInt, Number, PositiveInt, Scope, ToolResult

FEATURES = {
    "window_summary": "Window extent, level, recent statistics",
    "data_quality_profile": "Missingness, robust outlier candidates, flat runs",
    "basic_statistics": "Basic statistics with valid counts",
    "robust_distribution": "MAD, IQR, skew, tails and zero mass",
    "multiscale_trend": "Theil-Sen slopes and uncertainty across trailing scales",
    "seasonality_spectrum": "Periodogram candidate periods; not a causal diagnosis",
    "autocorrelation_profile": "ACF and PACF at valid lags",
    "robust_decomposition": "Robust STL and exact reconstruction diagnostics",
    "change_point_regime": "Deterministic piecewise level segmentation and regime summaries",
    "volatility_profile": "Trailing standard deviation and robust relative volatility",
    "ramp_turning_points": "Finite adjacent ramps and turning points",
    "covariate_association": "Historical paired Pearson and Spearman associations",
    "lead_lag_scan": "Historical lag correlations; positive lag means covariate leads target",
    "time_series_to_text": "Deterministic text derived from numeric facts",
    "time_series_to_image": "Plot with the exact data and text-equivalent summary",
}
KNOWLEDGE = {
    "list_topics": "List frozen topics",
    "search_general": "Lexical general knowledge retrieval",
    "search_domain": "Lexical domain knowledge retrieval",
    "get_entry": "Expand a knowledge entry",
    "explain_variable": "Variable meaning, units and lineage",
}
SAMPLES = {
    "search_target_shape": "Search frozen training history shapes",
    "search_context_conditioned": "Search shape plus explicitly selected context variables",
    "get_case": "Expand a frozen training case",
    "compare_case": "Compare training histories and realized futures",
}
CONTEXT = {
    "list_variables": "List current Episode variable cards",
    "get_variable_slice": "Read exact public covariates",
    "get_event_timeline": "Read event intervals",
    "future_shift_profile": "Future covariate shift vs history",
    "calendar_profile": "Local-time calendar attributes (weekend is not a holiday claim)",
    "build_context_card": "Collect only specified observation/evidence references",
}
SANDBOX = {
    "adopt_candidate": "Adopt a model candidate",
    "submit_full_forecast": "Submit an independent forecast",
    "constant_forecast": "Submit an Agent-chosen constant over the exact scored timeline; explicit unit required",
    "add_offset": "Add values within a range",
    "scale_values": "Scale within a range",
    "adjust_segments": "Apply non-overlapping numeric segments",
    "blend_forecasts": "Weighted source fusion",
    "shift_time": "Shift a local shape with explicit edge policy",
    "smooth_revision_delta": "Smooth only a revision delta",
    "replace_segment": "Replace a segment",
    "compare_versions": "Compare forecasts without truth",
    "validate_forecast": "Explicit validation using the public check profile",
}


def fields_for(tool_id):
    category, name = tool_id.split(".", 1)
    series = {"series_ref": (str, "episode.history.target"), "scope": (Scope, Field(default_factory=Scope))}
    if category == "model":
        return {
            "history_target_ref": (Literal["episode.history.target"], "episode.history.target"),
            "frequency": (str, ...),
            "forecast_gap": (NonnegativeInt, ...),
            "horizon_length": (PositiveInt, ...),
        }
    if category == "feature":
        fields = dict(series)
        if name in {"multiscale_trend"}:
            fields["scales"] = (list[PositiveInt], ...)
        if name in {
            "seasonality_spectrum",
            "autocorrelation_profile",
            "robust_decomposition",
            "change_point_regime",
        }:
            fields["missing_policy"] = (Literal["reject", "linear"], "reject")
        if name == "autocorrelation_profile":
            fields["max_lag"] = (PositiveInt, 24)
        if name == "robust_decomposition":
            fields["period"] = (AnnotatedPeriod, ...)
        if name == "change_point_regime":
            fields.update(
                {
                    "min_segment": (PositiveInt, 8),
                    "max_changes": (NonnegativeInt, 4),
                    "penalty": (Number, Field(default=3.0, gt=0)),
                }
            )
        if name == "volatility_profile":
            fields["window"] = (PositiveInt, 12)
        if name in {"covariate_association", "lead_lag_scan"}:
            fields["variables"] = (list[str], ...)
        if name == "lead_lag_scan":
            fields["max_lag"] = (NonnegativeInt, 12)
        return fields
    if category == "knowledge":
        if name.startswith("search_"):
            return {"query": (str, Field(min_length=1)), "k": (PositiveInt, Field(default=5, le=50))}
        if name == "get_entry":
            return {"entry_id": (str, ...)}
        if name == "explain_variable":
            return {"variable": (str, ...)}
        return {}
    if category == "context":
        if name in {"get_variable_slice", "future_shift_profile"}:
            fields = {"variables": (list[str], ...)}
            if name == "get_variable_slice":
                fields.update(
                    {
                        "scope": (Scope, Field(default_factory=Scope)),
                        "aggregate_steps": (PositiveInt, 1),
                        "aggregation": (Literal["mean", "sum", "min", "max", "last"], "mean"),
                    }
                )
            return fields
        if name == "get_event_timeline":
            return {"event_types": (list[str], Field(default_factory=list))}
        if name == "build_context_card":
            return {"refs": (list[str], ...)}
        return {}
    if category == "sample":
        if name.startswith("search_"):
            fields = {
                "k": (PositiveInt, Field(default=5, le=50)),
                "weights": (list[Number], Field(default_factory=lambda: [0.2, 0.6, 0.2])),
            }
            if name == "search_context_conditioned":
                fields.update(
                    {
                        "variables": (list[str], ...),
                        "context_weight": (Number, Field(default=0.25, ge=0, le=1)),
                    }
                )
            return fields
        return {"sample_id": (str, ...)} if name == "get_case" else {"sample_ids": (list[str], ...)}
    if category == "sandbox":
        if name == "compare_versions":
            return {"forecast_ids": (list[str], ...)}
        if name == "validate_forecast":
            return {"forecast_id": (str, ...), "checks": (list[str] | None, None)}
    if category == "registry":
        if name == "describe_tool":
            return {"tool_id": (str, ...)}
        if name == "list_tools":
            return {"category": (str | None, None)}
        return {}
    raise ValueError(tool_id)


AnnotatedPeriod = Annotated[int, Field(strict=True, ge=2)]


class Registry:
    def __init__(self):
        descriptions = {
            f"model.{m}_forecast": f"Frozen target-only {m} zero-shot forecast"
            for m in ("chronos", "timesfm", "sundial", "moirai")
        }
        for prefix, group in [
            ("feature", FEATURES),
            ("knowledge", KNOWLEDGE),
            ("sample", SAMPLES),
            ("context", CONTEXT),
            ("sandbox", SANDBOX),
        ]:
            descriptions.update({f"{prefix}.{k}": v for k, v in group.items()})
        descriptions.update(
            {
                "registry.list_tools": "List tool cards",
                "registry.describe_tool": "Get exact schema",
                "registry.list_domain_extensions": "List installed domain extension tools",
            }
        )
        self.manifests, self.schemas, self.extensions = {}, {}, {}
        code_digest = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest()
        for tool_id, description in descriptions.items():
            name = tool_id.split(".")[1]
            is_forecast = tool_id.startswith("sandbox.") and name not in {
                "compare_versions",
                "validate_forecast",
            }
            schema = (
                ForecastArgs
                if is_forecast
                else create_model(tool_id.replace(".", "_"), __base__=Contract, **fields_for(tool_id))
            )
            self.schemas[tool_id] = schema
            self.manifests[tool_id] = {
                "tool_id": tool_id,
                "version": "1.0.0",
                "category": tool_id.split(".")[0],
                "description": description,
                "input_schema": forecast_schema(name) if is_forecast else schema.model_json_schema(),
                "output_schema": ToolResult.model_json_schema(),
                "temporal_scope": "current_episode_and_frozen_train",
                "unit_policy": "preserve_and_report",
                "side_effect": "state" if tool_id.startswith("sandbox.") else "none",
                "parallel_safe": not tool_id.startswith("sandbox."),
                "implementation_digest": "sha256:" + code_digest,
                "limitations": ["Descriptive evidence is not a causal claim; no private target access"],
                "error_codes": ["INVALID_ARGUMENT", "PERMISSION_DENIED", "INSUFFICIENT_POINTS"],
            }

    @property
    def snapshot(self):
        return digest(self.manifests)

    def validate(self, tool_id, arguments):
        if tool_id not in self.schemas:
            raise AgentError("Unknown tool", "UNKNOWN_TOOL")
        data = dict(arguments)
        if self.schemas[tool_id] is ForecastArgs:
            expected = tool_id.split(".")[1]
            if "operation" in data and data["operation"] != expected:
                raise AgentError("Sandbox operation conflicts with tool name")
            data["operation"] = expected
        return self.schemas[tool_id].model_validate(data).model_dump()

    def cards(self, *, detailed=True):
        cards = [
            {
                "tool_id": m["tool_id"],
                "category": m["category"],
                "description": m["description"],
                "required": m["input_schema"].get("required", []),
                "arguments": compact_contract(m["input_schema"]),
                "schema_ref": m["tool_id"],
            }
            for m in self.manifests.values()
        ]
        for card in cards:
            if card["tool_id"].startswith("sandbox."):
                card["required"] = [k for k in card["required"] if k != "operation"]
            if card["tool_id"].startswith("sample.search_"):
                card["argument_note"] = (
                    "weights: exactly three nonnegative values summing to one (level, normalized shape, ramp); omit to use defaults."
                )
        order = getattr(self, "model_order", ())
        if order:
            by_id = {c["tool_id"]: c for c in cards}
            cards = [by_id[k] for k in order] + [c for c in cards if c["tool_id"] not in order]
        if not detailed:
            for card in cards:
                arguments = card["arguments"]
                card["arguments"] = {k: v for k, v in arguments.items() if k in card["required"]}
                card["optional_fields"] = [k for k in arguments if k not in card["required"]]
                if not card["required"]:
                    card["default_arguments"] = {}
        return cards

    def register_extension(
        self, tool_id, schema, handler, description, implementation_digest, *, category="context"
    ):
        if not tool_id.startswith("domain.") or tool_id in self.manifests:
            raise ValueError("Domain extensions use unique domain.* IDs")
        if not isinstance(schema, type) or not issubclass(schema, Contract):
            raise ValueError("Extension needs a strict Contract")
        if category not in {"model", "feature", "knowledge", "sample", "context", "sandbox"}:
            raise ValueError("Domain tools must belong to one of the six capability families")
        if category in {"model", "sandbox"}:
            raise ValueError(
                "This read-only extension interface cannot create model candidates or sandbox state"
            )
        self.schemas[tool_id] = schema
        self.extensions[tool_id] = handler
        self.manifests[tool_id] = {
            "tool_id": tool_id,
            "version": "1.0.0",
            "category": category,
            "origin": "domain_extension",
            "description": description,
            "input_schema": schema.model_json_schema(),
            "output_schema": {"type": "object"},
            "temporal_scope": "current_episode_only",
            "unit_policy": "preserve_and_report",
            "side_effect": "none",
            "parallel_safe": True,
            "implementation_digest": implementation_digest,
            "limitations": ["Trusted plugin; deployment isolation required"],
        }

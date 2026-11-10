"""Public, operation-specific wire schemas. Runtime validation remains authoritative."""

from copy import deepcopy

from .contracts import (
    Action,
    BatchArgs,
    CheckArgs,
    DeclareNeedArgs,
    EvidenceArgs,
    FinalizeArgs,
    ForecastArgs,
    StateRequestArgs,
)

FORECAST_FIELDS = {
    "constant_forecast": {"constant_value", "gap_constant_value", "unit"},
    "submit_full_forecast": {"scored_values", "scored_timestamps", "gap_values", "gap_timestamps", "unit"},
    "adopt_candidate": {"candidate_id"},
    "add_offset": {"parent_id", "range", "offset"},
    "scale_values": {"parent_id", "range", "factor"},
    "adjust_segments": {"parent_id", "range", "segments"},
    "blend_forecasts": {"source_ids", "weights", "base_id", "range"},
    "shift_time": {"parent_id", "range", "steps", "fill_policy", "fill_values"},
    "smooth_revision_delta": {"parent_id", "proposed_id", "range", "window", "method"},
    "replace_segment": {"parent_id", "source_id", "range", "values"},
}
REQUIRED = {
    "constant_forecast": ["constant_value", "unit"],
    "submit_full_forecast": ["scored_values", "scored_timestamps", "unit"],
    "adopt_candidate": ["candidate_id"],
    "add_offset": ["parent_id", "offset", "declared_effect"],
    "scale_values": ["parent_id", "factor", "declared_effect"],
    "adjust_segments": ["parent_id", "segments", "declared_effect"],
    "blend_forecasts": ["source_ids", "weights", "declared_effect"],
    "shift_time": ["parent_id", "steps", "fill_policy", "declared_effect"],
    "smooth_revision_delta": ["parent_id", "proposed_id", "declared_effect"],
    "replace_segment": ["parent_id", "declared_effect"],
}
COMMON = {
    "operation",
    "declared_effect",
    "evidence_refs",
    "rationale_origin",
    "rationale_scope",
    "correction_reason_code",
}


def nonnull(schema):
    value = deepcopy(schema)
    if "anyOf" in value:
        choices = [v for v in value["anyOf"] if v.get("type") != "null"]
        if len(choices) == 1:
            value = choices[0]
        else:
            value["anyOf"] = choices
    value.pop("default", None)
    return value


def forecast_schema(operation, task=None):
    schema = ForecastArgs.model_json_schema()
    schema["properties"] = {
        k: v for k, v in schema["properties"].items() if k in COMMON | FORECAST_FIELDS[operation]
    }
    schema["properties"]["operation"] = {"type": "string", "const": operation}
    schema["required"] = ["operation", *REQUIRED[operation]]
    for k in schema["required"]:
        schema["properties"][k] = nonnull(schema["properties"][k])
    if task and operation == "submit_full_forecast":
        for name, length in [
            ("scored_values", task["horizon_length"]),
            ("scored_timestamps", task["horizon_length"]),
            ("gap_values", task["forecast_gap"]),
            ("gap_timestamps", task["forecast_gap"]),
        ]:
            prop = schema["properties"][name]
            choices = prop.get("anyOf", [prop])
            for choice in choices:
                if choice.get("type") == "array":
                    choice.update(minItems=length, maxItems=length)
    return schema


def action_schema(task, max_calls=4):
    root = Action.model_json_schema()
    definitions = root.pop("$defs", {})
    payloads = {
        "DECLARE_NEED": DeclareNeedArgs,
        "CALL_TOOL": BatchArgs,
        "REGISTER_EVIDENCE": EvidenceArgs,
        "CHECK_FORECAST": CheckArgs,
        "REQUEST_STATE": StateRequestArgs,
        "FINALIZE": FinalizeArgs,
    }
    branches = []
    for kind in [*payloads, "CREATE_FORECAST"]:
        schemas = (
            [forecast_schema(op, task) for op in FORECAST_FIELDS]
            if kind == "CREATE_FORECAST"
            else [payloads[kind].model_json_schema()]
        )
        for schema in schemas:
            definitions.update(schema.pop("$defs", {}))
        payload = {"anyOf": schemas} if kind == "CREATE_FORECAST" else schemas[0]
        if kind == "CALL_TOOL":
            payload["properties"]["calls"]["maxItems"] = max_calls
        branch = deepcopy(root)
        branch["properties"]["action_type"] = {"type": "string", "const": kind}
        branch["properties"]["payload"] = payload
        branch["required"] = ["action_type", "state_version", "payload"]
        branches.append(branch)

    def concise(value):
        if isinstance(value, dict):
            return {k: concise(v) for k, v in value.items() if k not in {"title", "default"}}
        if isinstance(value, list):
            return [concise(x) for x in value]
        return value

    # Some chat templates enumerate only root.properties; a bare root anyOf
    # otherwise renders environment_action() as a zero-argument function.
    return concise({**root, "anyOf": branches, "$defs": definitions})


def compact_contract(schema):
    """Show parameter types/defaults without repeating shared definitions on every card."""

    def describe(s):
        if "$ref" in s:
            return s["$ref"].rsplit("/", 1)[-1]
        if "anyOf" in s:
            return "|".join(describe(x) for x in s["anyOf"])
        if "enum" in s:
            return "|".join(str(x) for x in s["enum"])
        if "const" in s:
            return str(s["const"])
        if s.get("type") == "array":
            return "list[" + describe(s.get("items", {})) + "]"
        return s.get("type", "object")

    return {
        k: {
            "type": describe(s),
            **({"default": s["default"]} if "default" in s else {}),
            **{c: s[c] for c in ("minimum", "maximum", "minItems", "maxItems", "description") if c in s},
        }
        for k, s in schema.get("properties", {}).items()
        if k != "operation"
    }


def wire_action_schema(task, max_calls=4):
    """Portable provider-facing subset; semantic operation matching stays in the Runner.

    The local native tool parser mishandles root discriminated unions. Publish a
    flat typed payload instead, while retaining bounded prediction arrays. This is
    intentionally not a claim of provider-enforced full semantic validation.
    """
    root = Action.model_json_schema()
    definitions = root.pop("$defs", {})
    properties = {}
    for model in (
        ForecastArgs,
        BatchArgs,
        CheckArgs,
        DeclareNeedArgs,
        EvidenceArgs,
        StateRequestArgs,
        FinalizeArgs,
    ):
        schema = model.model_json_schema()
        definitions.update(schema.get("$defs", {}))
        properties.update(schema["properties"])
    bounded = forecast_schema("submit_full_forecast", task)["properties"]
    for name in ("scored_values", "scored_timestamps", "gap_values", "gap_timestamps"):
        properties[name] = bounded[name]
    for name in ("offset", "factor", "values", "fill_values"):
        for choice in properties[name].get("anyOf", [properties[name]]):
            if choice.get("type") == "array":
                choice["maxItems"] = max(task["horizon_length"], task["forecast_gap"])
    properties["calls"]["maxItems"] = max_calls
    root["properties"]["payload"] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    root["required"] = ["action_type", "state_version", "payload"]
    root["$defs"] = definitions
    return root

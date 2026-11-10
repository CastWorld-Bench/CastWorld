"""Observable loop diagnostics, without changing score weights or rewarding tool quantity."""

from collections import Counter


def interaction_diagnostics(export):
    state, trace = export["state"], export["trace"]
    counts, information_turns = Counter(), []
    seen_calls = set()
    seen_substantive_calls = set()
    repeated_calls = 0
    substantive_calls = 0
    repeated_substantive_calls = 0
    invalid_rounds = 0
    state_read_rounds = 0
    for row in trace:
        if not row["event"]["valid"]:
            invalid_rounds += row["event"].get("actor") != "environment_capacity"
            continue
        if (row["event"].get("action") or {}).get("action_type") == "REQUEST_STATE":
            state_read_rounds += 1
        for record in row["event"].get("tool_records", []):
            if record.get("status") not in {"ok", "partial"}:
                continue
            namespace = record["tool_id"].split(".", 1)[0]
            counts[namespace] += 1
            if namespace != "registry":
                substantive_calls += 1
                # Missing hashes are treated as distinct calls; only demonstrably
                # identical successful calls count as redundant.
                if record.get("arguments_hash"):
                    signature = (record["tool_id"], record["arguments_hash"])
                    repeated_substantive_calls += signature in seen_substantive_calls
                    seen_substantive_calls.add(signature)
            if record.get("information"):
                information_turns.append(row["turn"])
                if record.get("arguments_hash"):
                    signature = (record["tool_id"], record["arguments_hash"])
                    repeated_calls += signature in seen_calls
                    seen_calls.add(signature)
    forecasts = state["forecasts"]
    versions = [f for f in forecasts.values() if f["kind"] != "model_candidate"]
    evidence_then_revision = 0
    for version in versions:
        previous = forecasts.get(version.get("previous_agent_forecast_id"))
        if not previous or previous["scored_values"] == version["scored_values"]:
            continue
        if any(previous["created_turn"] < t < version["created_turn"] for t in information_turns):
            evidence_then_revision += 1
    final = forecasts.get((state.get("final") or {}).get("forecast_id"))
    identical_models = [
        f.get("model_tool")
        for f in forecasts.values()
        if f["kind"] == "model_candidate" and final and f["scored_values"] == final["scored_values"]
    ]
    return {
        "version": "observable_loop_diagnostics/1.2",
        "repeated_successful_information_calls": repeated_calls,
        "successful_substantive_tool_calls": substantive_calls,
        "repeated_successful_substantive_tool_calls": repeated_substantive_calls,
        "invalid_action_rounds": invalid_rounds,
        "state_read_rounds": state_read_rounds,
        "successful_calls_by_namespace": dict(counts),
        "information_tool_calls": len(information_turns),
        "nonmodel_information_tool_calls": sum(
            counts[c] for c in ("feature", "context", "knowledge", "sample", "domain")
        ),
        "explicit_evidence_count": len(state["evidence"]),
        "agent_forecast_versions": len(versions),
        "new_information_then_changed_prediction": evidence_then_revision,
        "final_identical_model_tools": identical_models,
        "notice": "Descriptive only: more calls/changes are not intrinsically better. No causal inference or hidden reasoning judgment.",
    }

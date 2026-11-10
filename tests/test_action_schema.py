import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from castbench.action_schema import FORECAST_FIELDS, action_schema, forecast_schema, wire_action_schema
from castbench.capacity import reduce_packet
from castbench.catalog import Registry
from castbench.common import AgentError, finite_array
from castbench.contracts import Action, ForecastArgs
from castbench.providers import OpenAICompatibleProvider, ProviderProfile, ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


@pytest.mark.parametrize("horizon,gap", [(24, 0), (96, 0), (96, 57)])
def test_exact_output_contract(horizon, gap):
    schema = action_schema({"horizon_length": horizon, "forecast_gap": gap})
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    action = {
        "action_type": "CREATE_FORECAST",
        "state_version": 0,
        "payload": {
            "operation": "submit_full_forecast",
            "scored_values": [1.0] * horizon,
            "scored_timestamps": ["public timestamp"] * horizon,
            "unit": "u",
            "gap_values": [1.0] * gap,
            "gap_timestamps": ["public timestamp"] * gap,
        },
    }
    assert validator.is_valid(action)
    for n in [horizon - 1, horizon + 1, 1000]:
        bad = deepcopy(action)
        bad["payload"]["scored_values"] = [1.0] * n
        assert not validator.is_valid(bad)
    bad = deepcopy(action)
    bad["payload"]["scored_values"][0] = True
    assert not validator.is_valid(bad)


@pytest.mark.parametrize("operation", list(FORECAST_FIELDS))
def test_operation_fields_match_runtime_allowlist(operation):
    schema = forecast_schema(operation)
    fields = set(schema["properties"])
    common = {
        "operation",
        "declared_effect",
        "evidence_refs",
        "rationale_origin",
        "rationale_scope",
        "correction_reason_code",
    }
    assert fields == common | FORECAST_FIELDS[operation]
    for field in set(ForecastArgs.model_fields) - fields:
        # All these fields are known to the broad model, but intentionally absent on this operation.
        assert field not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["operation"]["const"] == operation


def test_action_payload_correlation_and_batch_limit():
    v = Draft202012Validator(action_schema({"horizon_length": 96, "forecast_gap": 0}))
    a = {
        "action_type": "CREATE_FORECAST",
        "state_version": 0,
        "payload": {"operation": "adopt_candidate", "candidate_id": "forecast_1"},
    }
    assert v.is_valid(a)
    a["action_type"] = "FINALIZE"
    assert not v.is_valid(a)
    a = {
        "action_type": "CALL_TOOL",
        "state_version": 0,
        "payload": {"calls": [{"call_id": "x", "tool_id": "feature.basic_statistics"}] * 5},
    }
    assert not v.is_valid(a)


def test_cards_explain_optional_arguments_and_operation_requirements():
    cards = {c["tool_id"]: c for c in Registry().cards()}
    blend = cards["sandbox.blend_forecasts"]
    assert "source_ids" in blend["required"]
    assert "operation" not in blend["required"]
    assert "parent_id" not in blend["arguments"]
    assert "scored_values" not in blend["arguments"]
    assert cards["context.get_variable_slice"]["arguments"]["variables"]["type"] == "list[string]"
    assert "ramp" in cards["sample.search_target_shape"]["argument_note"]


def test_provider_schema_used_on_identical_count_and_generate_request(bundle):
    p = OpenAICompatibleProvider(
        ProviderProfile(endpoint="http://localhost", model="test", structured_action_schema=True)
    )
    messages = [{"role": "user", "content": json.dumps({"episode": bundle.episode.model_dump()})}]
    r = p.request(messages, Action.model_json_schema())
    assert r == p.request(messages, Action.model_json_schema())
    wire = r["tools"][0]["function"]["parameters"]
    assert "anyOf" not in wire
    assert (
        wire["properties"]["payload"]["properties"]["scored_values"]["maxItems"]
        == bundle.episode.task.horizon_length
    )
    assert r["parallel_tool_calls"] is False
    p.client.close()


def test_completion_rescue_schema_cannot_be_replaced_by_provider_native_schema(bundle):
    from castbench.completion_rescue import action_schema

    p = OpenAICompatibleProvider(
        ProviderProfile(endpoint="http://localhost", model="test", structured_action_schema=True)
    )
    task = bundle.episode.task
    rescue = {
        "candidate_ids": ["model_candidate:test"],
        "horizon_length": task.horizon_length,
        "unit": task.target_unit,
    }
    schema = action_schema(rescue, 7, bundle.episode.forecast_timeline.scored_timestamps)
    messages = [{"role": "user", "content": json.dumps({"episode": bundle.episode.model_dump()})}]
    request = p.request(messages, schema)
    wire = request["tools"][0]["function"]["parameters"]
    assert wire == schema
    assert wire["properties"]["action_type"]["const"] == "CREATE_FORECAST"
    p.client.close()


def test_pending_repair_is_lossless_even_after_capacity_warning(bundle, tmp_path):
    from conftest import submit

    store = Store(tmp_path / "state")
    runner = Runner(bundle, store, ScriptedProvider([submit(bundle, [1.0])]), "r", protocol_repair=True)
    runner.step()
    packet = runner.packet(runner.state())
    pending = packet["pending_protocol_repair_feedback"]
    compressed = reduce_packet(packet)["pending_protocol_repair_feedback"]
    assert all(compressed[k] == v for k, v in pending.items())
    reply = store.trace("r")[0]["response"]
    assert "schema_query" not in reply["recovery"]
    assert "do not spend" in reply["recovery"]["hint"]
    assert "candidate_id" not in pending["feedback"]["schemas"]["CREATE_FORECAST"]["properties"]


def test_portable_schema_bounds_without_root_union():
    schema = wire_action_schema({"horizon_length": 96, "forecast_gap": 0})
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)
    a = {
        "action_type": "CREATE_FORECAST",
        "state_version": 0,
        "payload": {"operation": "submit_full_forecast", "scored_values": [1.0] * 96},
    }
    assert v.is_valid(a)  # Missing timestamps/unit is deliberately left to semantic runtime validation.
    a["payload"]["scored_values"].append(1.0)
    assert not v.is_valid(a)


def test_experimental_native_payload_schema_is_opt_in():
    p = OpenAICompatibleProvider(ProviderProfile(endpoint="http://localhost", model="test"))
    outer = Action.model_json_schema()
    request = p.request([], outer)
    assert request["tools"][0]["function"]["parameters"] == outer
    assert p.profile.structured_action_schema is False
    p.client.close()


@pytest.mark.parametrize("actual,expected", [(95, 96), (97, 96), (1, 0)])
def test_length_feedback_states_received_and_required_counts(actual, expected):
    with pytest.raises(AgentError, match=f"exactly {expected} finite values; received {actual} values"):
        finite_array([1.0] * actual, expected)
    assert finite_array([1.0] * expected, expected).size == expected


def test_nonfinite_feedback_is_not_misreported_as_length_error():
    with pytest.raises(AgentError, match="NaN or infinity"):
        finite_array([float("nan")], 1)

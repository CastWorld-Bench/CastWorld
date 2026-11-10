import json

import pytest
from conftest import call, final, run_actions
from pydantic import ValidationError

from castbench.action_schema import forecast_schema
from castbench.catalog import Registry
from castbench.common import AgentError, json_text
from castbench.context_memory import replay_group
from castbench.contracts import ForecastArgs
from castbench.forecast import create_forecast


def test_constant_submission_exact_timeline_and_no_model_dependency(tmp_path, bundle):
    store, runner = run_actions(
        tmp_path,
        bundle,
        [
            {
                "action_type": "CREATE_FORECAST",
                "payload": {
                    "operation": "constant_forecast",
                    "constant_value": 17.5,
                    "unit": bundle.episode.task.target_unit,
                },
            },
            final(),
        ],
    )
    state = runner.state()
    assert not state["errors"]
    forecast = state["forecasts"]["forecast_1"]
    assert forecast["scored_values"] == [17.5] * bundle.episode.task.horizon_length
    assert forecast["gap_values"] is None
    assert forecast["parents"] == []
    assert store.replay("test") == state
    recipe = store.trace("test")[0]["response"]["read_forecast"]
    assert runner._read_state(recipe, state)["scored_values"] == forecast["scored_values"]


def test_latest_cursor_after_rejected_action_has_current_state(tmp_path, bundle):
    from conftest import submit

    from castbench.providers import ScriptedProvider
    from castbench.runner import Runner
    from castbench.storage import Store

    runner = Runner(
        bundle,
        Store(tmp_path / "cursor"),
        ScriptedProvider([submit(bundle, values=[1])]),
        "cursor",
        protocol_repair=True,
    )
    runner.step()
    messages = runner.messages(runner.state())
    latest = json.loads(messages[-1]["content"])["current_action_cursor"]
    assert latest["state_version"] == 1
    assert latest["active_forecast_id"] is None
    assert latest["repair_pending"]["intent"]["kind"] == "forecast_creation"
    assert runner.state()["forecasts"] == {}


def test_inspection_cannot_claim_to_repair_failed_forecast_creation(tmp_path, bundle):
    from conftest import submit

    store, runner = run_actions(
        tmp_path,
        bundle,
        [
            submit(bundle),
            submit(bundle, values=[1]),
            call("sandbox.validate_forecast", {"forecast_id": "forecast_1"}),
            final(),
        ],
        protocol_repair=True,
    )
    failed_retry = store.trace("test")[2]["response"]
    assert failed_retry["code"] == "REPAIR_NOT_RESOLVED"
    assert runner.state()["protocol_alarms"][0]["status"] == "failed"
    assert len(runner.state()["forecasts"]) == 1
    assert not runner.state()["validations"]


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "0"])
def test_constant_rejects_invalid_numeric_value(value):
    with pytest.raises(ValidationError):
        ForecastArgs(operation="constant_forecast", constant_value=value, unit="x")


def test_constant_requires_explicit_value_and_unit(bundle):
    for args in [
        ForecastArgs(operation="constant_forecast", unit=bundle.episode.task.target_unit),
        ForecastArgs(operation="constant_forecast", constant_value=1),
    ]:
        with pytest.raises(AgentError):
            create_forecast(args, {"forecasts": {}, "active_forecast_id": None}, bundle)
    with pytest.raises(ValidationError):
        ForecastArgs(operation="constant_forecast", constant_value=1, offset=10)
    assert set(forecast_schema("constant_forecast")["required"]) == {"operation", "constant_value", "unit"}


def test_progressive_catalog_does_not_hide_tools_or_required_types():
    registry = Registry()
    detailed, compact = registry.cards(), registry.cards(detailed=False)
    assert [x["tool_id"] for x in compact] == [x["tool_id"] for x in detailed]
    assert len(json_text(compact)) < len(json_text(detailed))
    for full, brief in zip(detailed, compact):
        assert brief["required"] == full["required"]
        assert set(brief["arguments"]) == set(full["required"])
        assert set(brief["arguments"]) | set(brief["optional_fields"]) == set(full["arguments"])
        if "default_arguments" in brief:
            registry.validate(brief["tool_id"], brief["default_arguments"])


def test_rejected_array_compacts_in_context_but_feedback_remains():
    raw = {"sensitive_to_repetition": [11] * 2000}

    class NeverReplay:
        def assistant_message(self, value):
            raise AssertionError("Rejected raw arrays must not enter native history")

    row = {
        "turn": 1,
        "event": {
            "valid": False,
            "action": {
                "action_type": "CREATE_FORECAST",
                "payload": {"operation": "submit_full_forecast", "scored_values": [11] * 2000},
            },
        },
        "response": {"status": "agent_error", "message": "Expected 96 points", "repair": {"example": "kept"}},
    }
    result = replay_group(NeverReplay(), raw, row)
    content = json.loads(result[0]["content"])
    assert content["prior_action_response"] == row["response"]
    assert content["rejected_action"]["read_original"] == {"section": "history", "offset": 0, "limit": 1}
    assert len(json_text(result)) < 1500
    assert len(raw["sensitive_to_repetition"]) == 2000
    assert content["rejected_arguments"]["scored_values"]["items"] == 2000


def test_constant_gap_is_explicit_not_silently_filled(bundle):
    assert bundle.episode.task.forecast_gap > 0
    args = ForecastArgs(
        operation="constant_forecast",
        constant_value=12.5,
        gap_constant_value=11.5,
        unit=bundle.episode.task.target_unit,
    )
    forecast = create_forecast(args, {"forecasts": {}, "active_forecast_id": None}, bundle)
    assert forecast["gap_values"] == [11.5] * bundle.episode.task.forecast_gap
    assert forecast["scored_values"] == [12.5] * bundle.episode.task.horizon_length


def test_constant_obeys_same_hard_constraints(bundle):
    from castbench.contracts import Constraint

    copy = bundle.model_copy(deep=True)
    copy.constraints = [Constraint(constraint_id="nonnegative", kind="minimum", value=0, hard=True)]
    copy.default_check_profile = ["timeline", "finite", "domain"]
    with pytest.raises(AgentError, match="Hard constraint"):
        create_forecast(
            ForecastArgs(
                operation="constant_forecast", constant_value=-1, unit=copy.episode.task.target_unit
            ),
            {"forecasts": {}, "active_forecast_id": None},
            copy,
        )


def test_result_read_recipe_uses_real_authorized_artifact(tmp_path, bundle):
    store, runner = run_actions(
        tmp_path,
        bundle,
        [
            call("feature.basic_statistics"),
            {
                "action_type": "CREATE_FORECAST",
                "payload": {
                    "operation": "constant_forecast",
                    "constant_value": 20,
                    "unit": bundle.episode.task.target_unit,
                },
            },
            final(),
        ],
    )
    result = store.trace("test")[0]["response"]["results"][0]
    recipe = result["read_result"]
    assert recipe["ref"] == result["artifact_ref"]
    value = runner._read_state(recipe, runner.state())
    assert value["encoding"] == "json_text"
    child = value["children"][0]
    assert child["read"]["path"] == ["structured_result", child["key"]]
    runner._read_state(child["read"], runner.state())

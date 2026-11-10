from copy import deepcopy

import pytest
from conftest import call, final, submit

from castbench.common import InfrastructureError
from castbench.evaluator import aggregate, score_run
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


def setup(tmp_path, bundle, actions, **kwargs):
    store = Store(tmp_path / "repair")
    runner = Runner(bundle, store, ScriptedProvider(actions), "repair", protocol_repair=True, **kwargs)
    return store, runner


def short(bundle):
    return submit(bundle, [20.0] * (bundle.episode.task.horizon_length - 1))


def test_short_horizon_corrected_counts_alarm_not_error(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, runner = setup(tmp_path, bundle, [short(bundle), submit(bundle), final()])
    runner.run()
    state = runner.state()
    assert state["errors"] == []
    assert state["protocol_alarms"][0]["status"] == "recovered"
    assert state["repair_pending"] is None
    export = store.export("repair", include_raw=True)
    assert export["trace"][0]["event"]["valid"] is False
    detail = export["trace"][0]["response"]["protocol_alarm"]["feedback"]
    assert detail["forecast_contract"]["scored_count"] == bundle.episode.task.horizon_length
    assert detail["issues"][0]["received_count"] == bundle.episode.task.horizon_length - 1
    assert any(x["action_type"] == "CREATE_FORECAST" for x in detail["correct_format_examples"])
    score = score_run(export, truth)
    assert score["ProtocolAlarmCount"] == 1 and score["ErrorActionCount"] == 0
    assert score["RawInvalidActionCount"] == 1
    assert score["process_items"]["action"] == pytest.approx(97.5)
    assert store.replay("repair") == state


def test_failed_retry_counts_once_and_later_errors_count(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, runner = setup(
        tmp_path, bundle, [short(bundle), short(bundle), short(bundle), submit(bundle), final()]
    )
    runner.run()
    state = runner.state()
    assert [e["turn"] for e in state["errors"]] == [2, 3]
    assert len(state["protocol_alarms"]) == 1
    assert state["protocol_alarms"][0]["status"] == "failed"
    score = score_run(store.export("repair"), truth)
    assert score["ErrorActionCount"] == 2 and score["RawInvalidActionCount"] == 3
    assert score["ProtocolRepairFailureCount"] == 1


def test_unrelated_success_cannot_launder_error(tmp_path, bundle):
    store, runner = setup(
        tmp_path, bundle, [short(bundle), call("feature.basic_statistics"), submit(bundle), final()]
    )
    runner.run()
    assert runner.state()["errors"][0]["code"] == "REPAIR_NOT_RESOLVED"
    assert not runner.state()["observations"]
    assert store.trace("repair")[1]["event"]["valid"] is False


def test_bad_forecast_quality_is_not_protocol_error(tmp_path, bundle):
    b = bundle.model_copy(deep=True)
    b.constraints = []
    _, runner = setup(tmp_path, b, [submit(b, [1e9] * b.episode.task.horizon_length), final()])
    runner.run()
    assert runner.state()["protocol_alarms"] == []
    assert runner.state()["errors"] == []


def test_model_error_feedback_is_executable_and_retains_selected_tool(tmp_path, bundle):
    from test_tools import FakeModelService

    task = bundle.episode.task
    good = call(
        "model.moirai_forecast",
        {
            "frequency": task.frequency,
            "forecast_gap": task.forecast_gap,
            "horizon_length": task.horizon_length,
        },
    )
    store, runner = setup(
        tmp_path, bundle, [call("model.moirai_forecast"), good], model_service=FakeModelService()
    )
    runner.step()
    feedback = store.trace("repair")[0]["response"]["protocol_alarm"]["feedback"]
    assert runner.packet(runner.state())["pending_protocol_repair_feedback"]["feedback"] == feedback
    example = feedback["correct_format_examples"][0]
    assert example["payload"]["calls"][0]["tool_id"] == "model.moirai_forecast"
    assert any(x["location"] == ["payload", "calls", 0, "arguments", "frequency"] for x in feedback["issues"])
    from castbench.contracts import Action

    Action.model_validate(example)
    runner.registry.validate("model.moirai_forecast", example["payload"]["calls"][0]["arguments"])
    runner.step()
    assert runner.state()["protocol_alarms"][0]["status"] == "recovered"
    assert runner.state()["forecasts"]["forecast_1"]["kind"] == "model_candidate"


@pytest.mark.parametrize(
    "bad",
    [
        {
            "action_type": "CALL_TOOL",
            "payload": {"calls": [{"tool_id": {}, "call_id": "x", "arguments": {}}]},
        },
        {"action_type": "CALL_TOOL", "payload": {"calls": []}},
        {"action_type": "REQUEST_STATE", "payload": {"section": {}}},
    ],
)
def test_malformed_nested_parameters_do_not_crash_alarm(tmp_path, bundle, bad):
    _, runner = setup(tmp_path, bundle, [bad, bad, submit(bundle), final()])
    runner.run()
    assert len(runner.state()["protocol_alarms"]) == 1
    assert len(runner.state()["errors"]) == 1


@pytest.mark.parametrize("first", ["bad_forecast", "no_forecast"])
def test_budget_boundary_reserves_exactly_one_retry(tmp_path, bundle_and_truth, first):
    bundle, truth = bundle_and_truth
    b = bundle.model_copy(deep=True)
    b.budget.max_rounds = 1
    initial = (
        short(b)
        if first == "bad_forecast"
        else {"action_type": "REQUEST_STATE", "payload": {"section": "summary"}}
    )
    store, runner = setup(tmp_path, b, [initial, submit(b)])
    runner.step()
    assert runner.state()["status"] == "running"
    if first == "bad_forecast":
        assert runner.state()["repair_pending"]
    else:
        assert runner.state()["completion_rescue"]["status"] == "pending"
    runner.run()
    assert runner.state()["rounds_used"] == 2
    assert runner.state()["status"] == "finished"
    if first == "bad_forecast":
        assert runner.state()["protocol_alarms"][0]["status"] == "recovered"
    else:
        assert runner.state()["completion_rescue"]["status"] == "recovered"
    assert score_run(store.export("repair"), truth)["completed"]


def test_failed_completion_rescue_cannot_extend_again(tmp_path, bundle):
    b = bundle.model_copy(deep=True)
    b.budget.max_rounds = 1
    no_forecast = {"action_type": "REQUEST_STATE", "payload": {"section": "summary"}}
    _, runner = setup(tmp_path, b, [no_forecast, no_forecast])
    runner.run()
    assert runner.state()["rounds_used"] == 2
    assert runner.state()["status"] == "finished"
    assert runner.state()["final"]["forecast_id"] is None
    assert runner.state()["completion_rescue"]["status"] == "failed"
    assert len(runner.state()["errors"]) == 1


def test_completion_rescue_is_independent_of_an_already_used_alarm(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    b = bundle.model_copy(deep=True)
    b.budget.max_rounds = 3
    actions = [
        short(b),
        submit(b),
        {"action_type": "REQUEST_STATE", "payload": {"section": "summary"}},
    ]
    # The existing Agent forecast means completion rescue is unnecessary.
    store, runner = setup(tmp_path, b, actions)
    runner.run()
    assert runner.state()["completion_rescue"] is None
    assert score_run(store.export("repair"), truth)["completed"]

    # Consume the protocol alarm on a non-forecast action, then exhaust the
    # ordinary budget without a prediction. A separate narrowed completion turn remains.
    bad = {"action_type": "REQUEST_STATE", "payload": {"section": "not-a-section"}}
    fixed = {"action_type": "REQUEST_STATE", "payload": {"section": "summary"}}
    store, runner = setup(tmp_path / "second", b, [bad, fixed, fixed, submit(b)])
    runner.run()
    assert runner.state()["protocol_alarms"][0]["status"] == "recovered"
    assert runner.state()["completion_rescue"]["status"] == "recovered"
    assert runner.state()["rounds_used"] == 4
    assert score_run(store.export("repair"), truth)["completed"]


def test_completion_rescue_native_schema_only_allows_forecast_creation(tmp_path, bundle):
    class SchemaRecorder(ScriptedProvider):
        def request(self, messages, schema):
            self.schemas = getattr(self, "schemas", []) + [deepcopy(schema)]
            return super().request(messages, schema)

    b = bundle.model_copy(deep=True)
    b.budget.max_rounds = 1
    provider = SchemaRecorder(
        [
            {"action_type": "REQUEST_STATE", "payload": {"section": "summary"}},
            submit(b),
        ]
    )
    store = Store(tmp_path / "schema")
    runner = Runner(b, store, provider, "schema", protocol_repair=True)
    runner.run()
    rescue_schema = provider.schemas[-1]
    assert rescue_schema["properties"]["action_type"]["const"] == "CREATE_FORECAST"
    assert set(rescue_schema["properties"]["payload"]["oneOf"][0]["properties"]) <= {
        "operation",
        "candidate_id",
        "constant_value",
        "unit",
        "scored_values",
        "scored_timestamps",
    }


def test_finalize_without_prediction_can_create_on_retry(tmp_path, bundle):
    _, runner = setup(tmp_path, bundle, [final("absent"), submit(bundle), final()])
    runner.run()
    assert runner.state()["protocol_alarms"][0]["status"] == "recovered"
    assert not runner.state()["errors"]


def test_transaction_restart_preserves_one_chance(tmp_path, bundle):
    fired = []

    def fault(point):
        if point == "after_provider_accept" and not fired:
            fired.append(True)
            raise InfrastructureError("simulated interruption")

    actions = [short(bundle), submit(bundle), final()]
    store, runner = setup(tmp_path, bundle, actions, fault_hook=fault)
    with pytest.raises(InfrastructureError):
        runner.run()
    resumed = Runner(bundle, store, ScriptedProvider(actions), "repair", protocol_repair=True)
    resumed.run()
    assert len(resumed.state()["protocol_alarms"]) == 1
    assert resumed.state()["protocol_alarms"][0]["status"] == "recovered"
    assert not resumed.state()["errors"]
    assert store.replay("repair") == resumed.state()


def test_infrastructure_pause_does_not_consume_retry(tmp_path, bundle):
    store, runner = setup(tmp_path, bundle, [short(bundle), submit(bundle), final()])
    runner.step()
    before = runner.state()
    respond = runner.provider.respond

    def fail(request):
        raise InfrastructureError("network unavailable")

    runner.provider.respond = fail
    with pytest.raises(InfrastructureError):
        runner.step()
    assert runner.state() == before
    runner.provider.respond = respond
    runner.run()
    assert runner.state()["protocol_alarms"][0]["status"] == "recovered"


def test_aggregation_keeps_alarm_metrics_and_separates_policies(tmp_path, bundle_and_truth):
    b, truth = bundle_and_truth
    store, runner = setup(tmp_path, b, [short(b), submit(b), final()])
    runner.run()
    score = score_run(store.export("repair"), truth)
    result = aggregate([score], bootstrap_replicates=0)
    domain = result["datasets"][score["dataset_id"]]
    assert domain["alarm_sample_rate"] == 1 and domain["repair_success_rate"] == 1
    assert domain["alarm_total"] == 1 and domain["error_action_total"] == 0
    assert domain["completion_rescue_total"] == 0
    legacy = deepcopy(score)
    for key in (
        "CompletionRescueCount",
        "CompletionRescueSuccessCount",
        "CompletionRescueFailureCount",
    ):
        legacy.pop(key)
    assert aggregate([legacy], bootstrap_replicates=0)["datasets"][score["dataset_id"]][
        "completion_rescue_total"
    ] == 0
    other = deepcopy(score)
    other["episode_id"] += "_different"
    other["protocol_repair_profile_hash"] = "different_policy"
    with pytest.raises(ValueError, match="protocol_repair_profile_hash"):
        aggregate([score, other], bootstrap_replicates=0)


def test_score_reports_completion_rescue_without_changing_forecast(tmp_path, bundle_and_truth):
    b, truth = bundle_and_truth
    no_forecast = {"action_type": "REQUEST_STATE", "payload": {"section": "summary"}}
    store, runner = setup(
        tmp_path / "completion-score",
        b,
        [no_forecast] * b.budget.max_rounds + [submit(b)],
    )
    runner.run()
    score = score_run(store.export("repair"), truth)
    assert score["CompletionRescueCount"] == 1
    assert score["CompletionRescueSuccessCount"] == 1
    assert score["CompletionRescueFailureCount"] == 0


def test_long_history_compaction_cannot_remove_pending_feedback(tmp_path, bundle):
    import json
    from types import SimpleNamespace

    from test_context_memory import CountedScript

    from castbench.context_memory import build_messages
    from castbench.providers import SYSTEM_PROMPT

    actions = [short(bundle)]
    provider = CountedScript(actions)
    store = Store(tmp_path / "memory")
    runner = Runner(bundle, store, provider, "memory", protocol_repair=True)
    runner.step()
    packet = runner.packet(runner.state())
    provider.profile = SimpleNamespace(
        context_window_tokens=18000,
        max_output_tokens=1000,
        context_safety_tokens=100,
        context_trigger_fraction=0.1,
        token_count_mode="test_estimate",
    )
    messages, _ = build_messages(provider, store, "memory", packet, SYSTEM_PROMPT)
    current = json.loads(messages[1]["content"])
    assert (
        current["pending_protocol_repair_feedback"]["feedback"]["forecast_contract"]["scored_count"]
        == bundle.episode.task.horizon_length
    )


def test_mixed_batch_cannot_drop_failing_tool_on_retry(tmp_path, bundle):
    bad = {
        "action_type": "CALL_TOOL",
        "payload": {
            "calls": [
                call("feature.basic_statistics")["payload"]["calls"][0],
                {
                    "call_id": "two",
                    "tool_id": "feature.multiscale_trend",
                    "arguments": {},
                    "purpose": "trend",
                },
            ]
        },
    }
    store, runner = setup(tmp_path, bundle, [bad, call("feature.basic_statistics"), submit(bundle), final()])
    runner.run()
    assert runner.state()["errors"][0]["code"] == "REPAIR_NOT_RESOLVED"
    assert not runner.state()["observations"]
    examples = store.trace("repair")[0]["response"]["protocol_alarm"]["feedback"]["correct_format_examples"]
    assert all(e["action_type"] == "CALL_TOOL" and len(e["payload"]["calls"]) == 2 for e in examples)


def test_empty_response_gets_one_alarm_then_error(tmp_path, bundle):
    raw = {
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "", "tool_calls": []}}
        ]
    }
    _, runner = setup(tmp_path, bundle, [raw, raw, submit(bundle), final()])
    runner.run()
    assert len(runner.state()["protocol_alarms"]) == 1
    assert len(runner.state()["errors"]) == 1


def test_forecast_gap_length_and_no_padding(tmp_path, bundle):
    b = bundle
    assert b.episode.task.forecast_gap > 0
    bad = submit(
        b,
        gap_values=[1.0] * (b.episode.task.forecast_gap - 1),
        gap_timestamps=b.episode.forecast_timeline.gap_timestamps,
    )
    _, runner = setup(tmp_path, b, [bad, submit(b), final()])
    runner.step()
    assert not runner.state()["forecasts"]
    runner.run()
    assert runner.state()["forecasts"]["forecast_1"]["gap_values"] is None
    assert runner.state()["protocol_alarms"][0]["status"] == "recovered"


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("CALL_TOOL", {"calls": {}}),
        (
            "CALL_TOOL",
            {
                "calls": [
                    {"tool_id": "model.chronos_forecast", "arguments": None, "call_id": "x", "purpose": []}
                ]
            },
        ),
        ("DECLARE_NEED", {"description": {}, "kind": [], "scope": None}),
        ("REGISTER_EVIDENCE", {"source_observation_ids": {}, "statement": []}),
        ("CREATE_FORECAST", {"operation": {}, "source_ids": [{}]}),
        ("CHECK_FORECAST", {"forecast_id": {}, "checks": {}}),
        ("REQUEST_STATE", {"section": ["artifact"], "ref": {}}),
        ("FINALIZE", {"forecast_id": {}, "stop_reason": []}),
    ],
)
def test_bad_types_across_all_actions_raise_alarm_not_host_exception(tmp_path, bundle, kind, payload):
    _, runner = setup(tmp_path, bundle, [{"action_type": kind, "payload": payload}])
    runner.step()
    assert len(runner.state()["protocol_alarms"]) == 1
    assert runner.state()["errors"] == []
    assert runner.state()["status"] == "running"


@pytest.mark.parametrize("wrong_length", [95, 97])
def test_exact_96_point_forecast_contract(tmp_path, wrong_length):
    from castbench.common import load_json
    from castbench.contracts import PublicBundle
    from castbench.data import build_packages, synthetic_fixture

    frame, config = synthetic_fixture()
    config["horizon_length"] = 96
    config["train_end"] = frame["date"].iloc[70].isoformat()
    config["episode_origins"] = [{"timestamp": frame["date"].iloc[100].isoformat(), "split": "dev"}]
    build_packages(frame, config, tmp_path / "public", tmp_path / "private")
    b = PublicBundle.model_validate(load_json(next((tmp_path / "public" / "dev").glob("*.json"))))
    store, runner = setup(tmp_path, b, [submit(b, [20.0] * wrong_length), submit(b), final()])
    runner.step()
    assert not runner.state()["forecasts"]
    diag = store.trace("repair")[0]["response"]["protocol_alarm"]["feedback"]
    assert diag["issues"][0]["required_count"] == 96
    assert diag["issues"][0]["received_count"] == wrong_length
    runner.run()
    assert len(runner.state()["forecasts"]["forecast_1"]["scored_values"]) == 96
    assert not runner.state()["errors"]


def test_mixed_model_batch_feedback_never_drops_nonmodel_call(tmp_path, bundle):
    batch = {
        "action_type": "CALL_TOOL",
        "payload": {
            "calls": [
                {"call_id": "a", "tool_id": "model.moirai_forecast", "arguments": {}, "purpose": "model"},
                {"call_id": "b", "tool_id": "feature.basic_statistics", "arguments": {}, "purpose": "stats"},
            ]
        },
    }
    store, runner = setup(tmp_path, bundle, [batch])
    runner.step()
    info = store.trace("repair")[0]["response"]["protocol_alarm"]["feedback"]
    assert len(info["selected_model_call_fragments"]) == 1
    assert all(len(e["payload"]["calls"]) == 2 for e in info["correct_format_examples"])


def test_forecast_repair_examples_do_not_offer_unrelated_schema_query(tmp_path, bundle):
    store, runner = setup(tmp_path, bundle, [short(bundle)])
    runner.step()
    examples = store.trace("repair")[0]["response"]["protocol_alarm"]["feedback"]["correct_format_examples"]
    assert examples and all(e["action_type"] == "CREATE_FORECAST" for e in examples)

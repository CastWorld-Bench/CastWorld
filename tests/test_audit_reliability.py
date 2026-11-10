import json
from copy import deepcopy

import numpy as np
import pytest
from conftest import call, final, run_actions, submit
from pydantic import ValidationError

from castbench.common import AgentError, InfrastructureError
from castbench.contracts import PublicBundle
from castbench.data import build_packages, synthetic_fixture
from castbench.evaluator import aggregate, metrics, score_run
from castbench.forecast import diff_report
from castbench.providers import ScriptedProvider, decode
from castbench.runner import Runner
from castbench.storage import Store
from castbench.tools import _shape_distance


@pytest.mark.parametrize(
    "raw",
    [
        {"choices": [None]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {"tool_calls": [None]}}]},
        {"choices": [{"message": {"tool_calls": [{"type": "function", "function": None}]}}]},
    ],
)
def test_malformed_complete_envelope_is_charged_and_recovers(tmp_path, bundle, raw):
    store, runner = run_actions(tmp_path, bundle, [raw, submit(bundle), final()])
    assert runner.state()["rounds_used"] == 3
    assert len(runner.state()["errors"]) == 1
    assert not store.trace("test")[0]["event"]["valid"]
    assert runner.state()["status"] == "finished"


def test_duplicate_action_keys_are_not_silently_overwritten(bundle):
    provider = ScriptedProvider([submit(bundle)])
    raw = provider.respond({"messages": [{}, {"content": json.dumps({"state": {"state_version": 0}})}]})
    function = raw["choices"][0]["message"]["tool_calls"][0]["function"]
    function["arguments"] = '{"state_version":99,' + function["arguments"][1:]
    with pytest.raises(AgentError):
        decode(raw)


def test_chronological_versions_survive_json_key_sorting(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    bundle = bundle.model_copy(
        update={"budget": bundle.budget.model_copy(update={"max_rounds": 20, "state_summary_limit": 3})}
    )
    values = [20, 21, 22, 23, 24, 25, 26, 27, 28, 27, 28, 29]
    store, runner = run_actions(
        tmp_path, bundle, [submit(bundle, [float(v)] * 12) for v in values] + [final("forecast_12")]
    )
    assert score_run(store.export("test"), truth)["unsupported_reversals"] == 2
    assert list(runner.packet(runner.state())["state"]["forecasts"]) == [
        "forecast_10",
        "forecast_11",
        "forecast_12",
    ]
    page = runner._read_state({"section": "forecasts", "offset": 0, "limit": 3}, runner.state())
    assert [item[0] for item in page] == ["forecast_1", "forecast_2", "forecast_3"]


def evidence(scope):
    return {
        "action_type": "REGISTER_EVIDENCE",
        "payload": {
            "source_observation_ids": ["observation_1"],
            "scope": scope,
            "application_scope": [{"timeline": "scored", "start": 0, "end": 12}],
            "relation_type": "direct_context",
            "statement": "Use the observed facts",
        },
    }


def test_evidence_cannot_expand_all_timeline_slice(tmp_path, bundle):
    actions = [
        call(
            "context.get_variable_slice",
            {"variables": ["temperature"], "scope": {"timeline": "all", "start": 0, "end": 4}},
        ),
        evidence({"timeline": "scored", "variables": ["temperature"]}),
        submit(bundle),
        final(),
    ]
    _, runner = run_actions(tmp_path, bundle, actions)
    assert not runner.state()["evidence"]
    assert len(runner.state()["errors"]) == 1


def test_numeric_covariate_observation_has_real_variable_scope(tmp_path, bundle):
    actions = [
        call(
            "feature.window_summary",
            {"series_ref": "covariate:temperature", "scope": {"timeline": "history"}},
        ),
        evidence({"timeline": "history", "variables": ["event"]}),
        submit(bundle),
        final(),
    ]
    _, runner = run_actions(tmp_path, bundle, actions)
    assert runner.state()["observations"]["observation_1"]["scope"]["variables"] == ["temperature"]
    assert not runner.state()["evidence"]


def test_explicit_target_need_accepts_implicit_history_tool(tmp_path, bundle):
    declared = {
        "action_type": "DECLARE_NEED",
        "payload": {
            "description": "Inspect target history",
            "kind": "feature",
            "scope": {"timeline": "history", "variables": ["load"]},
        },
    }
    action = call("feature.window_summary")
    action["payload"]["calls"][0].pop("purpose")
    action["payload"]["calls"][0]["need_ids"] = ["need_1"]
    _, runner = run_actions(tmp_path, bundle, [declared, action, submit(bundle), final()])
    assert not runner.state()["errors"]
    assert runner.state()["needs"]["need_1"]["status"] == "observed"


def test_shape_ramp_distance_never_bridges_a_missing_gap():
    # Only edges 0->1 and 3->4 are observed. The jump over index 2 is not a ramp.
    parts = _shape_distance([0, 1, np.nan, 100, 101], [0, 1, 2, 3, 4], 1.0)
    assert parts[2] == 0.0


@pytest.mark.parametrize("bad", ["not-a-number", True, float("inf")])
def test_training_covariate_values_must_match_variable_contract(bundle, bad):
    raw = bundle.model_dump()
    raw["samples"][0]["context"]["temperature"][0] = bad
    with pytest.raises(ValidationError):
        PublicBundle.model_validate(raw)


def test_builder_rejects_ambiguous_naive_origin(tmp_path):
    frame, config = synthetic_fixture()
    config["episode_origins"][0]["timestamp"] = "2024-01-08T12:00:00"
    with pytest.raises(ValueError, match="offset|timezone"):
        build_packages(frame, config, tmp_path / "pub", tmp_path / "private")


@pytest.mark.parametrize("name", ["../escape", "nested/data", "C:\\escape"])
def test_builder_dataset_id_cannot_escape_output_directory(tmp_path, name):
    frame, config = synthetic_fixture()
    config["dataset_id"] = name
    with pytest.raises(ValueError, match="dataset_id"):
        build_packages(frame, config, tmp_path / "pub", tmp_path / "private")


def test_replay_detects_response_tampering(tmp_path, bundle):
    store, _ = run_actions(tmp_path, bundle, [submit(bundle), final()])
    with store.connection() as db:
        db.execute("UPDATE commits SET response=? WHERE turn=1", ('{"status":"forged"}',))
    with pytest.raises(InfrastructureError, match="integrity"):
        store.replay("test")


def test_completed_resume_checks_integrity(tmp_path, bundle):
    actions = [submit(bundle), final()]
    store, _ = run_actions(tmp_path, bundle, actions)
    with store.connection() as db:
        state = deepcopy(store.get("test"))
        state["final"]["forecast_id"] = "forged"
        db.execute("UPDATE runs SET state=? WHERE run_id='test'", (json.dumps(state),))
    with pytest.raises(InfrastructureError, match="Checkpoint|integrity"):
        Runner(bundle, store, ScriptedProvider(actions), "test").run()


def test_overflowing_revision_is_rejected_without_pausing(tmp_path, bundle):
    edit = {
        "action_type": "CREATE_FORECAST",
        "payload": {
            "operation": "scale_values",
            "parent_id": "forecast_1",
            "factor": 1e308,
            "declared_effect": {
                "baseline_id": "forecast_1",
                "ranges": [{"end": 12}],
                "direction": "increase",
                "magnitude": "large",
            },
        },
    }
    _, runner = run_actions(tmp_path, bundle, [submit(bundle), edit, final()])
    assert runner.state()["final"]["forecast_id"] == "forecast_1"
    assert len(runner.state()["errors"]) == 1
    assert len(runner.state()["forecasts"]) == 1


def test_extreme_finite_rewrite_has_safe_difference_report(bundle):
    report = diff_report([1e308, -1e308], [-1e308, 1e308], bundle.revision_scale)
    assert report["direction"] == "mixed" and report["magnitude"] == "large"
    assert report["changed_indices"] == [0, 1]
    assert np.isfinite(report["mean_abs_normalized"])


def test_tiny_scale_does_not_overflow_normalized_score():
    result = metrics([1.0], [2.0], {"scale_abs": 1e-310, "scale_sq": 1e-310})
    assert result["ForecastScore"] < 1e-300
    assert result["MASE"] is None and "MASE_scientific" in result


def test_empty_check_does_not_earn_verification_points(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    action = {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_1", "checks": []}}
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), action, final()])
    assert runner.state()["errors"]
    assert score_run(store.export("test"), truth)["process_items"]["verify"] == 0


def test_aggregate_refuses_mixed_provider_results(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, _ = run_actions(tmp_path, bundle, [submit(bundle), final()])
    first = score_run(store.export("test"), truth)
    second = deepcopy(first)
    second["episode_id"] = "another_episode"
    second["provider_profile_hash"] = "sha256:different-api"
    with pytest.raises(ValueError, match="Provider|provider"):
        aggregate([first, second], bootstrap_replicates=0)


def test_aggregate_does_not_overflow_finite_mean(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, _ = run_actions(tmp_path, bundle, [submit(bundle), final()])
    first = score_run(store.export("test"), truth)
    first["MSE"] = 1e308
    second = deepcopy(first)
    second["episode_id"] = "another_episode"
    result = aggregate([first, second], bootstrap_replicates=0)
    assert result["overall"]["MSE"] == 1e308


def test_gap_application_range_is_validated(tmp_path, bundle):
    action = evidence({"timeline": "history", "variables": ["load"]})
    action["payload"]["application_scope"] = [{"timeline": "gap", "end": 999}]
    _, runner = run_actions(
        tmp_path, bundle, [call("feature.window_summary"), action, submit(bundle), final()]
    )
    assert not runner.state()["evidence"]
    assert len(runner.state()["errors"]) == 1


def test_builder_honors_explicit_check_profile(tmp_path):
    from castbench.common import load_json

    frame, config = synthetic_fixture()
    config["default_check_profile"] = ["timeline", "finite", "domain"]
    build_packages(frame, config, tmp_path / "public", tmp_path / "private")
    bundle = load_json(next((tmp_path / "public/dev").glob("*.json")))
    assert bundle["default_check_profile"] == config["default_check_profile"]


@pytest.mark.parametrize("bad_values", [[1.0], [None] * 12, ["20"] * 12])
def test_bad_model_adapter_is_not_attributed_to_agent(tmp_path, bundle, bad_values):
    class BadModel:
        profiles = {}

        def predict(self, name, episode):
            return {
                "status": "ok",
                "structured_result": {"gap_values": [20.0] * 2, "scored_values": bad_values},
            }

    action = call("model.chronos_forecast", {"frequency": "h", "forecast_gap": 2, "horizon_length": 12})
    with pytest.raises(InfrastructureError, match="invalid candidate"):
        run_actions(tmp_path, bundle, [action], model_service=BadModel())
    state = Store(tmp_path / "store").get("test")
    assert state["rounds_used"] == 0 and not state["forecasts"] and not state["observations"]


def test_randomized_local_operations_match_numpy_reference(bundle):
    from castbench.contracts import ForecastArgs
    from castbench.forecast import create_forecast
    from castbench.runner import initial_state

    rng = np.random.default_rng(20260918)
    for _ in range(80):
        original = rng.normal(20, 2, 12)
        state = initial_state(bundle.episode.episode_id)
        state["forecasts"]["forecast_1"] = {
            "scored_values": original.tolist(),
            "gap_values": None,
            "kind": "agent_generated",
        }
        state["active_forecast_id"] = "forecast_1"
        unchanged = deepcopy(state)
        start = int(rng.integers(0, 12))
        end = int(rng.integers(start + 1, 13))
        amount = float(rng.uniform(0.2, 1.8))
        for operation, key in [("add_offset", "offset"), ("scale_values", "factor")]:
            args = ForecastArgs.model_validate(
                {
                    "operation": operation,
                    "parent_id": "forecast_1",
                    key: amount,
                    "range": {"start": start, "end": end},
                    "declared_effect": {
                        "baseline_id": "forecast_1",
                        "ranges": [{"start": start, "end": end}],
                        "direction": "mixed",
                        "magnitude": "small",
                    },
                }
            )
            result = create_forecast(args, state, bundle)
            expected = original.copy()
            expected[start:end] = (
                expected[start:end] + amount if key == "offset" else expected[start:end] * amount
            )
            np.testing.assert_array_equal(result["scored_values"], expected)
            assert state == unchanged


def test_randomized_metrics_match_direct_formula():
    rng = np.random.default_rng(89)
    for length in [1, 2, 12, 96, 711]:
        y, p = rng.normal(size=(2, length))
        result = metrics(y.tolist(), p.tolist(), {"scale_abs": 2.0, "scale_sq": 3.0})
        assert result["MAE"] == pytest.approx(np.abs(y - p).mean())
        assert result["MSE"] == pytest.approx(((y - p) ** 2).mean())
        assert result["sMAPE"] == pytest.approx((2 * np.abs(y - p) / (np.abs(y) + np.abs(p))).mean())
        assert 0 <= result["ForecastScore"] <= 100


def test_recent_observations_and_needs_are_chronological(tmp_path, bundle):
    bundle = bundle.model_copy(
        update={"budget": bundle.budget.model_copy(update={"max_rounds": 20, "state_summary_limit": 3})}
    )
    _, runner = run_actions(
        tmp_path, bundle, [call("feature.window_summary")] * 11 + [submit(bundle), final()]
    )
    packet = runner.packet(runner.state())["state"]
    assert list(packet["observations"]) == ["observation_9", "observation_10", "observation_11"]
    assert list(packet["needs"]) == ["need_9", "need_10", "need_11"]


def test_equal_valid_need_and_purpose_paths_produce_same_forecast(tmp_path, bundle):
    # Explicit Need and atomic purpose are alternate interfaces, not alternate numerical pipelines.
    declared = {
        "action_type": "DECLARE_NEED",
        "payload": {
            "description": "History",
            "kind": "feature",
            "scope": {"timeline": "history", "variables": ["load"]},
        },
    }
    action = call("feature.window_summary")
    action["payload"]["calls"][0].pop("purpose")
    action["payload"]["calls"][0]["need_ids"] = ["need_1"]
    _, a = run_actions(tmp_path / "a", bundle, [declared, action, submit(bundle), final()])
    _, b = run_actions(tmp_path / "b", bundle, [call("feature.window_summary"), submit(bundle), final()])
    assert not a.state()["errors"] and not b.state()["errors"]
    assert (
        a.state()["forecasts"]["forecast_1"]["scored_values"]
        == b.state()["forecasts"]["forecast_1"]["scored_values"]
    )


def test_runner_owns_frozen_bundle_and_catalog_snapshot(tmp_path, bundle):
    from castbench.catalog import Registry

    registry = Registry()
    provider = ScriptedProvider([submit(bundle), final()])
    runner = Runner(bundle, Store(tmp_path), provider, "frozen", registry=registry)
    original = runner.packet(runner.state())
    bundle.episode.history.target_values[0] = 99999.0
    bundle.budget.max_rounds = 1
    registry.manifests["feature.window_summary"]["description"] = "changed after start"
    assert runner.packet(runner.state()) == original
    runner.run()
    assert runner.state()["rounds_used"] == 2


def test_stages_follow_actions_not_agent_labels(tmp_path, bundle):
    actions = [
        submit(bundle),
        call("feature.window_summary"),
        submit(bundle, [21.0] * 12),
        {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_2"}},
        final("forecast_2"),
    ]
    actions[1]["phase"] = "STOP"
    store, runner = run_actions(tmp_path, bundle, actions)
    assert runner.state()["cycle"] == 2
    assert runner.state()["phase"] == "STOP"
    assert [item["observed_stage"] for item in runner.state()["phase_history"]] == [
        "REVISE",
        "ACQUIRE",
        "REVISE",
        "VERIFY",
        "STOP",
    ]
    observed = store.trace("test")[1]["event"]["stage_transition"]
    assert observed["intent_stage"] == "STOP" and observed["observed_stage"] == "ACQUIRE"
    assert observed["intent_mismatch"] and not runner.state()["errors"]


def test_sandbox_and_direct_actions_have_equal_observed_stages(tmp_path, bundle):
    direct = [
        submit(bundle),
        {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_1"}},
        final(),
    ]
    alias = [
        call("sandbox.submit_full_forecast", submit(bundle)["payload"]),
        call("sandbox.validate_forecast", {"forecast_id": "forecast_1"}),
        final(),
    ]
    _, a = run_actions(tmp_path / "a", bundle, direct)
    _, b = run_actions(tmp_path / "b", bundle, alias)
    assert a.state()["phase_history"] == b.state()["phase_history"]


def test_invalid_action_does_not_commit_a_stage_transition(tmp_path, bundle):
    actions = [
        submit(bundle),
        call("feature.window_summary", {"series_ref": "episode.future.target"}),
        final(),
    ]
    _, runner = run_actions(tmp_path, bundle, actions)
    assert runner.state()["cycle"] == 1
    assert [item["turn"] for item in runner.state()["phase_history"]] == [1, 3]

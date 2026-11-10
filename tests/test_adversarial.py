from copy import deepcopy

import pytest
from conftest import call, final, run_actions, submit

from castbench.evaluator import aggregate, score_run
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": "CALL_TOOL", "payload": {"calls": None}},
        {
            "action_type": "REGISTER_EVIDENCE",
            "payload": {
                "source_observation_ids": [{}],
                "scope": {},
                "application_scope": [],
                "relation_type": "direct_context",
            },
        },
        {"action_type": "FINALIZE", "payload": {"forecast_id": {}, "stop_reason": "x"}},
        {"action_type": "REQUEST_STATE", "payload": {"section": "forecasts", "offset": True}},
        {"action_type": "DECLARE_NEED", "payload": {"description": {}, "kind": "feature"}},
        {
            "action_type": "CREATE_FORECAST",
            "payload": {"operation": "scale_values", "factor": "not-a-number"},
        },
    ],
)
def test_malformed_payloads_cost_a_round_not_infrastructure(tmp_path, bundle, action):
    store, runner = run_actions(tmp_path, bundle, [action, submit(bundle), final()])
    assert runner.state()["rounds_used"] == 3
    assert store.trace("test")[0]["response"]["status"] == "agent_error"


def test_invalid_information_attempts_not_dropped_from_score(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    bad = call("feature.window_summary")
    bad["payload"]["calls"][0].pop("purpose")
    store, _ = run_actions(tmp_path, bundle, [bad, call("feature.window_summary"), submit(bundle), final()])
    result = score_run(store.export("test"), truth)
    assert result["process_items"]["need"] == 50


def test_repeated_illegal_call_cannot_change_id_to_look_recovered(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    a = call("feature.window_summary", {"series_ref": "episode.future.target"})
    b = deepcopy(a)
    b["payload"]["calls"][0]["call_id"] = "new-id"
    store, _ = run_actions(tmp_path, bundle, [a, b, submit(bundle), final()])
    assert score_run(store.export("test"), truth)["trust_items"]["recovery"] == 50


def test_evidence_source_history_application_future(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    scale = bundle.revision_scale.S
    evidence = {
        "action_type": "REGISTER_EVIDENCE",
        "payload": {
            "source_observation_ids": ["observation_1"],
            "scope": {"timeline": "history", "start": 0, "end": 48, "variables": ["load"]},
            "application_scope": [{"timeline": "scored", "start": 0, "end": 12}],
            "relation_type": "historical_analogy",
            "statement": "Historical context is relevant, not a causal proof",
        },
    }
    revision = {
        "action_type": "CREATE_FORECAST",
        "payload": {
            "operation": "add_offset",
            "parent_id": "forecast_1",
            "offset": scale * 0.1,
            "declared_effect": {
                "baseline_id": "forecast_1",
                "ranges": [{"start": 0, "end": 12}],
                "direction": "increase",
                "magnitude": "small",
            },
            "evidence_refs": ["evidence_1"],
        },
    }
    actions = [call("feature.window_summary"), evidence, submit(bundle), revision, final("forecast_2")]
    store, runner = run_actions(tmp_path, bundle, actions)
    assert not runner.state()["errors"]
    score = score_run(store.export("test"), truth)
    assert score["process_items"]["evidence"] == 100
    assert score["trust_items"]["effect"] == 100


def test_mutating_readonly_batch_is_rejected_atomically(tmp_path, bundle):
    bad = call("feature.window_summary")
    bad["payload"]["calls"].append(
        {"call_id": "b", "tool_id": "sandbox.submit_full_forecast", "arguments": submit(bundle)["payload"]}
    )
    store, runner = run_actions(tmp_path, bundle, [bad, submit(bundle), final()])
    assert not runner.state()["needs"] and not runner.state()["observations"]
    assert store.trace("test")[0]["response"]["status"] == "agent_error"


def test_noop_does_not_increase_revision_denominator(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    actions = [submit(bundle), submit(bundle), final("forecast_2")]
    store, _ = run_actions(tmp_path, bundle, actions)
    score = score_run(store.export("test"), truth)
    assert score["revision_count"] == 0
    assert score["process_items"]["evidence"] is None


def test_sandbox_blend_and_compare(tmp_path, bundle):
    blend = call(
        "sandbox.blend_forecasts",
        {
            "source_ids": ["forecast_1", "forecast_2"],
            "weights": [0.5, 0.5],
            "declared_effect": {
                "baseline_id": "forecast_2",
                "ranges": [{"start": 0, "end": 12}],
                "direction": "decrease",
                "magnitude": "large",
            },
        },
    )
    actions = [
        submit(bundle, [20.0] * 12),
        submit(bundle, [24.0] * 12),
        blend,
        call("sandbox.compare_versions", {"forecast_ids": ["forecast_1", "forecast_3"]}),
        final("forecast_3"),
    ]
    store, runner = run_actions(tmp_path, bundle, actions)
    assert runner.state()["forecasts"]["forecast_3"]["scored_values"] == [22.0] * 12
    assert not runner.state()["validations"]


def test_null_complete_response_preserved_on_resume(tmp_path, bundle):
    store = Store(tmp_path)
    actions = [submit(bundle), submit(bundle), final()]
    runner = Runner(bundle, store, ScriptedProvider(actions), "r")
    aid = store.prepare("r", 1, {})
    store.receive(aid, None, complete=True)
    runner.run()
    assert store.trace("r")[0]["response"]["status"] == "agent_error"
    assert len(store.export("r", True)["provider_archive"]) == 3


def test_tampered_trace_rejected(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, _ = run_actions(tmp_path, bundle, [submit(bundle), final()])
    exported = store.export("test")
    exported["trace"][0]["response"]["status"] = "fabricated"
    with pytest.raises(ValueError, match="integrity"):
        score_run(exported, truth)


def test_dataset_macro_not_point_pool(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, _ = run_actions(tmp_path, bundle, [submit(bundle), final()])
    score = score_run(store.export("test"), truth)
    second = deepcopy(score)
    second.update(dataset_id="different", episode_id="another", TotalScore=0)
    result = aggregate([score, second], bootstrap_replicates=50)
    assert result["overall"]["TotalScore"] == score["TotalScore"] / 2
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([score, score])

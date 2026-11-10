import pytest
from conftest import call, final, run_actions, submit

from castbench.common import AgentError, InfrastructureError
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


def test_direct_prediction_without_models(tmp_path, bundle):
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), final()])
    state = runner.state()
    assert state["final"]["forecast_id"] == "forecast_1"
    assert state["rounds_used"] == 2 and not state["observations"]
    assert store.replay("test") == state


@pytest.mark.parametrize("boundary", ["after_provider_accept", "before_commit", "after_commit"])
def test_crash_resume_exactly_once(tmp_path, bundle, boundary):
    actions = [submit(bundle), final()]
    provider = ScriptedProvider(actions)
    store = Store(tmp_path)
    fired = []

    def fault(point):
        if point == boundary and not fired:
            fired.append(True)
            raise InfrastructureError("injected crash")

    runner = Runner(bundle, store, provider, "r", fault_hook=fault)
    with pytest.raises(InfrastructureError):
        runner.run()
    resumed = Runner(bundle, store, provider, "r")
    resumed.run()
    assert resumed.state()["rounds_used"] == 2
    assert len(resumed.state()["forecasts"]) == 1
    assert len(store.export("r", True)["provider_archive"]) == 2


def test_first_complete_invalid_response_wins(tmp_path, bundle):
    store = Store(tmp_path)
    provider = ScriptedProvider([submit(bundle), submit(bundle), final()])
    runner = Runner(bundle, store, provider, "r")
    aid = store.prepare("r", 1, {})
    bad = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "I refuse"}}]}
    store.receive(aid, bad)
    duplicate = store.prepare("r", 1, {})
    good = provider.respond(provider.request(runner.messages(runner.state()), {}))
    store.receive(duplicate, good)
    runner.run()
    assert store.trace("r")[0]["response"]["status"] == "agent_error"
    assert runner.state()["rounds_used"] == 3


def test_incomplete_response_not_accepted(tmp_path):
    store = Store(tmp_path)
    aid = store.prepare("r", 1, {})
    store.receive(aid, {"partial": "tool_call"}, complete=False)
    assert store.accepted("r", 1) is None


def test_cross_run_artifact_denied(tmp_path, bundle):
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), final()])
    ref = runner.state()["forecasts"]["forecast_1"]["artifact_ref"]
    assert store.read_blob("test", ref)
    with pytest.raises(AgentError, match="authorized"):
        store.read_blob("other", ref)


def test_need_observed_not_automatically_resolved(tmp_path, bundle):
    store, runner = run_actions(tmp_path, bundle, [call("feature.window_summary"), submit(bundle), final()])
    assert runner.state()["needs"]["need_1"]["status"] == "observed"


def test_need_terminal_update_is_atomic(tmp_path, bundle):
    actions = [call("feature.window_summary"), submit(bundle), submit(bundle), final()]
    actions[1]["need_updates"] = [{"need_id": "need_1", "status": "resolved", "resolution_note": "enough"}]
    actions[2]["need_updates"] = [
        {"need_id": "need_1", "status": "abandoned", "resolution_note": "illegal reopening"}
    ]
    store, runner = run_actions(tmp_path, bundle, actions)
    assert runner.state()["needs"]["need_1"]["status"] == "resolved"
    assert len(runner.state()["forecasts"]) == 1
    assert store.trace("test")[2]["response"]["status"] == "agent_error"


def test_readonly_batch_deterministic_order(tmp_path, bundle):
    first = call("feature.window_summary")
    first["payload"]["calls"].append(
        {"call_id": "two", "tool_id": "context.calendar_profile", "purpose": "Calendar", "arguments": {}}
    )
    store, runner = run_actions(tmp_path, bundle, [first, submit(bundle), final()])
    results = store.trace("test")[0]["response"]["results"]
    assert [x["call_id"] for x in results] == ["one", "two"]
    assert runner.state()["rounds_used"] == 3


def test_budget_never_selects_model_or_best(tmp_path, bundle):
    bundle = bundle.model_copy(update={"budget": bundle.budget.model_copy(update={"max_rounds": 2})})
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), submit(bundle, [999.0] * 12)])
    assert runner.state()["final"] == {
        "forecast_id": "forecast_2",
        "forced_by_budget": True,
        "stop_reason": "round_budget_exhausted",
    }


def test_no_forecast_budget_finishes(tmp_path, bundle):
    bundle = bundle.model_copy(update={"budget": bundle.budget.model_copy(update={"max_rounds": 1})})
    _, runner = run_actions(tmp_path, bundle, [call("feature.window_summary")])
    assert runner.state()["final"]["forecast_id"] is None


def test_stale_version_costs_round(tmp_path, bundle):
    action = submit(bundle)
    action["state_version"] = 5
    store, runner = run_actions(tmp_path, bundle, [action, submit(bundle), final()])
    assert store.trace("test")[0]["response"]["code"] == "STATE_VERSION"
    assert runner.state()["rounds_used"] == 3


def test_changed_config_cannot_resume(tmp_path, bundle):
    store = Store(tmp_path)
    Runner(bundle, store, ScriptedProvider([submit(bundle), final()]), "r")
    with pytest.raises(InfrastructureError):
        Runner(bundle, store, ScriptedProvider([submit(bundle, [99.0] * 12), final()]), "r")


def test_writer_lock_prevents_second_writer(tmp_path):
    store = Store(tmp_path)
    with store.writer("same"):
        with pytest.raises(InfrastructureError):
            with store.writer("same"):
                pass


def test_unknown_private_state_rejected(tmp_path, bundle):
    actions = [{"action_type": "REQUEST_STATE", "payload": {"section": "truth"}}, submit(bundle), final()]
    store, runner = run_actions(tmp_path, bundle, actions)
    assert store.trace("test")[0]["response"]["code"] == "PERMISSION_DENIED"

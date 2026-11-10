import json
from copy import deepcopy

import pytest
from conftest import final, submit

from castbench.common import InfrastructureError
from castbench.evaluator import score_run
from castbench.providers import ProviderProfile, ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store

TRUNCATED = {
    "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "incomplete"}}]
}


class CapacityScript(ScriptedProvider):
    def __init__(self, actions, overflow=False, overflow_recovery=False):
        super().__init__(actions)
        self.profile = ProviderProfile(
            endpoint="http://local.test",
            model="test",
            capacity_recovery=True,
            context_window_tokens=49152,
            max_output_tokens=16384,
        )
        self.overflow, self.overflow_recovery = overflow, overflow_recovery

    def count_input_tokens(self, request):
        packet = json.loads(request["messages"][1]["content"])
        if packet["state"]["state_version"] and self.overflow:
            if self.overflow_recovery or not packet.get("capacity_index_view"):
                return 100000
        return 1000


def make(tmp_path, bundle, actions, **options):
    provider = CapacityScript(actions, **options)
    runner = Runner(bundle, Store(tmp_path / "store"), provider, "test", protocol_repair=True)
    return runner, provider


def test_output_warning_once_then_recover(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    runner, _ = make(tmp_path, bundle, [TRUNCATED, submit(bundle), final()])
    runner.run()
    state = runner.state()
    assert state["capacity_alarm"]["status"] == "recovered"
    assert state["errors"] == state["protocol_alarms"] == []
    score = score_run(runner.store.export("test", include_raw=True), truth)
    assert score["CapacityAlarmCount"] == 1 and score["completed"]
    assert score["CapacityAlarmPenaltyEquivalent"] == 0.05
    assert score["ProviderResponseCount"] == 3


def test_second_truncation_errors_episode_not_batch(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    runner, _ = make(tmp_path, bundle, [TRUNCATED, TRUNCATED])
    runner.run()
    assert len(runner.state()["errors"]) == 1
    score = score_run(runner.store.export("test"), truth)
    assert score["CapacityFailureCount"] == 1 and not score["completed"]
    assert score["TotalScore"] == 0


def test_input_warning_preserves_state_then_recover(tmp_path, bundle):
    runner, provider = make(tmp_path, bundle, [submit(bundle), None, final()], overflow=True)
    runner.step()
    forecasts = deepcopy(runner.state()["forecasts"])
    runner.step()
    assert runner.state()["capacity_alarm"]["kind"] == "input"
    assert runner.state()["forecasts"] == forecasts
    assert not runner.store.has_accepted("test", 2)
    messages = runner.messages(runner.state())
    assert any("capacity_warning" in m.get("content", "") for m in messages)
    runner.run()
    assert runner.state()["final"]["forecast_id"] == "forecast_1"
    assert runner.store.replay("test") == runner.state()


def test_input_warning_does_not_spend_the_remaining_agent_round(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    bundle.budget.max_rounds = 2
    runner, _ = make(tmp_path, bundle, [submit(bundle), None, final()], overflow=True)
    runner.run()
    score = score_run(runner.store.export("test"), truth)
    assert score["rounds"] == 2 and score["journal_transitions"] == 3
    assert not score["forced_by_budget"]
    assert score["completed"] and score["CapacityAlarmCount"] == 1


@pytest.mark.parametrize("first", ["input", "output"])
def test_capacity_allowance_shared(tmp_path, bundle, first):
    actions = [submit(bundle), None, TRUNCATED] if first == "input" else [TRUNCATED, submit(bundle)]
    runner, provider = make(tmp_path, bundle, actions, overflow=first == "input")
    runner.step()
    if first == "input":
        runner.step()
    else:
        runner.step()
        provider.overflow = provider.overflow_recovery = True
    runner.step()
    assert runner.state()["final"]["stop_reason"] == "capacity_exhausted"
    assert len(runner.state()["errors"]) == 1


def test_initial_unfit_task_remains_infrastructure(tmp_path, bundle):
    runner, provider = make(tmp_path, bundle, [])
    provider.count_input_tokens = lambda request: 100000
    with pytest.raises(InfrastructureError, match="Initial public task"):
        runner.step()
    assert runner.state()["state_version"] == 0


def test_pending_warning_gets_boundary_retry(tmp_path, bundle):
    bundle.budget.max_rounds = 1
    runner, _ = make(tmp_path, bundle, [TRUNCATED, submit(bundle)])
    runner.run()
    assert runner.state()["rounds_used"] == 2
    assert runner.state()["final"]["forecast_id"] == "forecast_1"


def test_accepted_truncation_crash_is_not_double_warning(tmp_path, bundle):
    runner, provider = make(tmp_path, bundle, [TRUNCATED, submit(bundle), final()])

    def crash(point):
        if point == "after_provider_accept":
            raise RuntimeError("crash")

    runner.fault_hook = crash
    with pytest.raises(RuntimeError):
        runner.step()
    runner.fault_hook = lambda point: None
    runner.run()
    assert runner.state()["capacity_alarm"]["turn"] == 1
    assert runner.state()["errors"] == []


def test_prior_protocol_errors_and_terminal_capacity_are_scoreable(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    bad = submit(bundle, forecast_id="invalid")
    runner, _ = make(tmp_path, bundle, [bad, bad, TRUNCATED, TRUNCATED])
    runner.run()
    score = score_run(runner.store.export("test"), truth)
    assert score["ErrorActionCount"] == 2
    assert score["ProtocolAlarmCount"] == score["CapacityAlarmCount"] == 1
    assert not score["completed"]


def test_input_repeats_after_delivered_warning_without_another_api_call(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    runner, provider = make(tmp_path, bundle, [TRUNCATED, submit(bundle)])
    runner.step()
    runner.step()
    provider.overflow = provider.overflow_recovery = True
    runner.run()
    score = score_run(runner.store.export("test"), truth)
    assert score["ProviderResponseCount"] == 2
    assert score["EnvironmentCapacityEventCount"] == 1
    assert score["CapacityFailureCount"] == 1 and not score["completed"]
    assert runner.state()["forecasts"]["forecast_1"]["scored_values"]


def test_cannot_deliver_warning_is_not_agent_error(tmp_path, bundle):
    runner, _ = make(tmp_path, bundle, [submit(bundle)], overflow=True, overflow_recovery=True)
    runner.step()
    runner.step()
    with pytest.raises(InfrastructureError, match="has not received"):
        runner.step()
    assert runner.state()["status"] == "running" and runner.state()["errors"] == []


def test_warning_does_not_erase_pending_protocol_intent(tmp_path, bundle):
    bad = submit(bundle, forecast_id="invalid")
    runner, _ = make(tmp_path, bundle, [bad, TRUNCATED, submit(bundle), final()])
    runner.run()
    state = runner.state()
    assert state["protocol_alarms"][0]["status"] == "recovered"
    assert state["capacity_alarm"]["status"] == "recovered"
    assert state["errors"] == []


def test_one_warning_is_not_reset_after_recovery(tmp_path, bundle):
    runner, _ = make(tmp_path, bundle, [TRUNCATED, submit(bundle), TRUNCATED])
    runner.run()
    assert runner.state()["capacity_alarm"]["status"] == "failed"
    assert runner.state()["capacity_alarm"]["turn"] == 1
    assert runner.state()["capacity_alarm"]["recovery_turn"] == 2
    assert len(runner.state()["errors"]) == 1


def test_failed_capacity_episode_does_not_stop_other_jobs(tmp_path, bundle):
    from castbench.scheduler import run_jobs

    first, _ = make(tmp_path / "one", bundle, [TRUNCATED, TRUNCATED])
    second, _ = make(tmp_path / "two", bundle, [submit(bundle), final()])
    completed = []
    run_jobs([first, second], lambda runner: runner.run(), 2, on_complete=completed.append)
    assert first.state()["status"] == second.state()["status"] == "finished"
    assert first.state()["final"]["forecast_id"] is None
    assert second.state()["final"]["forecast_id"] is not None

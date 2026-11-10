import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from conftest import final, submit

from castbench.common import InfrastructureError
from castbench.context_memory import build_messages
from castbench.providers import SYSTEM_PROMPT, OpenAICompatibleProvider, ProviderProfile, ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


def test_action_schemas_are_discoverable(tmp_path, bundle):
    runner = Runner(bundle, Store(tmp_path / "store"), ScriptedProvider([]), "schemas")
    schema = runner._read_state({"section": "schema", "ref": "CALL_TOOL"}, runner.state())
    assert schema["required"] == ["calls"]
    assert "purpose" in schema["$defs"]["ToolCall"]["properties"]
    envelope = runner._read_state({"section": "schema", "ref": "ACTION"}, runner.state())
    assert "action_type" in envelope["required"]


def test_constraint_feedback_is_not_mislabelled_as_syntax(tmp_path, bundle):
    from conftest import call
    from test_tools import FakeModelService

    from castbench.contracts import Constraint

    b = bundle.model_copy(deep=True)
    b.constraints = [Constraint(constraint_id="lower", kind="minimum", value=30.0, hard=True)]
    task = b.episode.task
    actions = [
        call(
            "model.chronos_forecast",
            {
                "frequency": task.frequency,
                "forecast_gap": task.forecast_gap,
                "horizon_length": task.horizon_length,
            },
        ),
        {
            "action_type": "CREATE_FORECAST",
            "payload": {"operation": "adopt_candidate", "candidate_id": "forecast_1"},
        },
    ]
    store = Store(tmp_path / "hard")
    runner = Runner(b, store, ScriptedProvider(actions), "hard", model_service=FakeModelService())
    runner.step()
    runner.step()
    response = store.trace("hard")[1]["response"]
    assert response["code"] == "HARD_CONSTRAINT"
    assert "numerical forecast violates" in response["recovery"]["hint"]
    assert "hard_constraints" in response["recovery"]
    assert "CHECK_FORECAST" not in response["recovery"]["hint"]
    assert response["recovery"]["hard_constraints"][0]["value"] == 30.0
    assert runner.state()["active_forecast_id"] is None
    assert runner.state()["forecasts"]["forecast_1"]["scored_values"] == [20.0] * task.horizon_length
    assert "plus optional rationale" not in SYSTEM_PROMPT


def test_uniform_point_views_keep_full_model_artifacts(tmp_path, bundle):
    from conftest import call, run_actions
    from test_tools import FakeModelService

    class RichModel(FakeModelService):
        def predict(self, name, episode):
            result = super().predict(name, episode)
            result["structured_result"]["distribution"] = [0.5] * (1 if name == "chronos" else 10000)
            return result

    task = bundle.episode.task
    actions = [
        call(
            f"model.{name}_forecast",
            {
                "frequency": task.frequency,
                "forecast_gap": task.forecast_gap,
                "horizon_length": task.horizon_length,
            },
        )
        for name in ("chronos", "timesfm", "sundial", "moirai")
    ]
    actions += [
        {
            "action_type": "CREATE_FORECAST",
            "payload": {"operation": "adopt_candidate", "candidate_id": "forecast_4"},
        },
        final("forecast_5"),
    ]
    store, runner = run_actions(tmp_path, bundle, actions, model_service=RichModel())
    for row in store.trace("test")[:4]:
        result = row["response"]["results"][0]
        assert result["presentation"] == "uniform_point_forecast_view"
        assert set(result["result"]["structured_result"]) == {"gap_values", "scored_values"}
        original = store.read_blob("test", result["artifact_ref"])
        assert "distribution" in original["structured_result"]
        forecast = store.read_blob("test", result["forecast_artifact_ref"])
        assert forecast["forecast_id"] == result["forecast_id"]
        assert forecast["kind"] == "model_candidate"
        assert result["artifact_ref"] != result["forecast_artifact_ref"]
        assert (
            runner.state()["observations"][result["observation_id"]]["artifact_ref"] == result["artifact_ref"]
        )
        assert (
            result["result"]["structured_result"]["scored_values"]
            == original["structured_result"]["scored_values"]
        )
    assert runner.state()["final"]["forecast_id"] == "forecast_5"


def test_large_history_has_authorized_original_artifact(tmp_path, bundle):
    from test_provider_history_wire import native

    action = {
        "action_type": "DECLARE_NEED",
        "state_version": 0,
        "payload": {"kind": "context", "description": "Remember this evidence question"},
    }
    raw = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "recorded " * 5000,
                    "tool_calls": native(json.dumps(action)),
                },
            }
        ]
    }
    provider = ScriptedProvider(
        [raw, {"action_type": "REQUEST_STATE", "payload": {"section": "history", "offset": 0, "limit": 1}}]
    )
    store = Store(tmp_path / "store")
    runner = Runner(bundle, store, provider, "archive")
    runner.step()
    runner.step()
    item = store.trace("archive")[1]["response"]["data"]["turns"][0]
    original = store.read_blob("archive", item["artifact_ref"])
    assert original["provider_response"] == raw
    assert runner.state()["errors"] == []


def test_model_order_balanced_over_24_days(tmp_path, bundle):
    from collections import Counter

    import pandas as pd

    store = Store(tmp_path / "orders")
    counts = [Counter() for _ in range(4)]
    first = []
    for day in range(24):
        b = bundle.model_copy(deep=True)

        def shift(s):
            return (pd.Timestamp(s) + pd.Timedelta(days=day)).isoformat()

        e = b.episode
        e.forecast_origin = shift(e.forecast_origin)
        e.history.timestamps = [shift(t) for t in e.history.timestamps]
        e.forecast_timeline.gap_timestamps = [shift(t) for t in e.forecast_timeline.gap_timestamps]
        e.forecast_timeline.scored_timestamps = [shift(t) for t in e.forecast_timeline.scored_timestamps]
        runner = Runner(b, store, ScriptedProvider([]), f"order-{day}")
        order = runner.registry.model_order
        first.append(order[0])
        for index, model in enumerate(order):
            counts[index][model] += 1
    assert all(set(c.values()) == {6} for c in counts)
    assert len(set(first[:4])) == 4


class CountedScript(ScriptedProvider):
    def count_input_tokens(self, request):
        return len(json.dumps(request, ensure_ascii=False)) // 4


def history_runner(tmp_path, bundle):
    actions = [
        {
            "action_type": "DECLARE_NEED",
            "payload": {
                "description": "Distinct evidence question " + str(i) + ": " + "long reasoning " * 100,
                "kind": "context",
            },
        }
        for i in range(7)
    ] + [submit(bundle), final()]
    provider = CountedScript(actions)
    store = Store(tmp_path / "store")
    runner = Runner(bundle, store, provider, "memory")
    for _ in range(7):
        runner.step()
    return runner, provider


def test_no_fixed_four_round_amnesia_and_replay(tmp_path, bundle):
    runner, provider = history_runner(tmp_path, bundle)
    messages = runner.messages(runner.state())
    assert len([m for m in messages if m["role"] == "assistant"]) == 7
    original = runner.store.export("memory", include_raw=True)
    provider.profile = SimpleNamespace(
        context_window_tokens=13000,
        max_output_tokens=1000,
        context_safety_tokens=100,
        context_trigger_fraction=0.8,
        token_count_mode="test_estimate",
    )
    compact, report = build_messages(
        provider, runner.store, "memory", runner.packet(runner.state()), SYSTEM_PROMPT
    )
    assert report["compacted_rounds"] > 0
    assert report["input_token_count"] <= report["input_hard_limit"]
    memory = json.loads(compact[1]["content"])["conversation_memory"]
    assert memory["turns"][0]["turn"] == 1
    assert memory["turns"][0]["read_original"]["offset"] == 0
    assert runner.store.export("memory", include_raw=True) == original
    assert build_messages(provider, runner.store, "memory", runner.packet(runner.state()), SYSTEM_PROMPT) == (
        compact,
        report,
    )
    calls = [c["id"] for m in compact for c in m.get("tool_calls", [])]
    replies = [m["tool_call_id"] for m in compact if m["role"] == "tool"]
    assert calls == replies


def test_context_failure_is_not_agent_error(tmp_path, bundle):
    runner, provider = history_runner(tmp_path, bundle)
    before = deepcopy(runner.state())
    provider.profile = SimpleNamespace(
        context_window_tokens=1000,
        max_output_tokens=500,
        context_safety_tokens=100,
        context_trigger_fraction=0.8,
        token_count_mode="test_estimate",
    )
    with pytest.raises(InfrastructureError, match="no Agent round charged"):
        runner.messages(runner.state())
    assert runner.state() == before


def test_history_retrieval_includes_original_response_and_reasoning(tmp_path, bundle):
    runner, _ = history_runner(tmp_path, bundle)
    result = runner._read_state({"section": "history", "offset": 0, "limit": 1}, runner.state())
    assert result["total_turns"] == 7
    assert result["turns"][0]["provider_response"] == runner.store.accepted("memory", 1)
    assert result["turns"][0]["action"]["payload"]["description"].endswith("long reasoning ")
    assert result["next_offset"] == 1


def test_context_configuration_rejects_impossible_reserve():
    with pytest.raises(ValueError, match="leave room"):
        ProviderProfile(endpoint="https://example.test", model="test", context_window_tokens=8192)


def test_llama_counts_template_with_tools_and_reasoning(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "test-only")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if request.url.path == "/apply-template":
            assert body["tools"][0]["function"]["name"] == "environment_action"
            assert body["messages"][0]["reasoning_content"] == "actual returned text"
            return httpx.Response(200, json={"prompt": "rendered template"})
        assert body["parse_special"] and body["add_special"]
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            ProviderProfile(
                endpoint="http://local.test/v1/chat/completions", model="test", token_count_mode="llama_cpp"
            ),
            client,
        )
        request = provider.request(
            [{"role": "assistant", "content": None, "reasoning_content": "actual returned text"}], {}
        )
        assert provider.count_input_tokens(request) == 3
        assert len(seen) == 2


def test_counterbalanced_model_cards_do_not_change_tools(tmp_path, bundle):
    runner = Runner(bundle, Store(tmp_path / "store"), ScriptedProvider([]), "order")
    cards = runner.packet(runner.state())["tool_cards"]
    assert [c["tool_id"] for c in cards[:4]] == list(runner.registry.model_order)
    assert set(c["tool_id"] for c in cards) == set(runner.registry.manifests)
    assert len(set(runner.registry.model_order)) == 4


def test_failed_action_is_not_silently_repaired(tmp_path, bundle):
    bad = submit(bundle, forecast_id="made_up_output_field")
    store = Store(tmp_path / "store")
    runner = Runner(bundle, store, ScriptedProvider([bad, submit(bundle), final()]), "errors")
    runner.step()
    assert runner.state()["forecasts"] == {}
    row = store.trace("errors")[0]
    assert not row["event"]["valid"] and row["response"]["status"] == "agent_error"
    assert "recovery" in row["response"]
    runner.run()
    assert len(runner.state()["errors"]) == 1
    assert runner.state()["final"]["forecast_id"] == "forecast_1"

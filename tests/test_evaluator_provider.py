import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from conftest import call, final, run_actions, submit

from castbench.common import AgentError, InfrastructureError, TransientInfrastructureError
from castbench.evaluator import metrics, score_run
from castbench.providers import OpenAICompatibleProvider, ProviderProfile, decode


def test_perfect_direct_golden(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    actions = [
        submit(bundle, truth["values"]),
        {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_1"}},
        final(),
    ]
    store, _ = run_actions(tmp_path, bundle, actions)
    score = score_run(store.export("test"), truth)
    assert score["ForecastScore"] == 100
    assert score["TotalScore"] == 100
    assert score["ProcessCoverage"] == pytest.approx(0.45)
    assert score["TrustCoverage"] == pytest.approx(0.85)


def test_no_check_golden(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store, _ = run_actions(tmp_path, bundle, [submit(bundle, truth["values"]), final()])
    score = score_run(store.export("test"), truth)
    assert score["ProcessScore"] == pytest.approx(100 * (0.20 + 0.15) / (0.20 + 0.10 + 0.15))


def test_robust_v12_scores_with_canonical_interaction_diagnostics(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    actions = [
        call("feature.window_summary"),
        submit(bundle, truth["values"]),
        final(),
    ]
    store, _ = run_actions(tmp_path, bundle, actions)
    score = score_run(store.export("test"), truth, score_profile="robust-v1.2")
    diagnostics = score["interaction_diagnostics"]
    assert score["schema_version"] == "castbench.score/1.2"
    assert diagnostics["successful_substantive_tool_calls"] == 1
    assert diagnostics["repeated_successful_substantive_tool_calls"] == 0
    assert score["EfficiencyScore"] == 100


def test_check_alias_equal_scores(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    actions = [
        submit(bundle, truth["values"]),
        {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_1"}},
        final(),
    ]
    a, _ = run_actions(tmp_path / "a", bundle, actions)
    actions[1] = call("sandbox.validate_forecast", {"forecast_id": "forecast_1"})
    b, _ = run_actions(tmp_path / "b", bundle, actions)
    sa, sb = score_run(a.export("test"), truth), score_run(b.export("test"), truth)
    assert sa["TotalScore"] == sb["TotalScore"]
    assert sa["process_items"] == sb["process_items"]


def test_private_truth_noninterference(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    actions = [call("context.future_shift_profile", {"variables": ["temperature"]}), submit(bundle), final()]
    a, _ = run_actions(tmp_path / "a", bundle, actions)
    altered = deepcopy(truth)
    altered["values"] = [x + 100 for x in truth["values"]]
    b, _ = run_actions(tmp_path / "b", bundle, actions)
    assert a.trace("test") == b.trace("test")
    assert score_run(a.export("test"), truth)["MSE"] != score_run(b.export("test"), altered)["MSE"]


def test_budget_no_forecast_scores_zero(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    bundle = bundle.model_copy(update={"budget": bundle.budget.model_copy(update={"max_rounds": 1})})
    store, _ = run_actions(tmp_path, bundle, [call("feature.window_summary")])
    assert score_run(store.export("test"), truth)["TotalScore"] == 0


@pytest.mark.parametrize("count,expected", [(0, 100), (1, 50), (2, 0)])
def test_permission_penalty_cannot_be_diluted(tmp_path, bundle_and_truth, count, expected):
    bundle, truth = bundle_and_truth
    actions = [{"action_type": "REQUEST_STATE", "payload": {"section": "truth"}}] * count + [
        submit(bundle),
        final(),
    ]
    store, _ = run_actions(tmp_path, bundle, actions)
    assert score_run(store.export("test"), truth)["trust_items"]["permissions"] == expected


def test_mse_mae_smape_hand_calculation():
    result = metrics([0, 2], [0, 4], {"scale_abs": 2, "scale_sq": 4})
    assert result["MSE"] == 2 and result["MAE"] == 1
    assert result["sMAPE"] == pytest.approx(1 / 3)
    assert result["MASE"] == 0.5
    assert result["RMSSE"] == pytest.approx(np.sqrt(0.5))


def test_unit_scale_invariant():
    a = metrics([1, 2, 3], [2, 3, 5], {"scale_abs": 2, "scale_sq": 3})
    b = metrics([1000, 2000, 3000], [2000, 3000, 5000], {"scale_abs": 2000, "scale_sq": 3_000_000})
    assert a["ForecastScore"] == pytest.approx(b["ForecastScore"])
    assert a["sMAPE"] == pytest.approx(b["sMAPE"])


def test_extreme_metrics_do_not_look_perfect():
    result = metrics([1e308], [-1e308], {"scale_abs": 1, "scale_sq": 1})
    assert result["sMAPE"] == 2 and result["ForecastScore"] < 1e-300
    assert result["MSE"] is None and "MSE_scientific" in result


def test_reasoning_provider_replay_is_not_fabricated():
    profile = ProviderProfile(endpoint="https://example.test/v1/chat/completions", model="test")
    provider = OpenAICompatibleProvider(profile)
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "actual returned provider field",
                    "tool_calls": [],
                }
            }
        ]
    }
    assert provider.assistant_message(raw)["reasoning_content"] == "actual returned provider field"
    raw["choices"][0]["message"].pop("reasoning_content")
    assert "reasoning_content" not in provider.assistant_message(raw)


def test_http_body_and_response(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake-test-only")
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}]
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderProfile(endpoint="https://example.test/v1/chat/completions", model="frozen-test"),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = provider.request([{"role": "user", "content": "test"}], {"type": "object"})
    raw = provider.respond(request)
    assert captured[0]["stream"] is False
    assert captured[0]["tool_choice"]["function"]["name"] == "environment_action"
    with pytest.raises(AgentError):
        decode(raw)


def test_streaming_provider_reassembles_reasoning_and_tool_call(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake")
    events = [
        {
            "id": "chat-1",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning_content": "check "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "done",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "environment_",
                                    "arguments": '{"action_',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "action",
                                    "arguments": 'type":"FINALIZE"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 12},
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    )
    provider = OpenAICompatibleProvider(
        ProviderProfile(endpoint="https://example.test", model="m", stream_responses=True),
        client,
    )
    request = provider.request([{"role": "user", "content": "{}"}], {"type": "object"})
    raw = provider.respond(request)
    message = raw["choices"][0]["message"]
    assert request["stream"] is True
    assert message["reasoning_content"] == "check done"
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"] == {
        "name": "environment_action",
        "arguments": '{"action_type":"FINALIZE"}',
    }
    assert raw["choices"][0]["finish_reason"] == "tool_calls"
    assert raw["usage"] == {"total_tokens": 12}


def test_streaming_http_524_is_transient_and_keeps_body_readable(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(524, text="gateway timeout"))
    )
    provider = OpenAICompatibleProvider(
        ProviderProfile(endpoint="https://example.test", model="m", stream_responses=True),
        client,
    )
    request = provider.request([{"role": "user", "content": "{}"}], {"type": "object"})
    with pytest.raises(TransientInfrastructureError, match="524.*gateway timeout"):
        provider.respond(request)


def test_http_400_is_not_agent_error(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(400, text="invalid API config"))
    )
    p = OpenAICompatibleProvider(ProviderProfile(endpoint="https://example.test", model="m"), client)
    with pytest.raises(InfrastructureError):
        p.respond({})


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503, 524, 599])
def test_retryable_http_status_is_classified_as_transient(monkeypatch, status):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text="temporary upstream failure")
        )
    )
    provider = OpenAICompatibleProvider(
        ProviderProfile(endpoint="https://example.test", model="m"), client
    )
    with pytest.raises(TransientInfrastructureError, match=str(status)):
        provider.respond({})


def test_nonretryable_http_status_remains_fatal_infrastructure(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="invalid credential"))
    )
    provider = OpenAICompatibleProvider(
        ProviderProfile(endpoint="https://example.test", model="m"), client
    )
    with pytest.raises(InfrastructureError) as error:
        provider.respond({})
    assert not isinstance(error.value, TransientInfrastructureError)


def test_reject_provider_semantic_override():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(
            ProviderProfile(endpoint="https://example.test", model="m", extra_body={"tools": []})
        )

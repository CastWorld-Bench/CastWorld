import json
import sys
from copy import deepcopy

import httpx
import pytest
from conftest import final, submit

from castbench.common import InfrastructureError
from castbench.models import ModelProfile, ModelService
from castbench.providers import OpenAICompatibleProvider, ProviderProfile, ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


def model_profile():
    return ModelProfile(
        name="chronos",
        engine="chronos_original",
        python_executable="test-only-python",
        checkpoint="test-only-checkpoint",
        revision="test-frozen-revision",
        checkpoint_digest="sha256:test-only",
        max_context=512,
        max_horizon=128,
        dependency_versions={"chronos-forecasting": "test-only"},
    )


def test_provider_snapshots_caller_profile():
    profile = ProviderProfile(
        endpoint="https://example.test/v1/chat/completions",
        model="frozen",
        extra_body={"thinking": {"type": "enabled"}},
    )
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        provider = OpenAICompatibleProvider(profile, client)
        frozen = deepcopy(provider.manifest())
        profile.model = "different"
        profile.extra_body["thinking"]["type"] = "disabled"
        profile.assistant_passthrough_fields.clear()
        assert provider.manifest() == frozen
        assert provider.request([], {})["model"] == "frozen"


def test_model_service_snapshots_caller_profile(tmp_path):
    profile = model_profile()
    service = ModelService([profile], Store(tmp_path / "store"))
    frozen = deepcopy(service.profiles["chronos"].model_dump())
    profile.seed = 17
    profile.dependency_versions.clear()
    assert service.profiles["chronos"].model_dump() == frozen


def test_npu_profile_requires_explicit_pool(tmp_path):
    raw = model_profile().model_dump()
    raw["device"] = "npu:0"
    raw["python_executable"] = sys.executable
    raw["checkpoint"] = str(tmp_path)
    profile = ModelProfile.model_validate(raw)
    service = ModelService([profile], Store(tmp_path))
    service.profiles = {name: profile for name in ("chronos", "timesfm", "sundial", "moirai")}
    with pytest.raises(InfrastructureError, match="device pool"):
        service.preflight()


@pytest.mark.parametrize("devices", [[-1], [1, 1], [True]])
def test_npu_pool_rejects_invalid_ids(tmp_path, devices):
    with pytest.raises(ValueError, match="npu_devices"):
        ModelService([], Store(tmp_path), npu_devices=devices)


def test_profile_rejects_unresolved_or_unknown_device():
    raw = model_profile().model_dump()
    for device in ("auto", "npu:4", "cuda"):
        raw["device"] = device
        with pytest.raises(ValueError, match="accelerator"):
            ModelProfile.model_validate(raw)


def test_moirai_short_history_uses_explicit_patch():
    raw = model_profile().model_dump()
    raw.update(name="moirai", engine="moirai_1")
    profile = ModelProfile.model_validate(raw)
    assert profile.moirai_patch_size == 16
    raw.update(moirai_patch_size=8)
    assert ModelProfile.model_validate(raw).moirai_patch_size == 8
    raw.update(name="chronos", engine="chronos_original")
    with pytest.raises(ValueError, match="only to MOIRAI"):
        ModelProfile.model_validate(raw)


@pytest.mark.parametrize("component", ["provider", "models"])
def test_between_round_config_drift_pauses_without_charging(tmp_path, bundle, component):
    provider = ScriptedProvider([submit(bundle), final()])
    store = Store(tmp_path / "store")
    models = ModelService([model_profile()], store)
    runner = Runner(bundle, store, provider, "test", model_service=models)
    runner.step()
    frozen_actions = deepcopy(provider.actions)
    if component == "provider":
        provider.actions[1]["payload"]["stop_reason"] = "changed after start"
    else:
        models.profiles["chronos"].seed = 17
    with pytest.raises(InfrastructureError, match="configuration changed"):
        runner.step()
    assert runner.state()["rounds_used"] == 1
    assert not store.has_accepted("test", 2)
    provider.actions = frozen_actions
    models.profiles["chronos"].seed = 0
    runner.run()
    assert runner.state()["status"] == "finished"
    assert runner.state()["rounds_used"] == 2


def test_http_multiround_retry_reasoning_and_archive(tmp_path, bundle, monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "fake-test-only")
    monkeypatch.setattr("castbench.runner.time.sleep", lambda _: None)
    actions = [
        submit(bundle),
        {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_1"}},
        final(),
    ]
    requests, returned = [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["authorization"] == "Bearer fake-test-only"
        assert body["model"] == "frozen-test"
        assert body["max_completion_tokens"] == 8192
        packet = json.loads(body["messages"][1]["content"])
        turn = packet["state"]["state_version"]
        if len(requests) == 1:
            raise httpx.ReadTimeout("test transport interruption", request=request)
        history = body["messages"][2:]
        if turn:
            cursor = json.loads(history[-1]["content"])["current_action_cursor"]
            assert cursor["state_version"] == turn
            assert history[-1]["role"] == "user"
            history = history[:-1]
        assert len(history) == turn * 2
        for index in range(turn):
            assistant, tool = history[index * 2 : index * 2 + 2]
            assert assistant["reasoning_content"] == f"test returned explanation {index}"
            assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == f"call_{index}"
            assert tool["role"] == "tool"
            assert json.loads(tool["content"])["state_version"] == index + 1
        action = {**actions[turn], "state_version": turn}
        raw = {
            "id": f"completion_{turn}",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": f"test returned explanation {turn}",
                        "tool_calls": [
                            {
                                "id": f"call_{turn}",
                                "type": "function",
                                "function": {"name": "environment_action", "arguments": json.dumps(action)},
                            }
                        ],
                    },
                }
            ],
        }
        returned.append(raw)
        return httpx.Response(200, json=raw)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            ProviderProfile(
                endpoint="https://example.test/v1/chat/completions",
                model="frozen-test",
                context_window_tokens=131072,
                output_token_parameter="max_completion_tokens",
            ),
            client,
        )
        store = Store(tmp_path / "store")
        runner = Runner(bundle, store, provider, "test")
        runner.run()
        assert len(requests) == 4
        assert requests[0] == requests[1]
        assert runner.state()["rounds_used"] == 3
        assert runner.state()["status"] == "finished"
        for turn, raw in enumerate(returned, 1):
            assert store.accepted("test", turn) == raw
        Runner(bundle, store, provider, "test").run()
        assert len(requests) == 4

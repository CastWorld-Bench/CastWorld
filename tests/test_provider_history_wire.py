import json
from copy import deepcopy

import httpx
import pytest
from conftest import final, submit

from castbench.providers import (
    OpenAICompatibleProvider,
    ProviderProfile,
    ScriptedProvider,
    replayable_tool_calls,
)
from castbench.runner import Runner
from castbench.storage import Store


def native(arguments):
    return [
        {"id": "call", "type": "function", "function": {"name": "environment_action", "arguments": arguments}}
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"value":-0.',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":1e999}',
        '{"value":' + "9" * 400 + "}",
        '{"x":1,"x":2}',
        "[]",
        "null",
        '"string"',
    ],
)
def test_damaged_arguments_cannot_be_replayed(arguments):
    assert not replayable_tool_calls(native(arguments))


def test_syntactically_valid_agent_mistake_remains_in_history():
    calls = native('{"unknown_but_valid_json":true}')
    original = deepcopy(calls)
    assert replayable_tool_calls(calls)
    assert calls == original
    assert replayable_tool_calls([])
    assert not replayable_tool_calls(calls + calls)


@pytest.mark.parametrize("finish_reason", ["length", "tool_calls"])
def test_truncated_native_response_is_archived_but_does_not_poison_next_http_request(
    tmp_path, bundle, monkeypatch, finish_reason
):
    monkeypatch.setenv("CASTBENCH_API_KEY", "test-only")
    damaged = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Actual returned explanation for test",
                    "tool_calls": native('{"action_type":"CREATE_FORECAST","payload":{"value":-0.'),
                },
            }
        ]
    }
    scripted = ScriptedProvider([damaged, submit(bundle), final()])
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        for message in body["messages"]:
            for call in message.get("tool_calls", []):
                # Reproduce the server-side history parser that raised HTTP 500 in the real run.
                assert isinstance(json.loads(call["function"]["arguments"]), dict)
        return httpx.Response(200, json=scripted.respond(body))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            ProviderProfile(
                endpoint="https://example.test/v1/chat/completions",
                model="test",
                context_window_tokens=131072,
            ),
            client,
        )
        store = Store(tmp_path / "store")
        runner = Runner(bundle, store, provider, "wire-test")
        runner.run()
        assert runner.state()["rounds_used"] == 3
        assert runner.state()["final"]["forecast_id"] == "forecast_1"
        assert len(runner.state()["errors"]) == 1
        assert store.accepted("wire-test", 1) == damaged
        assert len(requests) == 3
        history = requests[1]["messages"][2:]
        assert len(history) == 2 and history[0]["role"] == "user"
        assert json.loads(history[-1]["content"])["current_action_cursor"]["state_version"] == 1
        feedback = json.loads(history[0]["content"])
        assert feedback["raw_response_archived"] is True
        assert feedback["prior_action_response"]["status"] == "agent_error"
        assert store.replay("wire-test") == runner.state()

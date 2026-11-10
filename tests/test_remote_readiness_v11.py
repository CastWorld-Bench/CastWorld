import json
from copy import deepcopy

import httpx
import pytest
from conftest import call, final, run_actions, submit

from castbench.common import InfrastructureError, json_text
from castbench.context_memory import replay_group
from castbench.domain_knowledge import knowledge_for
from castbench.evaluator import aggregate, metrics, score_run
from castbench.providers import OpenAICompatibleProvider, ProviderProfile
from castbench.tools import _lexical


def profile(**kwargs):
    return ProviderProfile(endpoint="https://example.test/v1/chat/completions", model="test", **kwargs)


def test_auto_choice_and_omitted_parallel_field():
    p = OpenAICompatibleProvider(profile(tool_choice_mode="auto", parallel_tool_calls=None))
    request = p.request([], {"type": "object"})
    assert request["tool_choice"] == "auto" and "parallel_tool_calls" not in request
    assert request["tools"][0]["function"]["name"] == "environment_action"
    p.client.close()


@pytest.mark.parametrize("reason", ["aborted", "insufficient_system_resource"])
def test_provider_interruption_is_infrastructure(monkeypatch, reason):
    monkeypatch.setenv("CASTBENCH_API_KEY", "test")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"choices": [{"finish_reason": reason, "message": {"role": "assistant"}}]}
            )
        )
    ) as client:
        p = OpenAICompatibleProvider(profile(), client)
        with pytest.raises(InfrastructureError, match="interrupted"):
            p.respond({})


def test_missing_thinking_field_not_fabricated(monkeypatch):
    monkeypatch.setenv("CASTBENCH_API_KEY", "test")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "text"}}]}
            )
        )
    ) as client:
        p = OpenAICompatibleProvider(profile(require_reasoning_content=True), client)
        with pytest.raises(InfrastructureError, match="omitted"):
            p.respond({})


def test_response_error_redacts_credential(monkeypatch):
    key = "example-sensitive-key"
    monkeypatch.setenv("CASTBENCH_API_KEY", key)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400, text=key))) as client:
        p = OpenAICompatibleProvider(profile(), client)
        with pytest.raises(InfrastructureError) as error:
            p.respond({})
        assert key not in str(error.value)


def test_native_rejected_history_keeps_reasoning_and_feedback():
    p = OpenAICompatibleProvider(profile(replay_rejected_messages=True))
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "returned reason",
        "tool_calls": [
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "environment_action", "arguments": '{"invalid_field":true}'},
            }
        ],
    }
    row = {"event": {"valid": False}, "response": {"status": "protocol_warning"}}
    group = replay_group(p, {"choices": [{"message": msg}]}, row)
    assert group[0] == msg and group[1]["role"] == "tool"
    assert group[1]["tool_call_id"] == "call1"
    assert json.loads(group[1]["content"]) == row["response"]
    msg["tool_calls"] = None
    group = replay_group(p, {"choices": [{"message": msg}]}, row)
    assert group[0]["reasoning_content"] == "returned reason"
    assert group[1]["role"] == "user"
    msg["tool_calls"] = [
        {
            "id": "bad",
            "type": "function",
            "function": {"name": "environment_action", "arguments": '{"truncated":'},
        }
    ]
    group = replay_group(p, {"choices": [{"message": msg}]}, row)
    assert len(group) == 1 and group[0]["role"] == "user"
    p.client.close()


@pytest.mark.parametrize(
    "domain,query,entry_topic",
    [
        ("bike", "commuting", "通勤与休闲"),
        ("bike", "weather temperature rain", "天气舒适性"),
        ("windy", "icing", "结冰"),
        ("windy", "wind speed power", "风资源"),
        ("gef", "heating cooling", "取暖与制冷"),
        ("sunny", "solar radiation irradiance", "辐射与形态"),
    ],
)
def test_pack_aliases_are_relevant_not_global(domain, query, entry_topic):
    entries, _ = knowledge_for(domain, [])
    entry = next(x for x in entries if x["topic"].endswith("/" + entry_topic))
    assert _lexical(query, " ".join(entry["search_aliases"])) > 0
    assert _lexical("unrelatedxyz", " ".join(entry["search_aliases"])) == 0
    assert entry["language"] == "zh-CN"


def test_zero_metrics_robust_and_legacy_are_explicit():
    scale = {"scale_abs": 1, "scale_sq": 1}
    old = metrics([0, 0], [1e-8, -1e-8], scale)
    new = metrics([0, 0], [1e-8, -1e-8], scale, "robust-v1.1")
    assert old["sMAPE"] == new["sMAPE"] == 2
    assert 83 < old["ForecastScore"] < 84
    assert new["ForecastScore"] > 99.999
    assert metrics([1e308], [-1e308], scale, "robust-v1.1")["ForecastScore"] < 1e-300
    with pytest.raises(ValueError):
        metrics([0], [0], scale, "unknown")


def test_coverage_and_completed_metrics(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    actions = [
        submit(bundle),
        {"action_type": "CHECK_FORECAST", "payload": {"forecast_id": "forecast_1"}},
        final(),
    ]
    store, _ = run_actions(tmp_path, bundle, actions)
    old = score_run(store.export("test"), truth)
    new = score_run(store.export("test"), truth, "robust-v1.1")
    assert old["process_items"]["stop"] == 100
    assert new["process_items"]["stop"] is None
    assert new["trust_items"]["provenance"] is None
    assert new["ProcessCoverage"] < old["ProcessCoverage"]
    assert not new["coverage_details"]["reference_integrity_evaluable"]
    failed = deepcopy(new)
    failed.update(episode_id="failed-test", completed=False, TotalScore=0, ForecastScore=0)
    for k in ("MAE", "MSE", "MASE", "RMSSE", "sMAPE"):
        failed[k] = None
    summary = aggregate([new, failed], bootstrap_replicates=0)["datasets"][new["dataset_id"]]
    assert summary["MAE"] is None
    assert summary["valid_only"]["metrics"]["MAE"] == new["MAE"]
    assert summary["valid_only"]["n_valid"] == 1
    assert summary["valid_only"]["n_planned"] == 2
    assert summary["completion_rate"] == 0.5
    old["episode_id"] = "legacy-other"
    with pytest.raises(ValueError, match="evaluator_profile"):
        aggregate([old, new])


def test_nested_artifact_pages_preserve_every_item_without_inline_explosion(tmp_path, bundle, monkeypatch):
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), final()])
    cases = [{"id": i, "context": {"wind": list(range(10000))}} for i in range(100)]
    monkeypatch.setattr(store, "read_blob", lambda run_id, ref: {"cases": cases})
    request = {"section": "artifact", "ref": "authorized", "path": ["cases"], "limit": 256}
    indices, offset = [], 0
    while offset < len(cases):
        page = runner._read_state({**request, "offset": offset}, runner.state())
        assert page["encoding"] == "indexed_items"
        assert len(json_text(page).encode("utf-8")) <= bundle.budget.max_inline_bytes
        indices.extend(item["index"] for item in page["items"])
        assert page["next_offset"] > offset
        offset = page["next_offset"]
        for item in page["items"]:
            assert item["deferred"]
            read = {**item["read"], "path": [*item["read"]["path"], "context", "wind"], "offset": 96, "limit": 96}
            actual = runner._read_state(read, runner.state())
            assert actual["values"] == list(range(96, 192))
    assert indices == list(range(100))


def test_large_flat_array_pages_make_progress_and_small_slices_unchanged(tmp_path, bundle, monkeypatch):
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), final()])
    data = ["x" * 100 for _ in range(256)]
    monkeypatch.setattr(store, "read_blob", lambda run_id, ref: data)
    request = {"section": "artifact", "ref": "authorized", "limit": 256}
    page = runner._read_state(request, runner.state())
    assert 0 < page["next_offset"] < 256
    assert len(json_text(page).encode("utf-8")) <= bundle.budget.max_inline_bytes
    assert all(item["value"] == "x" * 100 for item in page["items"])
    small = runner._read_state({**request, "offset": 252, "limit": 4}, runner.state())
    assert small["values"] == data[252:]
    assert small["next_offset"] == 256
    empty = runner._read_state({**request, "offset": 300}, runner.state())
    assert empty["values"] == [] and empty["next_offset"] == 256


def test_generated_nested_read_recipes_respect_configured_page_size(tmp_path, bundle, monkeypatch):
    bundle.budget.max_state_page = 4
    store, runner = run_actions(tmp_path, bundle, [submit(bundle), final()])
    monkeypatch.setattr(store, "read_blob", lambda run_id, ref: [{"values": list(range(10000))}])
    page = runner._read_state({"section": "artifact", "ref": "authorized", "limit": 4}, runner.state())
    recipe = page["items"][0]["read"]
    assert recipe["limit"] == 4
    child = runner._read_state(recipe, runner.state())["children"][0]["read"]
    assert child["limit"] == 4
    assert runner._read_state(child, runner.state())["values"] == list(range(4))


def test_evidence_scope_error_identifies_public_source_without_relaxing_check(tmp_path, bundle):
    bundle.budget.max_rounds = 2
    actions = [
        call("feature.window_summary"),
        {
            "action_type": "REGISTER_EVIDENCE",
            "payload": {
                "source_observation_ids": ["observation_1"],
                "scope": {"timeline": "scored"},
                "application_scope": [{"timeline": "scored", "start": 0, "end": 1}],
                "relation_type": "direct_context",
            },
        },
    ]
    store, runner = run_actions(tmp_path, bundle, actions)
    response = store.trace("test")[-1]["response"]
    assert "source observation_1 observed_scope=" in response["message"]
    assert "Register separate evidence entries" in response["message"]
    assert "application_scope" in response["message"]
    assert not runner.state()["evidence"]

import base64
import io
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import call, final, run_actions, submit
from PIL import Image

from castbench.common import InfrastructureError, load_json, write_json
from castbench.features import analyze
from castbench.models import ModelProfile, ModelService
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


def test_context_compaction_is_a_readonly_projection(tmp_path, bundle):
    actions = [submit(bundle), submit(bundle, [21.0] * 12), final("forecast_2")]
    store, runner = run_actions(tmp_path, bundle, actions)
    state = runner.state()
    before = deepcopy(state)
    packet = runner.packet(state)
    assert state == before
    assert "changed_indices" in state["forecasts"]["forecast_2"]["diff"]
    assert "changed_count" in packet["state"]["forecasts"]["forecast_2"]["diff"]
    packet["state"]["forecasts"]["forecast_2"]["diff"]["changed_count"] = 999
    assert state == before


def test_artifact_numeric_slice_can_recover_large_results(tmp_path, bundle):
    actions = [call("context.get_variable_slice", {"variables": ["temperature"]}), submit(bundle), final()]
    store, runner = run_actions(tmp_path, bundle, actions)
    obs = runner.state()["observations"]["observation_1"]
    payload = {
        "section": "artifact",
        "ref": obs["artifact_ref"],
        "path": ["structured_result", "variables", "temperature", "values"],
        "offset": 48,
        "limit": 14,
    }
    result = runner._read_state(payload, runner.state())
    assert result["values"] == bundle.variables[0].values[48:62]


def test_image_tool_is_nonblank():
    x = np.sin(np.arange(48))
    output = analyze("time_series_to_image", x, list(map(str, range(48))), {})
    image = Image.open(io.BytesIO(base64.b64decode(output["structured_result"]["image_png_base64"])))
    pixels = np.asarray(image.convert("RGB"))
    assert image.width >= 1000 and pixels.std() > 10
    assert output["structured_result"]["plot_data"]["values"] == x.tolist()


@pytest.mark.parametrize("bad", [False, True])
def test_model_worker_receives_target_only_and_bad_output_is_infra(tmp_path, bundle, monkeypatch, bad):
    profile = ModelProfile(
        name="chronos",
        engine="chronos_original",
        python_executable="fake",
        checkpoint="fake",
        revision="pinned",
        checkpoint_digest="sha256:fake",
        max_context=24,
        max_horizon=30,
        dependency_versions={"torch": "test"},
    )
    service = ModelService([profile], Store(tmp_path))
    requests = []

    def fake_run(command, **kwargs):
        request = load_json(command[-2])
        requests.append(request)
        write_json(command[-1], {"point_values": [1.0] * (1 if bad else 14)})
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("castbench.models.subprocess.run", fake_run)
    if bad:
        with pytest.raises(InfrastructureError, match="invalid output"):
            service.predict("chronos", bundle.episode)
    else:
        result = service.predict("chronos", bundle.episode)
        assert len(result["structured_result"]["gap_values"]) == 2
        assert len(result["structured_result"]["scored_values"]) == 12
        service.predict("chronos", bundle.episode)
        assert len(requests) == 1
    assert set(requests[0]) == {
        "profile",
        "target_values",
        "frequency",
        "start_timestamp",
        "prediction_length",
    }
    assert len(requests[0]["target_values"]) == 24


def test_batch_recovery_reuses_completed_subcalls(tmp_path, bundle, monkeypatch):
    import castbench.runner as runner_module

    original = runner_module.execute_readonly
    count = {"feature.window_summary": 0, "context.calendar_profile": 0}
    failed = []

    def injected(tool, *args, **kwargs):
        count[tool] += 1
        if tool == "context.calendar_profile" and not failed:
            failed.append(True)
            raise InfrastructureError("injected transient executor failure")
        return original(tool, *args, **kwargs)

    monkeypatch.setattr(runner_module, "execute_readonly", injected)
    batch = call("feature.window_summary")
    batch["payload"]["calls"].append(
        {"call_id": "two", "tool_id": "context.calendar_profile", "purpose": "Calendar", "arguments": {}}
    )
    store = Store(tmp_path)
    runner = Runner(bundle, store, ScriptedProvider([batch, submit(bundle), final()]), "r")
    with pytest.raises(InfrastructureError):
        runner.run()
    assert runner.state()["rounds_used"] == 0
    runner.run()
    assert count == {"feature.window_summary": 1, "context.calendar_profile": 2}
    assert runner.state()["rounds_used"] == 3


def test_independent_episode_scheduler(tmp_path, bundle):
    from castbench.scheduler import run_many

    store = Store(tmp_path)
    runners = [
        Runner(bundle, store, ScriptedProvider([submit(bundle), final()]), f"run-{i}") for i in range(3)
    ]
    result = run_many(runners, max_workers=3)
    assert [r["run_id"] for r in result] == ["run-0", "run-1", "run-2"]
    assert all(r["status"] == "finished" for r in result)
    assert all(store.get(f"run-{i}")["rounds_used"] == 2 for i in range(3))


def test_extension_output_contract_failure_is_infrastructure(tmp_path, bundle):
    from castbench.catalog import Registry
    from castbench.contracts import Contract

    registry = Registry()
    registry.register_extension(
        "domain.broken",
        Contract,
        lambda b, p: {"status": "invalid-status", "structured_result": {}},
        "Conformance failure injection",
        "sha256:test",
    )
    with pytest.raises(InfrastructureError, match="result contract"):
        run_actions(tmp_path, bundle, [call("domain.broken")], registry=registry)
    assert Store(tmp_path / "store").get("test")["rounds_used"] == 0


def test_export_includes_authorized_raw_tool_artifacts(tmp_path, bundle):
    store, runner = run_actions(
        tmp_path,
        bundle,
        [call("context.get_variable_slice", {"variables": ["temperature"]}), submit(bundle), final()],
    )
    exported = store.export("test")
    ref = runner.state()["observations"]["observation_1"]["artifact_ref"]
    assert (
        exported["artifacts"][ref]["structured_result"]["variables"]["temperature"]["values"]
        == bundle.variables[0].values
    )
    assert exported["operational_log"]


def test_irrelevant_numeric_operation_parameter_rejected():
    from pydantic import ValidationError

    from castbench.contracts import ForecastArgs

    with pytest.raises(ValidationError, match="does not apply"):
        ForecastArgs(operation="add_offset", parent_id="f1", offset=1.0, factor=2.0)


@pytest.mark.parametrize("corruption", ["missing_truth", "infinite_scale"])
def test_bad_evaluator_bundle_not_ranked_even_without_forecast(tmp_path, bundle_and_truth, corruption):
    from castbench.evaluator import score_run

    bundle, truth = bundle_and_truth
    bundle = bundle.model_copy(update={"budget": bundle.budget.model_copy(update={"max_rounds": 1})})
    store, _ = run_actions(tmp_path, bundle, [call("feature.window_summary")])
    private = deepcopy(truth)
    if corruption == "missing_truth":
        private["values"][0] = None
    else:
        private["normalization"]["scale_sq"] = float("inf")
    with pytest.raises(ValueError):
        score_run(store.export("test"), private)


def test_context_card_assembles_only_selected_observations(tmp_path, bundle):
    actions = [
        call("feature.window_summary"),
        call("context.future_shift_profile", {"variables": ["temperature"]}),
        call("context.build_context_card", {"refs": ["observation_1"]}),
        submit(bundle),
        final(),
    ]
    store, runner = run_actions(tmp_path, bundle, actions)
    ref = runner.state()["observations"]["observation_3"]["artifact_ref"]
    card = store.read_blob("test", ref)["structured_result"]
    assert card["target_state"][0]["facts"]["statistics"]["valid_count"] == 48
    assert card["future_covariate_state"] == []
    assert card["inferences"] == []

import pytest

from castbench.common import load_json
from castbench.contracts import PublicBundle
from castbench.data import build_packages, synthetic_fixture
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


@pytest.fixture
def bundle_and_truth(tmp_path):
    frame, config = synthetic_fixture()
    build_packages(frame, config, tmp_path / "public", tmp_path / "private")
    bundle = PublicBundle.model_validate(load_json(next((tmp_path / "public/dev").glob("*.json"))))
    truth = load_json(next((tmp_path / "private").glob("*.json")))
    return bundle, truth


@pytest.fixture
def bundle(bundle_and_truth):
    return bundle_and_truth[0]


def submit(bundle, values=None, **extra):
    return {
        "action_type": "CREATE_FORECAST",
        "payload": {
            "operation": "submit_full_forecast",
            "scored_values": values if values is not None else [20.0] * bundle.episode.task.horizon_length,
            "scored_timestamps": bundle.episode.forecast_timeline.scored_timestamps,
            "unit": bundle.episode.task.target_unit,
            **extra,
        },
    }


def final(fid="forecast_1"):
    return {"action_type": "FINALIZE", "payload": {"forecast_id": fid, "stop_reason": "done"}}


def call(tool, args=None, purpose="Inspect relevant facts"):
    c = {"call_id": "one", "tool_id": tool, "arguments": args or {}}
    if not tool.startswith("sandbox."):
        c["purpose"] = purpose
    return {"action_type": "CALL_TOOL", "payload": {"calls": [c]}}


def run_actions(tmp_path, bundle, actions, run_id="test", **kwargs):
    store = Store(tmp_path / "store")
    runner = Runner(bundle, store, ScriptedProvider(actions), run_id, **kwargs)
    runner.run()
    return store, runner

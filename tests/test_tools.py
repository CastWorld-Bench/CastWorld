import numpy as np
import pytest
from conftest import call, final, run_actions

from castbench.catalog import FEATURES, Registry
from castbench.common import AgentError, InfrastructureError
from castbench.features import analyze
from castbench.runner import initial_state
from castbench.storage import Store
from castbench.tools import ToolView, execute_readonly

PARAMS = {
    "multiscale_trend": {"scales": [12, 24]},
    "robust_decomposition": {"period": 12},
    "covariate_association": {"variables": ["temperature"]},
    "lead_lag_scan": {"variables": ["temperature"], "max_lag": 3},
}


@pytest.mark.parametrize("name", list(FEATURES))
def test_all_feature_algorithms(tmp_path, bundle, name):
    registry = Registry()
    tool = "feature." + name
    params = registry.validate(tool, PARAMS.get(name, {}))
    view = ToolView(bundle, initial_state(bundle.episode.episode_id), Store(tmp_path), "r")
    result = execute_readonly(tool, params, view, registry)
    assert result["status"] == "ok"
    assert result["structured_result"]
    assert result == execute_readonly(tool, params, view, registry)


@pytest.mark.parametrize(
    "name,params",
    [
        ("knowledge.list_topics", {}),
        ("knowledge.search_general", {"query": "seasonality"}),
        ("knowledge.search_domain", {"query": "load"}),
        ("knowledge.get_entry", {"entry_id": "general-season"}),
        ("knowledge.explain_variable", {"variable": "temperature"}),
        ("sample.search_target_shape", {}),
        ("sample.search_context_conditioned", {"variables": ["temperature"]}),
        ("sample.get_case", {"sample_id": "train_0"}),
        ("sample.compare_case", {"sample_ids": ["train_0", "train_12"]}),
        ("context.list_variables", {}),
        ("context.get_variable_slice", {"variables": ["temperature", "event"]}),
        ("context.get_event_timeline", {}),
        ("context.future_shift_profile", {"variables": ["temperature", "event"]}),
        ("context.calendar_profile", {}),
        ("context.build_context_card", {"refs": []}),
        ("registry.list_tools", {}),
        ("registry.describe_tool", {"tool_id": "feature.window_summary"}),
        ("registry.list_domain_extensions", {}),
    ],
)
def test_information_tools(tmp_path, bundle, name, params):
    registry = Registry()
    args = registry.validate(name, params)
    result = execute_readonly(
        name, args, ToolView(bundle, initial_state(bundle.episode.episode_id), Store(tmp_path), "r"), registry
    )
    assert result["status"] == "ok"


def test_46_core_catalog():
    r = Registry()
    assert len([k for k in r.manifests if not k.startswith("registry.")]) == 46
    assert len(r.cards()) == 49
    assert r.snapshot == Registry().snapshot


def test_explain_target_metadata_is_public_but_future_values_stay_private(tmp_path, bundle):
    r = Registry()
    view = ToolView(bundle, initial_state(bundle.episode.episode_id), Store(tmp_path), "r")
    result = execute_readonly(
        "knowledge.explain_variable", {"variable": bundle.episode.task.target_name}, view, r
    )
    assert result["structured_result"]["role"] == "prediction_target"
    assert "values" not in result["structured_result"]
    with pytest.raises(AgentError):
        view.variable(bundle.episode.task.target_name)


@pytest.mark.parametrize("kind", ["all_missing", "constant", "short", "internal_missing", "boundary_missing"])
def test_feature_edges(kind):
    x = np.sin(np.arange(48)).astype(float)
    if kind == "all_missing":
        x[:] = np.nan
    if kind == "constant":
        x[:] = 1
    if kind == "short":
        x = x[:3]
    if kind == "internal_missing":
        x[4] = np.nan
    if kind == "boundary_missing":
        x[0] = np.nan
    result = analyze(
        "autocorrelation_profile",
        x,
        list(map(str, range(len(x)))),
        {"max_lag": 5, "missing_policy": "reject"},
    )
    assert result["status"] == "not_applicable"


def test_sample_search_never_uses_future_target(tmp_path, bundle):
    r = Registry()
    p = r.validate("sample.search_target_shape", {})
    state = initial_state(bundle.episode.episode_id)
    a = execute_readonly("sample.search_target_shape", p, ToolView(bundle, state, Store(tmp_path), "r"), r)
    altered = bundle.model_copy(deep=True)
    for sample in altered.samples:
        sample.future_target = [99999.0] * len(sample.future_target)
    b = execute_readonly("sample.search_target_shape", p, ToolView(altered, state, Store(tmp_path), "r"), r)
    assert a == b


def test_future_target_request_cannot_use_series_ref(tmp_path, bundle):
    registry = Registry()
    p = registry.validate("feature.window_summary", {"series_ref": "episode.future.target"})
    with pytest.raises(AgentError, match="unauthorized"):
        execute_readonly(
            "feature.window_summary", p, ToolView(bundle, initial_state("e"), Store(tmp_path), "r"), registry
        )


def test_model_schema_rejects_covariates():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Registry().validate(
            "model.chronos_forecast",
            {"frequency": "h", "forecast_gap": 0, "horizon_length": 12, "covariates": [1, 2]},
        )


class FakeModelService:
    profiles = {}

    def predict(self, name, episode):
        return {
            "status": "ok",
            "structured_result": {
                "gap_values": [20.0] * episode.task.forecast_gap,
                "scored_values": [20.0] * episode.task.horizon_length,
            },
            "warnings": [],
        }


@pytest.mark.parametrize("model", ["chronos", "timesfm", "sundial", "moirai"])
def test_model_gateway_and_adoption(tmp_path, bundle, model):
    t = bundle.episode.task
    action = call(
        f"model.{model}_forecast",
        {"frequency": t.frequency, "forecast_gap": t.forecast_gap, "horizon_length": t.horizon_length},
    )
    actions = [action, call("sandbox.adopt_candidate", {"candidate_id": "forecast_1"}), final("forecast_2")]
    store, runner = run_actions(tmp_path, bundle, actions, model_service=FakeModelService())
    assert runner.state()["forecasts"]["forecast_1"]["kind"] == "model_candidate"
    assert runner.state()["final"]["forecast_id"] == "forecast_2"


def test_missing_model_pauses_without_charging_round(tmp_path, bundle):
    t = bundle.episode.task
    with pytest.raises(InfrastructureError):
        run_actions(
            tmp_path,
            bundle,
            [
                call(
                    "model.chronos_forecast",
                    {
                        "frequency": t.frequency,
                        "forecast_gap": t.forecast_gap,
                        "horizon_length": t.horizon_length,
                    },
                )
            ],
        )
    assert Store(tmp_path / "store").get("test")["rounds_used"] == 0

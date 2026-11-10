import pytest
from conftest import submit
from pydantic import ValidationError

from castbench.common import AgentError
from castbench.contracts import ForecastArgs
from castbench.forecast import create_forecast, diff_report
from castbench.runner import initial_state


def initial(bundle):
    s = initial_state(bundle.episode.episode_id)
    v = create_forecast(ForecastArgs.model_validate(submit(bundle)["payload"]), s, bundle)
    s["forecasts"]["f1"] = v
    s["active_forecast_id"] = "f1"
    return s


def args(operation, **extra):
    base = {
        "operation": operation,
        "parent_id": "f1",
        "declared_effect": {
            "baseline_id": "f1",
            "ranges": [{"start": 1, "end": 4}],
            "direction": "increase",
            "magnitude": "small",
        },
    }
    return ForecastArgs.model_validate({**base, **extra})


@pytest.mark.parametrize(
    "operation,extra,expected",
    [
        ("add_offset", {"offset": 2}, [22, 22, 22]),
        ("scale_values", {"factor": 2}, [40, 40, 40]),
        ("replace_segment", {"values": [1, 2, 3]}, [1, 2, 3]),
        ("adjust_segments", {"segments": [{"range": {"start": 1, "end": 4}, "offset": 3}]}, [23, 23, 23]),
    ],
)
def test_local_operations_preserve_outside(bundle, operation, extra, expected):
    state = initial(bundle)
    v = create_forecast(args(operation, range={"start": 1, "end": 4}, **extra), state, bundle)
    assert v["scored_values"][1:4] == expected
    assert v["scored_values"][:1] + v["scored_values"][4:] == [20] * 9
    assert state["forecasts"]["f1"]["scored_values"] == [20] * 12


@pytest.mark.parametrize(
    "case", ["overlap", "weights", "wrong_baseline", "length", "no_declaration", "shift_policy", "bad_range"]
)
def test_bad_revision_rejected(bundle, case):
    s = initial(bundle)
    s["forecasts"]["f2"] = {**s["forecasts"]["f1"], "scored_values": [25] * 12}
    p = args("add_offset", offset=1).model_dump()
    if case == "overlap":
        p.update(
            operation="adjust_segments",
            segments=[
                {"range": {"start": 1, "end": 4}, "offset": 1},
                {"range": {"start": 2, "end": 5}, "offset": 1},
            ],
        )
    if case == "weights":
        p.update(operation="blend_forecasts", source_ids=["f1", "f2"], weights=[0.3, 0.3])
    if case == "wrong_baseline":
        p["declared_effect"]["baseline_id"] = "f2"
    if case == "length":
        p["offset"] = [1]
    if case == "no_declaration":
        p["declared_effect"] = None
    if case == "shift_policy":
        p.update(operation="shift_time", steps=1)
    if case == "bad_range":
        p["range"] = {"start": 1, "end": 99}
    with pytest.raises((AgentError, ValidationError)):
        create_forecast(ForecastArgs.model_validate(p), s, bundle)


def test_shift_has_no_wrap(bundle):
    s = initial(bundle)
    s["forecasts"]["f1"]["scored_values"] = list(range(12))
    v = create_forecast(args("shift_time", steps=2, fill_policy="keep_parent"), s, bundle)
    assert v["scored_values"] == [0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_smoothing_only_delta_and_both_parents(bundle):
    s = initial(bundle)
    s["forecasts"]["f2"] = {**s["forecasts"]["f1"], "scored_values": [20, 20, 26] + [20] * 9}
    v = create_forecast(args("smooth_revision_delta", proposed_id="f2", window=3), s, bundle)
    assert v["scored_values"][:5] == [20, 22, 22, 22, 20]
    assert v["parents"] == ["f1", "f2"]


def test_noop_and_threshold_boundary(bundle):
    scale = bundle.revision_scale
    assert diff_report([1], [1], scale)["direction"] == "unchanged"
    assert diff_report([0], [0.25 * scale.S], scale)["magnitude"] == "moderate"
    assert diff_report([0], [scale.S], scale)["magnitude"] == "large"


def test_full_rewrite_still_evaluation_revision(bundle):
    s = initial(bundle)
    v = create_forecast(ForecastArgs.model_validate(submit(bundle, [21] * 12)["payload"]), s, bundle)
    assert v["parents"] == [] and v["baseline_id"] == "f1" and v["evaluation_revision"]

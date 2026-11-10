from copy import deepcopy

import pytest
from pydantic import ValidationError

from castbench.contracts import PublicBundle
from castbench.data import build_packages, synthetic_fixture
from castbench.evaluator import normalization


@pytest.mark.parametrize(
    "mutation",
    [
        "future_truth",
        "wrong_length",
        "duplicate",
        "wrong_origin",
        "naive",
        "infinite",
        "unknown_unit",
        "derived",
        "wrong_scale",
        "cross_cutoff",
    ],
)
def test_invalid_bundle_rejected(bundle, mutation):
    obj = bundle.model_dump()
    e = obj["episode"]
    if mutation == "future_truth":
        obj["future_truth"] = [1.0]
    if mutation == "wrong_length":
        e["history"]["target_values"].pop()
    if mutation == "duplicate":
        e["history"]["timestamps"][1] = e["history"]["timestamps"][0]
    if mutation == "wrong_origin":
        e["forecast_origin"] = e["history"]["timestamps"][0]
    if mutation == "naive":
        e["history"]["timestamps"][0] = e["history"]["timestamps"][0].split("+")[0]
    if mutation == "infinite":
        e["history"]["target_values"][0] = float("inf")
    if mutation == "unknown_unit":
        e["task"]["target_unit"] = ""
    if mutation == "derived":
        obj["variables"][0]["target_derived"] = True
    if mutation == "wrong_scale":
        obj["revision_scale"]["target_unit"] = "different"
    if mutation == "cross_cutoff":
        obj["samples"][0]["future_timestamps"][-1] = e["forecast_timeline"]["scored_timestamps"][-1]
    with pytest.raises((ValidationError, ValueError)):
        PublicBundle.model_validate(obj)


def test_missing_history_is_explicit(bundle):
    obj = bundle.model_dump()
    obj["episode"]["history"]["target_values"][3] = None
    assert PublicBundle.model_validate(obj).episode.history.target_values[3] is None


def test_normalization_does_not_bridge_missing_timestamps():
    result = normalization(
        ["2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z", "2024-01-01T03:00:00Z"],
        [1, 100, 102],
        "h",
        1,
        1,
        min_pairs=1,
    )
    assert result["pairs"] == 1 and result["scale_abs"] == 2


def test_constant_scale_and_unit_conversion():
    ts = [f"2024-01-01T0{i}:00:00Z" for i in range(5)]
    a = normalization(ts, [2] * 5, "h", 2, 0.01)
    b = normalization(ts, [2000] * 5, "h", 2, 10)
    assert a["method"] == "fallback"
    assert b["scale_abs"] == 1000 * a["scale_abs"]
    assert b["scale_sq"] == 1_000_000 * a["scale_sq"]


@pytest.mark.parametrize("mutation", ["overlap", "source_gap", "private_nested", "derived"])
def test_builder_preflight(tmp_path, mutation):
    frame, config = synthetic_fixture()
    public, private = tmp_path / "pub", tmp_path / "priv"
    if mutation == "overlap":
        config["episode_origins"].append(deepcopy(config["episode_origins"][0]))
    if mutation == "source_gap":
        frame = frame.drop(5)
    if mutation == "private_nested":
        private = public / "private"
    if mutation == "derived":
        config["variables"][0]["target_derived"] = True
    with pytest.raises(ValueError):
        build_packages(frame, config, public, private)
    assert not public.exists()

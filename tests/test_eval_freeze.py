import hashlib
import importlib.util
from pathlib import Path

import pytest

from castbench.common import load_json, write_json


def freezer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_joint_evaluation.py"
    spec = importlib.util.spec_from_file_location("freeze_joint_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.freeze


def inputs(tmp_path, count=50):
    project = tmp_path / "project"
    packs = tmp_path / "packs"
    for directory in ("src", "scripts"):
        write_json(project / directory / "marker.json", {"kind": directory})
    for name in ("pyproject.toml", "README.md"):
        (project / name).write_text("fixture", encoding="utf-8")
    for domain in ("bike", "gef", "sunny", "windy"):
        write_json(
            packs / "public" / domain / "manifest.json",
            {"episodes": [{"episode_id": f"{domain}.{i}"} for i in range(count)]},
        )
        write_json(packs / "evaluator_private" / domain / "marker.json", {"private": True})
    model = tmp_path / "models.json"
    provider = tmp_path / "provider.json"
    write_json(model, [{"checkpoint": "F:/weights"}])
    write_json(provider, {"api_key_env": "KEY_ONLY_NOT_SECRET"})
    return project, packs, model, provider


def test_freeze_exact200_and_source_independence(tmp_path):
    args = inputs(tmp_path)
    out = tmp_path / "frozen_eval"
    freezer()(*args, out)
    plan = load_json(out / "evaluation_plan.json")
    assert plan["samples_planned"] == 200
    assert plan["sample_concurrency"] == 2
    assert plan["model_worker_concurrency"] == 1
    for relative, expected in plan["snapshot_hashes"].items():
        assert hashlib.sha256((out / "frozen" / relative).read_bytes()).hexdigest() == expected
    write_json(args[0] / "src" / "marker.json", {"changed": True})
    assert load_json(out / "frozen" / "src" / "marker.json") == {"kind": "src"}
    with pytest.raises(ValueError, match="NEW output"):
        freezer()(*args, out)


def test_freeze_rejects_incomplete_set_before_writing(tmp_path):
    args = inputs(tmp_path, 49)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="exactly 50"):
        freezer()(*args, out)
    assert not out.exists()


def test_freeze_explicit_concurrency(tmp_path):
    args = inputs(tmp_path)
    out = tmp_path / "serial"
    freezer()(
        *args,
        out,
        sample_concurrency=500,
        model_worker_concurrency=7,
        npu_devices=[1, 2, 3, 4, 5, 6, 7],
    )
    plan = load_json(out / "evaluation_plan.json")
    assert plan["sample_concurrency"] == 500
    assert plan["model_worker_concurrency"] == 7
    assert plan["npu_devices"] == [1, 2, 3, 4, 5, 6, 7]


def test_freeze_bike_windy_exact100(tmp_path):
    args = inputs(tmp_path)
    out = tmp_path / "two_domains"
    freezer()(*args, out, domains=["bike", "windy"])
    plan = load_json(out / "evaluation_plan.json")
    assert plan["samples_planned"] == 100
    from castbench.protocol_repair import POLICY

    assert plan["protocol"] == POLICY["version"]
    assert set(plan["selected_episodes"]) == {"bike", "windy"}
    for part in ("public", "evaluator_private"):
        assert {p.name for p in (out / "frozen/packs" / part).iterdir()} == {"bike", "windy"}


@pytest.mark.parametrize("score_profile", ["robust-v1.1", "robust-v1.2"])
def test_freeze_small_readiness_set_is_explicit(tmp_path, score_profile):
    args = inputs(tmp_path)
    out = tmp_path / "four"
    freezer()(*args, out, domains=["bike", "windy"], count=2, score_profile=score_profile)
    plan = load_json(out / "evaluation_plan.json")
    assert plan["samples_planned"] == 4
    assert plan["selected_episodes"] == {"bike": ["bike.0", "bike.1"], "windy": ["windy.0", "windy.1"]}
    assert plan["count_per_domain"] == 2
    assert plan["score_profile"] == score_profile


def test_capacity_continuation_preserves_small_selection_and_score_profile(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from prepare_capacity_continuation import prepare

    args = inputs(tmp_path)
    source, target = tmp_path / "original", tmp_path / "continued"
    freezer()(*args, source, domains=["windy"], count=1, score_profile="robust-v1.1")
    prepare(source, target, args[0], args[3])
    before, after = load_json(source / "evaluation_plan.json"), load_json(target / "evaluation_plan.json")
    assert after["selected_episodes"] == before["selected_episodes"]
    assert after["samples_planned"] == 1
    assert after["score_profile"] == "robust-v1.1"


@pytest.mark.parametrize("domains", [[], ["bike", "bike"], ["invalid"]])
def test_freeze_invalid_domains(tmp_path, domains):
    args = inputs(tmp_path)
    out = tmp_path / "bad_domains"
    with pytest.raises(ValueError, match="domains"):
        freezer()(*args, out, domains=domains)
    assert not out.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_freeze_rejects_bad_concurrency_before_writing(tmp_path, value):
    args = inputs(tmp_path)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="positive integer"):
        freezer()(*args, out, sample_concurrency=value)
    assert not out.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_freeze_rejects_bad_model_worker_concurrency_before_writing(tmp_path, value):
    args = inputs(tmp_path)
    out = tmp_path / "bad_model_workers"
    with pytest.raises(ValueError, match="positive integer"):
        freezer()(*args, out, model_worker_concurrency=value)
    assert not out.exists()


@pytest.mark.parametrize("value", [[-1], [1, 1], [True]])
def test_freeze_rejects_bad_npu_devices_before_writing(tmp_path, value):
    args = inputs(tmp_path)
    out = tmp_path / "bad_npu_devices"
    with pytest.raises(ValueError, match="npu_devices"):
        freezer()(*args, out, npu_devices=value)
    assert not out.exists()

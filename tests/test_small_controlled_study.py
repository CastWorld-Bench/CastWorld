import json

import pytest

from castbench.catalog import Registry
from scripts.run_small_controlled_study import filter_registry, load_panel


def test_load_explicit_panel(tmp_path):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"episodes": ["a.test.1", "b.test.2"]}), encoding="utf-8")
    assert load_panel(path) == ["a.test.1", "b.test.2"]


@pytest.mark.parametrize("value", [[], ["duplicate.test.1", "duplicate.test.1"], [3]])
def test_reject_invalid_panel(tmp_path, value):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_panel(path)


def test_model_sandbox_ablation_really_removes_information_tools():
    registry = filter_registry(Registry(), "model_sandbox")
    assert all(x.split(".", 1)[0] in {"model", "sandbox", "registry"} for x in registry.manifests)
    assert not any(x["tool_id"].startswith("feature.") for x in registry.cards())
    assert filter_registry(Registry(), "full").manifests.keys() >= registry.manifests.keys()

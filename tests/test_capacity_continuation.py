from copy import deepcopy

import pytest
from conftest import final, submit

from castbench.common import InfrastructureError, digest, write_json
from castbench.continuation import continuation_summary, preserved_score
from castbench.evaluator import aggregate, score_run
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.storage import Store


def configs():
    old = {
        "provider": {
            "adapter": "chat_completions/1.0",
            "profile": {
                "model": "same",
                "context_window_tokens": 32768,
                "max_output_tokens": 8192,
                "timeout_seconds": 600,
            },
        },
        "budget": 30,
    }
    new = deepcopy(old)
    new["provider"]["profile"].update(
        context_window_tokens=49152, max_output_tokens=16384, timeout_seconds=1200, capacity_recovery=True
    )
    return old, new


def test_explicit_migration_keeps_state_and_audit(tmp_path):
    store = Store(tmp_path)
    old, new = configs()
    old["revision_scale"] = {"magnitude_edges": [0.25, 1.0]}
    new["revision_scale"] = {"magnitude_edges": (0.25, 1.0)}
    state = {"status": "running", "state_version": 0}
    store.start("e", old, state)
    store.migrate_capacity_config("e", digest(old), new)
    assert store.get("e") == state and digest(store.config("e")) == digest(new)
    store.migrate_capacity_config("e", digest(old), new)
    log = store.export("e")["operational_log"]
    assert len(log) == 1 and log[0]["old_config"] == old


@pytest.mark.parametrize(
    "field,value", [("model", "other"), ("temperature", 0.9), ("context_window_tokens", 16384)]
)
def test_migration_rejects_noncapacity_or_lower_limit(tmp_path, field, value):
    store = Store(tmp_path)
    old, new = configs()
    new["provider"]["profile"][field] = value
    store.start("e", old, {"status": "running", "state_version": 0})
    with pytest.raises(InfrastructureError):
        store.migrate_capacity_config("e", digest(old), new)
    assert store.config("e") == old


def test_cannot_migrate_completed_or_pending_response(tmp_path):
    store = Store(tmp_path)
    old, new = configs()
    store.start("finished", old, {"status": "finished", "state_version": 0})
    with pytest.raises(InfrastructureError):
        store.migrate_capacity_config("finished", digest(old), new)
    store.start("pending", old, {"status": "running", "state_version": 0})
    aid = store.prepare("pending", 1, {})
    store.receive(aid, {})
    with pytest.raises(InfrastructureError, match="accepted response"):
        store.migrate_capacity_config("pending", digest(old), new)


def test_preserved_scores_unchanged_and_mixed_cohorts_explicit(tmp_path, bundle_and_truth):
    bundle, truth = bundle_and_truth
    store = Store(tmp_path / "store")
    runner = Runner(bundle, store, ScriptedProvider([submit(bundle), final()]), "test")
    runner.run()
    trace = store.export("test")
    score = score_run(trace, truth)
    eid = score["episode_id"]
    write_json(tmp_path / eid / "score.json", score)
    write_json(tmp_path / eid / "trace.json", trace)
    manifest = {"preserved_scores": {eid: digest(score)}}
    assert preserved_score(tmp_path, eid, manifest)
    other = {**score, "episode_id": "other", "provider_profile_hash": "other"}
    with pytest.raises(ValueError, match="Cannot mix"):
        aggregate([score, other])
    result = continuation_summary([score, other])
    assert len(result["frozen_profile_cohorts"]) == 2
    assert result["episodes"] == 2
    write_json(tmp_path / eid / "score.json", other)
    with pytest.raises(InfrastructureError):
        preserved_score(tmp_path, eid, manifest)

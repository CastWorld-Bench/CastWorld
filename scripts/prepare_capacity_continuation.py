"""Clone a stopped journal consistently and preserve completed scores; never overwrite the source."""

import argparse
import hashlib
import shutil
import sqlite3
from pathlib import Path

from freeze_joint_evaluation import freeze

from castbench.common import digest, load_json, write_json
from castbench.storage import Store


def prepare(source, output, project, provider):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and continuation must be separate sibling directories")
    old_plan = load_json(source / "evaluation_plan.json")
    for name, expected in old_plan["snapshot_hashes"].items():
        if hashlib.sha256((source / "frozen" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original snapshot changed: {name}")
    source_store = Store(source / "results/state")
    with (
        Store(source / "launch_lock").writer("frozen_evaluation"),
        source_store.writer("__joint_evaluation_coordinator__"),
    ):
        freeze(
            project,
            source / "frozen/packs",
            source / "frozen/models.json",
            provider,
            output,
            sample_concurrency=old_plan["sample_concurrency"],
            model_worker_concurrency=old_plan.get("model_worker_concurrency", 1),
            npu_devices=old_plan.get("npu_devices", []),
            domains=list(old_plan["selected_episodes"]),
            count=old_plan.get("count_per_domain", 50),
            score_profile=old_plan.get("score_profile", "legacy-v1"),
        )
        shutil.copytree(
            source / "results", output / "results", ignore=shutil.ignore_patterns("journal.sqlite*", "*.lock")
        )
        with source_store.connection() as src:
            with sqlite3.connect(output / "results/state/journal.sqlite") as dst:
                src.backup(dst)
        preserved, unfinished = {}, {}
        expected_ids = {eid for ids in old_plan["selected_episodes"].values() for eid in ids}
        with source_store.connection() as db:
            import json

            for eid, config, state in db.execute("SELECT run_id,config,state FROM runs"):
                if eid not in expected_ids:
                    raise ValueError("Unexpected Episode in the source journal")
                state = json.loads(state)
                score_path = source / "results" / eid / "score.json"
                if state["status"] == "finished":
                    if not score_path.exists():
                        raise ValueError(f"Finished Episode missing exported score: {eid}")
                    preserved[eid] = digest(load_json(score_path))
                else:
                    if score_path.exists():
                        raise ValueError("Unfinished Episode has a score")
                    unfinished[eid] = digest(json.loads(config))
        for name in ("execution_profile.json", "timing.json"):
            file = output / "results" / name
            if file.exists():
                file.rename(file.with_name("previous_" + name))
        manifest = {
            "source_run": str(source),
            "source_plan_hash": digest(old_plan),
            "preserved_scores": preserved,
            "unfinished_configs": unfinished,
            "notice": "Completed results are not regenerated; unfinished histories are retained. New capacity settings are explicitly audited at the continuation boundary.",
        }
        write_json(output / "continuation.json", manifest)
        plan = load_json(output / "evaluation_plan.json")
        plan["continuation"] = digest(manifest)
        write_json(output / "evaluation_plan.json", plan)
        return {"output": str(output), "preserved": len(preserved), "resumed": len(unfinished)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--provider", type=Path, required=True)
    a = p.parse_args()
    print(prepare(a.source, a.out, Path(__file__).resolve().parents[1], a.provider))

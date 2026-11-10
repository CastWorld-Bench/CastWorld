"""Audited capacity-profile continuation, not an implicit mixed-profile leaderboard."""

from collections import defaultdict
from pathlib import Path

from .common import InfrastructureError, digest, load_json
from .evaluator import aggregate, stable_mean


def preserved_score(directory, episode_id, manifest):
    expected = manifest.get("preserved_scores", {}).get(episode_id)
    if expected is None:
        return False
    score = load_json(Path(directory) / episode_id / "score.json")
    trace = load_json(Path(directory) / episode_id / "trace.json")
    if (
        digest(score) != expected
        or score["episode_id"] != episode_id
        or score["trace_hash"] != digest(trace["trace"])
    ):
        raise InfrastructureError("Preserved result or trace differs from the continuation manifest")
    if trace["state"]["status"] != "finished":
        raise InfrastructureError("Cannot skip an unfinished preserved Episode")
    return True


def continuation_summary(scores):
    cohorts = defaultdict(list)
    seen = set()
    for score in scores:
        key = score["episode_id"]
        if key in seen:
            raise ValueError("Duplicate Episode in continuation")
        seen.add(key)
        cohort = digest(
            {
                k: score[k]
                for k in (
                    "provider_profile_hash",
                    "evaluator_profile",
                    "evaluator_implementation_hash",
                    "protocol_repair_profile_hash",
                )
            }
        )
        cohorts[cohort].append(score)
    datasets = {}
    for dataset in sorted({s["dataset_id"] for s in scores}):
        rows = [s for s in scores if s["dataset_id"] == dataset]
        datasets[dataset] = {
            "episodes": len(rows),
            "valid_forecasts": sum(s["completed"] for s in rows),
            "error_actions": sum(s["ErrorActionCount"] for s in rows),
            "protocol_alarms": sum(s["ProtocolAlarmCount"] for s in rows),
            "capacity_alarms": sum(s.get("CapacityAlarmCount", 0) for s in rows),
            "capacity_failures": sum(s.get("CapacityFailureCount", 0) for s in rows),
            **{
                name: stable_mean([s[name] for s in rows]) if all(s[name] is not None for s in rows) else None
                for name in ("MSE", "MAE", "sMAPE", "MASE", "RMSSE")
            },
        }
    return {
        "notice": "Explicit mixed-capacity continuation. Old scores are unchanged. Cohorts are not directly comparable: domain composition differs. No pooled TotalScore leaderboard is claimed.",
        "episodes": len(scores),
        "valid_forecasts": sum(s["completed"] for s in scores),
        "datasets_descriptive_only": datasets,
        "frozen_profile_cohorts": {
            key: {
                "episodes": len(rows),
                "episode_ids": [s["episode_id"] for s in rows],
                "scores": aggregate(rows),
            }
            for key, rows in cohorts.items()
        },
    }

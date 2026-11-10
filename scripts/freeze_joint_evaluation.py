"""Freeze code, packs and profiles before starting a long local evaluation. Does not launch it."""

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from castbench.common import load_json, write_json
from castbench.evaluator import SCORE_PROFILES
from castbench.protocol_repair import POLICY


def freeze(
    project,
    packs,
    models,
    provider,
    output,
    sample_concurrency=2,
    model_worker_concurrency=1,
    npu_devices=None,
    domains=None,
    count=50,
    score_profile="legacy-v1",
):
    if type(count) is not int or not 1 <= count <= 50:
        raise ValueError("count must be an integer in 1..50")
    if score_profile not in SCORE_PROFILES:
        raise ValueError("Unknown scoring profile")
    if (
        isinstance(sample_concurrency, bool)
        or not isinstance(sample_concurrency, int)
        or sample_concurrency < 1
    ):
        raise ValueError("sample_concurrency must be a positive integer")
    if (
        isinstance(model_worker_concurrency, bool)
        or not isinstance(model_worker_concurrency, int)
        or model_worker_concurrency < 1
    ):
        raise ValueError("model_worker_concurrency must be a positive integer")
    npu_devices = [] if npu_devices is None else list(npu_devices)
    if any(type(device) is not int or device < 0 for device in npu_devices) or len(
        set(npu_devices)
    ) != len(npu_devices):
        raise ValueError("npu_devices must contain distinct nonnegative integers")
    domains = list(("bike", "gef", "sunny", "windy") if domains is None else domains)
    if (
        not domains
        or len(set(domains)) != len(domains)
        or not set(domains) <= {"bike", "gef", "sunny", "windy"}
    ):
        raise ValueError("domains must be a nonempty, unique subset of bike/gef/sunny/windy")
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; never overwrite an existing evaluation")
    public = packs / "public"
    selected = {}
    for domain in domains:
        manifest = load_json(public / domain / "manifest.json")
        entries = manifest["episodes"]
        ids = [e["episode_id"] for e in entries]
        if len(ids) != 50 or len(set(ids)) != 50:
            raise ValueError(f"{domain}: expected exactly 50 distinct frozen test Episodes")
        selected[domain] = ids[:count]
    all_ids = [eid for ids in selected.values() for eid in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Episode IDs must be unique across domains")
    target = output / "frozen"
    target.mkdir(parents=True)
    for name in ("src", "scripts"):
        shutil.copytree(project / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for part in ("public", "evaluator_private"):
        for domain in domains:
            shutil.copytree(packs / part / domain, target / "packs" / part / domain)
    for file in ("pyproject.toml", "README.md"):
        shutil.copy2(project / file, target / file)
    write_json(target / "models.json", load_json(models))
    write_json(target / "provider.json", load_json(provider))
    hashes = {
        str(p.relative_to(target)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_started",
        "samples_planned": sum(len(ids) for ids in selected.values()),
        "selected_episodes": selected,
        "protocol": POLICY["version"],
        "alarm_error_equivalent": POLICY["warning_penalty_error_equivalent"],
        "execution": "bounded concurrent Episodes, independent Agent state; frozen model-output cache only",
        "sample_concurrency": sample_concurrency,
        "model_worker_concurrency": model_worker_concurrency,
        "npu_devices": npu_devices,
        "python_executable": sys.executable,
        "snapshot_hashes": hashes,
        "notice": "Private targets are for offline scoring only. No weights or samples tuned on test scores.",
    }
    plan.update(count_per_domain=count, score_profile=score_profile)
    write_json(output / "evaluation_plan.json", plan)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--provider", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample-concurrency", type=int, default=2)
    p.add_argument("--model-worker-concurrency", type=int, default=1)
    p.add_argument("--npu-devices", type=int, nargs="*", default=[])
    p.add_argument("--domains", nargs="+", choices=["bike", "gef", "sunny", "windy"])
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--score-profile", choices=SCORE_PROFILES, default="legacy-v1")
    a = p.parse_args()
    print(
        freeze(
            Path(__file__).resolve().parents[1],
            a.packs.resolve(),
            a.models,
            a.provider,
            a.out,
            a.sample_concurrency,
            a.model_worker_concurrency,
            a.npu_devices,
            a.domains,
            a.count,
            a.score_profile,
        )
    )


if __name__ == "__main__":
    main()

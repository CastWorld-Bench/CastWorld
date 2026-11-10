"""Run an explicitly listed, isolated development panel with real provider/tools.

Private targets are opened only after an Episode has finished. This utility is for
protocol and budget studies, not for replacing or continuing a frozen benchmark run.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from castbench.common import digest, load_json, write_json
from castbench.domain_packs import load_domain_episode
from castbench.evaluator import score_run
from castbench.models import ModelProfile, ModelService
from castbench.providers import OpenAICompatibleProvider, ProviderProfile
from castbench.runner import Runner
from castbench.storage import Store


def load_panel(path: Path) -> list[str]:
    value = load_json(path)
    episodes = value["episodes"] if isinstance(value, dict) else value
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Panel must be a non-empty JSON list or an object with episodes")
    if any(not isinstance(x, str) or x.count(".") < 2 for x in episodes):
        raise ValueError("Every panel item must be an Episode ID")
    if len(set(episodes)) != len(episodes):
        raise ValueError("Panel contains duplicate Episode IDs")
    return episodes


def filter_registry(registry, mode: str):
    """Create a true catalog ablation after the signed domain pack was verified."""
    if mode == "full":
        return registry
    allowed = {"model", "sandbox", "registry"}
    keep = {tool_id for tool_id, item in registry.manifests.items() if item["category"] in allowed}
    registry.manifests = {k: v for k, v in registry.manifests.items() if k in keep}
    registry.schemas = {k: v for k, v in registry.schemas.items() if k in keep}
    registry.extensions = {k: v for k, v in registry.extensions.items() if k in keep}
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Frozen evaluation root")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-concurrency", type=int, default=2)
    parser.add_argument("--catalog-mode", choices=["full", "model_sandbox"], default="full")
    parser.add_argument(
        "--server-reasoning-budget",
        type=int,
        default=-1,
        help="Declared llama.cpp server-side reasoning budget used for this condition",
    )
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Use a new output directory for every controlled condition")
    if args.sample_concurrency < 1:
        raise ValueError("sample-concurrency must be positive")
    episodes = load_panel(args.panel)
    snapshot = source / "frozen"
    profile = ProviderProfile.model_validate(load_json(args.provider))
    service = ModelService(
        [ModelProfile.model_validate(x) for x in load_json(snapshot / "models.json")],
        Store(args.model_cache.resolve()),
        max_parallel_workers=1,
    )
    service.preflight()
    out.mkdir(parents=True)
    store = Store(out / "state")
    write_json(
        out / "study_profile.json",
        {
            "schema_version": "castbench.controlled_study/1.0",
            "source": str(source),
            "source_plan_hash": digest(load_json(source / "evaluation_plan.json")),
            "panel": episodes,
            "provider": profile.model_dump(),
            "sample_concurrency": args.sample_concurrency,
            "catalog_mode": args.catalog_mode,
            "server_reasoning_budget": args.server_reasoning_budget,
            "notice": "Development diagnostic; results must not replace the frozen benchmark run.",
        },
    )

    def execute(eid: str):
        domain = eid.split(".", 1)[0]
        bundle, registry = load_domain_episode(snapshot / "packs/public" / domain, eid)
        registry = filter_registry(registry, args.catalog_mode)
        provider = OpenAICompatibleProvider(profile)
        started = time.monotonic()
        try:
            runner = Runner(
                bundle,
                store,
                provider,
                eid,
                registry=registry,
                model_service=service,
                protocol_repair=True,
            )
            while runner.state()["status"] == "running":
                runner.step()
            trace = store.export(eid, include_raw=True)
            score = score_run(
                trace,
                load_json(snapshot / "packs/evaluator_private" / domain / f"{eid}.json"),
            )
            destination = out / "results" / eid
            write_json(destination / "trace.json", trace)
            write_json(destination / "score.json", score)
            write_json(
                destination / "job_status.json",
                {
                    "status": "finished",
                    "episode_id": eid,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                f"{eid}: completed={score['completed']} rounds={score['rounds']} "
                f"calls={score['SuccessfulToolCallCount']} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            return score
        finally:
            provider.client.close()

    scores = []
    with ThreadPoolExecutor(max_workers=args.sample_concurrency) as pool:
        futures = {pool.submit(execute, eid): eid for eid in episodes}
        for future in as_completed(futures):
            scores.append(future.result())
    write_json(
        out / "study_summary.json",
        {
            "episodes": len(scores),
            "completed": sum(x["completed"] for x in scores),
            "capacity_alarms": sum(x["CapacityAlarmCount"] for x in scores),
            "capacity_failures": sum(x["CapacityFailureCount"] for x in scores),
            "protocol_alarms": sum(x["ProtocolAlarmCount"] for x in scores),
            "successful_tool_calls": sum(x["SuccessfulToolCallCount"] for x in scores),
            "scores": {x["episode_id"]: x for x in scores},
        },
    )


if __name__ == "__main__":
    main()

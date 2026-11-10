"""Real independent model probes plus autonomous Agent runs; never tune from labels."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import httpx

from castbench.common import (
    InfrastructureError,
    TransientInfrastructureError,
    digest,
    load_json,
    write_json,
)
from castbench.continuation import preserved_score
from castbench.domain_packs import load_domain_episodes
from castbench.evaluator import score_run
from castbench.models import ModelProfile, ModelService
from castbench.providers import OpenAICompatibleProvider, ProviderProfile
from castbench.runner import Runner
from castbench.scheduler import run_jobs
from castbench.storage import Store


def refresh_summary(directory, planned=None):
    summary = []
    for path in sorted(directory.glob("*/score.json")):
        trace = load_json(path.parent / "trace.json")
        summary.append(
            {
                "episode_id": trace["config"]["episode_id"],
                "final": trace["state"]["final"],
                "score": load_json(path),
            }
        )
    write_json(directory / "summary.json", summary)
    recoveries = [
        load_json(path)
        for path in sorted(directory.glob("*/transport_recovery.json"))
    ]
    write_json(
        directory / "progress.json",
        {
            "samples_planned": planned,
            "samples_scored": len(summary),
            "valid_forecasts": sum(x["score"]["completed"] for x in summary),
            "llm_interaction_rounds": (
                sum(x["score"]["rounds"] for x in summary)
                if all("rounds" in x["score"] for x in summary)
                else None
            ),
            "error_actions": sum(x["score"].get("ErrorActionCount", 0) for x in summary),
            "protocol_alarms": sum(x["score"].get("ProtocolAlarmCount", 0) for x in summary),
            "capacity_alarms": sum(x["score"].get("CapacityAlarmCount", 0) for x in summary),
            "capacity_failures": sum(x["score"].get("CapacityFailureCount", 0) for x in summary),
            "completion_rescues": sum(
                x["score"].get("CompletionRescueCount", 0) for x in summary
            ),
            "successful_completion_rescues": sum(
                x["score"].get("CompletionRescueSuccessCount", 0) for x in summary
            ),
            "failed_completion_rescues": sum(
                x["score"].get("CompletionRescueFailureCount", 0) for x in summary
            ),
            "provider_responses": sum(
                x["score"].get("ProviderResponseCount", x["score"].get("rounds", 0)) for x in summary
            ),
            "transport_retry_events": sum(len(item.get("events", [])) for item in recoveries),
            "transport_recovered_samples": sum(
                item.get("status") == "recovered" for item in recoveries
            ),
            "transport_exhausted_samples": sum(
                item.get("status") == "exhausted" for item in recoveries
            ),
            "successful_repairs": sum(x["score"].get("ProtocolRepairSuccessCount", 0) for x in summary),
            "requested_tool_calls": sum(x["score"].get("ToolCallRequestedCount", 0) for x in summary),
            "successful_tool_calls": sum(x["score"].get("SuccessfulToolCallCount", 0) for x in summary),
            "notice": "Counts cover exported, ended Episodes only; the current Episode is visible in the log/journal.",
        },
    )


def check_local_capacity(profile, concurrency):
    if profile.token_count_mode != "llama_cpp":
        return {"capacity_check": "remote provider; operator must provision API concurrency"}
    key = os.environ.get(profile.api_key_env)
    if not key:
        raise InfrastructureError("Missing local API key")
    origin = profile.endpoint.split("/v1/")[0]
    response = httpx.get(origin + "/props", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    response.raise_for_status()
    props = response.json()
    slots = props.get("total_slots", 0)
    context = props.get("default_generation_settings", {}).get("n_ctx", 0)
    if slots < 1 or context < profile.context_window_tokens:
        raise InfrastructureError(
            f"Local service has {slots} slots / {context} context per slot; "
            f"requires at least one slot / {profile.context_window_tokens} context per slot"
        )
    return {
        "slots": slots,
        "context_per_slot": context,
        "queued_episodes": max(0, concurrency - slots),
        "build_info": props.get("build_info"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--provider", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--domains", nargs="+", default=["bike", "gef", "sunny", "windy"])
    p.add_argument("--count", type=int, default=1)
    p.add_argument(
        "--sample-concurrency",
        type=int,
        default=2,
        help="Independent concurrent Episodes; local server needs matching full-context slots",
    )
    p.add_argument(
        "--model-worker-concurrency",
        type=int,
        default=1,
        help="Independent foundation-model subprocesses; tune separately from API/sample concurrency",
    )
    p.add_argument(
        "--npu-devices",
        type=int,
        nargs="*",
        default=[],
        help="Physical Ascend IDs leased one per isolated npu:0 worker",
    )
    p.add_argument("--models-only", action="store_true")
    p.add_argument(
        "--score-profile",
        choices=["legacy-v1", "robust-v1.1", "robust-v1.2"],
        default="legacy-v1",
    )
    p.add_argument("--continuation-manifest", type=Path)
    p.add_argument(
        "--protocol-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze one corrective action per Episode, with a 0.05-error alarm penalty",
    )
    p.add_argument(
        "--model-cache", type=Path, help="Optional shared cache of frozen model outputs; never Agent history"
    )
    args = p.parse_args()
    continuation = load_json(args.continuation_manifest) if args.continuation_manifest else {}
    if not 1 <= args.count <= 50:
        raise ValueError("count must be 1..50")
    if len(set(args.domains)) != len(args.domains):
        raise ValueError("Duplicate domains would repeat Episodes")
    if args.sample_concurrency < 1:
        raise ValueError("sample-concurrency must be positive")
    if args.model_worker_concurrency < 1:
        raise ValueError("model-worker-concurrency must be positive")
    if any(device < 0 for device in args.npu_devices) or len(set(args.npu_devices)) != len(
        args.npu_devices
    ):
        raise ValueError("npu-devices must be distinct nonnegative integers")
    store = Store(args.out / "state")
    cache = Store(args.model_cache) if args.model_cache else store
    # Share a worker gate, not Episode data. GPU decoding can overlap CPU model execution.
    service = ModelService(
        [ModelProfile.model_validate(x) for x in load_json(args.models)],
        cache,
        max_parallel_workers=args.model_worker_concurrency,
        npu_devices=args.npu_devices,
    )
    service.preflight()
    profile = ProviderProfile.model_validate(load_json(args.provider))
    capacity = check_local_capacity(profile, args.sample_concurrency) if not args.models_only else {}
    jobs = []
    for domain in args.domains:
        root = args.packs / "public" / domain
        manifest = load_json(root / "manifest.json")
        episode_ids = [entry["episode_id"] for entry in manifest["episodes"][: args.count]]
        # Shared knowledge and the sample library are verified once per domain.
        # Each returned Bundle and all Agent state remain Episode-scoped.
        loaded = load_domain_episodes(root, episode_ids)
        manifest_hash = digest(manifest)
        jobs.extend(
            (
                domain,
                episode_id,
                *loaded[episode_id],
                digest({"domain_pack_manifest": manifest_hash, "episode_id": episode_id}),
            )
            for episode_id in episode_ids
        )
    if len({eid for _, eid, _, _, _ in jobs}) != len(jobs):
        raise ValueError("Duplicate Episode IDs across packages")
    stop = Event()
    started = time.time()

    def check_stop():
        if (args.out.parent / "STOP_REQUESTED").exists() or (args.out / "STOP_REQUESTED").exists():
            stop.set()
        if stop.is_set():
            raise InfrastructureError("Stop requested; saved sample progress is retained")

    def execute(job):
        domain, eid, bundle, registry, bundle_hash = job
        provider = None
        dest = args.out / eid
        try:
            check_stop()
            if preserved_score(args.out, eid, continuation):
                print(f"Preserved completed {eid}; no model or LLM call", flush=True)
                return eid
            model_results = {}
            for name in service.profiles:
                check_stop()
                tick = time.monotonic()
                result = service.predict(name, bundle.episode)
                if result["status"] != "ok":
                    raise RuntimeError(f"{eid}/{name}: {result}")
                model_results[name] = result
                write_json(dest / "model_results.json", model_results)
                print(f"{eid} {name}: ok {time.monotonic() - tick:.1f}s", flush=True)
            if args.models_only:
                return eid
            provider = OpenAICompatibleProvider(profile)
            runner = Runner(
                bundle,
                store,
                provider,
                eid,
                model_service=service,
                registry=registry,
                protocol_repair=args.protocol_repair,
                continuation_from=continuation.get("unfinished_configs", {}).get(eid),
                verified_bundle_hash=bundle_hash,
            )
            while runner.state()["status"] == "running":
                check_stop()
                tick = time.monotonic()
                runner.step()
                state = runner.state()
                print(
                    f"{eid} round={state['rounds_used']} status={state['status']} errors={len(state['errors'])} alarms={len(state.get('protocol_alarms', []))} {time.monotonic() - tick:.1f}s",
                    flush=True,
                )
            trace = store.export(eid, include_raw=True)
            write_json(dest / "trace.json", trace)
            # Only this offline scoring block opens truth; it is never given to Runner or tools.
            truth = load_json(args.packs / "evaluator_private" / domain / f"{eid}.json")
            score = score_run(trace, truth, args.score_profile)
            write_json(dest / "score.json", score)
            recovery_path = dest / "transport_recovery.json"
            recovery = load_json(recovery_path) if recovery_path.exists() else {"events": []}
            if recovery["events"]:
                recovery["status"] = "recovered"
                recovery["recovered_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(recovery_path, recovery)
            write_json(
                dest / "job_status.json",
                {
                    "status": "finished",
                    "episode_id": eid,
                    "transport_retry_count": len(recovery["events"]),
                },
            )
            print(f"Finished {eid}", flush=True)
            return eid
        except Exception as exc:
            transient = isinstance(exc, TransientInfrastructureError)
            if not transient:
                stop.set()
            write_json(
                dest / "job_status.json",
                {
                    "status": "retrying_transport" if transient else "paused",
                    "episode_id": eid,
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        finally:
            if provider is not None:
                provider.client.close()

    # Only the coordinator refreshes aggregate files. Per-Episode outputs have unique paths.
    def completed(_):
        if not args.models_only:
            refresh_summary(args.out, len(jobs))

    execution = {
        "sample_concurrency": args.sample_concurrency,
        "model_worker_concurrency": args.model_worker_concurrency,
        "npu_devices": args.npu_devices,
        "capacity": capacity,
        "episode_ids": [eid for _, eid, _, _, _ in jobs],
    }
    if args.score_profile != "legacy-v1":
        execution["score_profile"] = args.score_profile
    with store.writer("__joint_evaluation_coordinator__"):
        frozen = args.out / "execution_profile.json"
        if frozen.exists() and load_json(frozen) != execution:
            raise InfrastructureError("Execution configuration changed; use a new run directory")
        write_json(frozen, execution)
        completed(None)
        try:
            retry_delays = tuple(
                min(
                    profile.transport_retry_initial_seconds * (2**index),
                    profile.transport_retry_max_seconds,
                )
                for index in range(max(1, profile.max_episode_transport_retries))
            )

            def transport_retry(job, exc, retry_count, delay):
                _, eid, _, _, _ = job
                recovery_path = args.out / eid / "transport_recovery.json"
                recovery = (
                    load_json(recovery_path)
                    if recovery_path.exists()
                    else {"episode_id": eid, "status": "retrying", "events": []}
                )
                recovery["status"] = "retrying"
                recovery["events"].append(
                    {
                        "retry_count": retry_count,
                        "retry_after_seconds": delay,
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                        "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                write_json(recovery_path, recovery)
                write_json(
                    args.out / eid / "job_status.json",
                    {
                        "status": "retrying_transport",
                        "episode_id": eid,
                        "retry_count": retry_count,
                        "retry_after_seconds": delay,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                )
                print(
                    f"{eid} transient provider failure; retry {retry_count}/"
                    f"{profile.max_episode_transport_retries} in {delay:.1f}s",
                    flush=True,
                )

            def transport_exhausted(job, exc, retry_count):
                _, eid, _, _, _ = job
                recovery_path = args.out / eid / "transport_recovery.json"
                recovery = (
                    load_json(recovery_path)
                    if recovery_path.exists()
                    else {"episode_id": eid, "events": []}
                )
                recovery.update(
                    {
                        "status": "exhausted",
                        "exhausted_utc": datetime.now(timezone.utc).isoformat(),
                        "final_error_type": type(exc).__name__,
                        "final_reason": str(exc),
                    }
                )
                write_json(recovery_path, recovery)
                write_json(
                    args.out / eid / "job_status.json",
                    {
                        "status": "paused_transport_exhausted",
                        "episode_id": eid,
                        "retry_count": retry_count,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                )

            run_jobs(
                jobs,
                execute,
                args.sample_concurrency,
                on_complete=completed,
                stop=stop,
                retry_if=lambda exc: isinstance(exc, TransientInfrastructureError),
                max_retries=profile.max_episode_transport_retries,
                retry_delays=retry_delays,
                on_retry=transport_retry,
                on_retry_exhausted=transport_exhausted,
                job_label=lambda job: f"{job[0]}/{job[1]}",
                continue_after_retry_exhaustion=True,
            )
        finally:
            completed(None)
            write_json(
                args.out / "timing.json",
                {
                    "started_unix": started,
                    "elapsed_seconds": time.time() - started,
                    "sample_concurrency": args.sample_concurrency,
                },
            )


if __name__ == "__main__":
    main()

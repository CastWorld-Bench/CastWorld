from __future__ import annotations

import argparse
from pathlib import Path

from .common import digest, load_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="castbench")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Offline synthetic end-to-end run; no LLM or weight downloads")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    build = sub.add_parser("build", help="Offline builder; requires an explicit frozen configuration")
    build.add_argument("--csv", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--public", type=Path, required=True)
    build.add_argument("--private", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("bundle", type=Path)
    catalog = sub.add_parser("catalog")
    catalog.add_argument("--out", type=Path, required=True)
    hash_command = sub.add_parser("checkpoint-hash", help="Hash a local frozen checkpoint directory")
    hash_command.add_argument("directory", type=Path)
    smoke = sub.add_parser("model-smoke", help="Explicitly test real model workers without calling an LLM")
    smoke.add_argument("--bundle", type=Path, required=True)
    smoke.add_argument("--models", type=Path, required=True)
    smoke.add_argument("--store", type=Path, required=True)
    smoke.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="Run or resume the same logical Episode")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--domain-pack", type=Path, help="Verified public pack containing this bundle")
    run.add_argument("--provider", type=Path, required=True)
    run.add_argument("--models", type=Path, required=True)
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--resource-policy", type=Path,
                     help="Explicit frozen resource protocol for robust-v1.3; old runs stay unchanged")
    run.add_argument(
        "--protocol-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="One protocol correction per Episode; --no-protocol-repair selects the strict track",
    )
    export = sub.add_parser("export")
    export.add_argument("--store", type=Path, required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--include-raw", action="store_true")
    score = sub.add_parser("score", help="Offline only: accepts a private evaluator record")
    score.add_argument("--trace", type=Path, required=True)
    score.add_argument("--private", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument(
        "--score-profile",
        choices=["legacy-v1", "robust-v1.1", "robust-v1.2"],
        default="legacy-v1",
    )
    agg = sub.add_parser("aggregate")
    agg.add_argument("scores", nargs="+", type=Path)
    agg.add_argument("--out", type=Path, required=True)
    calibrate = sub.add_parser("calibrate-efficiency", help="Freeze anchors from complete validation exports")
    calibrate.add_argument("traces", nargs="+", type=Path)
    calibrate.add_argument("--out", type=Path, required=True)
    bind = sub.add_parser("bind-efficiency-policy", help="Bind frozen validation anchors before test decisions")
    bind.add_argument("--profile", type=Path, required=True)
    bind.add_argument("--episodes", type=Path, required=True, help="JSON list of all planned test Episode IDs")
    bind.add_argument("--out", type=Path, required=True)
    cohort = sub.add_parser("score-cohort", help="Offline robust-v1.3, complete fixed sample manifest only")
    cohort.add_argument("--manifest", type=Path, required=True)
    cohort.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        import pandas as pd

        from .data import build_packages

        print(build_packages(pd.read_csv(args.csv), load_json(args.config), args.public, args.private))
    elif args.command == "validate":
        from .contracts import PublicBundle

        bundle = PublicBundle.model_validate(load_json(args.bundle))
        print(f"Valid public Episode: {bundle.episode.episode_id}")
    elif args.command == "catalog":
        from .catalog import Registry

        r = Registry()
        write_json(args.out, {"snapshot": r.snapshot, "tools": r.manifests})
        print(args.out.resolve())
    elif args.command == "checkpoint-hash":
        from .model_worker import checkpoint_hash

        if not args.directory.is_dir() or not any(p.is_file() for p in args.directory.rglob("*")):
            raise ValueError("Checkpoint directory is empty or missing")
        print(checkpoint_hash(args.directory))
    elif args.command in {"run", "model-smoke"}:
        from .contracts import PublicBundle
        from .models import ModelProfile, ModelService
        from .providers import OpenAICompatibleProvider, ProviderProfile
        from .runner import Runner
        from .storage import Store

        bundle = PublicBundle.model_validate(load_json(args.bundle))
        registry = None
        if args.command == "run" and args.domain_pack:
            from .domain_packs import load_domain_episode

            bundle, registry = load_domain_episode(args.domain_pack, bundle.episode.episode_id)
        store = Store(args.store)
        models = ModelService([ModelProfile.model_validate(p) for p in load_json(args.models)], store)
        models.preflight()
        if args.command == "model-smoke":
            results = {
                name: models.predict(name, bundle.episode)
                for name in ("chronos", "timesfm", "sundial", "moirai")
            }
            write_json(args.out, results)
            print(args.out.resolve())
            return
        provider = OpenAICompatibleProvider(ProviderProfile.model_validate(load_json(args.provider)))
        state = Runner(
            bundle,
            store,
            provider,
            args.run_id,
            model_service=models,
            registry=registry,
            protocol_repair=args.protocol_repair,
            resource_policy=load_json(args.resource_policy) if args.resource_policy else None,
        ).run()
        print(state["final"])
    elif args.command == "export":
        from .storage import Store

        write_json(args.out, Store(args.store).export(args.run_id, args.include_raw))
        print(args.out.resolve())
    elif args.command == "score":
        from .evaluator import score_run

        write_json(args.out, score_run(load_json(args.trace), load_json(args.private), args.score_profile))
        print(args.out.resolve())
    elif args.command == "aggregate":
        from .evaluator import aggregate

        write_json(args.out, aggregate([load_json(p) for p in args.scores]))
        print(args.out.resolve())
    elif args.command == "calibrate-efficiency":
        from .efficiency import calibrate_profile

        write_json(args.out, calibrate_profile([load_json(p) for p in args.traces]))
        print(args.out.resolve())
    elif args.command == "score-cohort":
        from .efficiency import score_cohort
        from .evaluator import aggregate

        manifest = load_json(args.manifest)
        if set(manifest) != {"schema_version", "profiles", "runs"} or (
            manifest["schema_version"] != "castbench.efficiency-cohort/1.0"
        ):
            raise ValueError("Invalid frozen efficiency cohort manifest")
        base = args.manifest.resolve().parent
        exports = [load_json(base / r["trace"]) for r in manifest["runs"]]
        private = [load_json(base / r["private"]) for r in manifest["runs"]]
        profiles = {d: load_json(base / p) for d, p in manifest["profiles"].items()}
        scores = score_cohort(exports, private, profiles)
        write_json(args.out, {"scores": scores, "summary": aggregate(scores),
                             "manifest_hash": digest(manifest)})
        print(args.out.resolve())
    elif args.command == "bind-efficiency-policy":
        from .efficiency import bind_test_policy

        write_json(args.out, bind_test_policy(load_json(args.profile), load_json(args.episodes)))
        print(args.out.resolve())
    else:
        from .contracts import PublicBundle
        from .data import build_packages, synthetic_fixture
        from .evaluator import score_run
        from .providers import ScriptedProvider
        from .runner import Runner
        from .storage import Store

        frame, config = synthetic_fixture()
        path = args.out
        public, private = path / "public", path / "private"
        if not public.exists():
            build_packages(frame, config, public, private)
        bundle = PublicBundle.model_validate(load_json(next((public / "dev").glob("*.json"))))
        e = bundle.episode
        h = e.history.target_values
        # Simple seasonal forecast generated by a scripted Agent, not a claimed TSFM result.
        values = [h[-24 + ((e.task.forecast_gap + i) % 24)] for i in range(e.task.horizon_length)]
        actions = [
            {
                "action_type": "CALL_TOOL",
                "phase": "ACQUIRE",
                "payload": {
                    "calls": [
                        {
                            "call_id": "c1",
                            "tool_id": "feature.window_summary",
                            "purpose": "Inspect observed history",
                            "arguments": {},
                        }
                    ]
                },
            },
            {
                "action_type": "CREATE_FORECAST",
                "phase": "REVISE",
                "payload": {
                    "operation": "submit_full_forecast",
                    "scored_values": values,
                    "scored_timestamps": e.forecast_timeline.scored_timestamps,
                    "unit": e.task.target_unit,
                },
            },
            {"action_type": "CHECK_FORECAST", "phase": "VERIFY", "payload": {"forecast_id": "forecast_1"}},
            {
                "action_type": "FINALIZE",
                "phase": "STOP",
                "payload": {"forecast_id": "forecast_1", "stop_reason": "Synthetic smoke test complete"},
            },
        ]
        store = Store(path / "state")
        Runner(bundle, store, ScriptedProvider(actions), "demo").run()
        exported = store.export("demo", include_raw=True)
        write_json(path / "trace.json", exported)
        score = score_run(exported, load_json(private / (e.episode_id + ".json")))
        write_json(path / "score.json", score)
        print(
            f"Offline demo complete: {path.resolve()}\nForecastScore={score['ForecastScore']:.4f}; TotalScore={score['TotalScore']:.4f}"
        )


if __name__ == "__main__":
    main()

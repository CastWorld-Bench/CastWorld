"""Resolve each frozen foundation model on NPU, then CUDA, then CPU by real inference."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from castbench.common import load_json, write_json
from castbench.domain_packs import load_domain_episode
from castbench.models import ModelProfile, ModelService
from castbench.storage import Store

PRIORITY = ("npu:0", "cuda:0", "cpu")


def parse_python_overrides(values):
    result = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("Interpreter overrides must use MODEL=/absolute/path/to/python")
        name, executable = value.split("=", 1)
        if name not in {"chronos", "timesfm", "sundial", "moirai"} or name in result:
            raise ValueError(f"Invalid or duplicate model interpreter override: {name}")
        path = Path(executable).resolve()
        if not path.is_file():
            raise ValueError(f"Missing interpreter for {name}: {path}")
        result[name] = str(path)
    return result


def dependency_versions(python, names):
    code = (
        "import importlib.metadata as m,json;"
        "print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"
    )
    installed = json.loads(subprocess.check_output([python, "-c", code], text=True))
    normalized = {re.sub(r"[-_.]+", "-", key).lower(): value for key, value in installed.items()}
    return {name: normalized[re.sub(r"[-_.]+", "-", name).lower()] for name in names}


def candidate_profile(base, device, executable):
    raw = base.model_dump()
    dependency_names = list(base.dependency_versions)
    if device == "npu:0" and "torch-npu" not in dependency_names:
        dependency_names.append("torch-npu")
    raw.update(
        device=device,
        python_executable=executable,
        dependency_versions=dependency_versions(executable, dependency_names),
    )
    return ModelProfile.model_validate(raw)


def resolve(profiles, episode, cache, npu_device, npu_pythons=None, gpu_pythons=None):
    npu_pythons = npu_pythons or {}
    gpu_pythons = gpu_pythons or {}
    selected, report = [], {"priority": list(PRIORITY), "models": {}}
    for base in profiles:
        attempts = []
        for device in PRIORITY:
            executable = {
                "npu:0": npu_pythons.get(base.name, base.python_executable),
                "cuda:0": gpu_pythons.get(base.name, base.python_executable),
                "cpu": base.python_executable,
            }[device]
            started = time.monotonic()
            try:
                candidate = candidate_profile(base, device, executable)
                service = ModelService(
                    [candidate],
                    Store(cache / base.name / device.replace(":", "_")),
                    max_parallel_workers=1,
                    npu_devices=[npu_device] if device == "npu:0" else [],
                )
                result = service.predict(base.name, episode)
                if result["status"] != "ok":
                    raise RuntimeError(f"probe returned {result['status']}")
                attempts.append(
                    {
                        "device": device,
                        "python_executable": executable,
                        "status": "passed",
                        "elapsed_seconds": time.monotonic() - started,
                        "cache_key": result["structured_result"]["cache_key"],
                    }
                )
                selected.append(candidate.model_dump())
                report["models"][base.name] = {"selected": device, "attempts": attempts}
                break
            except Exception as exc:
                attempts.append(
                    {
                        "device": device,
                        "python_executable": executable,
                        "status": "failed",
                        "elapsed_seconds": time.monotonic() - started,
                        "reason": f"{type(exc).__name__}: {str(exc)[-2000:]}",
                    }
                )
        else:
            report["models"][base.name] = {"selected": None, "attempts": attempts}
            raise RuntimeError(f"No supported accelerator backend for {base.name}")
    return selected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--domain", choices=["bike", "gef", "sunny", "windy"], required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--npu-device", type=int, default=0)
    parser.add_argument("--npu-python", action="append", default=[])
    parser.add_argument("--gpu-python", action="append", default=[])
    args = parser.parse_args()
    if args.npu_device < 0:
        raise ValueError("npu-device must be nonnegative")
    root = args.packs / "public" / args.domain
    episode_id = args.episode_id
    if episode_id is None:
        episode_id = load_json(root / "manifest.json")["episodes"][0]["episode_id"]
    bundle, _ = load_domain_episode(root, episode_id)
    profiles = [ModelProfile.model_validate(raw) for raw in load_json(args.models)]
    selected, report = resolve(
        profiles,
        bundle.episode,
        args.cache,
        args.npu_device,
        parse_python_overrides(args.npu_python),
        parse_python_overrides(args.gpu_python),
    )
    report.update(episode_id=episode_id, npu_probe_device=args.npu_device)
    write_json(args.out, selected)
    write_json(args.report, report)
    print(args.out)


if __name__ == "__main__":
    main()

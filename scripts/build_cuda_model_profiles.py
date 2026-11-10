"""Build frozen CUDA model profiles for a migrated CastWorld checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from castbench.common import load_json, write_json
from castbench.model_worker import checkpoint_hash
from castbench.models import ModelProfile


MODEL_ENV = {
    "chronos": "tsfm",
    "timesfm": "tsfm",
    "sundial": "sundial",
    "moirai": "moirai",
}


def normalized_distributions(python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json;"
        "print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"
    )
    raw = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    return {re.sub(r"[-_.]+", "-", key).lower(): value for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    template = args.template or root / "checkpoints/tsfm/model_profiles.castbench.json"
    out = args.out or root / "checkpoints/tsfm/model_profiles.cuda.castbench.json"
    profiles = []
    for raw in load_json(template):
        name = raw["name"]
        python = root / ".envs" / MODEL_ENV[name] / "bin" / "python"
        checkpoint = root / "checkpoints" / "tsfm" / "weights" / name
        if not python.is_file() or not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing runtime for {name}: {python} / {checkpoint}")
        installed = normalized_distributions(python)
        dependencies = {
            package: installed[re.sub(r"[-_.]+", "-", package).lower()]
            for package in raw["dependency_versions"]
        }
        raw.update(
            python_executable=str(python.resolve()),
            checkpoint=str(checkpoint.resolve()),
            checkpoint_digest=checkpoint_hash(checkpoint),
            device="cuda:0",
            dependency_versions=dependencies,
        )
        profiles.append(ModelProfile.model_validate(raw).model_dump())

    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, profiles)
    print(out)


if __name__ == "__main__":
    main()

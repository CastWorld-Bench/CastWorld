"""Offline checks. Runs under the same interpreter as this script."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
checks = [
    [sys.executable, "-m", "compileall", "-q", "src", "tests"],
    [sys.executable, "-m", "ruff", "check", "--config", "pyproject.toml", "src", "tests", "scripts"],
    [sys.executable, "-m", "pytest", "-q", "-W", "error", "--junitxml=runs/verification/junit.xml"],
]
for command in checks:
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
print("All offline checks passed. This does not certify live LLM or model-weight deployments.")

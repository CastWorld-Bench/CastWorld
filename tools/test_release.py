"""Run the released offline test suite with external network access disabled."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ)
args = [sys.executable, '-m', 'pytest', '-p', 'tools.offline_test_guard', 'tests', '-q']
print('External connections disabled; local test servers remain available.', flush=True)
args += sys.argv[1:]
raise SystemExit(subprocess.run(args, cwd=ROOT, env=env).returncode)

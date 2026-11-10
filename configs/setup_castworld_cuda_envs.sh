#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${CASTWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA_EXE:-/opt/castworld/anaconda3/bin/conda}"
ENV_ROOT="${CASTWORLD_ENV_ROOT:-$ROOT/.envs}"

if [[ ! -x "$CONDA" ]]; then
  echo "Missing conda executable: $CONDA" >&2
  exit 1
fi

mkdir -p "$ENV_ROOT" "$ROOT/logs/migration"

create_env() {
  local name="$1"
  if [[ ! -x "$ENV_ROOT/$name/bin/python" ]]; then
    "$CONDA" create -p "$ENV_ROOT/$name" python=3.11 pip -y
  fi
}

for name in core tsfm sundial moirai; do
  create_env "$name"
done

"$ENV_ROOT/core/bin/python" -m pip install -e "${ROOT}[test]"

"$ENV_ROOT/tsfm/bin/python" -m pip install \
  'numpy==1.26.4' 'torch==2.7.1' 'accelerate==1.6.0' \
  'chronos-forecasting==1.5.3' 'huggingface-hub==0.36.2' \
  'safetensors==0.8.0' 'transformers==4.51.3' 'timesfm==3.0.2' 'psutil==7.2.2'

"$ENV_ROOT/sundial/bin/python" -m pip install \
  'numpy==1.26.4' 'torch==2.7.1' 'huggingface-hub==0.36.2' \
  'safetensors==0.8.0' 'transformers==4.40.1'

"$ENV_ROOT/moirai/bin/python" -m pip install \
  'numpy==1.26.4' 'pandas==2.1.4' 'scipy==1.11.4' 'torch==2.4.1' \
  'nvidia-nvjitlink-cu12==12.1.105' 'einops==0.7.0' 'gluonts==0.14.4' \
  'lightning==2.6.6' 'datasets==2.17.1' 'jax==0.6.1' 'jaxlib==0.6.1' \
  'uni2ts==1.2.0' 'huggingface-hub==1.32.0' 'safetensors==0.8.0'

cd "$ROOT"
"$ENV_ROOT/core/bin/python" scripts/build_cuda_model_profiles.py --root "$ROOT"
"$ENV_ROOT/core/bin/python" -m pytest -q

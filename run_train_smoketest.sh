#!/usr/bin/env bash
set -euo pipefail

# Helper to run a short smoke-test of the training pipeline
# Usage: ./run_train_smoketest.sh [--recordings_basedir <path>] [additional train.py args]

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OPENPILOT_PIPELINE="$REPO_ROOT/openpilot-pipeline"
VENV_PY="$REPO_ROOT/.venv/bin/python3"

RECORDINGS_BASEDIR="$REPO_ROOT/DATA/syagai_1st/larning_data"

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recordings_basedir)
      RECORDINGS_BASEDIR="$2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

export PYTHONPATH="$OPENPILOT_PIPELINE"
# Set dummy wandb env so train.py doesn't KeyError; we still pass --no_wandb for offline
export WANDB_ENTITY="local"
export WANDB_PROJECT="local"

# Ensure expected model path exists under openpilot-pipeline/common/models
mkdir -p "$OPENPILOT_PIPELINE/common/models"
if [ ! -f "$OPENPILOT_PIPELINE/common/models/supercombo.onnx" ]; then
  if [ -f "$REPO_ROOT/base/supercombo.onnx" ]; then
    cp "$REPO_ROOT/base/supercombo.onnx" "$OPENPILOT_PIPELINE/common/models/supercombo.onnx"
    echo "Copied base/supercombo.onnx -> openpilot-pipeline/common/models/supercombo.onnx"
  else
    echo "Warning: supercombo.onnx not found in base/; train may fail if model missing"
  fi
fi

echo "Running smoke-test train.py with recordings_basedir=$RECORDINGS_BASEDIR"
"$VENV_PY" "$OPENPILOT_PIPELINE/train/train.py" --recordings_basedir "$RECORDINGS_BASEDIR" --batch_size 1 --epochs 1 --seq_len 100 --no_wandb --date_it smoke_test "${EXTRA_ARGS[@]:-}"
#!/usr/bin/env bash
set -euo pipefail

# Helper to run a short smoke-test of the training pipeline
# Usage: ./run_train_smoketest.sh [--recordings_basedir <path>] [additional train.py args]

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OPENPILOT_PIPELINE="$REPO_ROOT/openpilot-pipeline"
VENV_PY="$REPO_ROOT/.venv/bin/python3"

RECORDINGS_BASEDIR="$REPO_ROOT/DATA/syagai_1st/larning_data"

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recordings_basedir)
      RECORDINGS_BASEDIR="$2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

export PYTHONPATH="$OPENPILOT_PIPELINE"
# Set dummy wandb env so train.py doesn't KeyError; we still pass --no_wandb for offline
export WANDB_ENTITY="local"
export WANDB_PROJECT="local"

# Ensure expected model path exists under openpilot-pipeline/common/models
mkdir -p "$OPENPILOT_PIPELINE/common/models"
if [ ! -f "$OPENPILOT_PIPELINE/common/models/supercombo.onnx" ]; then
  if [ -f "$REPO_ROOT/base/supercombo.onnx" ]; then
    cp "$REPO_ROOT/base/supercombo.onnx" "$OPENPILOT_PIPELINE/common/models/supercombo.onnx"
    echo "Copied base/supercombo.onnx -> openpilot-pipeline/common/models/supercombo.onnx"
  else
    echo "Warning: supercombo.onnx not found in base/; train may fail if model missing"
  fi
fi

echo "Running smoke-test train.py with recordings_basedir=$RECORDINGS_BASEDIR"
"$VENV_PY" "$OPENPILOT_PIPELINE/train/train.py" --recordings_basedir "$RECORDINGS_BASEDIR" --batch_size 1 --epochs 1 --seq_len 100 --no_wandb --date_it smoke_test "${EXTRA_ARGS[@]:-}"
#!/usr/bin/env bash
set -euo pipefail

# Helper to run a short smoke-test of the training pipeline
# Usage: ./run_train_smoketest.sh [--recordings_basedir <path>] [additional train.py args]

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OPENPILOT_PIPELINE="$REPO_ROOT/openpilot-pipeline"
VENV_PY="$REPO_ROOT/.venv/bin/python3"

RECORDINGS_BASEDIR="$REPO_ROOT/DATA/syagai_1st/larning_data"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recordings_basedir)
      RECORDINGS_BASEDIR="$2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

export PYTHONPATH="$OPENPILOT_PIPELINE"
# Set dummy wandb env so train.py doesn't KeyError; we still pass --no_wandb for offline
export WANDB_ENTITY="local"
export WANDB_PROJECT="local"

# Ensure expected model path exists under openpilot-pipeline/common/models
mkdir -p "$OPENPILOT_PIPELINE/common/models"
if [ ! -f "$OPENPILOT_PIPELINE/common/models/supercombo.onnx" ]; then
  if [ -f "$REPO_ROOT/base/supercombo.onnx" ]; then
    cp "$REPO_ROOT/base/supercombo.onnx" "$OPENPILOT_PIPELINE/common/models/supercombo.onnx"
    echo "Copied base/supercombo.onnx -> openpilot-pipeline/common/models/supercombo.onnx"
  else
    echo "Warning: supercombo.onnx not found in base/; train may fail if model missing"
  fi
fi

echo "Running smoke-test train.py with recordings_basedir=$RECORDINGS_BASEDIR"
"$VENV_PY" "$OPENPILOT_PIPELINE/train/train.py" --recordings_basedir "$RECORDINGS_BASEDIR" --batch_size 1 --epochs 1 --seq_len 100 --no_wandb --date_it smoke_test "${EXTRA_ARGS[@]:-}"

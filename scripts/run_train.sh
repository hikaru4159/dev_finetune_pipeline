#!/usr/bin/env bash
set -euo pipefail

# run_train.sh
# Convenience wrapper to run the training script with the same environment
# that succeeded during investigation. It sets PYTHONPATH, dummy WANDB env
# variables, ensures the ONNX model is available at the expected path, and
# launches the training script under the project's venv.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPILOT_PIPELINE="$REPO_ROOT/openpilot-pipeline"
VENV_PY="$REPO_ROOT/.venv/bin/python3"

# defaults
RECORDINGS_BASEDIR="$REPO_ROOT/DATA/syagai_1st/larning_data"
EPOCHS=1
BATCH_SIZE=1
SEQ_LEN=100
DATE_IT="smoke_test"

print_usage(){
  cat <<EOF
Usage: $0 [--recordings_basedir PATH] [--epochs N] [--batch_size N] [--seq_len N] [--date_it NAME] [--extra args...]

Runs train.py with safe defaults used during the recent successful run.
EOF
}

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recordings_basedir) RECORDINGS_BASEDIR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --seq_len) SEQ_LEN="$2"; shift 2;;
    --date_it) DATE_IT="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

export PYTHONPATH="$OPENPILOT_PIPELINE"
# avoid KeyError in train.py
export WANDB_ENTITY="local"
export WANDB_PROJECT="local"

# ensure expected model path
mkdir -p "$OPENPILOT_PIPELINE/common/models"
if [ ! -f "$OPENPILOT_PIPELINE/common/models/supercombo.onnx" ]; then
  if [ -f "$REPO_ROOT/base/supercombo.onnx" ]; then
    cp "$REPO_ROOT/base/supercombo.onnx" "$OPENPILOT_PIPELINE/common/models/supercombo.onnx"
    echo "Copied supercombo.onnx to expected path"
  else
    echo "Warning: base/supercombo.onnx not found; train may fail"
  fi
fi

echo "Launching training: epochs=$EPOCHS batch_size=$BATCH_SIZE seq_len=$SEQ_LEN recordings=$RECORDINGS_BASEDIR"
"$VENV_PY" "$OPENPILOT_PIPELINE/train/train.py" \
  --recordings_basedir "$RECORDINGS_BASEDIR" \
  --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --seq_len "$SEQ_LEN" --no_wandb --date_it "$DATE_IT" "${EXTRA_ARGS[@]:-}"

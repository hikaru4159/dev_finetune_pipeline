#!/bin/bash
# Lightweight wrapper to run only the dataset conversion step with simplified flags.
# By default this script prints the command (dry-run). Use --run to execute.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
DATA_DIR_DEFAULT="$SCRIPT_DIR/DATA/syagai_1st/larning_data"
OUT_ROOT_DEFAULT="/tmp/supercombo_dataset_larning_all"
JOBS=3
DRY_RUN=1

usage(){
  cat <<EOF
Usage: $0 [--run] [--data DIR] [--out DIR] [--python PY] [--jobs N]

Options:
  --run         Actually execute the conversion (default: dry-run)
  --data DIR    Data directory (default: $DATA_DIR_DEFAULT)
  --out DIR     Output root (default: $OUT_ROOT_DEFAULT)
  --python PY   Python executable to use (default: $VENV_PYTHON)
  --jobs N      Number of parallel jobs (default: $JOBS)
  -h, --help    Show this help
EOF
}

# parse args
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run) DRY_RUN=0; shift ;;
    --data) DATA_DIR="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --python) VENV_PYTHON="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

DATA_DIR=${DATA_DIR:-$DATA_DIR_DEFAULT}
OUT_ROOT=${OUT_ROOT:-$OUT_ROOT_DEFAULT}

CMD=("$VENV_PYTHON" "$SCRIPT_DIR/supercombo_dataset_package/batch_run_processing.py" \
     --data-dir "$DATA_DIR" --out-root "$OUT_ROOT" --python "$VENV_PYTHON" --jobs "$JOBS")

echo "[DRY_RUN=${DRY_RUN}] Command:" \
     "${CMD[@]}"

if [[ $DRY_RUN -eq 0 ]]; then
  echo "Running..."
  exec "${CMD[@]}"
else
  echo "Dry run. Add --run to execute."
fi

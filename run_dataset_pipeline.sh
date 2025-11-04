#!/bin/bash
# Simple launcher script for dataset processing and desire completion
# Usage: ./run_dataset_pipeline.sh [OPTIONS]
#
# This script ensures proper Python environment, handles path setup,
# and provides simple commands for common operations.

set -e  # Exit on any error

# Configuration - adjust these paths as needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
DATA_DIR="$PROJECT_ROOT/DATA/syagai_1st/larning_data"
OUTPUT_ROOT="/tmp/supercombo_dataset_larning_all"
JOBS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Check dependencies
check_deps() {
    log_info "Checking dependencies..."
    
    # Check Python venv
    if [[ ! -f "$VENV_PYTHON" ]]; then
        log_error "Virtual environment Python not found: $VENV_PYTHON"
        log_info "Please ensure .venv is set up with required packages"
        exit 1
    fi
    
    # Check data directory
    if [[ ! -d "$DATA_DIR" ]]; then
        log_error "Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Test Python environment
    if ! "$VENV_PYTHON" -c "import cv2, numpy, zstandard" &>/dev/null; then
        log_error "Missing required Python packages (cv2, numpy, zstandard)"
        log_info "Install with: $VENV_PYTHON -m pip install opencv-python numpy zstandard"
        exit 1
    fi
    
    log_success "Dependencies OK"
}

# Clean previous outputs
clean_outputs() {
    log_info "Cleaning previous outputs..."
    if [[ -d "$OUTPUT_ROOT" ]]; then
        rm -rf "$OUTPUT_ROOT"
        log_success "Cleaned $OUTPUT_ROOT"
    else
        log_info "Output directory does not exist, nothing to clean"
    fi
}

# Run full dataset conversion
run_conversion() {
    log_info "Starting dataset conversion..."
    log_info "Data dir: $DATA_DIR"
    log_info "Output dir: $OUTPUT_ROOT"
    log_info "Jobs: $JOBS"
    
    cd "$PROJECT_ROOT"
    
    "$VENV_PYTHON" supercombo_dataset_package/batch_run_processing.py \
        --data-dir "$DATA_DIR" \
        --out-root "$OUTPUT_ROOT" \
        --python "$VENV_PYTHON" \
        --jobs "$JOBS"
    
    if [[ $? -eq 0 ]]; then
        log_success "Dataset conversion completed"
    else
        log_error "Dataset conversion failed"
        exit 1
    fi
}

# Supplement desire for specific runs
supplement_desire() {
    local run_name="$1"
    if [[ -z "$run_name" ]]; then
        log_error "Usage: supplement_desire <run_name>"
        exit 1
    fi
    
    local run_path="$DATA_DIR/$run_name"
    local out_path="$OUTPUT_ROOT/$run_name"
    
    if [[ ! -d "$run_path" ]]; then
        log_error "Run directory not found: $run_path"
        exit 1
    fi
    
    log_info "Supplementing desire for run: $run_name"
    
    cd "$PROJECT_ROOT"
    "$VENV_PYTHON" supercombo_dataset_package/pipeline_adapter/generate_synthetic_labels.py \
        --run "$run_path" \
        --out "$out_path"
    
    if [[ $? -eq 0 ]]; then
        log_success "Desire supplementation completed for $run_name"
    else
        log_error "Desire supplementation failed for $run_name"
        exit 1
    fi
}

# Check status of processed runs
check_status() {
    log_info "Checking dataset status..."
    
    if [[ ! -d "$OUTPUT_ROOT" ]]; then
        log_warn "Output directory does not exist: $OUTPUT_ROOT"
        return
    fi
    
    cd "$PROJECT_ROOT"
    "$VENV_PYTHON" - << EOF "$OUTPUT_ROOT"
import os, glob
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/supercombo_dataset_larning_all"

print(f"=== Dataset Status Report ===")
print(f"Output root: {root}")

if not os.path.exists(root):
    print("❌ Output directory does not exist")
    sys.exit(1)

dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
if not dirs:
    print("❌ No run directories found")
    sys.exit(1)

print(f"📁 Found {len(dirs)} run directories")

required_files = [
    'input_imgs.npy', 'big_input_imgs.npy', 'desire.npy', 
    'traffic_convention.npy', 'lateral_control_params.npy', 
    'prev_desired_curv.npy', 'nav_features.npy', 
    'nav_instructions.npy', 'features_buffer.npy'
]

complete_runs = []
incomplete_runs = []
missing_desire = []

for run_dir in sorted(dirs):
    run_path = os.path.join(root, run_dir)
    missing = []
    
    for req_file in required_files:
        file_path = os.path.join(run_path, req_file)
        if not os.path.exists(file_path):
            missing.append(req_file)
    
    if not missing:
        complete_runs.append(run_dir)
    else:
        incomplete_runs.append((run_dir, missing))
        if 'desire.npy' in missing:
            missing_desire.append(run_dir)

print(f"\\n✅ Complete runs: {len(complete_runs)}")
for run in complete_runs[:5]:  # Show first 5
    print(f"   ✓ {run}")
if len(complete_runs) > 5:
    print(f"   ... and {len(complete_runs) - 5} more")

if incomplete_runs:
    print(f"\\n❌ Incomplete runs: {len(incomplete_runs)}")
    for run, missing in incomplete_runs[:3]:  # Show first 3
        print(f"   ✗ {run} (missing: {', '.join(missing)})")
    if len(incomplete_runs) > 3:
        print(f"   ... and {len(incomplete_runs) - 3} more")

if missing_desire:
    print(f"\\n🔍 Runs missing desire.npy: {len(missing_desire)}")
    for run in missing_desire:
        print(f"   - {run}")

print(f"\\n📊 Summary: {len(complete_runs)}/{len(dirs)} runs complete")
EOF
}

# Show usage
show_usage() {
    echo "Dataset Processing Pipeline Launcher"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  check-deps                    Check dependencies and environment"
    echo "  clean                        Clean previous outputs"
    echo "  convert                      Run full dataset conversion"
    echo "  full                         Clean + convert (complete pipeline)"
    echo "  status                       Check status of processed runs"
    echo "  supplement <run_name>        Supplement desire for specific run"
    echo "  help                         Show this help"
    echo ""
    echo "Configuration:"
    echo "  Data dir:    $DATA_DIR"
    echo "  Output dir:  $OUTPUT_ROOT"
    echo "  Python:      $VENV_PYTHON"
    echo "  Jobs:        $JOBS"
    echo ""
    echo "Examples:"
    echo "  $0 full                                    # Complete pipeline"
    echo "  $0 convert                                 # Just conversion"
    echo "  $0 supplement 00000003--06297c7620--0     # Fix specific run"
    echo "  $0 status                                  # Check results"
}

# Main execution
main() {
    case "${1:-}" in
        "check-deps")
            check_deps
            ;;
        "clean")
            clean_outputs
            ;;
        "convert")
            check_deps
            run_conversion
            ;;
        "full")
            check_deps
            clean_outputs
            run_conversion
            check_status
            ;;
        "status")
            check_status
            ;;
        "supplement")
            check_deps
            supplement_desire "$2"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "")
            log_error "No command specified"
            show_usage
            exit 1
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
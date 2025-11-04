#!/usr/bin/env python3
"""
Batch process runs under a DATA directory to generate supercombo dataset files (including features_buffer)
using the existing integration runner.

Usage:
  python batch_run_processing.py --data-dir DATA/syagai_1st --out-root /tmp/supercombo_dataset_with_drivingvision_all 

This script will iterate over immediate subdirectories of the data directory and invoke
openpilot-pipeline/integrate_with_supercombo_adapter.py for each run.

It respects the environment variable DRIVING_VISION_ONNX; set it before running to enable features_buffer generation.
"""
import os
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
INTEGRATION_RUNNER = WORKSPACE_ROOT / 'openpilot-pipeline' / 'integrate_with_supercombo_adapter.py'


def list_runs(data_dir):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"data_dir {data_dir} does not exist")
    runs = [x for x in sorted(p.iterdir()) if x.is_dir()]
    return runs


def _run_single(r, out_root, python_exe=None, extra_env=None):
    python_exe = python_exe or sys.executable
    out_dir = Path(out_root) / r.name
    cmd = [python_exe, str(INTEGRATION_RUNNER), '--run-dir', str(r), '--out-dir', str(out_dir)]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print('\n=== Processing run:', r)
    print('CMD:', ' '.join(cmd))
    p = subprocess.run(cmd, env=env)
    return (r, p.returncode, out_dir)


def run_for_each(runs, out_root, python_exe=None, extra_env=None, jobs=1):
    os.makedirs(out_root, exist_ok=True)
    results = []
    if jobs is None or jobs <= 1:
        for r in runs:
            results.append(_run_single(r, out_root, python_exe=python_exe, extra_env=extra_env))
        return results

    # parallel execution
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_run_single, r, out_root, python_exe, extra_env): r for r in runs}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                r = futures.get(fut)
                print(f"Run {r} raised: {e}")
                results.append((r, -1, Path(out_root) / r.name))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='DATA/syagai_1st', help='Top-level data directory containing runs')
    p.add_argument('--out-root', default='/tmp/supercombo_dataset_with_drivingvision_all', help='Root output directory')
    p.add_argument('--python', default=None, help='Python executable to run the integration script (default: current)')
    p.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs to run (default 1)')
    args = p.parse_args()

    runs = list_runs(args.data_dir)
    print('Found', len(runs), 'runs in', args.data_dir)
    results = run_for_each(runs, args.out_root, python_exe=args.python, jobs=args.jobs)

    print('\nSummary:')
    for r, rc, out in results:
        print(f"{r.name}: rc={rc} -> {out}")


if __name__ == '__main__':
    main()

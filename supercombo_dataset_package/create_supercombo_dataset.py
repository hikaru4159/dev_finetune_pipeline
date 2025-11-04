#!/usr/bin/env python3
"""
Batch wrapper to create supercombo-style datasets from a root DATA folder.

This script re-uses the existing functions in
`supercombo_dataset_package/transform_to_supercombo_dataset_gui.py` with minimal
changes. It finds log directories under a given root and writes per-run
numpy datasets to an output root.

Usage:
  python create_supercombo_dataset.py --data-root /path/to/DATA/syagai_1st --out-root /tmp/supercombo_dataset

"""
import os
import argparse
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from transform_to_supercombo_dataset_gui import find_log_dirs, main as transform_main


def run_batch(data_root, out_root, max_workers=None):
    # transform_main expects args with log_dir and out_dir; but it is written to
    # walk a given log_dir and find rlog files. We'll call it once per data_root
    # parent to let it find contained runs.
    class Args:
        def __init__(self, log_dir, out_dir):
            self.log_dir = log_dir
            self.out_dir = out_dir

    # If the provided data_root directly contains runs (e.g., 2023-.. dirs), call main
    args = Args(data_root, out_root)
    print(f"[INFO] Launching transform for data_root={data_root}, out_root={out_root}")
    transform_main(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True, help='Root folder containing run folders (e.g. DATA/syagai_1st)')
    p.add_argument('--out-root', required=True, help='Where to write per-run numpy dataset folders')
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    run_batch(data_root, out_root)


if __name__ == '__main__':
    main()

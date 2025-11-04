#!/usr/bin/env python3
"""
Small wrapper to run the minimal adapter training on one or more npy run folders.

Usage:
  python train_adapter.py --npy-folders /tmp/supercombo_dataset_with_drivingvision_all/2023-11-22--06-10-53--21 --epochs 1

This script reuses `delta_finetune_harness.train_adapter`.
"""
import argparse
import os
import sys
from pathlib import Path

# add package path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from delta_finetune_harness import train_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npy-folders', nargs='+', required=True, help='One or more run folders produced by the dataset step')
    p.add_argument('--onnx', required=True, help='Path to supercombo.onnx (not used directly in minimal example but kept for API parity)')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    for f in args.npy_folders:
        if not os.path.isdir(f):
            print('skip non-dir', f)
            continue
        print('Training adapter on', f)
        train_adapter(args.onnx, f, epochs=args.epochs, device=args.device)


if __name__ == '__main__':
    main()

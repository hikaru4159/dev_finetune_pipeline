#!/usr/bin/env python3
"""
Create sliding-window samples from per-run features_buffer.npy and desire.npy.

Produces a directory of .npz files per-run (or a combined npz) containing:
  X: (N, D)  # averaged features per window
  Y: (N, C)  # desire one-hot at window end (averaged or argmax)

Usage:
  python create_windowed_dataset.py --data-root /tmp/supercombo_dataset_with_drivingvision_all --out /tmp/windowed_samples --window 5 --stride 1
"""
import argparse
import os
from pathlib import Path
import numpy as np


def process_run(run_folder, window=5, stride=1, out_dir=None):
    run_folder = Path(run_folder)
    fb_p = run_folder / 'features_buffer.npy'
    des_p = run_folder / 'desire.npy'
    if not fb_p.exists() or not des_p.exists():
        return 0
    fb = np.load(fb_p)  # (1, T, D)
    des = np.load(des_p)  # (1, 100, C)
    fb = fb[0]  # (T, D)
    des = des[0]  # (100, C)

    T = fb.shape[0]
    samples_X = []
    samples_Y = []
    for start in range(0, T - window + 1, stride):
        end = start + window
        x_win = np.mean(fb[start:end], axis=0)  # (D,)
        # map Y: choose desire at the window end index relative to 100 frames
        # align end to last 100 frames mapping by taking min(end, des.shape[0]-1)
        y_idx = min(end-1, des.shape[0]-1)
        y = des[y_idx]
        samples_X.append(x_win)
        samples_Y.append(y)

    if not samples_X:
        return 0

    X = np.stack(samples_X)
    Y = np.stack(samples_Y)
    if out_dir is None:
        out_dir = run_folder
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / (run_folder.name + '_samples.npz')
    np.savez_compressed(out_p, X=X, Y=Y)
    return X.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()

    runs = sorted([p for p in Path(args.data_root).iterdir() if p.is_dir()])
    total = 0
    for r in runs:
        n = process_run(r, window=args.window, stride=args.stride, out_dir=args.out)
        print(f"{r.name}: {n} samples")
        total += n
    print('Total samples:', total)


if __name__ == '__main__':
    main()

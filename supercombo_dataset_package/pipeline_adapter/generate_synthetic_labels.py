#!/usr/bin/env python3
"""
Wrapper to generate missing labels (desire, prev_desired_curv surrogate) for runs.

It re-uses the existing `desire_analysis` scripts in this workspace to infer
desire labels from images or logs when the original log lacks those events.

Usage:
  python generate_synthetic_labels.py --run /path/to/run --out /path/to/out

The script will attempt to:
- extract images from camera hevc if needed
- run desire_from_images.extract_desire_from_images or desire_from_log_features
  to produce `desire.npy` compatible with supercombo expectations
"""
import os
import argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
sys.path.insert(0, ROOT)

from desire_analysis import desire_from_images
# delay import of desire_from_log_features because it loads capnp schemas at import time


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def generate(run_dir, out_dir):
    ensure_dir(out_dir)
    # try log-based inference first
    try:
        print(f"[INFO] Attempting desire inference from log for {run_dir}")
        # Import here to avoid capnp schema load at module import time
        from desire_analysis.desire_from_log_features import extract_log_features_with_time, infer_desire_from_features
        rlog = os.path.join(run_dir, 'rlog')
        if not os.path.exists(rlog) and os.path.exists(rlog + '.zst'):
            rlog = rlog + '.zst'
        features, times = extract_log_features_with_time(rlog)
        labels = infer_desire_from_features(features)
        # labels is array of strings; convert to one-hot using DESIRE_LABELS from module
        from desire_analysis.desire_from_images import DESIRE_LABELS
        label_to_idx = {n: i for i, n in enumerate(DESIRE_LABELS)}
        onehot = np.zeros((1, labels.shape[0], len(DESIRE_LABELS)), dtype=np.float32)
        for i, lab in enumerate(labels):
            idx = label_to_idx.get(lab, 0)
            onehot[0, i, idx] = 1.0
        # pad/truncate to 100 frames to match supercombo
        if onehot.shape[1] < 100:
            pad = np.zeros((1, 100 - onehot.shape[1], onehot.shape[2]), dtype=np.float32)
            onehot = np.concatenate([pad, onehot], axis=1)
        else:
            onehot = onehot[:, -100:, :]
        np.save(os.path.join(out_dir, 'desire.npy'), onehot)
        print('[INFO] desire.npy written (from log)')
        return True
    except Exception as e:
        print(f"[WARN] log-based desire inference failed: {e}")

    # fallback: image based
    try:
        print(f"[INFO] Attempting desire inference from images for {run_dir}")
        # need to extract images using desire_from_images.extract_images_from_hevc
        # Prefer standard/front camera images (fcamera) over wide/big images (ecamera)
        # to avoid using big_input_imgs for desire inference.
        hevc_candidates = ['fcamera.hevc', 'ecamera.hevc']
        hevc_path = None
        for f in hevc_candidates:
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                hevc_path = p
                break
        if not hevc_path:
            raise FileNotFoundError('no hevc file found for images')
        tmp_img_dir = os.path.join(out_dir, '_tmp_imgs')
        ensure_dir(tmp_img_dir)
        desire_from_images.extract_images_from_hevc(hevc_path, tmp_img_dir, fps=10)
        arr = desire_from_images.extract_desire_from_images(tmp_img_dir, n_frames=100)
        np.save(os.path.join(out_dir, 'desire.npy'), arr)
        print('[INFO] desire.npy written (from images)')
        return True
    except Exception as e:
        print(f"[ERROR] image-based desire inference failed: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    ok = generate(args.run, args.out)
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

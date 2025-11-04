#!/usr/bin/env python3
"""
Evaluate adapter_head.pt on run folders using features_buffer.npy and desire.npy.
Saves a CSV summary to /tmp/adapter_eval_summary.csv

Usage:
  python eval_adapters.py --runs /tmp/supercombo_dataset_with_drivingvision_all/*
"""
import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class AdapterHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def evaluate_one(run_folder):
    run_folder = Path(run_folder)
    fb_p = run_folder / 'features_buffer.npy'
    des_p = run_folder / 'desire.npy'
    adapter_p = run_folder / 'adapter_head.pt'
    if not fb_p.exists() or not des_p.exists() or not adapter_p.exists():
        return None

    fb = np.load(fb_p)  # (1, T, 512)
    des = np.load(des_p)  # (1, 100, C)
    # collapse time dim in same way as training (mean)
    X = np.mean(fb, axis=1)  # (1, 512)
    Y = np.mean(des, axis=1)  # (1, C)

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    in_dim = X_t.shape[1]
    out_dim = Y_t.shape[1]
    model = AdapterHead(in_dim, out_dim)
    model.load_state_dict(torch.load(adapter_p, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        logits = model(X_t)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, Y_t).item()

        # compute top1 acc: compare argmax of logits vs argmax of Y_t
        pred = torch.argmax(torch.sigmoid(logits), dim=1)
        true = torch.argmax(Y_t, dim=1)
        acc = (pred == true).float().mean().item()

    return {
        'run': run_folder.name,
        'loss': loss,
        'top1': acc,
        'adapter': str(adapter_p),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', default='/tmp/adapter_eval_summary.csv')
    args = p.parse_args()

    rows = []
    for r in args.runs:
        res = evaluate_one(r)
        if res is None:
            print('skip', r)
            continue
        print(f"{res['run']}: loss={res['loss']:.6f} top1={res['top1']:.3f}")
        rows.append(res)

    if rows:
        # save CSV
        import csv
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run', 'loss', 'top1', 'adapter'])
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print('Wrote summary to', args.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Train an adapter using precomputed windowed samples (.npz files).

Each .npz should contain X: (N,D) and Y: (N,C)

Usage:
  python train_samples_adapter.py --samples /tmp/windowed_samples/*.npz --epochs 5 --batch-size 4
"""
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


class SamplesDataset(Dataset):
    def __init__(self, patterns):
        files = []
        for p in patterns:
            files.extend(sorted(glob.glob(p)))
        self.X = []
        self.Y = []
        for f in files:
            d = np.load(f)
            self.X.append(d['X'])
            self.Y.append(d['Y'])
        if not self.X:
            raise RuntimeError('No sample files found')
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.Y[idx].astype(np.float32)


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


def train(samples_patterns, epochs=5, batch_size=4, lr=1e-3, device='cpu', out_dir='/tmp/samples_adapter_ckpt'):
    ds = SamplesDataset(samples_patterns)
    n = len(ds)
    # simple split
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    sample_x, sample_y = ds[0]
    in_dim = sample_x.shape[0]
    out_dim = sample_y.shape[0]

    model = AdapterHead(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(out_dir, exist_ok=True)
    best_val = float('inf')
    best_path = None

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        count = 0
        for Xb, Yb in train_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, Yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
            count += Xb.size(0)
        train_loss = total_loss / max(1, count)

        # val
        model.eval()
        vloss = 0.0
        vcount = 0
        vacc = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                logits = model(Xb)
                loss = loss_fn(logits, Yb)
                vloss += loss.item() * Xb.size(0)
                preds = torch.argmax(torch.sigmoid(logits), dim=1)
                trues = torch.argmax(Yb, dim=1)
                vacc += (preds == trues).float().sum().item()
                vcount += Xb.size(0)
        val_loss = vloss / max(1, vcount)
        val_acc = vacc / max(1, vcount)
        print(f"epoch {ep}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(out_dir, f'best_samples_adapter_ep{ep}.pt')
            torch.save(model.state_dict(), best_path)

    print('Best val loss', best_val, 'saved at', best_path)
    return best_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--out-dir', default='/tmp/samples_adapter_ckpt')
    args = p.parse_args()
    train(args.samples, epochs=args.epochs, batch_size=args.batch_size, out_dir=args.out_dir)

#!/usr/bin/env python3
"""
Train adapter with multiple runs mixed in minibatches.

Usage example:
  python train_multi_adapter.py --data-root /tmp/supercombo_dataset_with_drivingvision_all --epochs 5 --batch-size 2

Notes:
- This is a minimal, CPU-friendly trainer for quick experimentation.
- Each run folder is treated as one training sample (features_buffer averaged over time),
  so with many runs you'd get proper batches; here we have a small number of runs.
"""
import argparse
import os
from pathlib import Path
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class RunsDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.runs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        p = self.runs[idx]
        fb_p = p / 'features_buffer.npy'
        des_p = p / 'desire.npy'
        fb = np.load(fb_p)  # (1,T,D)
        des = np.load(des_p)  # (1,100,C)
        X = np.mean(fb, axis=1).astype(np.float32)  # (1,D)
        # convert desire (one-hot over time) to single class label via argmax over channel
        # des shape is (1,100,C) -> take mean over time -> (1,C) -> argmax -> scalar index
        try:
            probs = np.mean(des, axis=1)  # (1,C)
            label = int(np.argmax(probs, axis=1)[0])
        except Exception:
            # fallback: if shape unexpected, force class 0
            label = 0
        return {'run': p.name, 'X': X[0], 'Y': label, 'path': str(p)}


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


def collate_fn(batch):
    X = torch.from_numpy(np.stack([b['X'] for b in batch])).float()
    Y = torch.tensor([b['Y'] for b in batch], dtype=torch.long)
    runs = [b['run'] for b in batch]
    paths = [b['path'] for b in batch]
    return {'X': X, 'Y': Y, 'runs': runs, 'paths': paths}


def train(data_root, epochs=5, batch_size=2, lr=1e-3, device='cpu', out_dir='/tmp/multi_adapter_ckpt'):
    ds = RunsDataset(data_root)
    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    if len(train_ds) == 0:
        raise RuntimeError('Not enough runs for training; need at least 1 training sample')

    # build dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # sample one to get dims
    sample = ds[0]
    in_dim = sample['X'].shape[0]
    # out_dim is number of classes. infer from any desire.npy in dataset
    # find first desire.npy and read shape
    out_dim = None
    for p in ds.runs:
        try:
            des = np.load(p / 'desire.npy')
            if des.ndim == 3:
                out_dim = des.shape[2]
            else:
                out_dim = des.shape[-1]
            break
        except Exception:
            continue
    if out_dim is None:
        raise RuntimeError('Could not determine number of classes (no valid desire.npy found)')

    model = AdapterHead(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # compute class frequencies across the full dataset and build weights (inverse freq)
    counts = np.zeros(out_dim, dtype=np.float64)
    for p in ds.runs:
        try:
            des = np.load(p / 'desire.npy')
            if des.ndim == 3:
                am = np.argmax(np.mean(des, axis=1), axis=1)
            else:
                am = np.argmax(np.mean(des, axis=0), axis=0)
            for a in am.ravel():
                if 0 <= a < out_dim:
                    counts[int(a)] += 1
        except Exception:
            continue
    # avoid division by zero
    eps = 1e-6
    inv = 1.0 / (counts + eps)
    # normalize weights to have mean 1
    weights = inv / np.mean(inv)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    os.makedirs(out_dir, exist_ok=True)
    best_val = float('inf')
    best_path = None

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)  # long labels
            opt.zero_grad()
            logits = model(X)  # (B, out_dim)
            loss = loss_fn(logits, Y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * X.size(0)
            count += X.size(0)
        train_loss = total_loss / max(1, count)

        # validation
        model.eval()
        vloss = 0.0
        vcount = 0
        vacc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                logits = model(X)
                loss = loss_fn(logits, Y)
                vloss += loss.item() * X.size(0)
                preds = torch.argmax(logits, dim=1)
                trues = Y
                vacc += (preds == trues).float().sum().item()
                vcount += X.size(0)
        val_loss = vloss / max(1, vcount)
        val_acc = vacc / max(1, vcount)

        print(f"epoch {ep}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.3f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(out_dir, f'best_adapter_ep{ep}.pt')
            torch.save(model.state_dict(), best_path)

    print('Best val loss', best_val, 'saved at', best_path)
    return best_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out-dir', default='/tmp/multi_adapter_ckpt')
    args = p.parse_args()
    train(args.data_root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, out_dir=args.out_dir)

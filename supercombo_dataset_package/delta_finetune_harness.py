"""
Minimal delta-finetune harness (PyTorch) that keeps original ONNX model intact.

Approach:
- Use ONNX Runtime to run the base `supercombo.onnx` as a frozen feature extractor.
- Train a small PyTorch adapter (linear layers) on top of the ONNX outputs to predict the desired policy outputs.
- This script is a minimal example and must be adapted to your exact training loop / loss.

Note: This keeps the ONNX model file unchanged for reuse with comma.ai stacks.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import onnxruntime as ort
except Exception:
    ort = None


class NpyDataset(Dataset):
    def __init__(self, folder):
        # expects folder containing npy files produced by transform script
        self.folder = folder
        self.examples = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        # simple grouping by run: we expect files like input_imgs.npy etc in same folder
        # for now assume folder is the single-run folder (one example)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # load known names
        base = self.folder
        def load(name):
            p = os.path.join(base, name + '.npy')
            return np.load(p) if os.path.exists(p) else None
        input_imgs = load('input_imgs')
        big_input_imgs = load('big_input_imgs')
        features_buffer = load('features_buffer')
        desire = load('desire')
        return {
            'input_imgs': input_imgs,
            'big_input_imgs': big_input_imgs,
            'features_buffer': features_buffer,
            'desire': desire,
        }


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


def run_onnx_session(onnx_path, inputs_dict):
    if ort is None:
        raise RuntimeError('onnxruntime not installed')
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    out = sess.run(None, inputs_dict)
    return out


def train_adapter(onnx_path, npy_run_folder, epochs=3, lr=1e-3, device='cpu'):
    ds = NpyDataset(npy_run_folder)
    sample = ds[0]
    # For simplicity, use features_buffer as input to adapter if available
    fb = sample['features_buffer']
    if fb is None or fb.size == 0:
        raise RuntimeError('features_buffer missing in dataset; cannot train adapter in this minimal example')
    # features_buffer shape: (1, T, 512) -> collapse time dim by mean
    X = torch.from_numpy(np.mean(fb, axis=1)).float().to(device)  # (1, 512)
    y = sample['desire']
    if y is None:
        raise RuntimeError('desire missing; cannot train adapter')
    # collapse desire to average one-hot
    Y = torch.from_numpy(np.mean(y, axis=1)).float().to(device)  # (1, 8)

    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    model = AdapterHead(in_dim, out_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optim.step()
        print(f"epoch {ep+1}/{epochs} loss={loss.item():.6f}")

    # save adapter weights
    torch.save(model.state_dict(), os.path.join(npy_run_folder, 'adapter_head.pt'))
    print(f"Adapter saved to {os.path.join(npy_run_folder, 'adapter_head.pt')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--npy-run-folder', required=True, help='Folder with npy files for one run')
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    train_adapter(args.onnx, args.npy_run_folder, epochs=args.epochs)


if __name__ == '__main__':
    main()

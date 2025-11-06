#!/usr/bin/env python3
"""
Train adapter with temporal sequences (frame-by-frame learning).

改善点（本格的PyTorch学習環境）:
- 時系列を保持: 各フレームを独立したサンプルとして学習（2run × 99frames = 198サンプル）
- 完全ONNX互換: 残差層を含む完全なtemporal_policy.temporal_hydra構造
- GPU対応: CUDA自動検出、Mixed Precision Training (AMP)
- TensorBoard: 学習経過の可視化
- Data Augmentation: ドロップアウト、ノイズ注入
- 正則化: L2正則化、Early Stopping、重み付きCrossEntropy、Gradient Clipping
- 学習率スケジューリング: CosineAnnealingLR

Usage example:
  python train_multi_adapter.py --data-root tmp/full_npy --epochs 50 --batch-size 64 --lr 0.001
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARN] TensorBoard not available. Install with: pip install tensorboard")


class TemporalDataset(Dataset):
    """時系列を保持したデータセット: 各フレームを独立サンプルとして扱う"""
    def __init__(self, root, use_augmentation=True):
        self.root = Path(root)
        self.runs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        self.use_augmentation = use_augmentation
        
        # 全フレームをロード
        self.samples = []
        for run_dir in self.runs:
            fb_p = run_dir / 'features_buffer.npy'
            des_p = run_dir / 'desire.npy'
            if not fb_p.exists() or not des_p.exists():
                continue
            
            fb = np.load(fb_p)  # (1, 99, 512)
            des = np.load(des_p)  # (1, 100, 8)
            
            # 時系列の最初99フレームを使用（features_bufferに合わせる）
            for t in range(min(fb.shape[1], des.shape[1])):
                X = fb[0, t, :].astype(np.float32)  # (512,)
                Y = np.argmax(des[0, t, :])  # scalar (0-7)
                self.samples.append({'X': X, 'Y': Y, 'run': run_dir.name, 't': t})
        
        print(f"[INFO] Loaded {len(self.samples)} temporal samples from {len(self.runs)} runs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = sample['X'].copy()
        Y = sample['Y']
        
        # Data Augmentation（学習時のみ）
        if self.use_augmentation:
            # ガウシアンノイズ注入（標準偏差0.01）
            noise = np.random.randn(*X.shape).astype(np.float32) * 0.01
            X = X + noise
        
        return {'X': X, 'Y': Y}


class AdapterHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        """
        ONNX temporal_policy.temporal_hydra完全互換モデル構造:
        - in_layer: 512 -> 32
        - res_layer: 32 -> 32 -> 32 (残差接続)
        - final_layer: 32 -> 8
        
        この構造はbase/supercombo.onnxのtemporal_policy.temporal_hydraと完全一致
        """
        # Input layer: 512 -> 32
        self.in_layer = nn.Linear(in_dim, 32)
        
        # Residual layers: 32 -> 32 -> 32
        self.res_layer_0 = nn.Linear(32, 32)
        self.res_layer_2 = nn.Linear(32, 32)
        
        # Final layer: 32 -> 8
        self.final_layer = nn.Linear(32, out_dim)
        
        # Activations and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input layer
        x = self.in_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Residual block
        identity = x
        x = self.res_layer_0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res_layer_2(x)
        x = x + identity  # 残差接続
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final layer
        x = self.final_layer(x)
        
        return x


def collate_fn(batch):
    X = torch.stack([torch.from_numpy(b['X']) for b in batch]).float()
    Y = torch.tensor([b['Y'] for b in batch], dtype=torch.long)
    return {'X': X, 'Y': Y}


def train(data_root, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4, device='auto', out_dir='/tmp/multi_adapter_ckpt', patience=10, use_amp=True, tensorboard=True):
    """
    本格的PyTorch学習環境での訓練
    
    Args:
        data_root: データセットルートディレクトリ
        epochs: 最大エポック数
        batch_size: バッチサイズ
        lr: 学習率
        weight_decay: L2正則化係数
        device: 'auto' (CUDA自動検出), 'cuda', 'cpu'
        out_dir: 出力ディレクトリ
        patience: Early stopping patience
        use_amp: Mixed Precision Training使用
        tensorboard: TensorBoard使用
    """
    # Device selection (CUDA自動検出)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    # TensorBoard setup
    writer = None
    if tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(out_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[INFO] TensorBoard logging to: {log_dir}")
        print(f"  Run: tensorboard --logdir={log_dir}")
    elif tensorboard and not TENSORBOARD_AVAILABLE:
        print(f"[WARN] TensorBoard requested but not available")
    
    # Dataset作成（時系列保持）
    full_ds = TemporalDataset(data_root, use_augmentation=True)
    n = len(full_ds)
    
    if n < 10:
        print(f"[WARN] Very few samples ({n}). Consider adding more run data.")
    
    # Train/Val分割（80/20）
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    
    print(f"[INFO] Train samples: {n_train}, Val samples: {n_val}")
    
    # DataLoader (num_workers自動設定)
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device=='cuda'))
    
    # Model
    in_dim = 512  # features_buffer dimension
    out_dim = 8   # desire classes
    model = AdapterHead(in_dim, out_dim, dropout=0.3).to(device)
    
    print(f"\n[INFO] Model structure (完全ONNX互換):")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} ({param.numel()} params)")
        total_params += param.numel()
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer with weight decay (L2 regularization)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler (CosineAnnealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/10)
    
    # Mixed Precision Training (AMP)
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    if scaler:
        print(f"[INFO] Mixed Precision Training (AMP) enabled")
    
    # クラス重み計算（不均衡対策）
    class_counts = np.zeros(out_dim, dtype=np.float64)
    for sample in full_ds.samples:
        class_counts[sample['Y']] += 1
    
    # 逆頻度重み（頻度の低いクラスに高い重み）
    eps = 1e-6
    inv_freq = 1.0 / (class_counts + eps)
    weights = inv_freq / np.mean(inv_freq)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"\n[INFO] Class distribution: {class_counts}")
    print(f"[INFO] Class weights: {weights}")
    
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    
    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_path = None
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for ep in range(1, epochs+1):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_loss = 0.0
        count = 0
        correct = 0
        
        for batch_idx, batch in enumerate(train_loader):
            X = batch['X'].to(device, non_blocking=True)
            Y = batch['Y'].to(device, non_blocking=True)
            
            opt.zero_grad()
            
            # Mixed Precision Training
            if scaler:
                with autocast():
                    logits = model(X)
                    loss = loss_fn(logits, Y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X)
                loss = loss_fn(logits, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            
            total_loss += loss.item() * X.size(0)
            count += X.size(0)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == Y).sum().item()
        
        train_loss = total_loss / max(1, count)
        train_acc = correct / max(1, count)
        
        # Validation
        model.eval()
        vloss = 0.0
        vcount = 0
        vcorrect = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device, non_blocking=True)
                Y = batch['Y'].to(device, non_blocking=True)
                
                if scaler:
                    with autocast():
                        logits = model(X)
                        loss = loss_fn(logits, Y)
                else:
                    logits = model(X)
                    loss = loss_fn(logits, Y)
                
                vloss += loss.item() * X.size(0)
                vcount += X.size(0)
                
                preds = torch.argmax(logits, dim=1)
                vcorrect += (preds == Y).sum().item()
        
        val_loss = vloss / max(1, vcount)
        val_acc = vcorrect / max(1, vcount)
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        epoch_time = time.time() - epoch_start
        print(f"Epoch {ep:3d}/{epochs} [{epoch_time:.1f}s] | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"LR={current_lr:.6f}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, ep)
            writer.add_scalar('Loss/val', val_loss, ep)
            writer.add_scalar('Accuracy/train', train_acc, ep)
            writer.add_scalar('Accuracy/val', val_acc, ep)
            writer.add_scalar('Learning_rate', current_lr, ep)
        
        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_path = os.path.join(out_dir, f'best_adapter_ep{ep}.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  → ✓ Best model saved (val_acc={val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping at epoch {ep} (patience={patience})")
            break
    
    total_time = time.time() - start_time
    
    if writer:
        writer.close()
    
    print(f"\n{'='*80}")
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Best val_acc: {best_val_acc:.3f}")
    print(f"  Best model: {best_path}")
    print(f"{'='*80}")
    
    return best_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--device', default='auto', help='Device: auto, cuda, cpu')
    p.add_argument('--out-dir', default='/tmp/multi_adapter_ckpt')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    p.add_argument('--use-amp', action='store_true', default=True, help='Use Mixed Precision Training')
    p.add_argument('--no-amp', action='store_false', dest='use_amp', help='Disable Mixed Precision Training')
    p.add_argument('--tensorboard', action='store_true', default=True, help='Use TensorBoard')
    p.add_argument('--no-tensorboard', action='store_false', dest='tensorboard', help='Disable TensorBoard')
    args = p.parse_args()
    
    train(args.data_root, epochs=args.epochs, batch_size=args.batch_size, 
          lr=args.lr, weight_decay=args.weight_decay, device=args.device, 
          out_dir=args.out_dir, patience=args.patience, 
          use_amp=args.use_amp, tensorboard=args.tensorboard)

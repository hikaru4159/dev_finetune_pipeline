#!/usr/bin/env python3
"""
DATAディレクトリ配下のrunを一括でnpyデータセット化し、AdapterHead追加学習、ONNX変換まで自動実行するスクリプト。

Usage:
  python run_full_pipeline_with_onnx.py --data-dir DATA/itsdata --out-root tmp/full_npy --epochs 1 --batch-size 1 --adapter-out tmp/full_adapter_ckpt --pth-out nets/model_itr/test_adapter_ep1.pt --onnx-out nets/model_itr/test_adapter_ep1.onnx --base-onnx base/supercombo.onnx
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
BATCH_RUN_SCRIPT = THIS_DIR / 'batch_run_processing.py'
TRAIN_SCRIPT = THIS_DIR / 'train_multi_adapter.py'
ONNX_EXPORT_SCRIPT = WORKSPACE_ROOT / 'openpilot-pipeline' / 'train' / 'export_pth_to_onnx.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--adapter-out', required=True)
    p.add_argument('--pth-out', required=True)
    p.add_argument('--onnx-out', required=True)
    p.add_argument('--base-onnx', required=True)
    args = p.parse_args()

    # 1. データセット一括変換
    print(f"\n=== データセット変換: {args.data_dir} → {args.out_root} ===")
    subprocess.run([sys.executable, str(BATCH_RUN_SCRIPT), '--data-dir', args.data_dir, '--out-root', args.out_root], check=True)

    # 2. AdapterHead追加学習
    print(f"\n=== AdapterHead追加学習: {args.out_root} → {args.adapter_out} ===")
    subprocess.run([sys.executable, str(TRAIN_SCRIPT), '--data-root', args.out_root, '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--out-dir', args.adapter_out], check=True)

    # 3. ONNX変換
    print(f"\n=== ONNX変換: {args.pth_out} + {args.base_onnx} → {args.onnx_out} ===")
    subprocess.run([sys.executable, str(ONNX_EXPORT_SCRIPT), args.pth_out, args.onnx_out, args.base_onnx], check=True)

    print("\n=== 完了 ===")

if __name__ == '__main__':
    main()

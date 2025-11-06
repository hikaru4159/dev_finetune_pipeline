#!/usr/bin/env python3
"""
DATAディレクトリ配下のrunを一括でnpyデータセット化し、AdapterHead追加学習、ONNX変換まで自動実行するスクリプト。

機能：
- データセット一括変換（desire画像推論による自動補完を含む）
- AdapterHead追加学習（複数run対応、重み付けCrossEntropyLoss）
- PyTorch重み→ONNX変換（4テンソル置換）

Usage:
  python run_full_pipeline_with_onnx.py \\
    --data-dir DATA/itsdata \\
    --out-root tmp/full_npy \\
    --epochs 5 \\
    --batch-size 2 \\
    --adapter-out tmp/full_adapter_ckpt \\
    --pth-out tmp/full_adapter_ckpt/best_adapter_ep5.pt \\
    --onnx-out nets/model_itr/test_adapter_final.onnx \\
    --base-onnx base/supercombo.onnx

オプション：
  --data-dir: 入力データディレクトリ（run1, run2, ... を含む）
  --out-root: データセット出力ルート（各runのnpyファイルが生成される）
  --epochs: 学習エポック数（デフォルト: 1）
  --batch-size: バッチサイズ（デフォルト: 1）
  --adapter-out: AdapterHeadチェックポイント出力ディレクトリ
  --pth-out: 最終PyTorch重みファイルパス（参考：実際はbest_adapter_ep{epochs}.ptとして保存）
  --onnx-out: 出力ONNXファイルパス
  --base-onnx: ベースsupercombo.onnxファイルパス
  --learning-rate: 学習率（デフォルト: 0.001）
  --weight-decay: 重み減衰（デフォルト: 0.0001）
  --patience: Early stopping patience（デフォルト: 5）
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
import time

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
BATCH_RUN_SCRIPT = THIS_DIR / 'batch_run_processing.py'
TRAIN_SCRIPT = THIS_DIR / 'train_multi_adapter.py'
ONNX_EXPORT_SCRIPT = WORKSPACE_ROOT / 'openpilot-pipeline' / 'train' / 'export_pth_to_onnx.py'


def main():
    p = argparse.ArgumentParser(
        description='一括学習パイプライン：データ変換→学習→ONNX変換',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例：
  python run_full_pipeline_with_onnx.py \\
    --data-dir DATA/itsdata \\
    --out-root tmp/full_npy \\
    --epochs 5 \\
    --batch-size 2 \\
    --adapter-out tmp/full_adapter_ckpt \\
    --pth-out tmp/full_adapter_ckpt/best_adapter_ep5.pt \\
    --onnx-out nets/model_itr/test_adapter_final.onnx \\
    --base-onnx base/supercombo.onnx
        """
    )
    p.add_argument('--data-dir', required=True, help='入力データディレクトリ')
    p.add_argument('--out-root', required=True, help='データセット出力ルート')
    p.add_argument('--epochs', type=int, default=1, help='学習エポック数')
    p.add_argument('--batch-size', type=int, default=1, help='バッチサイズ')
    p.add_argument('--adapter-out', required=True, help='AdapterHeadチェックポイント出力ディレクトリ')
    p.add_argument('--pth-out', required=False, default=None, help='最終PyTorch重みファイルパス（省略可、自動検出）')
    p.add_argument('--onnx-out', required=True, help='出力ONNXファイルパス')
    p.add_argument('--base-onnx', required=True, help='ベースsupercombo.onnxファイルパス')
    p.add_argument('--learning-rate', type=float, default=0.001, help='学習率')
    p.add_argument('--weight-decay', type=float, default=0.0001, help='重み減衰（L2正則化）')
    p.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = p.parse_args()

    # 出力ディレクトリ作成
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.adapter_out, exist_ok=True)
    os.makedirs(Path(args.onnx_out).parent, exist_ok=True)

    # 時間計測開始
    start_time = time.time()

    try:
        # 1. データセット一括変換（desire画像推論を含む）
        print("\n" + "=" * 80)
        print(f"ステップ 1/3: データセット変換")
        print(f"  入力: {args.data_dir}")
        print(f"  出力: {args.out_root}")
        print(f"  注意: desire補完は画像から光学フロー推論で自動実行されます")
        print("=" * 80)
        
        step1_start = time.time()
        result = subprocess.run(
            [sys.executable, str(BATCH_RUN_SCRIPT), '--data-dir', args.data_dir, '--out-root', args.out_root],
            check=True,
            capture_output=False
        )
        step1_time = time.time() - step1_start
        print(f"\n✓ データセット変換完了 ({step1_time:.1f}秒)")

        # 2. AdapterHead追加学習
        print("\n" + "=" * 80)
        print(f"ステップ 2/3: AdapterHead追加学習")
        print(f"  入力: {args.out_root}")
        print(f"  出力: {args.adapter_out}")
        print(f"  設定: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        print("=" * 80)
        
        step2_start = time.time()
        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            '--data-root', args.out_root,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--out-dir', args.adapter_out,
            '--lr', str(args.learning_rate),
            '--weight-decay', str(args.weight_decay),
            '--patience', str(args.patience)
        ]
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        step2_time = time.time() - step2_start
        print(f"\n✓ AdapterHead学習完了 ({step2_time:.1f}秒)")

        # 実際に保存されたPyTorch重みファイルを検索
        # train_multi_adapter.pyは best_adapter_ep{epochs}.pt として保存する
        actual_pth_path = Path(args.adapter_out) / f'best_adapter_ep{args.epochs}.pt'
        
        # ファイルの存在確認
        if not actual_pth_path.exists():
            # 最後の手段：adapter_out内の最新のbest_adapter_*.ptを検索
            best_files = list(Path(args.adapter_out).glob('best_adapter_ep*.pt'))
            if best_files:
                actual_pth_path = max(best_files, key=lambda p: p.stat().st_mtime)
                print(f"[INFO] 最新のチェックポイントを使用: {actual_pth_path}")
            else:
                raise FileNotFoundError(f"PyTorch重みファイルが見つかりません: {args.adapter_out}")
        else:
            print(f"[INFO] PyTorch重みファイル: {actual_pth_path}")
        
        # もしユーザーがpth-outを指定していた場合は情報メッセージ
        if args.pth_out and str(actual_pth_path) != args.pth_out:
            print(f"[INFO] 注意: 指定されたパス '{args.pth_out}' の代わりに '{actual_pth_path}' を使用します")


        # 3. ONNX変換（8テンソル置換）
        print("\n" + "=" * 80)
        print(f"ステップ 3/3: ONNX変換")
        print(f"  PyTorch重み: {actual_pth_path}")
        print(f"  ベースONNX: {args.base_onnx}")
        print(f"  出力ONNX: {args.onnx_out}")
        print(f"  注意: temporal_policy.temporal_hydraの8テンソルを置換します")
        print("=" * 80)
        
        step3_start = time.time()
        result = subprocess.run(
            [sys.executable, str(ONNX_EXPORT_SCRIPT), str(actual_pth_path), args.onnx_out, args.base_onnx],
            check=True,
            capture_output=False
        )
        step3_time = time.time() - step3_start
        print(f"\n✓ ONNX変換完了 ({step3_time:.1f}秒)")

        # 完了メッセージ
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("✓✓✓ 全パイプライン完了 ✓✓✓")
        print(f"  合計時間: {total_time:.1f}秒")
        print(f"  出力ONNX: {args.onnx_out}")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\n✗ エラー発生: {e}")
        print(f"  コマンド: {e.cmd}")
        print(f"  終了コード: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
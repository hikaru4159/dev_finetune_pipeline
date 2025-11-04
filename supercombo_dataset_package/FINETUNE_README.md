Supercombo fine-tuning helper
=================================

目的
----
このディレクトリには、openpilot 0.9.6 の supercombo モデル（`base/supercombo.onnx`）をそのまま再利用しつつ、追加学習（微調整）に必要なデータセット作成と最小限の学習ハーネスを収めています。ONNX の入出力形状は変更せず、comma.ai のスタックで再利用できることを重視します。

含まれるスクリプト
-------------------
- `create_supercombo_dataset.py`
  - `DATA/syagai_1st` のようなルートから各実行ログを探索し、既存の `transform_to_supercombo_dataset_gui.py` を呼び出して各ラン毎に numpy 形式の dataset を生成します。
- `inspect_supercombo_onnx.py`
  - ONNX モデルの入力/出力名と形状を確認します。生成データが期待形状に合うか必ずチェックしてください。
- `delta_finetune_harness.py`
  - ONNX をそのまま推論用に使い、PyTorch ベースの小さな adapter を学習する最小ハーネス（雛形）。ONNX 自体は書き換えず、adapter を別ファイルとして保存します。

前提/環境
---------
- `supercombo_dataset_package/README.md` に従って仮想環境を構築してください（pycapnp, numpy, opencv, zstandard, onnxruntime, torch 等が必要です）。
- FFmpeg が必要です（画像抽出用）。

基本的なワークフロー
-------------------
1. 環境をセットアップ
   - supercombo_dataset_package/setup_env/setup_supercombo_env.sh を実行し、venv を有効化
2. データ変換
   - 例: python create_supercombo_dataset.py --data-root DATA/syagai_1st --out-root /tmp/supercombo_dataset
   - これにより、各ランごとに npy ファイル群（`input_imgs.npy`, `big_input_imgs.npy`, `desire.npy`, `features_buffer.npy`, ...） が生成されます。
3. ONNX 形状確認
   - python inspect_supercombo_onnx.py --onnx base/supercombo.onnx
   - 表示された入力名・形状が生成した npy の shape と一致することを確認
4. 微調整（最小例）
   - python delta_finetune_harness.py --onnx base/supercombo.onnx --npy-run-folder /tmp/supercombo_dataset/2023-... --epochs 5
   - このスクリプトは最小限の adapter を学習して `adapter_head.pt` を保存します。実運用の学習ループやバッチ処理、学習率スケジューラなどは各自で拡張してください。

注意点
---------
- ONNX の入出力形状は厳格に維持してください。出力を意図的に変える場合は comma.ai の再利用性を損なうため、別名のモデルとして管理してください。
- 既存の `transform_to_supercombo_dataset_gui.py` の抽出ロジックをそのまま利用しています。ログの欠損や rlog のフォーマット違いにより一部の値（lateral params 等）がゼロ埋めされる場合があります。

次のステップ
-------------
- 複数ラン（バッチ）学習向けに `delta_finetune_harness.py` を DataLoader 対応に拡張
- テストケース（小さなサンプルデータ）を追加し、自動で形状整合をチェックするスクリプトを追加

ライセンス
-------
このリポジトリに合わせて利用してください。

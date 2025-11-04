#!/bin/bash
# supercombo_dataset_package 最小環境をワンクリックで再構築するスクリプト
# 実行例: bash setup_supercombo_env.sh
#
# [注意] GUI利用時はtkが必要です。tkはpipではなくシステムパッケージです。
# 例: sudo apt install python3-tk

set -e

# 1. Python仮想環境作成
echo "[1/4] Python仮想環境の作成..."
python3 -m venv venv
source venv/bin/activate

# 2. 必要パッケージインストール
echo "[2/4] 必要パッケージのインストール..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. capnpスキーマの依存チェック
echo "[3/4] capnpスキーマ依存ファイルの存在確認..."
for f in ../openpilot_096/cereal/log.capnp ../openpilot_096/cereal/car.capnp ../openpilot_096/cereal/custom.capnp ../openpilot_096/cereal/legacy.capnp ../openpilot_096/cereal/include/c++.capnp; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] $f が存在しません。リポジトリを正しく取得してください。"
    exit 1
  fi
done

echo "[4/4] 構築完了！ 仮想環境を有効化するには: source venv/bin/activate"
echo "supercombo_dataset_package/transform_to_supercombo_dataset_gui.py を実行できます。"

#!/bin/bash
# openpilot_patch_apply.sh
# openpilotディレクトリを公式openpilotサブモジュール化後、差分パッチを適用するスクリプト

set -e

if [ ! -d openpilot ]; then
  git submodule add https://github.com/commaai/openpilot.git openpilot
else
  echo "openpilotディレクトリは既に存在します。"
fi

echo "公式との差分パッチを適用します..."
patch -p1 -d openpilot < openpilot_patch_full.patch
echo "パッチ適用完了"

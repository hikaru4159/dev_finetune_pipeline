#!/bin/bash
# comma2k19_patch_apply.sh
# comma2k19ディレクトリを公式comma2k19サブモジュール化後、差分パッチを適用するスクリプト

set -e

if [ ! -d comma2k19 ]; then
  git submodule add https://github.com/commaai/comma2k19.git comma2k19
else
  echo "comma2k19ディレクトリは既に存在します。"
fi

echo "公式との差分パッチを適用します..."
patch -p1 -d comma2k19 < comma2k19_patch_full.patch
echo "パッチ適用完了"

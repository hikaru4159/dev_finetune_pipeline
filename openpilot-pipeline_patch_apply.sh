#!/bin/bash
# openpilot-pipelineパッチ適用スクリプト
# サブモジュール初期化・更新後に実行してください

set -e

echo "パッチ適用開始..."
patch -p1 -d openpilot-pipeline < openpilot-pipeline_patch_full.patch
echo "パッチ適用完了"

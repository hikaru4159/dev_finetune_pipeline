# Dataset Processing Pipeline - 使用ガイド

desire補完スクリプトの実行失敗を防ぎ、コマンドを簡素化するための統合スクリプトです。

## 📁 作成されたファイル

1. **`run_dataset_pipeline.sh`** - メインの統合スクリプト（bash）
2. **`desire_supplement.py`** - 改良版desire補完スクリプト（Python）

## 🚀 基本的な使用方法

### 1. 依存関係チェック
```bash
./run_dataset_pipeline.sh check-deps
```

### 2. 完全なパイプライン実行（推奨）
```bash
./run_dataset_pipeline.sh full
```
これは以下を自動で実行します：
- 依存関係チェック
- 既存出力のクリーンアップ  
- データセット変換
- 結果確認

### 3. データセット変換のみ
```bash
./run_dataset_pipeline.sh convert
```

### 4. 結果確認
```bash
./run_dataset_pipeline.sh status
```

### 5. 特定ランの個別補完
```bash
./run_dataset_pipeline.sh supplement 00000003--06297c7620--0
```

## 🔧 設定

スクリプト上部で以下の設定を変更可能：

```bash
DATA_DIR="$PROJECT_ROOT/DATA/syagai_1st/larning_data"  # 入力データディレクトリ
OUTPUT_ROOT="/tmp/supercombo_dataset_larning_all"      # 出力ディレクトリ
JOBS=3                                                 # 並列ジョブ数
```

## 📋 改善点

### 前の問題点
- Python環境の不一致
- パス設定エラー  
- 依存関係の欠落
- エラー時の診断が困難

### 改善された機能
✅ **自動依存関係チェック**
- 必要なPythonパッケージの確認
- 仮想環境パスの検証
- データディレクトリの存在確認

✅ **堅牢なエラーハンドリング**
- 複数のフォールバック方法（log-based → image-based → zero-filled）
- 詳細なエラーメッセージ
- 部分的な成功でも継続実行

✅ **省コマンド化**
- 1つのコマンドで完全なパイプライン実行
- 設定の自動検出
- カラー出力で結果が分かりやすい

✅ **バッチ処理サポート**
- 並列処理対応
- 進捗表示
- 詳細な結果レポート

## 🐍 Python版（高度な用途）

個別制御が必要な場合は、Python版を直接使用：

```bash
# 単一ラン処理
python desire_supplement.py --run /path/to/run --out /path/to/output

# バッチ処理  
python desire_supplement.py --batch /path/to/data --out-root /path/to/output --jobs 3

# 画像ベース強制（ログベースをスキップ）
python desire_supplement.py --run /path/to/run --out /path/to/output --force-image

# 依存関係チェック
python desire_supplement.py --check-deps
```

## 📊 出力例

### 成功時
```
[INFO] Checking dependencies...
[SUCCESS] Dependencies OK
[INFO] Starting dataset conversion...
[SUCCESS] Dataset conversion completed
=== Dataset Status Report ===
📁 Found 21 run directories
✅ Complete runs: 21
📊 Summary: 21/21 runs complete
```

### エラー時
```
[ERROR] Missing dependencies: zstandard
[INFO] Install with: python -m pip install zstandard
```

## 🔍 トラブルシューティング

### 1. 依存関係エラー
```bash
./run_dataset_pipeline.sh check-deps
# 不足パッケージが表示されるのでインストール
```

### 2. データディレクトリが見つからない
```bash
# スクリプト内のDATA_DIRパスを確認・修正
```

### 3. 個別ランの処理失敗
```bash
# 特定ランを個別で処理
./run_dataset_pipeline.sh supplement <run_name>
```

### 4. メモリ不足
```bash
# 並列ジョブ数を減らす（スクリプト内のJOBS設定を変更）
```

## 📝 ログ確認

実行ログで以下を確認：
- `[SKIP]` - 入力不足でスキップされたラン（bootなど）
- `✓` - 正常処理されたラン  
- `✗` - 処理に失敗したラン
- `Shapes:` - 生成されたnpyファイルの形状確認

これで、desire補完の実行失敗を大幅に減らし、簡単なコマンドで安定した処理が可能になります。
- **capnpスキーマ依存チェックは相対パス依存**
   - `setup_supercombo_env.sh`は必ず`supercombo_dataset_package/setup_env`ディレクトリ内で実行してください。
   - 他のディレクトリで実行するとcapnpスキーマファイルの存在チェックが失敗します。
   - スクリプト先頭やREADMEでもこの点を強調しています。
# supercombo_dataset_package GUI版 データセット変換ツール

## 概要

このツールは、OpenPilotログ・映像データからsupercomboモデル用のデータセットをGUIで簡単に作成できるPythonスクリプトです。

- 前方カメラ・ワイドカメラ画像、各種センサ・制御データを時系列で抽出
- 必要な入力データをONNXモデル形式に合わせて自動生成
- GUIで入力・出力ディレクトリを直感的に選択可能

## 必要環境

- Python 3.8以降
- Linux推奨
- 依存パッケージ: numpy, opencv-python, zstandard, pycapnp, tkinter（GUI用）
- FFmpeg（画像抽出用、システムにインストール必須）


## 初期環境構築手順（setup_envの利用）

1. ディレクトリ移動
   ```bash
   cd supercombo_dataset_package/setup_env
   ```
2. 仮想環境の作成と依存パッケージのインストール
   ```bash
   bash setup_supercombo_env.sh
   ```
   - 必要なPython仮想環境と依存パッケージが自動でインストールされます
3. 仮想環境の有効化
   ```bash
   source venv/bin/activate
   ```
4. 以降は仮想環境有効化後にGUIやCLIでスクリプトを実行してください

---

## GUIの起動方法

1. 仮想環境を有効化
   ```
   source supercombo_dataset_package/setup_env/venv/bin/activate
   ```
2. GUIモードでスクリプトを起動
   ```
   python supercombo_dataset_package/transform_to_supercombo_dataset_gui.py --gui
   ```

## 使い方

1. **入力フォルダ**を選択
   - rlog.zst, qlog.zst, dcamera.hevc, ecamera.hevc等が格納されたディレクトリ、またはその親ディレクトリを指定
2. **出力フォルダ**を選択
   - 変換後のデータセットを保存するディレクトリを指定
3. **「データセット作成」ボタン**を押す
   - 進捗・ログがダイアログで表示されます
   - 完了後、出力先に各種npy/jsonファイルが生成されます


## データセット内容・データ要素情報

| 入力名                  | 形状（Shape）           | 備考・データ形成方法                                                                                   |
|-------------------------|------------------------|--------------------------------------------------------------------------------------------------------|
| input_imgs              | (1, 12, 128, 256)      | 前方カメラ画像。YUV420のYチャンネルのみ抽出し、float32型で0.0～1.0に正規化。12フレーム分を時系列で格納。      |
| big_input_imgs          | (1, 12, 128, 256)      | ワイドカメラ画像。YUV420のYチャンネルのみ抽出し、float32型で0.0～1.0に正規化。12フレーム分を時系列で格納。    |
| desire                  | (1, 100, 8)            | 操作意図。OpenPilot公式8ラベル（none, turnLeft, turnRight, laneChangeLeft, laneChangeRight, keepLeft, keepRight, keepLane）をone-hotで100フレーム分（過去5秒）float32型で格納。**rlogにdesireイベントが無い場合は、画像から光学フロー推論で自動補完**。|
---
### desireラベル定義（OpenPilot公式/モデル準拠）
```python
DESIRE_LABELS = [
   "none",             # 何も意図しない
   "turnLeft",         # 左折
   "turnRight",        # 右折
   "laneChangeLeft",   # 左車線変更
   "laneChangeRight",  # 右車線変更
   "keepLeft",         # 左寄り走行
   "keepRight",        # 右寄り走行
   "keepLane"          # 車線維持
]
```
※順序・意味は公式capnp定義に準拠。停止・加速・減速はdesireには含まれず、longitudinalPlan等で管理。

---
### desire自動補完機能（画像ベース推論）

rlogにdesireイベントが存在しない場合、**画像から光学フロー（Farneback法）による自動補完**を実行します：

**処理の流れ：**
1. カメラ映像（fcamera.hevc/ecamera.hevc）から10fps（100msec間隔）で画像抽出
2. 画像を**256x512にリサイズ**（OpenPilotモデル入力サイズに最適化）
3. 連続フレーム間でFarneback光学フローを計算
4. 横方向の動き（flow_x）から車線変更・右左折を推定：
   - 横方向移動が閾値以上 → 車線変更（左/右）
   - 動きが小さい → 車線維持（keepLane）
5. 不足フレームはkeepLaneでパディング

**検出精度：**
- 車線変更（左/右）：適度な横方向移動を検出
- 車線維持：通常走行時のデフォルト
- 画像サイズ最適化により、元解像度（1208x1928）より高精度

**フォールバック：**
- 画像推論が失敗した場合、全フレームをkeepLaneで補完

---
### 解析知見・補足
- desireはlogにイベントが無い場合でも、画像時系列から未来の動きを見て過去の操作意図を推定可能
- 画像差分・車線検出・方向判定等で車線変更/左折/右折/車線維持などを推定
- 公式モデル・supercomboモデルともこの8ラベルで統一
- ルールベース/AIベースどちらでも推定ロジック実装可能
- **本実装では光学フローベースのルールベース推論を採用**（高速・安定）
| traffic_convention      | (1, 2)                 | 交通規則。右側通行: `[1.0, 0.0]`、左側通行: `[0.0, 1.0]` のone-hotベクトルをfloat32型で格納。                |
| lateral_control_params  | (1, 2)                 | 横制御パラメータ。`[v_ego, steer_delay]`（車速[m/s], ステアリング遅延[秒]）をfloat32型で格納。                |
| prev_desired_curv       | (1, 100, 1)            | 直近の目標曲率。float32型で100フレーム分（過去5秒）時系列で格納。                                           |
| nav_features            | (1, 256)               | ナビゲーション特徴量。地図・経路情報を埋め込みベクトル化し、float32型で格納。                                 |
| nav_instructions        | (1, 150)               | ナビ命令。案内文や経路指示をone-hotまたは埋め込みベクトル化し、float32型で格納。                              |
| features_buffer         | (1, 99, 512)           | RNN隠れ状態。モデルの中間特徴をfloat32型で99フレーム分（時系列履歴）格納。                                   |

## logファイル（rlog, qlog）におけるデータ存在状況

| 入力名                  | rlog存在 | qlog存在 | 備考・補足 |
|-------------------------|----------|----------|-----------------------------|
| input_imgs              | ×        | ×        | 画像は映像ファイルから抽出   |
| big_input_imgs          | ×        | ×        | 画像は映像ファイルから抽出   |
| desire                  | ×        | ×        | desireイベント無し、**画像から光学フロー推論で自動補完** |
| traffic_convention      | ○        | ○        | onroadEvents等から取得       |
| lateral_control_params  | △        | △        | carState, lateralPlanはあるが値が全て0 |
| prev_desired_curv       | △        | △        | lateralPlanはあるが値が全て0 |
| nav_features            | ×        | ×        | navFeaturesイベント無し、ゼロ埋め |
| nav_instructions        | ×        | ×        | navInstructionsイベント無し、ゼロ埋め |
| features_buffer         | ×        | ×        | featuresBufferイベント無し、ゼロ埋め |

---

### 補足
- 画像データ（input_imgs, big_input_imgs）は映像ファイル（dcamera.hevc, ecamera.hevc等）から抽出
- **desire**：logにイベント無し→**画像から光学フロー推論で自動補完**（車線変更・keepLane等を検出）
- nav_features, nav_instructions, features_bufferはlogファイルにイベント自体が存在しないためゼロ埋め
- traffic_conventionはonroadEvents等から取得できる
- lateral_control_params, prev_desired_curvはイベント型はあるが値が全て0（抽出失敗 or データ未記録）

## 注意事項


- **環境構築時の注意点（このPCでのトラブル例）**

- **コマンド実行時のディレクトリ位置に注意**
   - `setup_supercombo_env.sh`は必ず`supercombo_dataset_package/setup_env`ディレクトリ内で実行してください。
   - 仮想環境の有効化（`source venv/bin/activate`）も同じく`setup_env`内で行うとパスの問題が起きません。
   - スクリプト実行時は、プロジェクトルート（`supercombo_dataset_package_min`）または`supercombo_dataset_package`からパスを指定してください。

- **相対パス・絶対パスの混同に注意**
   - スクリプトやシェルの実行場所によっては、相対パスが正しく解決されずエラーになることがあります。
   - 例：`openpilot_096/cereal/log.capnp`が見つからない場合は、実行ディレクトリを見直してください。

- **仮想環境の場所**
   - 仮想環境（venv）は`supercombo_dataset_package/setup_env/venv`に作成されます。
   - 別ディレクトリで`source venv/bin/activate`を実行しても失敗します。

- **GUI起動時の注意**
   - GUIはリモート環境やX11転送が無効な場合は起動できません。ローカルPCで実行してください。

- **FFmpegのインストール**
   - 画像抽出にはFFmpegが必要です。`sudo apt install ffmpeg`等で事前にインストールしてください。

---

## 一括学習パイプライン（追加学習自動化）

データ変換→AdapterHead学習→ONNX変換を一括実行するスクリプトを提供しています。

### 使い方

```bash
# 仮想環境を有効化
source supercombo_dataset_package/setup_env/venv/bin/activate

# 一括実行
python supercombo_dataset_package/run_full_pipeline_with_onnx.py \
  --data-dir DATA/itsdata \
  --out-root tmp/full_npy \
  --epochs 5 \
  --batch-size 2 \
  --adapter-out tmp/full_adapter_ckpt \
  --pth-out tmp/full_adapter_ckpt/best_adapter_ep5.pt \
  --onnx-out nets/model_itr/test_adapter_final.onnx \
  --base-onnx base/supercombo.onnx \
  --learning-rate 0.001 \
  --weight-decay 0.0001
```

### パイプラインの流れ

**ステップ 1: データセット変換**
- `DATA/itsdata`配下の全run（1, 2, ...）をnpyデータセットに変換
- **desire補完**：画像から光学フロー推論で自動生成（256x512リサイズ最適化）
- 不足フレームはkeepLaneでパディング
- 出力先: `tmp/full_npy/1`, `tmp/full_npy/2`, ...

**ステップ 2: AdapterHead追加学習**
- 複数runを一括読み込み（バッチ処理）
- AdapterHead構造: Linear(512→32)→ReLU→Linear(32→8)
- 重み付けCrossEntropyLoss（クラス不均衡対応）
- 学習率: 0.001（デフォルト）、重み減衰: 0.0001
- チェックポイント保存: `tmp/full_adapter_ckpt/adapter_ep*.pt`
- 最良モデル: `tmp/full_adapter_ckpt/best_adapter_ep5.pt`

**ステップ 3: ONNX変換**
- PyTorch重み（.pt）→ONNX（base/supercombo.onnx）に統合
- `temporal_policy.temporal_hydra`の4テンソルを置換：
  - `in_layer.weight/bias` (512→32)
  - `final_layer.weight/bias` (32→8)
- 出力ONNX: `nets/model_itr/test_adapter_final.onnx`

### オプション説明

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--data-dir` | 入力データディレクトリ（run1, run2, ...を含む） | 必須 |
| `--out-root` | データセット出力ルート | 必須 |
| `--epochs` | 学習エポック数 | 1 |
| `--batch-size` | バッチサイズ | 1 |
| `--adapter-out` | AdapterHeadチェックポイント出力ディレクトリ | 必須 |
| `--pth-out` | 最終PyTorch重みファイルパス | 必須 |
| `--onnx-out` | 出力ONNXファイルパス | 必須 |
| `--base-onnx` | ベースsupercombo.onnxファイルパス | 必須 |
| `--learning-rate` | 学習率 | 0.001 |
| `--weight-decay` | 重み減衰（L2正則化） | 0.0001 |

### 実行例（詳細設定）

```bash
# より多くのエポック＋大きいバッチサイズで学習
python supercombo_dataset_package/run_full_pipeline_with_onnx.py \
  --data-dir DATA/itsdata \
  --out-root tmp/training_data \
  --epochs 10 \
  --batch-size 4 \
  --adapter-out tmp/adapter_checkpoints \
  --pth-out tmp/adapter_checkpoints/best_adapter_ep10.pt \
  --onnx-out nets/model_itr/supercombo_finetuned.onnx \
  --base-onnx base/supercombo.onnx \
  --learning-rate 0.0005 \
  --weight-decay 0.0002
```

### 期待される出力

```
================================================================================
ステップ 1/3: データセット変換
  入力: DATA/itsdata
  出力: tmp/full_npy
  注意: desire補完は画像から光学フロー推論で自動実行されます
================================================================================
[INFO] Extracting desire from images: DATA/itsdata/1/fcamera.hevc
[INFO] Desire extracted from images, final shape: (1, 12, 8)
...
✓ データセット変換完了 (45.2秒)

================================================================================
ステップ 2/3: AdapterHead追加学習
  入力: tmp/full_npy
  出力: tmp/full_adapter_ckpt
  設定: epochs=5, batch_size=2, lr=0.001
================================================================================
Epoch 1/5, Train Loss: 0.234, Val Loss: 0.189, Val Acc: 0.875
...
✓ AdapterHead学習完了 (123.5秒)

================================================================================
ステップ 3/3: ONNX変換
  PyTorch重み: tmp/full_adapter_ckpt/best_adapter_ep5.pt
  ベースONNX: base/supercombo.onnx
  出力ONNX: nets/model_itr/test_adapter_final.onnx
  注意: temporal_policy.temporal_hydraの4テンソルを置換します
================================================================================
✓ ONNX変換完了 (2.3秒)

================================================================================
✓✓✓ 全パイプライン完了 ✓✓✓
  合計時間: 171.0秒
  出力ONNX: nets/model_itr/test_adapter_final.onnx
================================================================================
```

---

## トラブルシューティング

### desire補完が失敗する場合

- **現象**: desireが全てゼロ、またはkeepLaneのみ
- **原因**: 
  - カメラ映像ファイル（fcamera.hevc/ecamera.hevc）が存在しない
  - 画像抽出（ffmpeg）が失敗
  - 光学フロー計算でエラー
- **対策**:
  1. カメラファイルの存在確認: `ls DATA/itsdata/*/fcamera.hevc`
  2. ffmpegインストール確認: `ffmpeg -version`
  3. デバッグモードで実行: ログに`[INFO] Extracting desire from images`が出力されるか確認
  4. フォールバック: 自動的にkeepLaneで補完されます（学習は継続可能）

### ONNX変換でshape不一致エラー

- **現象**: `RuntimeError: shape mismatch`
- **原因**: AdapterHead構造とONNXのtemporal_hydra構造が不一致
- **対策**:
  - `train_multi_adapter.py`のAdapterHead構造を確認: 512→32→8
  - `export_pth_to_onnx.py`のweight_mappingを確認
  - base/supercombo.onnxのtemporal_policy.temporal_hydra構造を確認

### 学習精度が上がらない

- **現象**: val_accが0付近、またはlossが減少しない
- **原因**:
  - データ不足（run数が少ない）
  - desireクラスが偏っている（keepLaneのみ）
  - 学習率が不適切
- **対策**:
  1. データ確認: `python -c "import numpy as np; d=np.load('tmp/full_npy/1/desire.npy'); print(np.unique(np.argmax(d[0], axis=-1)))"`
  2. エポック数を増やす: `--epochs 10`
  3. 学習率を調整: `--learning-rate 0.0005`
  4. より多くのrunデータを追加

---
何か問題があれば、ログ出力やエラーダイアログの内容を確認してください。

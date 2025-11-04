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
| desire                  | (1, 100, 8)            | 操作意図。OpenPilot公式8ラベル（none, turnLeft, turnRight, laneChangeLeft, laneChangeRight, keepLeft, keepRight, keepLane）をone-hotで100フレーム分（過去5秒）float32型で格納。|
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
### 解析知見・補足
- desireはlogにイベントが無い場合でも、画像時系列から未来の動きを見て過去の操作意図を推定可能
- 画像差分・車線検出・方向判定等で車線変更/左折/右折/車線維持などを推定
- 公式モデル・supercomboモデルともこの8ラベルで統一
- ルールベース/AIベースどちらでも推定ロジック実装可能
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
| desire                  | ×        | ×        | desireイベント無し、ゼロ埋め |
| traffic_convention      | ○        | ○        | onroadEvents等から取得       |
| lateral_control_params  | △        | △        | carState, lateralPlanはあるが値が全て0 |
| prev_desired_curv       | △        | △        | lateralPlanはあるが値が全て0 |
| nav_features            | ×        | ×        | navFeaturesイベント無し、ゼロ埋め |
| nav_instructions        | ×        | ×        | navInstructionsイベント無し、ゼロ埋め |
| features_buffer         | ×        | ×        | featuresBufferイベント無し、ゼロ埋め |

---

### 補足
- 画像データ（input_imgs, big_input_imgs）は映像ファイル（dcamera.hevc, ecamera.hevc等）から抽出
- desire, nav_features, nav_instructions, features_bufferはlogファイルにイベント自体が存在しないためゼロ埋め
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


何か問題があれば、ログ出力やエラーダイアログの内容を確認してください。

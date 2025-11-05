# 仮想環境・パッケージ管理について

このリポジトリでは、Python仮想環境（.venv/）やsite-packages（パッケージ群）はgit管理対象外です。

## 初期セットアップ手順
1. Python 3.12 以上をインストール
2. 仮想環境作成
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 必要パッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```

## 再現性のためのファイル
- requirements.txt（必須パッケージ一覧）
- README.md（セットアップ手順・実行方法）

## 注意
- .venv/ や site-packages/ は .gitignore で除外されています。
- 仮想環境やパッケージは各自セットアップしてください。

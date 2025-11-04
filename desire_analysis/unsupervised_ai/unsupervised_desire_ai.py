from joblib import dump
# unsupervised_desire_ai.py
# 教師なし学習による操作意図（desire）判別AI装置サンプル
# クラスタリング＋特徴抽出＋クラスタ意味付け

# OpenPilot公式準拠 desireラベル定義と番号
DESIRE_LABELS = [
    "none",             # 0: 何も意図しない
    "turnLeft",         # 1: 左折
    "turnRight",        # 2: 右折
    "laneChangeLeft",   # 3: 左車線変更
    "laneChangeRight",  # 4: 右車線変更
    "keepLeft",         # 5: 左寄り走行
    "keepRight",        # 6: 右寄り走行
    "keepLane"          # 7: 車線維持
]
DESIRE_NONE = 0
DESIRE_TURN_LEFT = 1
DESIRE_TURN_RIGHT = 2
DESIRE_LANE_CHANGE_LEFT = 3
DESIRE_LANE_CHANGE_RIGHT = 4
DESIRE_KEEP_LEFT = 5
DESIRE_KEEP_RIGHT = 6
DESIRE_KEEP_LANE = 7

import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 入力画像ディレクトリ（抽出済み画像を利用）
HEVC_DATADIR = "/home/user1434407/dev-fine-dataset/desire_analysis/unsupervised_ai/hevc_datas"
IMG_DIR = "hevc_imgs"
# 画像特徴抽出（車線・動きベクトルに限定）
def extract_images_from_hevc(hevc_path, out_dir, fps=10):
    # ファイル名からサブディレクトリ名生成
    base = os.path.splitext(os.path.basename(hevc_path))[0]
    out_dir = os.path.join(out_dir, base)
    # 画像抽出済み判定: ディレクトリが存在し、かつjpg画像が10枚以上ある場合のみスキップ
    if os.path.exists(out_dir):
        jpg_files = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
        if len(jpg_files) >= 10:
            print(f"[INFO] 画像抽出済み: {out_dir} ({len(jpg_files)}枚)")
            return out_dir
        else:
            print(f"[WARN] 画像ディレクトリは存在しますが画像枚数が不足しています。再抽出します: {out_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)

    # .lockファイル削除処理（hevcファイルの親ディレクトリ内）
    hevc_dir = os.path.dirname(hevc_path)
    for f in os.listdir(hevc_dir):
        if f.endswith('.lock'):
            try:
                os.remove(os.path.join(hevc_dir, f))
                print(f"[INFO] .lockファイル削除: {f}")
            except Exception as e:
                print(f"[WARN] .lockファイル削除失敗: {f} ({e})")

    jpg_pattern = os.path.join(out_dir, "%06d.jpg")
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", hevc_path,
        "-vf", f"fps={fps}",
        jpg_pattern
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("[ERROR] ffmpeg failed:")
        print(result.stderr.decode())
        print(f"[ERROR] ffmpegコマンド: {' '.join(cmd)}")
        raise RuntimeError("ffmpeg error")
    print(f"[INFO] Extracted images to {out_dir}")
    return out_dir
    import subprocess
    # 出力ディレクトリがなければ自動生成
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    jpg_pattern = os.path.join(out_dir, "%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", hevc_path,
        "-vf", f"fps={fps}",
        jpg_pattern
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("[ERROR] ffmpeg failed:", result.stderr.decode())
        raise RuntimeError("ffmpeg error")
    print(f"[INFO] Extracted images to {out_dir}")
    return out_dir


# 画像特徴抽出（車線・動きベクトルに限定）
def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()

def extract_features(img_dir, future_offset=100):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if len(imgs) == 0:
        raise RuntimeError(f"画像が見つかりません: {img_dir}")
    max_idx = len(imgs) - future_offset
    def process_one(i):
        img = cv2.imread(imgs[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        hist = cv2.calcHist([gray], [0], None, [32], [0,256]).flatten()
        future_idx = i + future_offset
        future_img = cv2.imread(imgs[future_idx])
        future_gray = cv2.cvtColor(future_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, future_gray)
        diff_hist = cv2.calcHist([diff], [0], None, [16], [0,256]).flatten()
        feat = np.concatenate([hist, diff_hist])
        return feat
    features = [None] * max_idx
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_one, i): i for i in range(max_idx)}
        for idx, future in enumerate(as_completed(futures)):
            i = futures[future]
            features[i] = future.result()
            print_progress_bar(idx+1, max_idx, prefix='特徴抽出', suffix='完了', length=40)
    features = np.array(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

# PCAで次元圧縮
# KMeansでクラスタリング
# クラスタごとにdesireラベルを割り当て（意味付けは後処理で人手/ルールで対応）
def unsupervised_desire_clustering(features, n_clusters=None):
    # 少量データ時は主成分数を自動調整
    n_samples, n_feats = features.shape
    n_pca = min(16, n_samples, n_feats)
    pca = PCA(n_components=n_pca)
    reduced = pca.fit_transform(features)
    # クラスタ数自動推定（データ数が少ない場合は2〜8で最適を探索）
    if n_clusters is None:
        from sklearn.metrics import silhouette_score
        best_score = -1
        best_k = 2
        for k in range(2, min(8, n_samples)+1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(reduced)
            score = silhouette_score(reduced, labels) if len(set(labels)) > 1 else -1
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(reduced)
    return clusters, reduced, kmeans, n_clusters

if __name__ == "__main__":
    # hevc_datas配下を再帰探索し、全てのecamera.hevcファイルを収集
    hevc_infos = []  # (hevc_path, parent_dir) のリスト
    for root, dirs, files in os.walk(HEVC_DATADIR):
        for file in files:
            if file == "ecamera.hevc":
                parent_dir = os.path.basename(root)
                hevc_infos.append((os.path.join(root, file), parent_dir))
    total_files = len(hevc_infos)
    if total_files == 0:
        print("[INFO] 対象となるecamera.hevcファイルがありません。処理を終了します。")
    else:
        all_features = []
        frame_map = []
        img_path_map = []
        for idx, (hevc_path, parent_dir) in enumerate(hevc_infos):
            print_progress_bar(idx, total_files, prefix='全体進捗', suffix='処理中', length=40)
            # 画像抽出ディレクトリも親ディレクトリ名で区別
            out_img_dir = os.path.abspath(extract_images_from_hevc(hevc_path, os.path.join(IMG_DIR, parent_dir), fps=10))
            features = extract_features(out_img_dir, future_offset=100)
            all_features.append(features)
            for i in range(features.shape[0]):
                base = os.path.splitext(os.path.basename(hevc_path))[0]
                frame_map.append(f"{parent_dir}/{base}:frame{i}")
                img_path_map.append(os.path.join(out_img_dir, f"{i+1:06d}.jpg"))
        print_progress_bar(total_files, total_files, prefix='全体進捗', suffix='完了', length=40)
        all_features = np.concatenate(all_features, axis=0)
        print_progress_bar(0, 1, prefix='クラスタリング', suffix='開始', length=40)
        clusters, reduced, kmeans, n_clusters = unsupervised_desire_clustering(all_features, n_clusters=None)
        print_progress_bar(1, 1, prefix='クラスタリング', suffix='完了', length=40)
        desire_labels_result = [DESIRE_LABELS[c % len(DESIRE_LABELS)] for c in clusters]
        np.save("unsupervised_desire_clusters_all.npy", clusters)
        # HTML形式で画像サンプル付きラベル出力
        with open("unsupervised_desire_labels_all.html", "w") as f:
            f.write("<html><body><table border='1'><tr><th>Frame</th><th>Label</th><th>Sample</th></tr>\n")
            for idx2, label in enumerate(desire_labels_result):
                img_path = img_path_map[idx2]
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_large = cv2.resize(img, (150, 150))
                        _, buf = cv2.imencode('.jpg', img_large)
                        b64 = base64.b64encode(buf).decode('utf-8')
                        img_tag = f"<img src='data:image/jpeg;base64,{b64}' width='150' height='150'/>"
                    else:
                        img_tag = "(画像なし)"
                except Exception as e:
                    img_tag = f"(画像エラー: {e})"
                f.write(f"<tr><td>{frame_map[idx2]}</td><td>{label}</td><td>{img_tag}</td></tr>\n")
            f.write("</table></body></html>\n")
        dump(kmeans, "unsupervised_desire_kmeans_all.joblib")
        print(f"[INFO] 全データクラスタリング完了。推定クラスタ数: {n_clusters}。クラスタ番号・ラベル名・モデルファイルを保存しました。画像サンプルはHTMLで出力されます。")

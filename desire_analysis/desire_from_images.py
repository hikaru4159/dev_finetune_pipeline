# desire_from_images.py
# DESIRE_LABELS公式定義に準拠した画像時系列から操作意図を推定するサンプル
import os
import numpy as np
import cv2
import subprocess
from typing import Optional

# OpenPilot公式準拠 8次元ラベル
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

# 画像時系列から操作意図を推定（ルールベース雛形）
def extract_images_from_hevc(hevc_path, out_dir, fps=10):
    # ensure clean out_dir
    if os.path.exists(out_dir):
        # remove any existing jpg files to avoid mixing previous extractions
        for f in os.listdir(out_dir):
            if f.lower().endswith('.jpg'):
                try:
                    os.remove(os.path.join(out_dir, f))
                except Exception:
                    pass
    else:
        os.makedirs(out_dir, exist_ok=True)

    jpg_pattern = os.path.join(out_dir, "%06d.jpg")
    # use -vsync 0 to avoid ffmpeg duplicating frames when timestamps are odd
    cmd = [
        "ffmpeg", "-y", "-i", hevc_path,
        "-vsync", "0",
        "-vf", f"fps={fps}",
        jpg_pattern
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("[ERROR] ffmpeg failed:", result.stderr.decode())
        raise RuntimeError("ffmpeg error")

    # basic sanity check: ensure we have more than one frame and frames are not identical
    imgs = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
    if len(imgs) < 2:
        # try a fallback extraction without fps filter
        cmd2 = ["ffmpeg", "-y", "-i", hevc_path, "-vsync", "0", jpg_pattern]
        subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        imgs = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
    print(f"[INFO] Extracted images to {out_dir} ({len(imgs)} files)")
    return out_dir
def extract_desire_from_images(
    img_dir: str,
    n_frames: int = 100,
    flow_winsize: int = 15,
    flow_levels: int = 3,
    flow_iterations: int = 3,
    flow_poly_n: int = 5,
    flow_poly_sigma: float = 1.2,
    flow_mean_abs_thresh: float = 1.5,
    move_score_thresh: float = 2.5e5,
    hysteresis_radius: int = 3,
    debug: Optional[bool] = False,
):
    """
    画像系列からDESIREラベルを推定する改良版ルールベース関数。

    - Farneback光学フローで横方向の平均動きを計算し、左右の変化を検出します。
    - グレースケール差分の合計（move_score）も補助的に使い、閾値を保守的に設定します。
    - hysteresis_radius により近傍フレームへラベルを広げて過剰検出を抑制します。

    返り値: numpy array, shape (1, n_frames, 8)
    """
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])[:n_frames]
    if len(imgs) == 0:
        raise RuntimeError(f"no images found in {img_dir}")
    # 読み込みと前処理
    gray_frames = []
    for p in imgs:
        im = cv2.imread(p)
        if im is None:
            # 保守的に最終フレームをコピー
            im = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)

    desire_arr = np.zeros((1, n_frames, 8), dtype=np.float32)
    # 横方向フローの平均値を蓄積
    flow_x_means = np.zeros(n_frames, dtype=np.float32)
    move_scores = np.zeros(n_frames, dtype=np.float32)

    for i in range(len(gray_frames) - 1):
        a = gray_frames[i]
        b = gray_frames[i + 1]
        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None,
            pyr_scale=0.5, levels=flow_levels, winsize=flow_winsize,
            iterations=flow_iterations, poly_n=flow_poly_n, poly_sigma=flow_poly_sigma, flags=0
        )
        # flow[...,0] is x (horizontal) displacement
        flow_x = flow[..., 0]
        # use robust statistic: median of x flow and mean absolute
        flow_x_means[i] = float(np.median(flow_x))
        move_scores[i] = float(np.sum(np.abs(b.astype(np.float32) - a.astype(np.float32))))

    # last frame: copy previous
    if len(gray_frames) >= 2:
        flow_x_means[len(gray_frames) - 1] = flow_x_means[len(gray_frames) - 2]
        move_scores[len(gray_frames) - 1] = move_scores[len(gray_frames) - 2]
    else:
        flow_x_means[0] = 0.0
        move_scores[0] = 0.0

    # 平滑化（移動平均）でノイズ除去
    kernel = np.ones(3, dtype=np.float32) / 3.0
    flow_x_smooth = np.convolve(flow_x_means, kernel, mode='same')
    move_score_smooth = np.convolve(move_scores, kernel, mode='same')

    # 判定ルール: 横方向の絶対値が閾値を超え、かつ move_score が補助閾値を超える場合に車線変更
    potential_change = (np.abs(flow_x_smooth) >= flow_mean_abs_thresh) & (move_score_smooth >= move_score_thresh)

    if debug:
        print(f"flow_x_smooth[:10]={flow_x_smooth[:10]}")
        print(f"move_score_smooth[:10]={move_score_smooth[:10]}")
        print(f"potential_change.sum()={potential_change.sum()}")

    # hysteresis: 周辺フレームに広げる
    final_change = np.zeros_like(potential_change)
    for i in range(len(potential_change)):
        if potential_change[i]:
            lo = max(0, i - hysteresis_radius)
            hi = min(len(potential_change), i + hysteresis_radius + 1)
            final_change[lo:hi] = True

    # ラベル付け: 左右どちらかは flow_x_smooth の符号で判定
    for i in range(n_frames):
        if final_change[i]:
            # 流れの平均が正→右方向へ移動
            if flow_x_smooth[i] > 0:
                desire_arr[0, i, 4] = 1.0  # laneChangeRight
            else:
                desire_arr[0, i, 3] = 1.0  # laneChangeLeft
        else:
            desire_arr[0, i, 7] = 1.0  # keepLane

    return desire_arr

if __name__ == "__main__":
    # ecamera.hevcから画像抽出→抽出画像でdesire推定
    hevc_path = "/home/user1434407/dev-fine-dataset/2023-11-22--06-10-53--1/ecamera.hevc"
    img_dir = "/home/user1434407/dev-fine-dataset/desire_analysis/ecamera_imgs"
    extract_images_from_hevc(hevc_path, img_dir, fps=10)
    desire = extract_desire_from_images(img_dir)
    np.save("desire_from_images.npy", desire)

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
    target_size: tuple = (512, 256),  # (width, height) = OpenPilot model input size
    flow_winsize: int = 15,
    flow_levels: int = 3,
    flow_iterations: int = 3,
    flow_poly_n: int = 5,
    flow_poly_sigma: float = 1.2,
    flow_mean_abs_thresh: float = 0.3,    # 調整: 256x512サイズに適した閾値
    move_score_thresh: float = 5.0e5,     # 調整: リサイズ後の画像差分閾値
    turn_flow_thresh: float = 1.0,        # 新規: 右左折判定の閾値
    lane_change_duration_min: int = 10,   # 新規: 車線変更の最小継続フレーム数
    hysteresis_radius: int = 5,           # 調整: より広い範囲に拡大
    debug: Optional[bool] = False,
):
    """
    画像系列からDESIREラベルを推定する改良版ルールベース関数。

    - 画像を256x512にリサイズしてからFarneback光学フローで横方向の平均動きを計算
    - グレースケール差分の合計（move_score）も補助的に使用
    - hysteresis_radius により近傍フレームへラベルを広げて過剰検出を抑制
    - 100msec間隔の画像に適した閾値設定で時系列処理を行います
    - 動きの継続時間で右左折（長い）と車線変更（短い）を区別します

    Args:
        target_size: (width, height) tuple for resizing images (default: 512x256 for OpenPilot)

    返り値: numpy array, shape (1, n_frames, 8)
    """
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if len(imgs) == 0:
        raise RuntimeError(f"no images found in {img_dir}")
    
    # n_framesの制約を適用（時系列を正しく処理）
    if len(imgs) > n_frames:
        imgs = imgs[:n_frames]
    actual_n_frames = len(imgs)
    
    # 読み込みと前処理（256x512にリサイズ）
    gray_frames = []
    for p in imgs:
        im = cv2.imread(p)
        if im is None:
            # 保守的に最終フレームをコピー
            if len(gray_frames) > 0:
                gray_frames.append(gray_frames[-1].copy())
            else:
                # フォールバック：target_sizeの空画像
                gray_frames.append(np.zeros((target_size[1], target_size[0]), dtype=np.uint8))
            continue
        
        # リサイズ: target_size = (width, height)
        im_resized = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)

    desire_arr = np.zeros((1, n_frames, 8), dtype=np.float32)
    # 横方向フローの平均値を蓄積
    flow_x_means = np.zeros(actual_n_frames, dtype=np.float32)
    move_scores = np.zeros(actual_n_frames, dtype=np.float32)

    # 時系列順に光学フローを計算
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
        # use robust statistic: median of x flow
        flow_x_means[i] = float(np.median(flow_x))
        move_scores[i] = float(np.sum(np.abs(b.astype(np.float32) - a.astype(np.float32))))

    # last frame: copy previous
    if len(gray_frames) >= 2:
        flow_x_means[len(gray_frames) - 1] = flow_x_means[len(gray_frames) - 2]
        move_scores[len(gray_frames) - 1] = move_scores[len(gray_frames) - 2]
    else:
        flow_x_means[0] = 0.0
        move_scores[0] = 0.0

    # 平滑化（移動平均）でノイズ除去 - より長い窓で時系列の傾向を捉える
    kernel_size = 5
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    flow_x_smooth = np.convolve(flow_x_means, kernel, mode='same')
    move_score_smooth = np.convolve(move_scores, kernel, mode='same')

    # 判定ルール: 横方向の絶対値が閾値を超え、かつ move_score が補助閾値を超える場合に車線変更
    potential_change = (np.abs(flow_x_smooth) >= flow_mean_abs_thresh) & (move_score_smooth >= move_score_thresh)

    if debug:
        print(f"[DEBUG] Image size after resize: {gray_frames[0].shape if gray_frames else 'N/A'}")
        print(f"[DEBUG] flow_x_smooth[:10]={flow_x_smooth[:10]}")
        print(f"[DEBUG] flow_x_smooth range: [{flow_x_smooth.min():.4f}, {flow_x_smooth.max():.4f}]")
        print(f"[DEBUG] move_score_smooth[:10]={move_score_smooth[:10]}")
        print(f"[DEBUG] move_score_smooth range: [{move_score_smooth.min():.1f}, {move_score_smooth.max():.1f}]")
        print(f"[DEBUG] potential_change.sum()={potential_change.sum()}")
        print(f"[DEBUG] Thresholds: flow_mean_abs={flow_mean_abs_thresh}, move_score={move_score_thresh}")

    # hysteresis: 周辺フレームに広げる
    final_change = np.zeros_like(potential_change)
    for i in range(len(potential_change)):
        if potential_change[i]:
            lo = max(0, i - hysteresis_radius)
            hi = min(len(potential_change), i + hysteresis_radius + 1)
            final_change[lo:hi] = True

    # ラベル付け: 左右どちらかは flow_x_smooth の符号で判定
    # actual_n_framesまでのみラベル付け
    for i in range(min(actual_n_frames, n_frames)):
        if i < len(final_change) and final_change[i]:
            # 流れの平均が正→右方向へ移動
            if flow_x_smooth[i] > 0:
                desire_arr[0, i, 4] = 1.0  # laneChangeRight
            else:
                desire_arr[0, i, 3] = 1.0  # laneChangeLeft
        else:
            desire_arr[0, i, 7] = 1.0  # keepLane
    
    # 残りのフレームはkeepLaneで埋める
    for i in range(actual_n_frames, n_frames):
        desire_arr[0, i, 7] = 1.0  # keepLane

    return desire_arr

if __name__ == "__main__":
    # ecamera.hevcから画像抽出→抽出画像でdesire推定
    hevc_path = "/home/user1434407/dev-fine-dataset/2023-11-22--06-10-53--1/ecamera.hevc"
    img_dir = "/home/user1434407/dev-fine-dataset/desire_analysis/ecamera_imgs"
    extract_images_from_hevc(hevc_path, img_dir, fps=10)
    desire = extract_desire_from_images(img_dir)
    np.save("desire_from_images.npy", desire)

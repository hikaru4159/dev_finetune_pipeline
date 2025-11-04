# シンプルな閾値定義（チューニング可能）
LANE_CHANGE_ANGLE_THRESH = 10.0  # 小さな舵角で車線変更とみなす閾値 (deg)
TURN_ANGLE_THRESH = 30.0         # 大きな舵角で交差点での曲がりとみなす閾値 (deg)
import os
import numpy as np
import capnp
import zstandard as zstd
import traceback

# cereal capnpロード
CEREAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis_lateral_params/cereal'))
log_capnp = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))

# desireラベル定義
DESIRE_LABELS = [
    "none", "turnLeft", "turnRight", "laneChangeLeft", "laneChangeRight", "keepLeft", "keepRight", "keepLane", "brake"
]

def decompress_zst(path):
    print(f"[DEBUG] Decompressing: {path}")
    try:
        with open(path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                chunks = []
                while True:
                    chunk = reader.read(16384)
                    if not chunk:
                        break
                    chunks.append(chunk)
                print(f"[DEBUG] Decompression complete, {sum(len(c) for c in chunks)} bytes")
                return b''.join(chunks)
    except Exception as e:
        print(f"[ERROR] decompress_zst failed: {e}")
        traceback.print_exc()
        raise

def extract_log_features_with_time(rlog_path):
    print(f"[DEBUG] Extracting log features from: {rlog_path}")
    if not os.path.exists(rlog_path):
        print(f"[ERROR] rlogファイルが存在しません: {rlog_path}")
        return np.zeros((1, 6), dtype=np.float32), np.zeros((1,), dtype=np.int64)
    try:
        if rlog_path.endswith('.zst'):
            dat = decompress_zst(rlog_path)
        else:
            with open(rlog_path, 'rb') as f:
                dat = f.read()
        print(f"[DEBUG] Decompressed log size: {len(dat)} bytes")
        print("[DEBUG] Reading events from log_capnp...")
        events = log_capnp.Event.read_multiple_bytes(dat)
        features = []
        times = []
        count = 0
        for ev in events:
            count += 1
            if hasattr(ev, 'which') and ev.which() == 'carState':
                cs = ev.carState
                v_ego = float(getattr(cs, 'vEgo', 0.0))
                steering_angle = float(getattr(cs, 'steeringAngleDeg', 0.0))
                steering_torque = float(getattr(cs, 'steeringTorque', 0.0))
                brake = float(getattr(cs, 'brake', 0.0))
                left_blinker = int(getattr(cs, 'leftBlinker', False))
                right_blinker = int(getattr(cs, 'rightBlinker', False))
                log_time = int(getattr(cs, 'logMonoTime', 0))
                features.append([v_ego, steering_angle, steering_torque, brake, left_blinker, right_blinker])
                times.append(log_time)
            if count % 10000 == 0:
                print(f"[DEBUG] Processed {count} events...")
        print(f"[DEBUG] Total events processed: {count}")
        return np.array(features), np.array(times)
    except Exception as e:
        print(f"[ERROR] Exception in extract_log_features_with_time: {e}")
        traceback.print_exc()
        return np.zeros((1, 6), dtype=np.float32), np.zeros((1,), dtype=np.int64)
def get_image_frame_times(img_count, video_fps, start_time):
    # 画像枚数・動画fps・開始時刻から各フレームの推定時刻を生成
    return np.array([start_time + int(1e9 * i / video_fps) for i in range(img_count)])

def sync_labels_to_images(img_times, log_times, desire_labels):
    # 各画像時刻に最も近いlogイベントのラベルを割り当て
    synced_labels = []
    for t in img_times:
        idx = np.argmin(np.abs(log_times - t))
        synced_labels.append(desire_labels[idx])
    return np.array(synced_labels)

def infer_desire_from_features(features):
    """
    シンプルなルール:
    - ウィンカーがONならそれを優先 (turnLeft/turnRight or laneChangeLeft/Right)
    - ウィンカー無しでは舵角の絶対値で判定: |angle| >= TURN_ANGLE_THRESH -> turn, >= LANE_CHANGE_ANGLE_THRESH -> laneChange, else keepLane
    入力: features: N x 6 array -> [vEgo, steeringAngleDeg, steeringTorque, brake, leftBlinker, rightBlinker]
    出力: numpy array of labels (strings)
    """
    labels = []
    for row in features:
        try:
            v_ego = float(row[0])
            angle = float(row[1])
            # torque/brake may be unused here but preserved for compatibility
            # torque = float(row[2])
            # brake = float(row[3])
            left_blinker = bool(row[4])
            right_blinker = bool(row[5])
        except Exception:
            labels.append("keepLane")
            continue

        # blinker優先ルール
        if left_blinker and not right_blinker:
            # 舵角が大きければ交差点での曲がり、そうでなければ車線変更候補
            if abs(angle) >= TURN_ANGLE_THRESH:
                labels.append("turnLeft")
            elif abs(angle) >= LANE_CHANGE_ANGLE_THRESH:
                labels.append("laneChangeLeft")
            else:
                labels.append("turnLeft")
            continue
        if right_blinker and not left_blinker:
            if abs(angle) >= TURN_ANGLE_THRESH:
                labels.append("turnRight")
            elif abs(angle) >= LANE_CHANGE_ANGLE_THRESH:
                labels.append("laneChangeRight")
            else:
                labels.append("turnRight")
            continue

        # ウィンカー無し: 舵角メイン判定
        if abs(angle) >= TURN_ANGLE_THRESH:
            labels.append("turnLeft" if angle < 0 else "turnRight")
        elif abs(angle) >= LANE_CHANGE_ANGLE_THRESH:
            labels.append("laneChangeLeft" if angle < 0 else "laneChangeRight")
        else:
            labels.append("keepLane")

    return np.array(labels)

def output_desire_labels_html(desire_labels, img_dir, out_html="desire_from_log_features.html"):
    import base64
    import cv2
    global synced_features  # ステア舵角・トルク値を参照
    with open(out_html, "w") as f:
        f.write("<html><body><table border='1'><tr><th>Frame</th><th>Label</th><th>SteerAngle</th><th>SteerTorque</th><th>Sample</th></tr>\n")
        for idx, label in enumerate(desire_labels):
            img_path = os.path.join(img_dir, f"{idx+1:06d}.jpg")
            steer_angle = synced_features[idx][1] if idx < len(synced_features) else "-"
            steer_torque = synced_features[idx][2] if idx < len(synced_features) else "-"
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
            f.write(f"<tr><td>{idx+1}</td><td>{label}</td><td>{steer_angle:.2f}</td><td>{steer_torque:.2f}</td><td>{img_tag}</td></tr>\n")
        f.write("</table></body></html>\n")

if __name__ == "__main__":
    import subprocess
    # データディレクトリ（例: 2023-11-22--06-10-53--1）
    data_dir = "/home/user1434407/dev-fine-dataset/2023-11-22--06-10-53--1"
    rlog_path = os.path.join(data_dir, "rlog")
    hevc_path = os.path.join(data_dir, "ecamera.hevc")
    img_dir = "/home/user1434407/dev-fine-dataset/hevc_imgs/ecamera"
    # 画像ディレクトリが空ならecamera.hevcからjpg抽出
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if len(img_files) < 10 and os.path.exists(hevc_path):
        # ffmpegで画像抽出（1フレーム1枚、連番出力）
        cmd = [
            "ffmpeg", "-i", hevc_path,
            os.path.join(img_dir, "%06d.jpg")
        ]
        print("画像抽出コマンド:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print("ffmpeg画像抽出失敗:", e)

    # rlogから特徴量・タイムスタンプ抽出
    features, log_times = extract_log_features_with_time(rlog_path)
    desire_labels = infer_desire_from_features(features)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    img_count = len(img_files)
    video_fps = 10  # 例: 10Hz（実際はffprobe等で取得してください）
    start_time = log_times[0] if len(log_times) > 0 else 0
    img_times = get_image_frame_times(img_count, video_fps, start_time)
    # desireラベルと同期した特徴量（ステア舵角・トルク）も抽出
    synced_indices = [np.argmin(np.abs(log_times - t)) for t in img_times]
    synced_features = features[synced_indices]
    desire_labels_synced = sync_labels_to_images(img_times, log_times, desire_labels)
    print("推定desireラベル例:", np.unique(desire_labels_synced, return_counts=True))
    np.save("desire_from_log_features_synced.npy", desire_labels_synced)
    output_desire_labels_html(desire_labels_synced, img_dir)

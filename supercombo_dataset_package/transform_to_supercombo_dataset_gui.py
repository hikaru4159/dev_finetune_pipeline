#!/usr/bin/env python3
"""
GUI付き supercomboデータセット作成スクリプト
元のtransform_to_supercombo_dataset.pyの機能を維持しつつ、
入力フォルダ・出力フォルダをエクスプローラで選択できるUI（Tkinter）を追加。
"""
import os, sys, argparse, numpy as np, json, threading, gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox
# openpilot_096/openpilot/tools/lib までパスを通すことで__init__.py不要化
base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'openpilot_096/openpilot/tools/lib'))
sys.path.insert(0, os.path.join(base_dir, 'openpilot_096/cereal'))
from openpilot_096.openpilot.tools.lib.logreader import LogReader
# openpilot_096/openpilot/tools/lib までパスを通すことで__init__.py不要化
base_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(base_dir, 'openpilot_096/openpilot/tools/lib'))
sys.path.insert(0, os.path.join(base_dir, 'openpilot_096/cereal'))

# --- 1. 画像抽出 ---
def extract_input_imgs(log_dir, out_dir, n_frames=12, shape=(128,256)):
    import subprocess, glob, cv2
    # hevc_pathを引数で渡す。デフォルトはフロントカメラ `fcamera.hevc`（`dcamera` は使用しない）
    hevc_path = log_dir if isinstance(log_dir, str) and log_dir.endswith('.hevc') else os.path.join(log_dir, "fcamera.hevc")
    tmp_dir = os.path.join(out_dir, "_tmp_imgs")
    if os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            if f.lower().endswith('.jpg'):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except Exception:
                    pass
    else:
        os.makedirs(tmp_dir, exist_ok=True)
    jpg_pattern = os.path.join(tmp_dir, "%06d.jpg")
    # 10fps (100msecごと)で抽出
    cmd = [
        "ffmpeg", "-y", "-i", hevc_path,
        "-vsync", "0",
        "-vf", "fps=10",
        jpg_pattern
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    frames = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))
    frames = frames[-12:]
    # produce Y (luma) channel only to match supercombo expectation: (1,12,128,256)
    arr = np.zeros((1, 12, shape[0], shape[1]), dtype=np.float32)
    out_img_dir = os.path.join(out_dir, "output_imgs")
    os.makedirs(out_img_dir, exist_ok=True)
    # デフォルトの frame_times (nanoseconds 単位で 100ms 間隔)
    frame_times = [i * 1e8 for i in range(len(frames))]

    # 可能であれば rlog の camera timestamp を優先して使う
    try:
        # hevc の親ディレクトリに rlog(.zst) がある想定
        log_dir_parent = os.path.dirname(hevc_path)
        rlog_path = os.path.join(log_dir_parent, 'rlog.zst') if os.path.exists(os.path.join(log_dir_parent, 'rlog.zst')) else os.path.join(log_dir_parent, 'rlog')
        if os.path.exists(rlog_path):
            # LogReader を使って roadCameraState の timestampSof を収集
            lr = LogReader(rlog_path)
            ts_list = []
            for msg in lr:
                try:
                    if hasattr(msg, 'which') and msg.which() == 'roadCameraState':
                        ts = getattr(msg.roadCameraState, 'timestampSof', None)
                        if ts is not None:
                            ts_list.append(int(ts))
                except Exception:
                    continue
            if ts_list:
                # 最新の抽出されたフレーム数に合わせて末尾から採る
                take = min(len(frames), len(ts_list))
                if take > 0:
                    frame_times = ts_list[-take:]
                    print(f"[INFO] Using rlog timestamps for frame_times (source of truth) — {len(frame_times)} entries, avg interval ~{(np.mean(np.diff(frame_times))/1e9):.3f}s")
                    # ffprobe と rlog の差をチェック（ffprobe が返すfpsは raw hevc だと誤検出することがある）
                    try:
                        proc = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=avg_frame_rate","-of","default=noprint_wrappers=1:nokey=1", hevc_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        ff_avg = proc.stdout.strip()
                        if ff_avg:
                            # avg_frame_rate は "25/1" のように返る
                            if '/' in ff_avg:
                                num, den = ff_avg.split('/')
                                ff_fps = float(num) / float(den) if float(den) != 0 else float(num)
                            else:
                                ff_fps = float(ff_avg)
                            rlog_fps = len(frame_times) / ((frame_times[-1] - frame_times[0]) / 1e9) if len(frame_times) > 1 and (frame_times[-1] - frame_times[0])>0 else None
                            if rlog_fps and abs(rlog_fps - ff_fps) > 0.5:
                                print(f"[WARNING] ffprobe reports avg_frame_rate={ff_fps:.2f} but rlog camera timestamps indicate ~{rlog_fps:.2f} fps. Raw .hevc may lack timing metadata — using rlog timestamps.")
                    except Exception:
                        pass
    except Exception:
        # rlog が読めない場合はデフォルトの frame_times を使う
        pass
    for i, f in enumerate(frames):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (shape[1], shape[0]))
        # convert to luma (Y) channel using BGR->YUV and take Y, but cv2.cvtColor to GRAY is acceptable
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr[0, i] = y.astype(np.float32) / 255.0
        out_path = os.path.join(out_img_dir, f"{i:06d}.jpg")
        cv2.imwrite(out_path, img)
    import shutil
    for f in frames:
        try:
            os.remove(f)
        except Exception:
            pass
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    del img, frames
    gc.collect()
    # pad/truncate to exactly n_frames on time axis (pad at front with zeros)
    T = arr.shape[1]
    if T < n_frames:
        pad = np.zeros((1, n_frames - T, shape[0], shape[1]), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=1)
    elif T > n_frames:
        arr = arr[:, -n_frames:, :, :]
    return arr, frame_times
def extract_big_input_imgs(log_dir, out_dir, n_frames=12, shape=(128,256)):
    import subprocess, glob, cv2
    # hevc_pathを引数で渡す
    hevc_path = log_dir if isinstance(log_dir, str) and log_dir.endswith('.hevc') else os.path.join(log_dir, "ecamera.hevc")
    tmp_dir = os.path.join(out_dir, "_tmp_bigimgs")
    if os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            if f.lower().endswith('.jpg'):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except Exception:
                    pass
    else:
        os.makedirs(tmp_dir, exist_ok=True)
    jpg_pattern = os.path.join(tmp_dir, "%06d.jpg")
    # 10fps (100msecごと)で抽出
    cmd = [
        "ffmpeg", "-y", "-i", hevc_path,
        "-vsync", "0",
        "-vf", "fps=10",
        jpg_pattern
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    frames = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))
    frames = frames[-12:]
    arr = np.zeros((1, 12, shape[0], shape[1]), dtype=np.float32)
    out_img_dir = os.path.join(out_dir, "output_bigimgs")
    os.makedirs(out_img_dir, exist_ok=True)
    # デフォルトの frame_times
    frame_times = [i * 1e8 for i in range(len(frames))]
    # wide カメラ用に rlog の wideRoadCameraState タイムスタンプを優先取得
    try:
        log_dir_parent = os.path.dirname(hevc_path)
        rlog_path = os.path.join(log_dir_parent, 'rlog.zst') if os.path.exists(os.path.join(log_dir_parent, 'rlog.zst')) else os.path.join(log_dir_parent, 'rlog')
        if os.path.exists(rlog_path):
            lr = LogReader(rlog_path)
            ts_list = []
            for msg in lr:
                try:
                    if hasattr(msg, 'which') and msg.which() == 'wideRoadCameraState':
                        ts = getattr(msg.wideRoadCameraState, 'timestampSof', None)
                        if ts is not None:
                            ts_list.append(int(ts))
                except Exception:
                    continue
            if ts_list:
                take = min(len(frames), len(ts_list))
                if take > 0:
                    frame_times = ts_list[-take:]
                    print(f"[INFO] Using rlog wideRoadCameraState timestamps for frame_times — {len(frame_times)} entries, avg interval ~{(np.mean(np.diff(frame_times))/1e9):.3f}s")
                    try:
                        proc = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=avg_frame_rate","-of","default=noprint_wrappers=1:nokey=1", hevc_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        ff_avg = proc.stdout.strip()
                        if ff_avg:
                            if '/' in ff_avg:
                                num, den = ff_avg.split('/')
                                ff_fps = float(num) / float(den) if float(den) != 0 else float(num)
                            else:
                                ff_fps = float(ff_avg)
                            rlog_fps = len(frame_times) / ((frame_times[-1] - frame_times[0]) / 1e9) if len(frame_times) > 1 and (frame_times[-1] - frame_times[0])>0 else None
                            if rlog_fps and abs(rlog_fps - ff_fps) > 0.5:
                                print(f"[WARNING] ffprobe reports avg_frame_rate={ff_fps:.2f} but rlog wideRoadCameraState timestamps indicate ~{rlog_fps:.2f} fps. Raw .hevc may lack timing metadata — using rlog timestamps.")
                    except Exception:
                        pass
    except Exception:
        pass
    for i, f in enumerate(frames):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (shape[1], shape[0]))
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr[0, i] = y.astype(np.float32) / 255.0
        out_path = os.path.join(out_img_dir, f"{i:06d}.jpg")
        cv2.imwrite(out_path, img)
    import shutil
    for f in frames:
        try:
            os.remove(f)
        except Exception:
            pass
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    del img, frames
    gc.collect()
    # pad/truncate to exactly n_frames on time axis (pad at front)
    T = arr.shape[1]
    if T < n_frames:
        pad = np.zeros((1, n_frames - T, shape[0], shape[1]), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=1)
    elif T > n_frames:
        arr = arr[:, -n_frames:, :, :]
    return arr, frame_times
def extract_desire(log_dir, frame_times):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    DESIRE_NUM = 8
    # desire値を画像フレームタイミングに同期（最新100フレーム、直前値補完・許容時間1秒）
    vals = []
    try:
        lr = LogReader(rlog_path)
        msg_list = []
        for msg in lr:
            ts = getattr(msg, 'logMonoTime', None)
            if hasattr(msg, "desire") and ts is not None:
                # desire値を画像フレームタイミング（100msec）に同期（全フレーム分、直前値補完・許容時間150msec）
                d = int(msg.desire)
                onehot = np.zeros(DESIRE_NUM, dtype=np.float32)
                if 0 <= d < DESIRE_NUM:
                    onehot[d] = 1.0
                msg_list.append((ts, onehot))
        # 直前値補完（許容時間150msec=1.5e8）
        last_val = np.zeros(DESIRE_NUM, dtype=np.float32)
        last_ts = None
        for ft in frame_times:
            candidates = [(t, v) for t, v in msg_list if t <= ft]
            if candidates:
                t, v = candidates[-1]
                if last_ts is None or ft - t <= 1.5e8:
                    last_val = v
                    last_ts = t
                else:
                    last_val = np.zeros(DESIRE_NUM, dtype=np.float32)
            else:
                last_val = np.zeros(DESIRE_NUM, dtype=np.float32)
            vals.append(last_val)
    except Exception:
        pass
    arr = np.array(vals, dtype=np.float32)[None, ...] if vals else np.zeros((1,len(frame_times),DESIRE_NUM), dtype=np.float32)
    return arr
def extract_traffic_convention(log_dir):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    arr = np.array([[0, 1]], dtype=np.float32)
    try:
        lr = LogReader(rlog_path)
        for msg in lr:
            if hasattr(msg, "trafficConvention"):
                v = int(msg.trafficConvention)
                if v == 0:
                    arr[0, 0] = 1.0
                    arr[0, 1] = 0.0
                elif v == 1:
                    arr[0, 0] = 0.0
                    arr[0, 1] = 1.0
                break
    except Exception:
        pass
    return arr
def extract_lateral_control_params(log_dir, frame_times):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    vals = []
    try:
        lr = LogReader(rlog_path)
        msg_list = []
        for msg in lr:
            ts = getattr(msg, 'logMonoTime', None)
            w = msg.which() if hasattr(msg, 'which') else None
            v_ego = float(msg.carState.vEgo) if w == "carState" and hasattr(msg.carState, "vEgo") else None
            steer_delay = float(msg.lateralPlan.steerDelay) if w == "lateralPlan" and hasattr(msg.lateralPlan, "steerDelay") else None
            if v_ego is not None and steer_delay is not None and ts is not None:
                msg_list.append((ts, v_ego, steer_delay))
        # 直前値補完（許容時間150msec=1.5e8）
        last_val = np.zeros(2, dtype=np.float32)
        last_ts = None
        for ft in frame_times:
            candidates = [(t, v, s) for t, v, s in msg_list if t <= ft]
            if candidates:
                t, v, s = candidates[-1]
                if last_ts is None or ft - t <= 1.5e8:
                    last_val = np.array([v, s], dtype=np.float32)
                    last_ts = t
                else:
                    last_val = np.zeros(2, dtype=np.float32)
            else:
                last_val = np.zeros(2, dtype=np.float32)
            vals.append(last_val)
    except Exception:
        pass
    arr = np.array(vals, dtype=np.float32)[None, ...] if vals else np.zeros((1,len(frame_times),2), dtype=np.float32)
    return arr
def extract_prev_desired_curv(log_dir, frame_times):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    # desiredCurvature値を画像フレームタイミングに同期（最新100フレーム、直前値補完・許容時間1秒）
    vals = []
    try:
        lr = LogReader(rlog_path)
        msg_list = []
        for msg in lr:
            ts = getattr(msg, 'logMonoTime', None)
            w = msg.which() if hasattr(msg, 'which') else None
            if w == "lateralPlan" and hasattr(msg.lateralPlan, "desiredCurvature") and ts is not None:
                msg_list.append((ts, float(msg.lateralPlan.desiredCurvature)))
        # 直前値補完（許容時間150msec=1.5e8）
        last_val = 0.0
        last_ts = None
        for ft in frame_times:
            candidates = [(t, v) for t, v in msg_list if t <= ft]
            if candidates:
                t, v = candidates[-1]
                if last_ts is None or ft - t <= 1.5e8:
                    last_val = v
                    last_ts = t
                else:
                    last_val = 0.0
            else:
                last_val = 0.0
            vals.append([last_val])
    except Exception:
        pass
    arr = np.array(vals, dtype=np.float32)[None, ...] if vals else np.zeros((1,len(frame_times),1), dtype=np.float32)
    return arr
def extract_nav_features(log_dir):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    arr = np.zeros((1, 256), dtype=np.float32)
    try:
        lr = LogReader(rlog_path)
        for msg in lr:
            w = msg.which() if hasattr(msg, 'which') else None
            if w == "navFeatures" and hasattr(msg, "navFeatures"):
                vals = list(msg.navFeatures)
                arr[0, :min(256, len(vals))] = vals[:256]
                break
    except Exception:
        pass
    return arr
def extract_nav_instructions(log_dir):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    arr = np.zeros((1, 150), dtype=np.float32)
    try:
        lr = LogReader(rlog_path)
        for msg in lr:
            w = msg.which() if hasattr(msg, 'which') else None
            if w == "navInstructions" and hasattr(msg, "navInstructions"):
                vals = list(msg.navInstructions)
                arr[0, :min(150, len(vals))] = vals[:150]
                break
    except Exception:
        pass
    return arr
def extract_features_buffer(log_dir, frame_times):
    rlog_path = os.path.join(log_dir, "rlog.zst")
    # featuresBuffer値を画像フレームタイミングに同期（最新99フレーム、直前値補完・許容時間1秒）
    vals = []
    try:
        lr = LogReader(rlog_path)
        msg_list = []
        for msg in lr:
            ts = getattr(msg, 'logMonoTime', None)
            w = msg.which() if hasattr(msg, 'which') else None
            if w == "featuresBuffer" and hasattr(msg, "featuresBuffer") and ts is not None:
                arr_ = np.array(msg.featuresBuffer)
                if arr_.ndim == 1 and arr_.shape[0] == 512:
                    msg_list.append((ts, arr_))
                elif arr_.ndim == 2 and arr_.shape[1] == 512:
                    for row in arr_:
                        msg_list.append((ts, row))
        # 直前値補完（許容時間150msec=1.5e8）
        last_val = np.zeros(512, dtype=np.float32)
        last_ts = None
        for ft in frame_times:
            candidates = [(t, v) for t, v in msg_list if t <= ft]
            if candidates:
                t, v = candidates[-1]
                if last_ts is None or ft - t <= 1.5e8:
                    last_val = v
                    last_ts = t
                else:
                    last_val = np.zeros(512, dtype=np.float32)
            else:
                last_val = np.zeros(512, dtype=np.float32)
            vals.append(last_val)
    except Exception:
        pass
    arr = np.array(vals, dtype=np.float32)[None, ...] if vals else np.zeros((1,len(frame_times),512), dtype=np.float32)
    return arr
def save_dataset(out_dir, **kwargs):
    for k, v in kwargs.items():
        np.save(os.path.join(out_dir, f"{k}.npy"), v)
    meta = {k: v.shape for k,v in kwargs.items()}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

# --- 5. 実行 ---
def find_log_dirs(log_dir):
    # 再帰的にrlog.zstを含むディレクトリを探索
    result = []
    print(f"[DEBUG] find_log_dirs: Searching in {log_dir}")
    for root, dirs, files in os.walk(log_dir):
        print(f"[DEBUG] Checking {root}, files: {files}")
        if 'rlog.zst' in files:
            print(f"[DEBUG] find_log_dirs: rlog.zst found in {root}")
            result.append(root)
        elif 'rlog' in files:
            print(f"[DEBUG] find_log_dirs: rlog found in {root} (no rlog.zst)")
            result.append(root)
    if not result:
        print(f"[WARNING] find_log_dirs: No rlog.zst or rlog found under {log_dir}")
    return result

def process_log_dir(args_tuple):
    log_dir, out_dir_root = args_tuple
    import traceback
    run_name = os.path.basename(os.path.normpath(log_dir))
    # If caller already passed an out_dir that targets this exact run (e.g. /out_root/<run_name>),
    # avoid creating a double-nested folder by checking the end of out_dir_root.
    if os.path.basename(os.path.normpath(out_dir_root)) == run_name:
        out_dir = out_dir_root
    else:
        out_dir = os.path.join(out_dir_root, run_name)
    # Do NOT create out_dir here: only create it once we confirm required inputs exist.
    print(f"[DEBUG] process_log_dir: resolved out_dir = {out_dir} (not created yet)")
    try:
        # --- input_imgs用カメラファイル選択 ---
        # 明示的に front カメラは `fcamera.hevc` のみを使用する（dcamera は無視）
        input_camera_files = ["fcamera.hevc"]
        input_selected_file = None
        for fname in input_camera_files:
            fpath = os.path.join(log_dir, fname)
            if os.path.exists(fpath):
                input_selected_file = fpath
                print(f"[DEBUG] Using input camera file: {input_selected_file}")
                break
        if not input_selected_file:
            print(f"[SKIP] required camera file not found (fcamera.hevc) in {log_dir} - skipping this run")
            return

        # --- big_input_imgs用カメラファイル選択 ---
        big_camera_files = ["ecamera.hevc"]
        big_selected_file = None
        for fname in big_camera_files:
            fpath = os.path.join(log_dir, fname)
            if os.path.exists(fpath):
                big_selected_file = fpath
                print(f"[DEBUG] Using big camera file: {big_selected_file}")
                break
        if not big_selected_file:
            print(f"[WARNING] ecamera.hevc が存在しません: {log_dir}")

        # --- input_imgs画像サイズ取得 ---
        import cv2
        cap = cv2.VideoCapture(input_selected_file)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"inputカメラファイルから画像が取得できません: {input_selected_file}")
        h, w = frame.shape[:2]
        print(f"[DEBUG] Input camera image size: {h}x{w}")

        # --- big_input_imgs画像サイズ取得（あれば） ---
        if big_selected_file:
            cap_big = cv2.VideoCapture(big_selected_file)
            ret_big, frame_big = cap_big.read()
            cap_big.release()
            if not ret_big:
                print(f"[WARNING] bigカメラファイルから画像が取得できません: {big_selected_file}")
                h_big, w_big = 128, 256
            else:
                h_big, w_big = frame_big.shape[:2]
            print(f"[DEBUG] Big camera image size: {h_big}x{w_big}")
        else:
            h_big, w_big = 128, 256

        # create output dir now that required inputs exist
        os.makedirs(out_dir, exist_ok=True)

        # --- 全フレーム抽出 ---
        # For downstream compatibility, always produce (128,256) luma images
        input_imgs, frame_times = extract_input_imgs(input_selected_file, out_dir, shape=(128, 256))
        if big_selected_file:
            big_input_imgs, _ = extract_big_input_imgs(big_selected_file, out_dir, shape=(128, 256))
        else:
            # match extract_big_input_imgs output shape: (1, n_frames, H, W)
            big_input_imgs = np.zeros((1, 12, h_big, w_big), dtype=np.float32)

        print(f"[DEBUG] extract_desire: {log_dir}")
        desire = extract_desire(log_dir, frame_times)
        print(f"[DEBUG] extract_traffic_convention: {log_dir}")
        traffic_convention = extract_traffic_convention(log_dir)
        print(f"[DEBUG] extract_lateral_control_params: {log_dir}")
        lateral_control_params = extract_lateral_control_params(log_dir, frame_times)
        print(f"[DEBUG] extract_prev_desired_curv: {log_dir}")
        prev_desired_curv = extract_prev_desired_curv(log_dir, frame_times)
        print(f"[DEBUG] extract_nav_features: {log_dir}")
        nav_features = extract_nav_features(log_dir)
        print(f"[DEBUG] extract_nav_instructions: {log_dir}")
        nav_instructions = extract_nav_instructions(log_dir)
        print(f"[DEBUG] extract_features_buffer: {log_dir}")
        features_buffer = extract_features_buffer(log_dir, frame_times)
        print(f"[DEBUG] save_dataset: {out_dir}")

        # memo.txt仕様に合わせて最新Nフレームにスライス（shape厳密化）
        input_imgs_out = input_imgs
        big_input_imgs_out = big_input_imgs

        # pad/truncate desire to (1,100,8)
        DESIRE_NUM = 8
        T_des = desire.shape[1]
        if T_des >= 100:
            desire_out = desire[:, -100:, :]
        else:
            pad = np.zeros((1, 100 - T_des, DESIRE_NUM), dtype=np.float32)
            desire_out = np.concatenate([pad, desire], axis=1)

        # pad/truncate prev_desired_curv to (1,100,1)
        T_prev = prev_desired_curv.shape[1]
        if T_prev >= 100:
            prev_desired_curv_out = prev_desired_curv[:, -100:, :]
        else:
            pad = np.zeros((1, 100 - T_prev, 1), dtype=np.float32)
            prev_desired_curv_out = np.concatenate([pad, prev_desired_curv], axis=1)

        # pad/truncate features_buffer to (1,99,512)
        T_fb = features_buffer.shape[1]
        if T_fb >= 99:
            features_buffer_out = features_buffer[:, -99:, :]
        else:
            pad = np.zeros((1, 99 - T_fb, 512), dtype=np.float32)
            features_buffer_out = np.concatenate([pad, features_buffer], axis=1)

        # lateral_control_params expected as (1,2) (per-run params). Reduce time series to last known value.
        if lateral_control_params is None:
            lateral_control_params_out = np.zeros((1,2), dtype=np.float32)
        else:
            try:
                if lateral_control_params.ndim == 3 and lateral_control_params.shape[1] > 0:
                    last = lateral_control_params[0, -1, :]
                else:
                    # if shape is already (1,2)
                    last = lateral_control_params.flatten()[:2]
                lateral_control_params_out = np.array(last, dtype=np.float32).reshape(1,2)
            except Exception:
                lateral_control_params_out = np.zeros((1,2), dtype=np.float32)

        save_dataset(
            out_dir,
            input_imgs=input_imgs_out,
            big_input_imgs=big_input_imgs_out,
            desire=desire_out,
            traffic_convention=traffic_convention,
            lateral_control_params=lateral_control_params_out,
            prev_desired_curv=prev_desired_curv_out,
            nav_features=nav_features,
            nav_instructions=nav_instructions,
            features_buffer=features_buffer_out
        )
        # メモリ解放
        del input_imgs, big_input_imgs, desire, traffic_convention, lateral_control_params, prev_desired_curv, nav_features, nav_instructions, features_buffer
        gc.collect()
        print(f"[INFO] データセットを{out_dir}に保存しました")
    except Exception as e:
        print(f"[ERROR] {out_dir} で例外発生: {e}\n{traceback.format_exc()}")

def main(args):
    log_dirs = find_log_dirs(args.log_dir)
    out_dir_root = args.out_dir
    os.makedirs(out_dir_root, exist_ok=True)

    # 並列処理（プロセス数はCPUコア数に自動設定）
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_log_dir, (log_dir, out_dir_root)) for log_dir in log_dirs]
        for future in as_completed(futures):
            future.result()

# --- GUI部分 ---
def run_with_gui():
    def select_input_dir():
        path = filedialog.askdirectory(title="入力フォルダを選択")
        if path:
            input_var.set(path)
    def select_output_dir():
        path = filedialog.askdirectory(title="出力フォルダを選択")
        if path:
            output_var.set(path)
    def run_script():
        log_dir = input_var.get()
        out_dir = output_var.get()
        if not log_dir or not out_dir:
            messagebox.showerror("エラー", "入力・出力フォルダを選択してください")
            return
        def worker():
            import io, sys, os, traceback
            class DualWriter:
                def __init__(self, *writers):
                    self.writers = writers
                def write(self, s):
                    for w in self.writers:
                        w.write(s)
                def flush(self):
                    for w in self.writers:
                        w.flush()
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = DualWriter(buf, old_stdout)
            try:
                print(f"input_var: {input_var.get()}")
                print(f"output_var: {output_var.get()}")
                log_dir_norm = os.path.abspath(os.path.normpath(log_dir))
                out_dir_norm = os.path.abspath(os.path.normpath(out_dir))
                print(f"worker log_dir: {log_dir_norm}")
                print(f"worker out_dir: {out_dir_norm}")
                main_args = argparse.Namespace(log_dir=log_dir_norm, out_dir=out_dir_norm)
                try:
                    main(main_args)
                except Exception as e:
                    print(f"[ERROR] main() raised: {e}\n{traceback.format_exc()}")
                    raise
                finally:
                    sys.stdout = old_stdout
                messagebox.showinfo("完了", f"データセット作成が完了しました\n{out_dir_norm}\n\nログ:\n{buf.getvalue()}")
            except Exception as e:
                sys.stdout = old_stdout
                error_msg = f"{str(e)}\n\nログ:\n{buf.getvalue()}"
                print(f"[GUI ERROR] {error_msg}")
                messagebox.showerror("エラー", error_msg)
        threading.Thread(target=worker).start()
    root = tk.Tk()
    root.title("supercomboデータセット作成GUI")
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    tk.Label(root, text="入力フォルダ").grid(row=0, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=input_var, width=40).grid(row=0, column=1, padx=5)
    tk.Button(root, text="選択", command=select_input_dir).grid(row=0, column=2, padx=5)
    tk.Label(root, text="出力フォルダ").grid(row=1, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=output_var, width=40).grid(row=1, column=1, padx=5)
    tk.Button(root, text="選択", command=select_output_dir).grid(row=1, column=2, padx=5)
    tk.Button(root, text="データセット作成", command=run_script, width=20).grid(row=2, column=0, columnspan=3, pady=10)
    root.mainloop()

# --- main関数の分岐 ---
if __name__ == "__main__":
    if '--gui' in sys.argv:
        # GUI起動時はコマンドライン引数不要
        run_with_gui()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_dir', required=True, help='走行ログディレクトリ or その親ディレクトリ')
        parser.add_argument('--out_dir', required=True, help='出力ディレクトリ')
        args = parser.parse_args()
        main(args)
        # --- rlogイベント一覧抽出 ---
        import os
        rlog_path = os.path.join(args.log_dir, "rlog")
        if os.path.exists(rlog_path):
            try:
                from openpilot_096.openpilot.tools.lib.logreader import LogReader
                event_types = set()
                lr = LogReader(rlog_path)
                for msg in lr:
                    if hasattr(msg, "which"):
                        event_types.add(msg.which())
                print("rlogに記録されているイベント一覧:", event_types)
            except Exception as e:
                print("rlog解析エラー:", e)
        else:
            print("rlogファイルが存在しません:", rlog_path)

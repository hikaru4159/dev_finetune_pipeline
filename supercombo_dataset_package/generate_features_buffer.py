#!/usr/bin/env python3
"""
Generate features_buffer (1, T, 512) for a run using a driving_vision ONNX model.

Usage:
  python generate_features_buffer.py --onnx /path/to/driving_vision.onnx --run /path/to/run --out /path/to/out

The script attempts to use an existing `_dataset_frames` directory if present,
otherwise extracts frames from `fcamera.hevc` or `ecamera.hevc` with ffmpeg.

The ONNX must accept the same parsed input as supercombo/driving_vision (1,12,128,256)
or the script can be adapted by providing the proper preprocessing.
"""
import os
import argparse
import tempfile
import subprocess
import shutil
import numpy as np
import onnxruntime as ort
import cv2


def parse_image(frame):
    # ported from MTammvee/openpilot-supercombo-model
    H = (frame.shape[0]*2)//3
    W = frame.shape[1]
    parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H+H//4].reshape((-1, H//2, W//2))
    parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2, W//2))

    return parsed


def extract_frames_from_hevc(hevc_path, out_dir, fps=10, max_frames=400):
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", hevc_path,
        "-vf", f"fps={fps}", pattern
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
    return frames[:max_frames]


def load_frames(run_dir):
    # prefer _dataset_frames
    frames_dir = os.path.join(run_dir, '_dataset_frames')
    if os.path.isdir(frames_dir):
        imgs = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        if imgs:
            return imgs

    # fallback to hevc files
    for name in ('fcamera.hevc', 'ecamera.hevc'):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            tmpd = tempfile.mkdtemp(prefix='frames_')
            try:
                imgs = extract_frames_from_hevc(p, tmpd, fps=10)
                return imgs
            finally:
                # keep tmp for debugging; caller may remove it
                pass

    raise FileNotFoundError('no frames or hevc found in run dir')


def build_parsed_sequences(image_files, resize=(256,512)):
    # build parsed images as in the demo; return list of parsed frames
    parsed_list = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (resize[1], resize[0]))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        parsed = parse_image(img_yuv)
        parsed_list.append(parsed)
    return parsed_list


def run_onnx_for_features(onnx_path, parsed_list, out_dir, out_name=None):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    # choose output: prefer any output with last dim 512
    outputs = sess.get_outputs()
    chosen_output = None
    for o in outputs:
        # o.shape elements may be DimensionProto-like objects with 'dim_value',
        # or plain ints/None depending on onnxruntime/onnx version. Handle both.
        shape = []
        if o.shape is not None:
            # o.shape can be a tuple/list of ints or objects
            for d in o.shape:
                if hasattr(d, 'dim_value'):
                    shape.append(d.dim_value)
                else:
                    # d may be int or None
                    try:
                        shape.append(int(d) if d is not None else None)
                    except Exception:
                        shape.append(None)
        if shape and shape[-1] == 512:
            chosen_output = o.name
            break
    if out_name is not None:
        chosen_output = out_name
    if chosen_output is None:
        # fallback to first output
        chosen_output = outputs[0].name

    feats = []
    # create sliding windows of 12 parsed frames (supercombo input uses 2 frames * 6?)
    # The demo uses 2 parsed frames -> (1,12,128,256). We'll create windows of size 2
    window = 2
    for i in range(len(parsed_list)-window+1):
        window_parsed = parsed_list[i:i+window]
        arr = np.array(window_parsed)  # shape (2,6,H',W')
        # reshape to (1,12,128,256)
        # keep as uint8 because some driving_vision ONNX models expect uint8 inputs
        arr = arr.reshape((1,12,arr.shape[2],arr.shape[3])).astype('uint8')
        # Some driving_vision models expect multiple inputs (e.g. input_imgs and big_input_imgs).
        # Build a feed dict that supplies the same processed array to each model input that has
        # the matching shape (or at least the same number of channels/time dims).
        feed = {}
        for inp in sess.get_inputs():
            try:
                feed[inp.name] = arr
            except Exception:
                # fallback: ignore inputs we cannot fill
                pass
        out = sess.run([chosen_output], feed)
        feat = np.array(out[0]).squeeze()
        feats.append(feat)

    feats = np.array(feats)
    # feats shape might be (T,512) or (1,T,512) etc. Normalize to (1,T,512)
    if feats.ndim == 2:
        feats = feats[None, ...]
    if feats.shape[-1] != 512:
        D = feats.shape[-1]
        if D > 512:
            # truncate
            feats = feats[..., :512]
        elif D < 512:
            # pad with zeros
            pad_shape = list(feats.shape)
            pad_shape[-1] = 512 - D
            pad = np.zeros(tuple(pad_shape), dtype=feats.dtype)
            feats = np.concatenate([feats, pad], axis=-1)

    # pad/truncate to 99 frames
    T = feats.shape[1]
    if T < 99:
        pad = np.zeros((1, 99 - T, 512), dtype=np.float32)
        feats = np.concatenate([pad, feats], axis=1)
    elif T > 99:
        feats = feats[:, -99:, :]

    # save
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'features_buffer.npy'), feats.astype(np.float32))
    return os.path.join(out_dir, 'features_buffer.npy')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--output-name', default=None, help='ONNX output name that holds features')
    args = p.parse_args()

    imgs = load_frames(args.run)
    parsed = build_parsed_sequences(imgs)
    path = run_onnx_for_features(args.onnx, parsed, args.out, out_name=args.output_name)
    print('features_buffer saved to', path)


if __name__ == '__main__':
    main()

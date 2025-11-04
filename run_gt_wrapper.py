#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import onnxruntime as ort
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from gt_distill.generate_gt import generate_ground_truth


def make_session(path_to_model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path_to_model, providers=["CPUExecutionProvider"], sess_options=options)

    # Create a lightweight compatibility wrapper around the session to fill missing inputs
    class SessionCompat:
        def __init__(self, sess):
            self._sess = sess
            # record expected input shapes
            self._inputs = {inp.name: inp.shape for inp in sess.get_inputs()}

        def run(self, outputs, input_feed):
            feed = {}

            # Helper to create zero arrays with given shape and dtype
            def zeros_for(name):
                shape = self._inputs.get(name)
                if shape is None:
                    return None
                # replace dynamic dims (None) with 1
                shape_filled = [1 if (d is None or isinstance(d, str)) else d for d in shape]
                return np.zeros(tuple(shape_filled), dtype=np.float16)

            # copy provided inputs (cast to float16) but only if the model expects them
            for k, v in input_feed.items():
                if k not in self._inputs:
                    # skip unexpected input names (generate_ground_truth may pass 'initial_state')
                    continue
                # ensure numpy array
                arr = np.asarray(v)
                # cast to float16 if possible
                try:
                    arr = arr.astype(np.float16)
                except Exception:
                    pass
                feed[k] = arr

            # If big_input_imgs missing, duplicate input_imgs
            if 'big_input_imgs' not in feed and 'input_imgs' in feed:
                feed['big_input_imgs'] = feed['input_imgs']

            # Desire: model expects [1,100,8] in many models; if a shorter desire provided, expand it
            if 'desire' in feed:
                desire_shape = self._inputs.get('desire')
                if desire_shape and len(desire_shape) == 3:
                    T = 1 if (desire_shape[1] is None or isinstance(desire_shape[1], str)) else desire_shape[1]
                    # if provided desire is [1,8] expand to [1,T,8]
                    if feed['desire'].ndim == 2:
                        feed['desire'] = np.repeat(feed['desire'][:, None, :], T, axis=1).astype(np.float16)
                    elif feed['desire'].ndim == 3 and feed['desire'].shape[1] != T:
                        # if mismatch in time dim, try to tile or trim
                        curT = feed['desire'].shape[1]
                        if curT < T:
                            feed['desire'] = np.repeat(feed['desire'], int(np.ceil(T/curT)), axis=1)[:, :T, :].astype(np.float16)
                        else:
                            feed['desire'] = feed['desire'][:, :T, :].astype(np.float16)

            # fill other missing inputs with zeros according to expected shapes
            for expected in self._inputs.keys():
                if expected not in feed:
                    z = zeros_for(expected)
                    if z is not None:
                        feed[expected] = z

            return self._sess.run(outputs, feed)

    return SessionCompat(sess)


def find_segments(recordings_basedir):
    segments = []
    for dirpath, dirs, files in os.walk(recordings_basedir):
        if 'fcamera.hevc' in files or 'video.hevc' in files:
            segments.append(dirpath)
    return sorted(segments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings_basedir', required=True)
    parser.add_argument('--path_to_model', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    model_path = args.path_to_model
    recordings_basedir = args.recordings_basedir

    print('Using model:', model_path)
    print('Looking for segments under:', recordings_basedir)

    segments = find_segments(recordings_basedir)
    print(f'Found {len(segments)} segments to process (will use force={args.force})')

    # create session once and pass into generate_ground_truth by monkeypatching module-level usage
    model = make_session(model_path)

    created = 0
    for seg in tqdm(segments):
        try:
            generate_ground_truth(seg, model, force=args.force)
            out = os.path.join(seg, 'gt_distill.h5')
            if os.path.exists(out):
                created += 1
        except Exception as e:
            print(f'[ERROR] segment {seg}:', e)

    print(f'Done. gt_distill.h5 present in {created}/{len(segments)} segments')


if __name__ == '__main__':
    main()

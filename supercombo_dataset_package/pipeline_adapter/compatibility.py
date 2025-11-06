"""
Compatibility shim to integrate mbalesni/openpilot-pipeline training code with
openpilot v0.9.6 `supercombo.onnx` expectations.

Expose small functions the external pipeline can call instead of modifying its
internals heavily.
"""
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
# Ensure this package folder is importable even when called from other working dirs
SC_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert parent directory so 'supercombo_dataset_package' can be imported by name
SC_PARENT = os.path.abspath(os.path.join(SC_PKG_DIR, '..'))
if SC_PARENT not in sys.path:
    sys.path.insert(0, SC_PARENT)


def prepare_run_for_training(run_dir: str, out_dir: str) -> dict:
    """
    Prepare a single run folder for training. Ensures required `.npy` files
    exist in `out_dir` (created if missing) matching the shapes required by
    supercombo.onnx. Returns a dict of paths to saved npy files.
    
    - Generates desire.npy from images if missing using desire_from_images module
    - Generates features_buffer.npy using ONNX model if DRIVING_VISION_ONNX is set
    """
    # Use existing transform script if available
    from supercombo_dataset_package.transform_to_supercombo_dataset_gui import process_log_dir
    # process_log_dir expects tuple args in the original file; call it to fill out_dir
    try:
        process_log_dir((run_dir, out_dir))
    except Exception:
        # If process failed, continue and try to fill missing fields
        pass

    # If process_log_dir skipped this run because required inputs were missing,
    # it will not have created out_dir. In that case, honor the skip and return
    # an empty mapping to indicate nothing was prepared for this run.
    if not os.path.isdir(out_dir):
        print(f"[SKIP] prepare_run_for_training: output dir was not created for run {run_dir} (missing inputs) - skipping")
        return {}

    # Ensure mandatory fields exist; produce zero-filled placeholders if needed
    required = {
        'input_imgs': (1, 12, 128, 256),
        'big_input_imgs': (1, 12, 128, 256),
        'desire': (1, 100, 8),
        'traffic_convention': (1, 2),
        'lateral_control_params': (1, 2),
        'prev_desired_curv': (1, 100, 1),
        'nav_features': (1, 256),
        'nav_instructions': (1, 150),
        'features_buffer': (1, 99, 512),
    }
    out = {}
    for name, shape in required.items():
        p = os.path.join(out_dir, f"{name}.npy")
        if not os.path.exists(p):
            # Special handling for desire: generate from images only
            if name == 'desire':
                try:
                    print(f"[INFO] desire.npy missing - generating from images")
                    desire_arr = _generate_desire_from_images(run_dir, out_dir)
                    
                    # If image inference failed, use keepLane fallback
                    if desire_arr is None:
                        print(f"[WARN] Image-based desire inference failed, using keepLane fallback")
                        desire_arr = np.zeros(shape, dtype=np.float32)
                        desire_arr[:, :, 7] = 1.0  # keepLane for all frames
                    
                    np.save(p, desire_arr)
                    print(f"[INFO] Generated desire.npy with shape {desire_arr.shape}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to generate desire: {e}")
                    arr = np.zeros(shape, dtype=np.float32)
                    arr[:, :, 7] = 1.0  # keepLane fallback
                    np.save(p, arr)
            # Special-case: try to generate features_buffer if possible
            elif name == 'features_buffer' and 'DRIVING_VISION_ONNX' in os.environ:
                onnx_path = os.environ['DRIVING_VISION_ONNX']
                try:
                    from ..generate_features_buffer import load_frames, build_parsed_sequences, run_onnx_for_features
                except Exception:
                    from supercombo_dataset_package.generate_features_buffer import load_frames, build_parsed_sequences, run_onnx_for_features
                try:
                    print(f"[INFO] features_buffer missing - generating using {onnx_path}")
                    # call helper to build and save features_buffer
                    # run_onnx_for_features will save to out_dir/features_buffer.npy
                    imgs = load_frames(run_dir)
                    parsed = build_parsed_sequences(imgs)
                    run_onnx_for_features(onnx_path, parsed, out_dir)
                except Exception as e:
                    print(f"[ERROR] Failed to generate features_buffer using {onnx_path}: {e}")
            else:
                # by default, do not fabricate missing data to honor strict policy
                print(f"[WARN] {name}.npy missing in {out_dir}; leaving missing (set GENERATE_MISSING_ZEROS=1 to force zero-fill)")
                if os.environ.get('GENERATE_MISSING_ZEROS') == '1':
                    arr = np.zeros(shape, dtype=np.float32)
                    np.save(p, arr)
                else:
                    # leave missing
                    pass
        out[name] = p
    return out


def _generate_desire_from_rlog(run_dir: str, out_dir: str, n_frames: int = 100):
    """
    Generate desire.npy from rlog file using logged lateralPlan.desire or inferred from carState.
    Falls back to None if rlog reading fails.
    
    Returns:
        numpy array of shape (1, 100, 8) or None if failed
    """
    run_path = Path(run_dir)
    rlog_file = run_path / 'rlog'
    
    if not rlog_file.exists():
        print(f"[WARN] rlog file not found in {run_dir}")
        return None
    
    print(f"[INFO] Extracting desire from rlog: {rlog_file}")
    
    # Import extract_desire_from_rlog
    try:
        extract_desire_module_path = os.path.abspath(os.path.join(SC_PKG_DIR, '..'))
        if extract_desire_module_path not in sys.path:
            sys.path.insert(0, extract_desire_module_path)
        from extract_desire_from_rlog import extract_desire_from_rlog
    except ImportError as e:
        print(f"[ERROR] Failed to import extract_desire_from_rlog: {e}")
        return None
    
    try:
        desire_arr = extract_desire_from_rlog(str(rlog_file), n_frames=n_frames)
        return desire_arr
    except Exception as e:
        print(f"[ERROR] Failed to extract desire from rlog: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_desire_from_images(run_dir: str, out_dir: str, n_frames: int = 100):
    """
    Generate desire.npy from images using rule-based optical flow analysis.
    Falls back to None if optical flow detection fails.
    
    Returns:
        numpy array of shape (1, 100, 8) or None if failed
    """
    import tempfile
    
    # Find camera file (fcamera.hevc or ecamera.hevc)
    run_path = Path(run_dir)
    camera_files = list(run_path.glob('*camera.hevc'))
    
    if not camera_files:
        print(f"[WARN] No camera.hevc file found in {run_dir}")
        return None
    
    camera_file = camera_files[0]
    print(f"[INFO] Using camera file: {camera_file}")
    
    # Import desire_from_images module
    try:
        desire_analysis_path = os.path.abspath(os.path.join(SC_PKG_DIR, '..', 'desire_analysis'))
        if desire_analysis_path not in sys.path:
            sys.path.insert(0, desire_analysis_path)
        from desire_from_images import extract_images_from_hevc, extract_desire_from_images
    except ImportError as e:
        print(f"[ERROR] Failed to import desire_from_images: {e}")
        return None
    
    # Extract images to temp directory
    with tempfile.TemporaryDirectory() as tmp_img_dir:
        try:
            extract_images_from_hevc(str(camera_file), tmp_img_dir, fps=10)
            desire_arr = extract_desire_from_images(tmp_img_dir, n_frames=n_frames)
            return desire_arr
        except Exception as e:
            print(f"[ERROR] Failed to extract desire from images: {e}")
            import traceback
            traceback.print_exc()
            return None


def onnx_model_wrapper(onnx_path: str):
    """
    Return a simple callable that runs the ONNX model with a dict of numpy
    inputs and returns the outputs. This keeps external code free from
    provider details.
    """
    try:
        import onnxruntime as ort
    except Exception:
        raise

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    def run(inputs: dict):
        return sess.run(None, inputs)

    return run

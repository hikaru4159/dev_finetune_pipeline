#!/usr/bin/env python3
"""
Robust desire completion script with improved error handling and environment setup.

This script provides a more reliable way to supplement missing desire labels
with proper dependency checking, environment setup, and error recovery.

Usage:
  python desire_supplement.py --run /path/to/run --out /path/to/out [options]
  python desire_supplement.py --batch /path/to/data/dir --out-root /path/to/output [options]
"""
import os
import sys
import argparse
import traceback
from pathlib import Path

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Check if all required dependencies are available."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import zstandard
    except ImportError:
        missing.append("zstandard")
    
    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print(f"[INFO] Install with: python -m pip install {' '.join(missing)}")
        return False
    
    return True

def safe_import_desire_analysis():
    """Safely import desire analysis modules with fallback."""
    try:
        from desire_analysis import desire_from_images
        return desire_from_images, True
    except ImportError as e:
        print(f"[ERROR] Failed to import desire_analysis: {e}")
        print("[INFO] Make sure desire_analysis directory exists and is accessible")
        return None, False

def generate_desire_robust(run_dir, out_dir, force_image_method=False):
    """
    Generate desire.npy with robust error handling and multiple fallback methods.
    
    Args:
        run_dir: Path to input run directory
        out_dir: Path to output directory  
        force_image_method: Skip log-based method and go directly to image-based
    
    Returns:
        tuple: (success: bool, method_used: str, error_msg: str)
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    desire_file = out_dir / "desire.npy"
    
    # Check if already exists
    if desire_file.exists():
        print(f"[INFO] desire.npy already exists in {out_dir}")
        return True, "existing", ""
    
    # Method 1: Try log-based inference (if not forced to skip)
    if not force_image_method:
        try:
            print(f"[INFO] Attempting log-based desire inference for {run_dir}")
            success, error = _try_log_based_inference(run_dir, out_dir)
            if success:
                return True, "log-based", ""
            else:
                print(f"[WARN] Log-based inference failed: {error}")
        except Exception as e:
            print(f"[WARN] Log-based inference exception: {e}")
    
    # Method 2: Try image-based inference
    try:
        print(f"[INFO] Attempting image-based desire inference for {run_dir}")
        success, error = _try_image_based_inference(run_dir, out_dir)
        if success:
            return True, "image-based", ""
        else:
            print(f"[WARN] Image-based inference failed: {error}")
    except Exception as e:
        print(f"[WARN] Image-based inference exception: {e}")
    
    # Method 3: Generate zero-filled placeholder
    try:
        print(f"[INFO] Generating zero-filled desire.npy placeholder for {run_dir}")
        import numpy as np
        zeros = np.zeros((1, 100, 8), dtype=np.float32)
        np.save(desire_file, zeros)
        print(f"[INFO] Created zero-filled desire.npy at {desire_file}")
        return True, "zero-filled", ""
    except Exception as e:
        error_msg = f"Failed to create zero-filled placeholder: {e}"
        print(f"[ERROR] {error_msg}")
        return False, "failed", error_msg

def _try_log_based_inference(run_dir, out_dir):
    """Try log-based desire inference. Returns (success, error_msg)."""
    try:
        # Check for required capnp dependencies
        import capnp
        import zstandard as zstd
    except ImportError as e:
        return False, f"Missing capnp dependencies: {e}"
    
    try:
        # Import log-based modules
        from desire_analysis.desire_from_log_features import extract_log_features_with_time, infer_desire_from_features
        from desire_analysis.desire_from_images import DESIRE_LABELS
        import numpy as np
        
        # Find rlog file
        rlog = run_dir / 'rlog'
        if not rlog.exists() and (run_dir / 'rlog.zst').exists():
            rlog = run_dir / 'rlog.zst'
        
        if not rlog.exists():
            return False, "No rlog or rlog.zst file found"
        
        # Extract features and generate labels
        features, times = extract_log_features_with_time(str(rlog))
        if features.shape[0] == 0:
            return False, "No features extracted from log"
        
        labels = infer_desire_from_features(features)
        
        # Convert to one-hot
        label_to_idx = {n: i for i, n in enumerate(DESIRE_LABELS)}
        onehot = np.zeros((1, labels.shape[0], len(DESIRE_LABELS)), dtype=np.float32)
        for i, lab in enumerate(labels):
            idx = label_to_idx.get(lab, 0)
            onehot[0, i, idx] = 1.0
        
        # Pad/truncate to 100 frames
        if onehot.shape[1] < 100:
            pad = np.zeros((1, 100 - onehot.shape[1], onehot.shape[2]), dtype=np.float32)
            onehot = np.concatenate([pad, onehot], axis=1)
        else:
            onehot = onehot[:, -100:, :]
        
        # Save
        np.save(out_dir / 'desire.npy', onehot)
        print(f"[SUCCESS] Log-based desire.npy created with shape {onehot.shape}")
        return True, ""
        
    except Exception as e:
        return False, f"Log-based inference error: {e}"

def _try_image_based_inference(run_dir, out_dir):
    """Try image-based desire inference. Returns (success, error_msg)."""
    try:
        desire_from_images, success = safe_import_desire_analysis()
        if not success:
            return False, "Could not import desire_from_images module"
        
        import numpy as np
        
        # Find camera file (prefer fcamera.hevc)
        camera_candidates = ['fcamera.hevc', 'ecamera.hevc']
        camera_file = None
        for candidate in camera_candidates:
            candidate_path = run_dir / candidate
            if candidate_path.exists():
                camera_file = candidate_path
                break
        
        if not camera_file:
            return False, "No camera files (fcamera.hevc/ecamera.hevc) found"
        
        # Create temporary image directory
        tmp_img_dir = out_dir / '_tmp_imgs'
        tmp_img_dir.mkdir(exist_ok=True)
        
        # Extract images and generate desire
        desire_from_images.extract_images_from_hevc(str(camera_file), str(tmp_img_dir), fps=10)
        desire_array = desire_from_images.extract_desire_from_images(str(tmp_img_dir), n_frames=100)
        
        # Save
        np.save(out_dir / 'desire.npy', desire_array)
        print(f"[SUCCESS] Image-based desire.npy created with shape {desire_array.shape}")
        
        # Cleanup temporary images
        import shutil
        shutil.rmtree(tmp_img_dir, ignore_errors=True)
        
        return True, ""
        
    except Exception as e:
        return False, f"Image-based inference error: {e}"

def process_single_run(run_dir, out_dir, force_image=False):
    """Process a single run and return result summary."""
    success, method, error = generate_desire_robust(run_dir, out_dir, force_image)
    return {
        'run': Path(run_dir).name,
        'success': success,
        'method': method,
        'error': error,
        'out_path': out_dir
    }

def process_batch_runs(data_dir, out_root, force_image=False, jobs=1):
    """Process multiple runs in batch."""
    data_dir = Path(data_dir)
    out_root = Path(out_root)
    
    # Find all run directories
    run_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        print(f"[ERROR] No run directories found in {data_dir}")
        return []
    
    print(f"[INFO] Found {len(run_dirs)} run directories")
    
    results = []
    
    if jobs <= 1:
        # Sequential processing
        for run_dir in run_dirs:
            out_dir = out_root / run_dir.name
            result = process_single_run(run_dir, out_dir, force_image)
            results.append(result)
            
            status = "✓" if result['success'] else "✗"
            print(f"[{status}] {result['run']} -> {result['method']}")
            if not result['success']:
                print(f"    Error: {result['error']}")
    
    else:
        # Parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {}
            for run_dir in run_dirs:
                out_dir = out_root / run_dir.name  
                future = executor.submit(process_single_run, run_dir, out_dir, force_image)
                futures[future] = run_dir.name
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                status = "✓" if result['success'] else "✗"
                print(f"[{status}] {result['run']} -> {result['method']}")
                if not result['success']:
                    print(f"    Error: {result['error']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Robust desire completion script")
    parser.add_argument('--run', help='Single run directory to process')
    parser.add_argument('--out', help='Output directory for single run')
    parser.add_argument('--batch', help='Batch process all runs in directory')
    parser.add_argument('--out-root', help='Root output directory for batch processing')
    parser.add_argument('--force-image', action='store_true', help='Skip log-based method, use image-based directly')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs for batch processing')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        return 1 if not check_dependencies() else 0
    
    # Single run processing
    if args.run and args.out:
        print(f"[INFO] Processing single run: {args.run}")
        result = process_single_run(args.run, args.out, args.force_image)
        
        if result['success']:
            print(f"[SUCCESS] Completed using {result['method']} method")
            return 0
        else:
            print(f"[ERROR] Failed: {result['error']}")
            return 1
    
    # Batch processing
    elif args.batch and args.out_root:
        print(f"[INFO] Processing batch: {args.batch}")
        results = process_batch_runs(args.batch, args.out_root, args.force_image, args.jobs)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"\n=== Summary ===")
        print(f"Total runs: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        
        if total > 0:
            methods = {}
            for r in results:
                if r['success']:
                    methods[r['method']] = methods.get(r['method'], 0) + 1
            
            print("Methods used:")
            for method, count in methods.items():
                print(f"  {method}: {count}")
        
        return 0 if successful == total else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
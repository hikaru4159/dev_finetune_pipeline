#!/usr/bin/env python3
"""
Extract desire labels from rlog file using logged lateralPlan.desire values.
If not available, fall back to inferring from carState (steering angle, blinkers).
"""
import os
import sys
import numpy as np
import argparse

# Add capnp paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'openpilot'))

def extract_desire_from_rlog(rlog_path, n_frames=100, fps=10):
    """
    Extract desire labels from rlog file.
    
    Returns:
        numpy array of shape (1, n_frames, 8) with one-hot encoded desire labels
    """
    try:
        from tools.lib.logreader import LogReader
    except ImportError:
        print("[ERROR] Failed to import LogReader. Make sure openpilot tools are available.")
        # Fallback: return keepLane for all frames
        desire_arr = np.zeros((1, n_frames, 8), dtype=np.float32)
        desire_arr[:, :, 7] = 1.0  # keepLane
        return desire_arr
    
    desire_labels = []
    timestamps = []
    
    try:
        lr = LogReader(rlog_path)
        
        # First try to get desire from lateralPlan
        for msg in lr:
            if msg.which() == 'lateralPlan':
                timestamps.append(msg.logMonoTime / 1e9)
                desire = msg.lateralPlan.desire
                desire_labels.append(desire)
        
        # If lateralPlan.desire not available, infer from carState
        if len(desire_labels) == 0:
            print("[INFO] lateralPlan.desire not found, inferring from carState")
            for msg in lr:
                if msg.which() == 'carState':
                    timestamps.append(msg.logMonoTime / 1e9)
                    cs = msg.carState
                    
                    # Infer desire from steering angle and blinkers
                    angle = cs.steeringAngleDeg
                    left_blinker = cs.leftBlinker
                    right_blinker = cs.rightBlinker
                    v_ego = cs.vEgo
                    
                    # Thresholds
                    TURN_ANGLE_THRESH = 90.0
                    LANE_CHANGE_ANGLE_THRESH = 15.0
                    MIN_SPEED = 5.0  # m/s
                    
                    if v_ego < MIN_SPEED:
                        desire = 7  # keepLane at low speed
                    elif left_blinker:
                        if abs(angle) >= TURN_ANGLE_THRESH:
                            desire = 1  # turnLeft
                        else:
                            desire = 3  # laneChangeLeft
                    elif right_blinker:
                        if abs(angle) >= TURN_ANGLE_THRESH:
                            desire = 2  # turnRight
                        else:
                            desire = 4  # laneChangeRight
                    elif abs(angle) >= TURN_ANGLE_THRESH:
                        desire = 1 if angle < 0 else 2  # turnLeft or turnRight
                    elif abs(angle) >= LANE_CHANGE_ANGLE_THRESH:
                        desire = 3 if angle < 0 else 4  # laneChangeLeft or laneChangeRight
                    else:
                        desire = 7  # keepLane
                    
                    desire_labels.append(desire)
        
    except Exception as e:
        print(f"[ERROR] Failed to read rlog: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return keepLane for all frames
        desire_arr = np.zeros((1, n_frames, 8), dtype=np.float32)
        desire_arr[:, :, 7] = 1.0  # keepLane
        return desire_arr
    
    if len(desire_labels) == 0:
        print("[WARN] No desire labels found in rlog, using keepLane fallback")
        desire_arr = np.zeros((1, n_frames, 8), dtype=np.float32)
        desire_arr[:, :, 7] = 1.0  # keepLane
        return desire_arr
    
    # Convert to numpy
    desire_labels = np.array(desire_labels, dtype=np.int32)
    timestamps = np.array(timestamps)
    
    # Resample to match fps (10Hz = 0.1s intervals)
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration = end_time - start_time
    
    target_times = np.linspace(start_time, end_time, n_frames)
    resampled_desires = []
    
    for t in target_times:
        # Find closest timestamp
        idx = np.argmin(np.abs(timestamps - t))
        resampled_desires.append(desire_labels[idx])
    
    # Convert to one-hot encoding
    desire_arr = np.zeros((1, n_frames, 8), dtype=np.float32)
    for i, d in enumerate(resampled_desires):
        if 0 <= d < 8:
            desire_arr[0, i, d] = 1.0
        else:
            desire_arr[0, i, 7] = 1.0  # default to keepLane
    
    print(f"[INFO] Extracted desire labels: {np.unique(resampled_desires, return_counts=True)}")
    
    return desire_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rlog', required=True, help='Path to rlog file')
    parser.add_argument('--output', required=True, help='Output path for desire.npy')
    parser.add_argument('--n-frames', type=int, default=100, help='Number of frames')
    args = parser.parse_args()
    
    desire_arr = extract_desire_from_rlog(args.rlog, n_frames=args.n_frames)
    np.save(args.output, desire_arr)
    print(f"[INFO] Saved desire to {args.output}, shape: {desire_arr.shape}")

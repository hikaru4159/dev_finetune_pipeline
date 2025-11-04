#!/usr/bin/env python3
import sys
import os
import argparse
import traceback
from pathlib import Path




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    openpilot_pipeline = str(repo_root / 'openpilot-pipeline')
    # make sure modules import correctly
    if openpilot_pipeline not in sys.path:
        sys.path.insert(0, openpilot_pipeline)

    # import after ensuring package path
    from run_gt_wrapper import make_session
    from gt_distill.generate_gt import generate_ground_truth

    print('Running generate_ground_truth on segment:', args.segment)
    print('Using model:', args.model)

    model = None
    try:
        model = make_session(args.model)
    except Exception:
        print('Failed to create onnxruntime session:')
        traceback.print_exc()
        return

    try:
        generate_ground_truth(args.segment, model, force=True)
        print('Done without exception')
    except Exception:
        print('Exception during generate_ground_truth:')
        traceback.print_exc()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Inspect ONNX model inputs/outputs (names and shapes).

Usage:
  python inspect_supercombo_onnx.py --onnx base/supercombo.onnx

"""
import argparse
import onnx
import numpy as np


def inspect(path):
    m = onnx.load(path)
    graph = m.graph
    print(f"Model: {path}")
    print("Inputs:")
    for i in graph.input:
        name = i.name
        shape = None
        try:
            shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
        except Exception:
            shape = None
        print(f" - {name}: {shape}")
    print("Outputs:")
    for o in graph.output:
        name = o.name
        shape = None
        try:
            shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        except Exception:
            shape = None
        print(f" - {name}: {shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    args = p.parse_args()
    inspect(args.onnx)


if __name__ == '__main__':
    main()

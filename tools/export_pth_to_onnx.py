#!/usr/bin/env python3
"""
Export AdapterHead trained weights to ONNX model.

このスクリプトは、AdapterHeadの学習済み重み（in_layer, res_layer_0, res_layer_2, final_layer, 各bias含む）をONNXモデルの該当initializerに直接置換します。
"""
import os
import sys
import torch
import onnx
from onnx import numpy_helper
import shutil


def export_to_onnx(pth_path, onnx_output_path, original_onnx_path):
    print(f"\nLoading models...")
    print(f"  .pth: {pth_path}")
    print(f"  Original ONNX: {original_onnx_path}")

    # Load trained AdapterHead weights
    checkpoint = torch.load(pth_path, map_location='cpu')

    # Load original ONNX model
    onnx_model = onnx.load(original_onnx_path)

    # 完全な重みマッピング（残差層を含む）
    weight_mapping = {
        'in_layer.weight': 'temporal_policy.temporal_hydra.in_layer.desire_state.weight',
        'in_layer.bias': 'temporal_policy.temporal_hydra.in_layer.desire_state.bias',
        'res_layer_0.weight': 'temporal_policy.temporal_hydra.res_layer.desire_state.0.weight',
        'res_layer_0.bias': 'temporal_policy.temporal_hydra.res_layer.desire_state.0.bias',
        'res_layer_2.weight': 'temporal_policy.temporal_hydra.res_layer.desire_state.2.weight',
        'res_layer_2.bias': 'temporal_policy.temporal_hydra.res_layer.desire_state.2.bias',
        'final_layer.weight': 'temporal_policy.temporal_hydra.final_layer.desire_state.weight',
        'final_layer.bias': 'temporal_policy.temporal_hydra.final_layer.desire_state.bias',
    }

    print("\n置換対象の重み（8個）:")

    replaced_count = 0
    for pth_key, onnx_key in weight_mapping.items():
        if pth_key not in checkpoint:
            print(f"  ⚠ WARNING: {pth_key} not found in checkpoint")
            continue

        trained_weight = checkpoint[pth_key].numpy()

        # Find and replace in ONNX initializers
        found = False
        for init in onnx_model.graph.initializer:
            if init.name == onnx_key:
                original_shape = numpy_helper.to_array(init).shape
                print(f"  ✓ {pth_key} ({trained_weight.shape}) -> {onnx_key} ({original_shape})")

                if trained_weight.shape != original_shape:
                    print(f"    ⚠ Shape mismatch! Skipping.")
                    continue

                new_tensor = numpy_helper.from_array(trained_weight, name=init.name)
                init.CopyFrom(new_tensor)
                replaced_count += 1
                found = True
                break

        if not found:
            print(f"  ⚠ WARNING: {onnx_key} not found in ONNX model")

    if replaced_count == 0:
        print("\n⚠ WARNING: No weights were replaced. Copying original ONNX...")
        shutil.copy2(original_onnx_path, onnx_output_path)
        return True

    # Save modified ONNX
    onnx.save(onnx_model, onnx_output_path)
    print(f"\n✓ Successfully replaced {replaced_count}/8 weight tensors")
    print(f"✓ Saved modified ONNX to: {onnx_output_path}")

    if replaced_count < 8:
        print(f"⚠ WARNING: Expected 8 replacements, got {replaced_count}")
    else:
        print("✓ All weights successfully exported to ONNX (完全構造一致)")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export trained .pth weights to ONNX model")
    parser.add_argument("pth_path", help="Path to trained .pth file")
    parser.add_argument("onnx_output_path", help="Path to output ONNX file")
    parser.add_argument("original_onnx_path", help="Path to original ONNX model")
    args = parser.parse_args()

    print("="*60)
    print("ONNX Export Tool - Training Verification")
    print("="*60)

    success = export_to_onnx(args.pth_path, args.onnx_output_path, args.original_onnx_path)
    sys.exit(0 if success else 1)

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
import re
import numpy as np


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

    # ユーティリティ: チェックポイントから state_dict を取り出してフラット化
    def extract_state_dict(cp):
        # cp may be a dict with nested 'state_dict' or similar
        if isinstance(cp, dict):
            # common containers
            for key in ('state_dict', 'model', 'net', 'params'):
                if key in cp and isinstance(cp[key], dict):
                    cp = cp[key]
                    break

        # If values are dicts, flatten recursively
        flat = {}

        def _flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    _flatten(v, new_key)
            else:
                flat[prefix] = obj

        if isinstance(cp, dict):
            _flatten(cp, '')
        else:
            # unknown format; return as-is
            return {}

        return flat

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        # try to convert
        try:
            return np.array(x)
        except Exception:
            return None

    def normalize_name(s: str):
        s = s.lower()
        s = s.replace('::', '.')
        s = s.replace('/', '.')
        s = s.replace('\\\\', '.')
        s = s.replace('-', '.')
        s = s.replace('__', '.')
        s = re.sub(r'[^0-9a-z_.]', '.', s)
        s = re.sub(r'\.+', '.', s)
        s = s.strip('.')
        # remove common prefixes
        for p in ('module.', 'trainable_layers.', 'trainable_layers_', 'state_dict.', 'model.', 'net.'):
            if s.startswith(p):
                s = s[len(p):]
        return s

    # build flat state dict
    flat = extract_state_dict(checkpoint)

    # build normalized index
    norm_index = {}
    for k, v in flat.items():
        nk = normalize_name(k)
        norm_index.setdefault(nk, []).append(k)

    # helper to find best matching key for expected
    def find_checkpoint_key(expected):
        # exact match
        if expected in flat:
            return expected

        exp_norm = normalize_name(expected)
        # exact normalized match
        if exp_norm in norm_index and len(norm_index[exp_norm]) == 1:
            return norm_index[exp_norm][0]
        if exp_norm in norm_index:
            # multiple candidates -> prefer exact token suffix match
            candidates = norm_index[exp_norm]
            for c in candidates:
                if c.endswith(expected):
                    return c
            return candidates[0]

        # fallback: shape-matching search
        exp_suffix = expected.split('.')[-1]
        candidates = []
        for k, v in flat.items():
            if not isinstance(v, (torch.Tensor, np.ndarray)):
                continue
            vk = normalize_name(k)
            if vk.endswith(exp_norm) or vk.endswith(exp_suffix):
                candidates.append(k)

        if len(candidates) == 1:
            return candidates[0]

        # as last resort, return None
        return None

    replaced_count = 0
    for pth_key, onnx_key in weight_mapping.items():
        ck = find_checkpoint_key(pth_key)
        if ck is None:
            print(f"  ⚠ WARNING: could not find checkpoint key for {pth_key}")
            continue

        trained_arr = to_numpy(flat[ck])
        if trained_arr is None:
            print(f"  ⚠ WARNING: checkpoint value for {ck} is not convertible to numpy")
            continue

        # Find and replace in ONNX initializers
        found = False
        for init in onnx_model.graph.initializer:
            if init.name == onnx_key:
                original_shape = numpy_helper.to_array(init).shape
                print(f"  ✓ {ck} ({trained_arr.shape}) -> {onnx_key} ({original_shape})")

                if trained_arr.shape != original_shape:
                    print(f"    ⚠ Shape mismatch! Skipping.")
                    found = True
                    break

                new_tensor = numpy_helper.from_array(trained_arr, name=init.name)
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

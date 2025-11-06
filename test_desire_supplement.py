#!/usr/bin/env python3
import sys
sys.path.insert(0, 'supercombo_dataset_package')
from pipeline_adapter.compatibility import _generate_desire_from_images
import numpy as np

# Test desire generation for run 1
print("Testing desire generation for run 1...")
des1 = _generate_desire_from_images('DATA/itsdata/1', 'tmp/full_npy/1')
if des1 is not None:
    print(f'  Generated desire shape: {des1.shape}')
    print(f'  mean: {np.mean(des1, axis=1)}')
    print(f'  unique classes: {np.unique(np.argmax(des1, axis=2))}')
else:
    print('  Failed to generate desire')

print("\nTesting desire generation for run 2...")
des2 = _generate_desire_from_images('DATA/itsdata/2', 'tmp/full_npy/2')
if des2 is not None:
    print(f'  Generated desire shape: {des2.shape}')
    print(f'  mean: {np.mean(des2, axis=1)}')
    print(f'  unique classes: {np.unique(np.argmax(des2, axis=2))}')
else:
    print('  Failed to generate desire')

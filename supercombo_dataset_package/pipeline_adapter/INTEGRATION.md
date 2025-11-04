Integration notes — using mbalesni/openpilot-pipeline with v0.9.6 supercombo
=====================================================================

Goal
----
Make the external `mbalesni/openpilot-pipeline` training code produce datasets compatible with openpilot v0.9.6 `supercombo.onnx` without large edits to that repo. Use this package's `pipeline_adapter` as a minimal shim.

Recommended minimal approach
---------------------------
1. Clone or use `mbalesni/openpilot-pipeline` as usual.
2. Add the path to this repository to the Python path for the training script, for example by setting `PYTHONPATH` or performing `sys.path.insert(0, '/path/to/supercombo_dataset_package')` at the top of the training entrypoint.
3. At the point where the pipeline expects preprocessed run data, call the adapter:

   from supercombo_dataset_package.pipeline_adapter.compatibility import prepare_run_for_training
   prepare_run_for_training(run_dir, out_dir_for_run)

   This will call the existing `transform_to_supercombo_dataset_gui` processing and fill in any missing `.npy` files with zero placeholders. If `desire.npy` is missing, call the label generator:

   from supercombo_dataset_package.pipeline_adapter.generate_synthetic_labels import generate
   generate(run_dir, out_dir_for_run)

4. To run the model during training or evaluation, use the ONNX wrapper:

   from supercombo_dataset_package.pipeline_adapter.compatibility import onnx_model_wrapper
   run_onnx = onnx_model_wrapper('/full/path/to/base/supercombo.onnx')
   outputs = run_onnx(inputs_dict)

Notes & caveats
---------------
- The adapter fills missing fields with zeros to maintain shape compatibility. Zero-filled fields may bias training; prefer to synthesize labels (desire) with `generate_synthetic_labels.py` when possible.
- This shim intentionally avoids changing the external repo; instead it provides callable helpers the external code can import.
- If you prefer in-repo changes, patch the training entrypoint to call `prepare_run_for_training` before dataset construction.

Example (quick patch for external repo's runner)
------------------------------------------------
Insert near the top of the training job before dataset reading:

   import sys
   sys.path.insert(0, '/home/user1434407/dev_pipeline_base/supercombo_dataset_package')
   from pipeline_adapter.compatibility import prepare_run_for_training
   prepare_run_for_training(run_dir, out_dir_for_run)

This is a minimal change that avoids editing the rest of the pipeline.

# Model Export Guide

This guide explains the technical process of exporting PyTorch models to ONNX (Open Neural Network Exchange) format.

## Overview

The export process bridges the gap between research (PyTorch) and production (ONNX Runtime / TensorRT). It relies on `torch.onnx.export`, which uses "tracing" to record the operations performed by the model given a dummy input.

## Script Usage

The primary entry point is `scripts/04_export.sh`.

```bash
bash scripts/04_export.sh [CONFIG] [CHECKPOINT] [OUTPUT_PATH]
```

### Examples

Export with defaults:
```bash
bash scripts/04_export.sh
```

Export to a specific location:
```bash
bash scripts/04_export.sh \
    configs/clip_hateful_memes.py \
    work_dirs/clip_hateful_memes/best.pth \
    deploy_aws/production_model.onnx
```

## Technical Details

### The Export Pipeline

1.  **Model Reconstruction**: The script instantiates the PyTorch model using the same `config` used for training.
2.  **Weight Loading**: Weights are loaded from the `.pth` checkpoint.
3.  **Dummy Input Generation**:
    -   Because CLIP/ViLT models require both image and text inputs, the script generates:
        -   `input_ids`: Tensor of shape `(1, 77)` (standard CLIP text length).
        -   `attention_mask`: Tensor of shape `(1, 77)`.
        -   `pixel_values`: Tensor of shape `(1, 3, 224, 224)` (standard CLIP image size).
4.  **Tracing**:
    -   `torch.onnx.export()` executes the model with the dummy inputs.
    -   It records the sequence of operators (MatMul, Conv2d, Softmax, etc.) into a static graph.
5.  **Opset Version**: We target ONNX Opset 11 or higher to support modern Transformer operations.

### Verification

The script includes a verification step:
1.  Runs inference on the original PyTorch model with random data.
2.  Runs inference on the exported ONNX model with the same data using `onnxruntime`.
3.  Asserts that `np.allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)`.

### Common Pitfalls

-   **Dynamic Control Flow**: Tracing cannot capture `if/else` logic that depends on input values (unless `torch.jit.script` is used, but `export` is simpler for standard models). Ensure the model's forward pass is static regarding tensor shapes.
-   **Device mismatch**: Ensure the model is on CPU before exporting to avoid embedding CUDA-specific device instructions if targeting generic inference.

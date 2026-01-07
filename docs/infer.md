# Inference Guide

This guide details how to run inference on single samples using the Harmful Content Detection pipeline.

## Overview

Inference allows you to get predictions (label and confidence score) for a specific image-text pair using a trained model. This is useful for ad-hoc testing, debugging, or verifying model behavior on specific examples.

## Script Usage

The primary entry point is `scripts/03_infer.sh`.

```bash
bash scripts/03_infer.sh [CONFIG] [CHECKPOINT] [IMAGE_PATH] [TEXT]
```

### Examples

Run with default settings:
```bash
bash scripts/03_infer.sh
```

Run with custom inputs:
```bash
bash scripts/03_infer.sh \
    configs/clip_hateful_memes.py \
    work_dirs/clip_hateful_memes/latest.pth \
    data/hateful_memes/img/01235.png \
    "Look at this meme"
```

## Technical Details

### Workflow

1.  **Initialization**: `HCDInference` class (in `src/core/inference.py`) is initialized.
    -   It parses the config.
    -   Builds the model using `MODELS.build(cfg.model)`.
    -   Loads the checkpoint weights.
    -   Moves the model to the specified device and sets it to `eval()` mode.
    -   Builds the data transformation pipeline (resizing, normalization, tokenization) from `cfg.data.test.pipeline`.

2.  **Preprocessing**:
    -   The input image path is read.
    -   The input text is tokenized.
    -   Transforms are applied to create the input tensor dictionary.

3.  **Forward Pass**:
    -   The model receives the input tensors.
    -   A forward pass is executed inside a `torch.no_grad()` context.
    -   Logits are computed.

4.  **Post-processing**:
    -   Softmax is applied to logits to get probabilities.
    -   The class with the highest probability is selected.
    -   Result is formatted as a dictionary (Label, Score, Probabilities).

### Output

The tool prints the JSON-formatted result to the console.

```json
{
    "label": 1,
    "score": 0.982,
    "probabilities": [0.018, 0.982]
}
```
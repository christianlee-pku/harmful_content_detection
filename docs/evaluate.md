# Evaluation Guide

This guide details the evaluation process for the Harmful Content Detection pipeline.

## Overview

Evaluation measures the model's performance on a validation or test dataset. It computes metrics such as Accuracy and AUC-ROC (Area Under the Receiver Operating Characteristic Curve). The evaluation script runs in inference mode, disabling gradient calculation to save memory and computation.

## Script Usage

The primary entry point is `scripts/02_evaluate.sh`.

```bash
bash scripts/02_evaluate.sh [CONFIG_PATH] [CHECKPOINT_PATH]
```

### Examples

Evaluate the default config and latest checkpoint:
```bash
bash scripts/02_evaluate.sh
```

Evaluate a specific checkpoint:
```bash
bash scripts/02_evaluate.sh configs/clip_hateful_memes.py work_dirs/clip_hateful_memes/epoch_5.pth
```

## Technical Details

### Workflow

1.  **Initialization**: `tools/eval.py` loads the configuration and builds the model and dataset.
2.  **Checkpoint Loading**: The state dictionary from the specified checkpoint file is loaded into the model.
3.  **Data Pipeline**: The validation dataset pipeline is used. This typically involves deterministic transforms (e.g., resizing without random cropping) to ensure consistent results.
4.  **Inference Loop**:
    -   The model is set to `eval()` mode.
    -   `torch.no_grad()` context is used.
    -   Batches are processed, and logits/probabilities are collected.
5.  **Metric Calculation**:
    -   **Accuracy**: Comparison of predicted class indices vs. ground truth labels.
    -   **AUC-ROC**: Calculated using the raw probabilities for the positive class.

### Output Formats

The evaluation results are printed to stdout and can optionally be saved to a JSON file.

**JSON Output Schema:**

```json
{
  "accuracy": 0.754,
  "auc": 0.812
}
```
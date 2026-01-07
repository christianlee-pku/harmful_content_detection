# Deployment Guide

This guide details the architecture and process for deploying the Harmful Content Detection model to AWS SageMaker.

## Overview

We use AWS SageMaker's "Bring Your Own Model" (BYOM) paradigm. Specifically, we deploy the ONNX-exported model wrapped in a PyTorch inference container (or a generic ONNX Runtime container, though we use PyTorch here for easier pre/post-processing flexibility).

## Script Usage

The primary entry point is `scripts/05_deploy.sh`.

```bash
bash scripts/05_deploy.sh <ROLE_ARN> [ENDPOINT_NAME]
```

### Examples

Deploy with a specific role:
```bash
bash scripts/05_deploy.sh arn:aws:iam::123456789012:role/SageMakerRole
```

Deploy with a custom endpoint name:
```bash
bash scripts/05_deploy.sh arn:aws:iam::123456789012:role/SageMakerRole my-hcd-endpoint
```

## Technical Architecture

### Components

1.  **Model Artifact (`model.tar.gz`)**:
    -   Contains the `model.onnx` file.
    -   Contains a `code/` directory with `inference.py` and `requirements.txt`.
2.  **Inference Script (`inference.py`)**:
    -   Implements the SageMaker inference contract functions: `model_fn`, `input_fn`, `predict_fn`, and `output_fn`.
3.  **SageMaker Endpoint**:
    -   **Model**: The definition pointing to the S3 artifact and Docker image (PyTorch container).
    -   **Endpoint Config**: Specifies instance type (e.g., `ml.m5.xlarge`) and count.
    -   **Endpoint**: The HTTPS REST API exposed by AWS.

### The Inference Lifecycle

When a request hits the endpoint:

1.  **`input_fn`**:
    -   Receives the raw JSON payload.
    -   Decodes the Base64 image string into a PIL Image.
    -   Extracts the text string.
    -   Returns a dictionary `{'image': img, 'text': txt}`.
2.  **`predict_fn`**:
    -   **Preprocessing**: Uses `transformers.CLIPProcessor` to tokenize text and normalize images (mean/std) into tensors.
    -   **Inference**: Passes numpy arrays to the `onnxruntime.InferenceSession`.
    -   Returns raw logits.
3.  **`output_fn`**:
    -   Applies `softmax` to logits to get probabilities.
    -   Formats the result into JSON: `{'label': 1, 'score': 0.95}`.
    -   Returns the JSON string as the HTTP response.

### Security & IAM

-   **Execution Role**: The SageMaker service assumes an IAM role (passed via `ROLE_ARN`) to access the S3 bucket where the model artifact is stored.
-   **VPC**: By default, the endpoint runs in a service-managed VPC. For higher security, it can be configured to run within a private subnet.

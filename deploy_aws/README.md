# AWS SageMaker ONNX Deployment

This directory contains the artifacts and scripts required to deploy the ONNX-exported Harmful Content Detection model to AWS SageMaker with automated input processing.

## Structure

```
deploy_aws/
├── code/
│   ├── inference.py        # SageMaker inference entry point
│   └── requirements.txt    # Python dependencies
└── README.md               # This file
```

## Prerequisites

1.  **Exported Model**: Ensure you have `model.onnx` generated (see `tools/export.py`).
2.  **AWS Credentials**: Configured via `aws configure` or environment variables.
3.  **SageMaker SDK**: Installed via `pip install sagemaker`.

## Deployment Steps

### 1. Prepare Model Artifact

Create a `model.tar.gz` containing the ONNX model and the `code/` directory.

```bash
# Copy model.onnx to current dir or specify path
cp ../work_dirs/clip_hateful_memes/model.onnx .

# Tar it up
tar -czvf model.tar.gz model.onnx code/
```

### 2. Deploy

Use the provided `deploy.py` script:

```bash
python deploy.py --role <YOUR_AWS_IAM_ROLE> --endpoint-name hcd-endpoint
```

This script will automatically:
1. Package `model.onnx` and `code/` into `model.tar.gz`.
2. Upload and deploy to AWS SageMaker.

### 3. Invoke Endpoint

Use the provided `invoke_endpoint.py` script to test the deployed model:

```bash
python invoke_endpoint.py \
    --endpoint hcd-onnx-endpoint \
    --image ../data/hateful_memes/img/01235.png \
    --text "this is a test"
```

This script will:
1. Encode the image to Base64.
2. Send a JSON request to SageMaker.
3. Print the prediction result.

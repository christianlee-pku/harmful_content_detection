#!/usr/bin/env bash
# 05_deploy.sh
# Usage: bash scripts/05_deploy.sh <ROLE_ARN> [ENDPOINT_NAME]

set -e

if [ -z "$1" ]; then
    echo "Error: ROLE_ARN is required."
    echo "Usage: bash scripts/05_deploy.sh <ROLE_ARN> [ENDPOINT_NAME]"
    exit 1
fi

ROLE_ARN="$1"
ENDPOINT_NAME=${2:-"hcd-onnx-endpoint"}

echo "Deploying to SageMaker..."
echo "Role: $ROLE_ARN"
echo "Endpoint: $ENDPOINT_NAME"

# Ensure PYTHONPATH allows importing from src/ if needed, though deploy.py handles packaging
python deploy_aws/deploy.py --role "$ROLE_ARN" --endpoint-name "$ENDPOINT_NAME"
echo "Deployment triggered."
#!/usr/bin/env bash
# 04_export.sh
# Usage: bash scripts/04_export.sh [CONFIG] [CHECKPOINT] [OUTPUT_PATH]

set -e

CONFIG=${1:-"configs/clip_hateful_memes.py"}
CHECKPOINT=${2:-"work_dirs/clip_hateful_memes/latest.pth"}
OUTPUT=${3:-"deploy_aws/model.onnx"}

echo "Exporting model to ONNX..."
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"

python tools/export.py "$CONFIG" "$CHECKPOINT" "$OUTPUT"
echo "Export complete: $OUTPUT"
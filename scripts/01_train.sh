#!/usr/bin/env bash
# 01_train.sh
# Usage: bash scripts/01_train.sh [CONFIG_PATH] [NUM_GPUS]

set -e

CONFIG=${1:-"configs/clip_hateful_memes.py"}
GPUS=${2:-1}

echo "Starting training..."
echo "Config: $CONFIG"
echo "GPUs: $GPUS"

bash tools/dist_train.sh "$CONFIG" "$GPUS"
echo "Training complete."
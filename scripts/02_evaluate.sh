#!/usr/bin/env bash
# 02_evaluate.sh
# Usage: bash scripts/02_evaluate.sh [CONFIG_PATH] [CHECKPOINT_PATH]

set -e

CONFIG=${1:-"configs/clip_hateful_memes.py"}
CHECKPOINT=${2:-"work_dirs/clip_hateful_memes/latest.pth"}

echo "Starting evaluation..."
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"

python tools/eval.py "$CONFIG" "$CHECKPOINT"
echo "Evaluation complete."
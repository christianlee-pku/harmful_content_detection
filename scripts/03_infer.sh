#!/usr/bin/env bash
# 03_infer.sh
# Usage: bash scripts/03_infer.sh [CONFIG] [CHECKPOINT] [IMAGE_PATH] [TEXT]

set -e

CONFIG=${1:-"configs/clip_hateful_memes.py"}
CHECKPOINT=${2:-"work_dirs/clip_hateful_memes/latest.pth"}
IMAGE=${3:-"data/hateful_memes/img/01235.png"}
TEXT=${4:-"look at this meme"}
DEVICE="cpu"

echo "Starting inference..."
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Image: $IMAGE"
echo "Text: $TEXT"

python tools/infer.py "$CONFIG" "$CHECKPOINT" "$IMAGE" "$TEXT" --device $DEVICE
echo "Inference complete."
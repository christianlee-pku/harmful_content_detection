#!/usr/bin/env bash

# Usage Examples:
# 1. Single Node, 4 GPUs:
#    bash tools/dist_train.sh configs/clip_hateful_memes.py 4
#
# 2. Multi-Node (2 nodes, 8 GPUs total):
#    # On Node 0:
#    NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.1 bash tools/dist_train.sh configs/clip_hateful_memes.py 4
#    # On Node 1:
#    NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.1 bash tools/dist_train.sh configs/clip_hateful_memes.py 4
#
# 3. Custom Port:
#    PORT=29501 bash tools/dist_train.sh configs/clip_hateful_memes.py 4

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

if [ $# -lt 2 ]; then
    echo "Usage: bash $0 CONFIG GPUS [NNODES] [NODE_RANK] [PORT] [MASTER_ADDR] [PY_ARGS]"
    exit 1
fi

# Extract any additional python arguments (starting from $3 if NNODES etc are not provided, or after)
# Ideally, we assume user provides specific args or we shift.
# A simpler approach is to let optional args be optional variables and trailing args be pass-through.
shift 2

# We need to handle the optional args logic if we want to support passing them positionally.
# Alternatively, users can set NNODES etc as env vars.
# Let's assume standard usage: bash dist_train.sh config.py 4 --work-dir ...

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@}

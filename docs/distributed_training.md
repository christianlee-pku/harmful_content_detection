# Distributed Training Guide

This guide explains how to perform distributed training using the provided scripts.

## Overview

The system uses PyTorch's `DistributedDataParallel` (DDP) backend. The entry point `scripts/01_train.sh` wraps the underlying launcher to simplify usage.

## Usage

The basic syntax is:

```bash
bash scripts/01_train.sh <CONFIG_PATH> <NUM_GPUS>
```

### 1. Single Node, Multiple GPUs

To train on a single machine with 4 GPUs:

```bash
bash scripts/01_train.sh configs/clip_hateful_memes.py 4
```

To train on a single machine with 1 GPU (using DDP wrapper):

```bash
bash scripts/01_train.sh configs/clip_hateful_memes.py 1
```

### 2. Multi-Node Training

For training across multiple machines (nodes), you typically need to invoke the underlying tool `tools/dist_train.sh` directly with environment variables, as the helper script is optimized for single-node usage.

**Example: 2 Nodes, 8 GPUs total (4 per node)**

**On Node 0 (Master):**
```bash
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.1.100 \
PORT=29500 \
bash tools/dist_train.sh configs/clip_hateful_memes.py 4
```

**On Node 1 (Worker):**
```bash
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.1.100 \
PORT=29500 \
bash tools/dist_train.sh configs/clip_hateful_memes.py 4
```

## Troubleshooting

- **Address already in use**: Change the `PORT` environment variable.
- **NCCL timeout**: Ensure firewalls allow communication on the `PORT` and between nodes.
- **CUDA OOM**: Reduce `samples_per_gpu` in your configuration file.
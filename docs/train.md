# Training Guide

This guide details how to train models using the Harmful Content Detection pipeline.

## Overview

The training process is built on a distributed runner architecture that supports both single-GPU and multi-GPU setups. It leverages `torch.distributed` for process communication and follows a Registry-based design philosophy for component initialization.

## Script Usage

The primary entry point for training is the bash script `scripts/01_train.sh`.

```bash
bash scripts/01_train.sh [CONFIG_PATH] [NUM_GPUS]
```

### Examples

Train with default config on 1 GPU:
```bash
bash scripts/01_train.sh
```

Train with custom config on 4 GPUs:
```bash
bash scripts/01_train.sh configs/my_custom_config.py 4
```

## Configuration File Structure

The system uses a hierarchical configuration system. A typical config file (`configs/clip_hateful_memes.py`) includes:

### Model
Defines the architecture components.

```python
model = dict(
    type='MultimodalFusionModel',
    backbone=dict(type='CLIPBackbone', model_name='ViT-B/32'),
    head=dict(type='FusionHead', in_channels=512, hidden_channels=256, num_classes=2),
    loss=dict(type='CrossEntropyLoss')
)
```

### Data
Defines dataset paths, transforms, and loader settings.

```python
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='HatefulMemesDataset',
        ann_file='data/hateful_memes/train.jsonl',
        img_prefix='data/hateful_memes/img',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Tokenize', max_length=77),
            # ...
        ]
    ),
    val=dict(...)
)
```

### Optimizer & Scheduler
Defines optimization strategy.

```python
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
runner = dict(type='DistRunner', max_epochs=10)
```

## Technical Details

### Architecture

1.  **Entry Point**: `tools/train.py` parses the config and initializes the `Runner`.
2.  **Runner**: The `DistRunner` (or base runner) orchestrates the training loop.
    -   **Epochs & Iterations**: Manages the outer and inner loops.
    -   **Hooks**: Executes registered hooks (e.g., `CheckpointHook`, `LoggerHook`) at specific lifecycle events (`before_run`, `after_iter`, etc.).
3.  **Data Loading**: The `HatefulMemesDataset` is wrapped in a `DataLoader`. Transforms like `LoadImage` and `Tokenize` are applied on-the-fly or cached.

### Distributed Training

For multi-GPU training, the script uses `torch.distributed.launch` (or `torchrun`).

-   **Process Group**: Initializes the process group using the `nccl` backend for NVIDIA GPUs.
-   **Distributed Sampler**: Ensures each GPU receives a distinct slice of the dataset.

## Troubleshooting

-   **CUDA OOM**: Reduce `data.samples_per_gpu` in the config.
-   **NCCL Errors**: Ensure firewall rules allow communication on the distributed training port (default: 29500).
-   **Data Loading Bottleneck**: Increase `data.workers_per_gpu`.
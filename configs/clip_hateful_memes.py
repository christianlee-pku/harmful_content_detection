# Config for Hateful Memes Detection using CLIP
# Ordered by ML Pipeline sequence

# 1. Data Pipeline
data = dict(
    train=dict(
        type='HatefulMemesDataset',
        data_root='data/hateful_memes',
        annotation_file='train.jsonl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CLIPProcessorTransform', model_name='openai/clip-vit-base-patch32')
        ]
    ),
    val=dict(
        type='HatefulMemesDataset',
        data_root='data/hateful_memes',
        annotation_file='dev.jsonl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CLIPProcessorTransform', model_name='openai/clip-vit-base-patch32')
        ]
    ),
    dataloader=dict(
        batch_size=8,
        num_workers=4,
        shuffle=True
    )
)

# 2. Model Architecture
model = dict(
    type='MultimodalClassifier',
    backbone=dict(
        type='CLIP',
        model_name='openai/clip-vit-base-patch32',
        freeze_backbone=True
    ),
    head=dict(
        type='FusionHead',
        input_dim=1024, # 512 image + 512 text
        hidden_dim=512,
        num_classes=2,
        dropout=0.1
    ),
    loss=dict(
        type='CrossEntropyLoss',
        reduction='mean'
    )
)

# 3. Optimization & Schedule
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01
)

# 4. Runtime & Lifecycle
runtime = dict(
    max_epochs=1,
    work_dir='./work_dirs/clip_hateful_memes',
    seed=42
)

hooks = [
    dict(type='LoggerHook', interval=10),
    dict(type='CheckpointHook', interval=1)
]
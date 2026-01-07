import argparse
import os
import torch
import warnings

# Fix huggingface/tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.registry import MODELS, DATASETS, HOOKS
from src.core.runner.base_runner import BaseRunner
from src.optimizer.builder import build_optimizer
from src.datasets.builder import build_dataloader

# Import modules to trigger registration
import src.models.backbones.clip
import src.models.backbones.vilt
import src.models.heads.fusion_head
import src.models.losses.cross_entropy_loss
import src.models.builder
import src.datasets.hateful_memes
import src.datasets.pipelines.transforms
import src.core.hooks.logger
import src.core.hooks.checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.runtime.work_dir = args.work_dir

    # Init logger
    logger = get_logger(
        name="hcd", 
        log_file=os.path.join(cfg.runtime.work_dir, "train.log")
    )
    logger.info(f"Config:\n{cfg}")

    # Set seed
    if cfg.runtime.get('seed') is not None:
        torch.manual_seed(cfg.runtime.seed)

    # Build Model
    logger.info("Building model...")
    model = MODELS.build(cfg.model)

    # Build Datasets
    logger.info("Building datasets...")
    train_dataset = DATASETS.build(cfg.data.train)
    val_dataset = DATASETS.build(cfg.data.val) if cfg.data.get('val') else None

    # Build Dataloaders
    train_loader = build_dataloader(
        train_dataset, 
        batch_size=cfg.data.dataloader.batch_size,
        num_workers=cfg.data.dataloader.num_workers,
        shuffle=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = build_dataloader(
            val_dataset,
            batch_size=cfg.data.dataloader.batch_size,
            num_workers=cfg.data.dataloader.num_workers,
            shuffle=False
        )

    # Build Optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # Build Runner
    if args.launcher == 'pytorch':
        from src.core.runner.dist_runner import DistRunner
        runner_cls = DistRunner
    else:
        runner_cls = BaseRunner

    runner = runner_cls(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        work_dir=cfg.runtime.work_dir,
        max_epochs=cfg.runtime.max_epochs,
        logger=logger,
        seed=cfg.runtime.get('seed', 42)
    )

    # Register Hooks
    if cfg.get('hooks'):
        for hook_cfg in cfg.hooks:
            runner.register_hook(hook_cfg)

    # Resume if needed
    if args.resume_from:
        runner.resume(args.resume_from)

    # Start Training
    logger.info("Starting training...")
    runner.train()


if __name__ == '__main__':
    main()

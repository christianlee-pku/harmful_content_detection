import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.registry import HOOKS
from src.utils.logger import get_logger


class BaseRunner:
    """
    Base Runner that manages the training and validation loops.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        work_dir: str = "work_dirs",
        max_epochs: int = 10,
        logger: Optional[Any] = None,
        seed: int = 42,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.work_dir = work_dir
        self.max_epochs = max_epochs
        self.logger = logger or get_logger(name="runner")
        self.seed = seed

        # Training state
        self.epoch = 0
        self.iter = 0
        self.max_iters = (
            len(self.train_dataloader) * self.max_epochs
            if self.train_dataloader
            else 0
        )
        self.hooks: List[Any] = []
        
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

    def register_hook(self, hook_cfg: Dict[str, Any]):
        """Register a hook from a config dict."""
        hook = HOOKS.build(hook_cfg)
        self.hooks.append(hook)
        # Sort hooks by priority if we implement priority later
        # self.hooks.sort(key=lambda x: x.priority)

    def call_hook(self, fn_name: str):
        """Call a specific method on all registered hooks."""
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

    def train(self):
        """The main training loop."""
        self.call_hook("before_run")
        
        for self.epoch in range(self.epoch, self.max_epochs):
            self.call_hook("before_train_epoch")
            self.model.train()
            
            for i, data_batch in enumerate(self.train_dataloader):
                self.call_hook("before_train_iter")
                
                # Training step
                self.optimizer.zero_grad()
                
                # Use .module if wrapped in DDP
                model = self.model.module if hasattr(self.model, 'module') else self.model
                losses = model.train_step(data_batch, None)
                
                # Parse losses
                if isinstance(losses, dict):
                    total_loss = sum(losses.values())
                else:
                    total_loss = losses
                
                total_loss.backward()
                self.optimizer.step()
                
                self.iter += 1
                self.current_losses = losses # Store for hooks to access
                
                self.call_hook("after_train_iter")
                
            self.call_hook("after_train_epoch")
            
            # Validation
            if self.val_dataloader:
                self.val()
                
        self.call_hook("after_run")

    @torch.no_grad()
    def val(self):
        """The validation loop."""
        self.call_hook("before_val_epoch")
        self.model.eval()
        
        for i, data_batch in enumerate(self.val_dataloader):
            self.call_hook("before_val_iter")
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            predictions = model.val_step(data_batch)
            self.current_predictions = predictions
            
            self.call_hook("after_val_iter")
            
        self.call_hook("after_val_epoch")

    def save_checkpoint(self, filename: str, meta: Optional[Dict] = None):
        """Save a training checkpoint."""
        checkpoint = {
            "epoch": self.epoch + 1,
            "iter": self.iter,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "meta": meta or {},
        }
        filepath = os.path.join(self.work_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        if self.optimizer and checkpoint["optimizer"]:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint.get("epoch", 0)
        self.iter = checkpoint.get("iter", 0)
        self.logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        self.logger.info(f"Resuming from {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)


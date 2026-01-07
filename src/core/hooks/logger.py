import time
import torch
import datetime
from typing import Dict, Optional

from src.core.registry import HOOKS


@HOOKS.register_module()
class LoggerHook:
    """
    Logger hook that prints training status with structured formatting.
    Logs: Epoch, iter, lr, eta, time, data_time, memory, total & individual losses.
    """
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.start_time = 0
        self.iter_time = 0
        self.last_iter_time = 0
        self.data_time = 0
        self.start_iter_timestamp = 0
    
    def before_run(self, runner):
        self.start_time = time.time()
        self.last_iter_time = time.time()

    def before_train_iter(self, runner):
        self.start_iter_timestamp = time.time()
        # Data time is time since last iteration end (or start of run) until now
        self.data_time = self.start_iter_timestamp - self.last_iter_time

    def after_train_iter(self, runner):
        if runner.iter % self.interval != 0:
            self.last_iter_time = time.time()
            return

        # Calculate times
        now = time.time()
        iter_time = now - self.start_iter_timestamp
        self.last_iter_time = now
        
        # Calculate ETA
        # Average iter time could be smoothed, but using current for simplicity or simple moving average
        remaining_iters = runner.max_iters - runner.iter
        eta_seconds = remaining_iters * iter_time
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Get LR
        if runner.optimizer:
            lr = runner.optimizer.param_groups[0]['lr']
        else:
            lr = 0.0

        # Memory
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            mem_mb = 0

        # Losses
        losses = runner.current_losses
        if isinstance(losses, dict):
            total_loss = sum(l.item() for l in losses.values())
            # Format sub-losses, excluding 'loss' if it exists and is the only one (handled by total)
            # But typically 'loss' IS the total loss in some frameworks. Here we sum values.
            # If the model returns {'loss': tensor}, sum is that tensor.
            # We want to print: loss: <total> loss_cls: <val> ...
            
            sub_loss_str = []
            for k, v in losses.items():
                # If key is 'loss', check if it equals total_loss (it should). 
                # If we have multiple losses, 'loss' usually isn't one of them unless it's explicitly returned.
                # Let's just print all keys except 'loss' if we already have a total 'loss' label.
                if k == 'loss':
                    continue
                sub_loss_str.append(f"{k}: {v.item():.4f}")
            
            loss_str = f"loss: {total_loss:.4f}"
            if sub_loss_str:
                loss_str += " " + " ".join(sub_loss_str)
        else:
            total_loss = losses.item()
            loss_str = f"loss: {total_loss:.4f}"

        # INFO - Epoch(train) [5][2500/3517] lr: 1.8168e-04 eta: 22:37:17 time: 1.2090 data_time: 0.0404 memory: 24752 grad_norm: 1.7776 loss: 0.6155 loss_xx: 0.2851
        
        # Determine batches per epoch
        if runner.train_dataloader:
            batches_per_epoch = len(runner.train_dataloader)
        else:
            batches_per_epoch = 0
            
        current_iter_in_epoch = (runner.iter - 1) % batches_per_epoch + 1 if batches_per_epoch > 0 else runner.iter

        log_msg = (
            f"Epoch(train) [{runner.epoch + 1}][{current_iter_in_epoch}/{batches_per_epoch}] "
            f"lr: {lr:.4e} "
            f"eta: {eta_str} "
            f"time: {iter_time:.4f} "
            f"data_time: {self.data_time:.4f} "
            f"memory: {int(mem_mb)} "
            f"{loss_str}"
        )
        
        runner.logger.info(log_msg)

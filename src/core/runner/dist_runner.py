import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List, Union

from src.core.runner.base_runner import BaseRunner
from src.core.registry import RUNNERS


@RUNNERS.register_module()
class DistRunner(BaseRunner):
    """
    Runner that supports both Distributed Data Parallel (DDP) training and 
    Single-GPU training.
    """
    def __init__(self, gpu_ids: Optional[Union[int, List[int]]] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.gpu_ids = gpu_ids

        self._init_dist()
        self._wrap_model()

    def _init_dist(self):
        """Initialize distributed environment if available."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.distributed = True
            
            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
                self.logger.info(f"Initialized process group: rank={self.rank}, world_size={self.world_size}")
        else:
            self.logger.info("Distributed environment not detected. Running in single-process mode.")
            self.distributed = False

        if torch.cuda.is_available():
            if self.distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                # Single process mode
                # If gpu_ids is provided as int (e.g. 0), use that.
                # If list (e.g. [0]), use first.
                # If None, use default.
                
                if self.gpu_ids is not None:
                    if isinstance(self.gpu_ids, list):
                        gpu_id = self.gpu_ids[0]
                    else:
                        gpu_id = int(self.gpu_ids)
                    
                    torch.cuda.set_device(gpu_id)
                    self.device = torch.device(f"cuda:{gpu_id}")
                else:
                    # Default to current device (usually 0)
                    self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def _wrap_model(self):
        """Move model to device and wrap with DDP if distributed."""
        self.model = self.model.to(self.device)
        
        if self.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank, 
                find_unused_parameters=True
            )
            self.logger.info(f"Model wrapped with DistributedDataParallel on device {self.device}")
        else:
            self.logger.info(f"Model moved to device {self.device}")

    def save_checkpoint(self, filename: str, meta: Optional[dict] = None):
        """Save checkpoint only on rank 0."""
        if self.rank == 0:
            # Access underlying model for DDP
            model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
            
            checkpoint = {
                "epoch": self.epoch + 1,
                "iter": self.iter,
                "state_dict": model_state,
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "meta": meta or {},
            }
            filepath = os.path.join(self.work_dir, filename)
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
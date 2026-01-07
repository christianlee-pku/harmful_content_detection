import torch.optim as optim
from typing import Dict, Any, Iterable

from src.core.registry import OPTIMIZERS


def build_optimizer(model, cfg: Dict[str, Any]):
    """
    Build optimizer from config for a specific model.
    """
    optimizer_cfg = cfg.copy()
    optim_type = optimizer_cfg.pop('type')
    
    # Let's verify if OPTIMIZERS has it.
    optim_cls = OPTIMIZERS.get(optim_type)
    if optim_cls is None:
        # Fallback to torch.optim
        if hasattr(optim, optim_type):
            optim_cls = getattr(optim, optim_type)
        else:
            raise ValueError(f"Optimizer {optim_type} not found in registry or torch.optim")
            
    return optim_cls(model.parameters(), **optimizer_cfg)

# Register standard optimizers
# Use force=True or check before register if multiple imports are possible
if 'Adam' not in OPTIMIZERS.module_dict:
    OPTIMIZERS.register_module(module=optim.Adam, name='Adam')
if 'AdamW' not in OPTIMIZERS.module_dict:
    OPTIMIZERS.register_module(module=optim.AdamW, name='AdamW')
if 'SGD' not in OPTIMIZERS.module_dict:
    OPTIMIZERS.register_module(module=optim.SGD, name='SGD')
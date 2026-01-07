import torch.nn as nn
from typing import Optional

from src.core.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """
    Wrapper for torch.nn.CrossEntropyLoss to register it.
    """

    def __init__(self, weight: Optional[list] = None, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight)
        
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)

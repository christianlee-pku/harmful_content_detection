import torch
import torch.nn as nn
from typing import Dict, Union

from src.core.registry import MODELS
from src.core.base import BaseModel


@MODELS.register_module()
class FusionHead(BaseModel):
    """
    Simple fusion head that concatenates features and classifies.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], data_samples=None, mode='tensor') -> torch.Tensor:
        """
        Args:
            inputs: Tensor (fused features) or Dict (modality specific features to be fused)
        """
        if isinstance(inputs, dict):
            # Example simple fusion: Concatenate image and text embeddings
            # Assuming keys 'image_embeds' and 'text_embeds' exist
            if 'image_embeds' in inputs and 'text_embeds' in inputs:
                x = torch.cat([inputs['image_embeds'], inputs['text_embeds']], dim=-1)
            else:
                # Fallback or error if keys missing. 
                # For ViLT, inputs might be just a tensor already.
                raise ValueError(f"FusionHead expects 'image_embeds' and 'text_embeds' in input dict, got {inputs.keys()}")
        else:
            x = inputs

        return self.fc(x)

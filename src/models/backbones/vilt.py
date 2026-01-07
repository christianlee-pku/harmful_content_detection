import torch
import torch.nn as nn
from transformers import ViltModel
from typing import Optional
import os

from src.core.registry import MODELS


@MODELS.register_module()
class ViLT(nn.Module):
    """
    ViLT Backbone wrapper.
    """

    def __init__(self, model_name: str = "dandelin/vilt-b32-mlm", freeze_backbone: bool = True, local_files_only: bool = False):
        super().__init__()
        
        if os.path.isdir(model_name):
            self.model = ViltModel.from_pretrained(model_name, local_files_only=True)
        else:
            self.model = ViltModel.from_pretrained(model_name, local_files_only=local_files_only)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        return outputs.last_hidden_state[:, 0, :] # CLS token

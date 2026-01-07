import torch
import torch.nn as nn
from transformers import CLIPModel
from typing import Dict, Optional, Union
import os

from src.core.registry import MODELS


@MODELS.register_module()
class CLIP(nn.Module):
    """
    CLIP Backbone wrapper.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", freeze_backbone: bool = True, local_files_only: bool = False):
        super().__init__()
        
        # If model_name is a directory, load from there (Cloud/SageMaker scenario)
        if os.path.isdir(model_name):
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=True, use_safetensors=True)
        else:
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only, use_safetensors=True)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return {
            'image_embeds': outputs.image_embeds, # [batch_size, embed_dim]
            'text_embeds': outputs.text_embeds,   # [batch_size, embed_dim]
            'last_hidden_state': outputs.vision_model_output.last_hidden_state
        }

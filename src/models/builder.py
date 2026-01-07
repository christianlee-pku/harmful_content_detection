import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
from src.core.registry import MODELS
from src.core.base import BaseModel


@MODELS.register_module()
class MultimodalClassifier(BaseModel):
    """
    Standard classifier architecture: Backbone -> Head -> Loss
    """
    def __init__(self, backbone: Dict[str, Any], head: Dict[str, Any], loss: Dict[str, Any] = None, init_cfg: Dict = None):
        super().__init__(init_cfg)
        self.backbone = MODELS.build(backbone)
        self.head = MODELS.build(head)
        if loss:
            self.loss_module = MODELS.build(loss)
        else:
            self.loss_module = None

    def forward(self, inputs=None, data_samples=None, mode='tensor', **kwargs):
        """
        Multimodal forward pass.
        
        Args:
            inputs (Dict, optional): Dictionary containing 'pixel_values', 'input_ids', etc. (Standard training)
            data_samples (List, optional): Data samples containing labels (Standard training)
            mode (str): Execution mode ('tensor', 'loss', 'predict').
            **kwargs: Additional arguments, specifically to capture 'pixel_values', 'input_ids', 
                      'attention_mask' when passed as keyword arguments (e.g. from ONNX wrapper).
        """
        
        # Resolve inputs into a dictionary
        if isinstance(inputs, dict):
            input_dict = inputs
        else:
            # Fallback for ONNX export wrapper which passes args as kwargs
            # or any other caller using explicit kwargs
            input_dict = kwargs

        # Forward Backbone
        # The backbone (CLIP/ViLT) expects kwargs or specific args
        # We unpack input_dict to pass them
        if input_dict:
            features = self.backbone(**input_dict)
        else:
            # Fallback if someone called model(tensor) which isn't standard here but supported by base
            features = self.backbone(input_dict)

        # Forward Head
        logits = self.head(features)

        # Return based on mode
        if mode == 'tensor':
            return logits
        
        if mode == 'predict':
            return logits.argmax(dim=-1)

        if mode == 'loss':
            if self.loss_module is None:
                raise ValueError("Loss module is not defined but mode='loss' requested")
            
            labels = None
            if isinstance(data_samples, dict) and 'labels' in data_samples:
                 labels = data_samples['labels']
            
            if labels is None:
                 raise ValueError("Labels not found in data_samples for loss calculation")
            
            return {'loss_cls': self.loss_module(logits, labels)}
        
        return logits

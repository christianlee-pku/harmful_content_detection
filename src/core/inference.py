import torch
from typing import Union, List, Dict, Any
from PIL import Image

from src.utils.config import Config
from src.core.registry import MODELS, TRANSFORMS
from src.utils.logger import get_logger

# Import modules to trigger registration
import src.models.backbones.clip
import src.models.backbones.vilt
import src.models.heads.fusion_head
import src.models.losses.cross_entropy_loss
import src.models.builder
import src.datasets.hateful_memes
import src.datasets.pipelines.transforms


class HCDInference:
    """
    High-level Inference API for Harmful Content Detection.
    """
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cpu'):
        self.cfg = Config.fromfile(config_path)
        self.device = torch.device(device)
        self.logger = get_logger("inference")
        
        # Build model
        self.model = MODELS.build(self.cfg.model)
        
        # Load checkpoint
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Build transforms (pipeline)
        # We need to extract the pipeline config. Usually in data.test or data.val
        pipeline_cfg = self.cfg.data.get('test', self.cfg.data.get('val', {})).get('pipeline', [])
        
        self.transforms = []
        for transform_cfg in pipeline_cfg:
            # We don't need LoadImageFromFile if we pass PIL Image directly
            if transform_cfg['type'] == 'LoadImageFromFile':
                continue
            self.transforms.append(TRANSFORMS.build(transform_cfg))

    def predict(self, image: Union[str, Image.Image], text: str) -> Dict[str, Any]:
        """
        Predict harmfulness of an image-text pair.
        """
        # Prepare data dict
        data = {'text': text}
        
        if isinstance(image, str):
            try:
                img = Image.open(image).convert('RGB')
                data['img'] = img
            except Exception as e:
                raise ValueError(f"Could not load image from {image}: {e}")
        elif isinstance(image, Image.Image):
            data['img'] = image.convert('RGB')
        else:
            raise TypeError("Image must be file path or PIL Image")

        # Apply transforms
        for t in self.transforms:
            data = t(data)
            
        # Move tensors to device and batch
        inputs = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.unsqueeze(0).to(self.device) # Add batch dim
        
        # Forward pass
        with torch.no_grad():
            # MultimodalClassifier forward expects 'inputs' dict (unpacked inside if needed)
            # Or kwargs if we changed builder.py
            
            # Using keyword unpacking for safety as builder.py was updated to support **inputs
            # Wait, MultimodalClassifier.forward takes 'inputs' (dict), 'data_samples', 'mode'
            # It does NOT take **kwargs matching the backbone inputs directly.
            # So we should pass the dict as the first argument.
            
            logits = self.model(inputs, mode='tensor')
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            
        return {
            'label': pred_label,
            'score': probs[0, pred_label].item(),
            'probabilities': probs[0].tolist()
        }

from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, CLIPProcessor

from src.core.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile:
    """
    Load an image from file.
    """
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img_path' not in results:
            raise KeyError("img_path missing in results")
        
        try:
            img = Image.open(results['img_path']).convert('RGB')
            results['img'] = img
        except Exception as e:
            raise IOError(f"Failed to load image {results['img_path']}: {e}")
            
        return results


@TRANSFORMS.register_module()
class Tokenize:
    """
    Tokenize text using a pretrained tokenizer.
    """
    def __init__(self, tokenizer_name: str, max_length: int = 77, padding: str = 'max_length', truncation: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'text' not in results:
            raise KeyError("text missing in results")
        
        text = results['text']
        encoded = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        for k, v in encoded.items():
            results[k] = v.squeeze(0)
            
        return results


@TRANSFORMS.register_module()
class CLIPProcessorTransform:
    """
    Use CLIPProcessor to handle both Image and Text.
    This replaces manual Tokenize and separate Image transforms if using standard CLIP input.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        # Silence warning by using fast processor if available
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        text = results['text']
        image = results.get('img') # Should be loaded PIL image
        
        if image is None:
             raise ValueError("Image not found in results. Ensure LoadImageFromFile is run before.")
             
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
        
        # Unpack
        for k, v in inputs.items():
            results[k] = v.squeeze(0)
            
        return results

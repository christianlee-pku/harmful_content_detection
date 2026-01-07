import os
import json
import torch
from typing import List, Dict, Any, Optional
from PIL import Image

from src.core.registry import DATASETS
from src.core.base import BaseDataset


@DATASETS.register_module()
class HatefulMemesDataset(BaseDataset):
    """
    Hateful Memes Dataset.
    """
    def __init__(self, data_root: str, annotation_file: str, pipeline: List[Dict], test_mode: bool = False, **kwargs):
        self.annotation_file = annotation_file
        super().__init__(data_root, pipeline, test_mode, **kwargs)
        
        # Build pipeline transforms
        from src.core.registry import TRANSFORMS
        self.transforms = []
        for transform_cfg in pipeline:
            self.transforms.append(TRANSFORMS.build(transform_cfg))

    def load_data_list(self) -> List[Dict]:
        """
        Load annotations from jsonl file.
        """
        data_list = []
        ann_path = os.path.join(self.data_root, self.annotation_file)
        if not os.path.exists(ann_path):
             # Try absolute path if not in data_root
             if os.path.exists(self.annotation_file):
                 ann_path = self.annotation_file
             else:
                 raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # items have keys: id, img, label, text
                data_info = {
                    'img_path': os.path.join(self.data_root, item['img']),
                    'text': item['text'],
                    'labels': item['label'],
                    'id': item['id']
                }
                data_list.append(data_info)
        return data_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data_info = self.data_list[idx]
        
        # Create results dict that pipeline transforms will modify
        results = {
            'img_path': data_info['img_path'],
            'text': data_info['text'],
            'labels': data_info['labels'],
            'id': data_info['id']
        }
        
        # Apply transforms
        for transform in self.transforms:
            results = transform(results)
            
        # Filter out non-collatable types (like PIL images) 
        # to avoid DataLoader errors.
        collatable_results = {}
        for k, v in results.items():
            if isinstance(v, (torch.Tensor, str, int, float, dict, list)):
                collatable_results[k] = v
        
        return collatable_results

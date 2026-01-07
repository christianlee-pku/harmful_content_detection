from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BaseModel(nn.Module, ABC):
    """
    Base class for all models in the HCD framework.
    """

    def __init__(self, init_cfg: Optional[Dict] = None):
        super().__init__()
        self.init_cfg = init_cfg

    @abstractmethod
    def forward(self, inputs: Any, data_samples: Optional[List] = None, mode: str = 'tensor') -> Union[torch.Tensor, Dict, List]:
        """
        Standard forward interface.
        """
        pass

    def train_step(self, data: Dict[str, Any], optim_wrapper: Any) -> Dict[str, torch.Tensor]:
        """
        Default training step.
        """
        inputs, data_samples = self.data_preprocessor(data)
        losses = self(inputs, data_samples, mode='loss')
        return losses

    def val_step(self, data: Dict[str, Any]) -> List[Any]:
        """
        Default validation step.
        """
        inputs, data_samples = self.data_preprocessor(data)
        predictions = self(inputs, data_samples, mode='predict')
        return predictions

    def test_step(self, data: Dict[str, Any]) -> List[Any]:
        """
        Default test step.
        """
        return self.val_step(data)

    def data_preprocessor(self, data: Dict[str, Any]) -> tuple:
        """
        Simple data preprocessor that moves data to device and splits into inputs/data_samples.
        """
        device = next(self.parameters()).device
        
        # Keys expected by the model forward pass
        input_keys = ['pixel_values', 'input_ids', 'attention_mask', 'token_type_ids']
        
        inputs = {}
        data_samples = {'labels': None}
        
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            
            if k in input_keys:
                inputs[k] = v
            else:
                # Other metadata
                data_samples[k] = v
                
        return inputs, data_samples


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets in the HCD framework.
    """

    def __init__(self, 
                 data_root: str, 
                 pipeline: List[Dict], 
                 test_mode: bool = False,
                 metainfo: Optional[Dict] = None):
        self.data_root = data_root
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.metainfo = metainfo or {}
        
        self.data_list = self.load_data_list()

    @abstractmethod
    def load_data_list(self) -> List[Dict]:
        """
        Load annotations and return a list of data info dicts.
        """
        pass

    def __len__(self) -> int:
        return len(self.data_list)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index.
        """
        pass

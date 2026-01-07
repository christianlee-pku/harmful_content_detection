from typing import Dict, Any, Union
from torch.utils.data import Dataset, DataLoader

from src.core.registry import DATASETS


def build_dataset(cfg: Dict[str, Any]) -> Dataset:
    """Build dataset."""
    return DATASETS.build(cfg)


def build_dataloader(dataset: Dataset, 
                    batch_size: int = 1, 
                    num_workers: int = 0, 
                    shuffle: bool = False,
                    **kwargs) -> DataLoader:
    """Build dataloader."""
    # Custom collate_fn might be needed here to handle dicts from dataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs
    )

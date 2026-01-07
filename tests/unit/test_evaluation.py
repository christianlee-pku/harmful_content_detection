import os
import torch
import shutil
import pytest
from src.utils.config import Config
from src.core.registry import MODELS, DATASETS
from tools.eval import calculate_metrics

def test_calculate_metrics():
    # Test with dummy data
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]]) # Class 1, Class 0
    targets = torch.tensor([1, 0])
    
    metrics = calculate_metrics(preds, targets)
    assert metrics['accuracy'] == 1.0
    assert metrics['auc'] == 1.0
    
    # Test mismatch
    preds = torch.tensor([[0.1, 0.9], [0.1, 0.9]]) # Class 1, Class 1
    targets = torch.tensor([1, 0])
    metrics = calculate_metrics(preds, targets)
    assert metrics['accuracy'] == 0.5

def test_eval_script_flow(tmp_path):
    # This is a smoke test for the eval script logic (mocking parts)
    pass

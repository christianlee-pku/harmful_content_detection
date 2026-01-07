import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.core.runner.base_runner import BaseRunner
from src.core.hooks.checkpoint import CheckpointHook
from src.utils.logger import get_logger

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)
    
    def train_step(self, data, optim_wrapper):
        inputs = data['inputs']
        targets = data['targets']
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        return {'loss': loss}
        
    def data_preprocessor(self, data):
        return data['inputs'], data

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(20, 10)
        self.targets = torch.randn(20, 1)
    
    def __len__(self):
        return 20
    
    def __getitem__(self, idx):
        return {'inputs': self.data[idx], 'targets': self.targets[idx]}

def test_checkpoint_hook(tmp_path):
    # Setup
    work_dir = str(tmp_path / "work_dirs")
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=4)
    logger = get_logger(name="test_checkpoint")
    
    runner = BaseRunner(
        model=model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        work_dir=work_dir,
        max_epochs=2,
        logger=logger
    )
    
    # Add CheckpointHook
    hook = CheckpointHook(interval=1)
    runner.hooks.append(hook)
    
    # Run training
    runner.train()
    
    # Check if checkpoints exist
    assert os.path.exists(os.path.join(work_dir, "epoch_1.pth"))
    assert os.path.exists(os.path.join(work_dir, "epoch_2.pth"))
    assert os.path.exists(os.path.join(work_dir, "latest.pth"))
    
    # Verify content
    ckpt = torch.load(os.path.join(work_dir, "latest.pth"))
    assert ckpt['epoch'] == 2
    assert 'state_dict' in ckpt
    assert 'optimizer' in ckpt

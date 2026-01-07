import argparse
import os
import torch
from tqdm import tqdm
from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.registry import MODELS, DATASETS
from src.datasets.builder import build_dataloader

# Import modules to trigger registration
import src.models.backbones.clip
import src.models.backbones.vilt
import src.models.heads.fusion_head
import src.models.losses.cross_entropy_loss
import src.models.builder
import src.datasets.hateful_memes
import src.datasets.pipelines.transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs')
    return parser.parse_args()


def calculate_metrics(predictions, targets):
    """
    Calculate accuracy and ROC-AUC.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # predictions: list of tensors or tensor [N, num_classes] (logits or probs)
    # targets: list of tensors or tensor [N]
    
    if isinstance(predictions, list):
        predictions = torch.cat(predictions)
    if isinstance(targets, list):
        targets = torch.cat(targets)
        
    probs = torch.softmax(predictions, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    # Move to CPU numpy
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs_np = probs.cpu().numpy()[:, 1] # Prob of positive class
    
    acc = accuracy_score(targets_np, preds_np)
    try:
        auc = roc_auc_score(targets_np, probs_np)
    except ValueError:
        auc = 0.0 # Handle case with only one class
        
    return {'accuracy': acc, 'auc': auc}


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    work_dir = args.work_dir or cfg.runtime.work_dir
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # Init logger
    logger = get_logger(
        name="hcd_eval", 
        log_file=os.path.join(work_dir, "eval.log")
    )
    logger.info(f"Config:\n{cfg}")

    # Build Model
    logger.info("Building model...")
    model = MODELS.build(cfg.model)
    
    # Load Checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Build Test Dataset
    logger.info("Building test dataset...")
    # Use 'val' if 'test' is not in config, or explicitly 'test'
    test_cfg = cfg.data.get('test', cfg.data.get('val'))
    test_dataset = DATASETS.build(test_cfg)

    # Build Dataloader
    test_loader = build_dataloader(
        test_dataset, 
        batch_size=cfg.data.dataloader.batch_size,
        num_workers=cfg.data.dataloader.num_workers,
        shuffle=False
    )

    # Evaluation Loop
    logger.info("Starting evaluation...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, data_batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # Move data to device
            inputs, data_samples = model.data_preprocessor(data_batch)
            
            # Forward
            # mode='tensor' returns logits
            outputs = model(inputs, mode='tensor')
            
            all_predictions.append(outputs)
            
            if data_samples['labels'] is not None:
                all_targets.append(data_samples['labels'])

    # Metrics
    if all_targets:
        metrics = calculate_metrics(all_predictions, all_targets)
        logger.info(f"Test Results: {metrics}")
    else:
        logger.warning("No labels found in test set, skipping metrics.")

if __name__ == '__main__':
    main()

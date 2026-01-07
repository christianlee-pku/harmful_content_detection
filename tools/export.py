import argparse
import torch
import os
from src.utils.config import Config
from src.core.registry import MODELS
from src.utils.export_utils import export_to_onnx
from src.utils.logger import get_logger

# Import modules to trigger registration
import src.models.backbones.clip
import src.models.backbones.vilt
import src.models.heads.fusion_head
import src.models.losses.cross_entropy_loss
import src.models.builder

def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_file', help='output onnx file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    logger = get_logger("export")
    
    # Build model
    logger.info("Building model...")
    model = MODELS.build(cfg.model)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    # Handle state_dict key if it exists
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    logger.info("Creating dummy input...")
    batch_size = 1
    # Standard CLIP/ViLT input shapes
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 100, (batch_size, 77))
    attention_mask = torch.ones(batch_size, 77, dtype=torch.long)
    
    input_sample = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    logger.info(f"Exporting to {args.output_file}...")
    export_to_onnx(model, args.output_file, input_sample)
    logger.info("Export complete.")

if __name__ == '__main__':
    main()
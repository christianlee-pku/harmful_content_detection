import argparse
import os
import json
import torch
from src.core.inference import HCDInference

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on a single image-text pair')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image file path')
    parser.add_argument('text', help='text content')
    parser.add_argument('--device', default='cpu', help='device used for inference')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Inference API
    # Ensure PYTHONPATH includes current directory
    inferencer = HCDInference(args.config, args.checkpoint, args.device)
    
    # Run prediction
    print(f"Predicting for image: {args.image}")
    print(f"Text: {args.text}")
    
    result = inferencer.predict(args.image, args.text)
    
    # Print result
    print("\nResult:")
    print(json.dumps(result, indent=4))

if __name__ == '__main__':
    main()
import os
import json
import random
import argparse

def sample_jsonl(input_file, output_file, num_samples=5):
    """
    Sample lines from a jsonl file.
    """
    if not os.path.exists(input_file):
        print(f"Warning: Input file {input_file} not found. Skipping.")
        return

    with open(input_file, 'r') as f:
        lines = f.readlines()

    if len(lines) <= num_samples:
        sampled_lines = lines
    else:
        sampled_lines = random.sample(lines, num_samples)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for line in sampled_lines:
            f.write(line)
    
    print(f"Sampled {len(sampled_lines)} lines from {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Sample dataset for debugging.")
    parser.add_argument('--data-root', default='data/hateful_memes', help='Root directory of dataset')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    args = parser.parse_args()

    # Define files to sample
    files = ['train.jsonl', 'dev.jsonl']
    
    for filename in files:
        input_path = os.path.join(args.data_root, filename)
        # Create a tiny version
        output_filename = filename.replace('.jsonl', '_tiny.jsonl')
        output_path = os.path.join(args.data_root, output_filename)
        
        sample_jsonl(input_path, output_path, args.num_samples)

if __name__ == '__main__':
    main()

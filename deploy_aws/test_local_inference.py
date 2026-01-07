import sys
import os
import json
import base64
import torch

# Ensure we can import from code/ directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from inference import model_fn, input_fn, predict_fn, output_fn

def test_local_inference(model_dir, image_path, text="test text"):
    print("--- Starting Local Inference Test ---")
    
    # 1. Load Model
    print("1. Testing model_fn...")
    try:
        model_artifacts = model_fn(model_dir)
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # 2. Input Processing
    print("2. Testing input_fn...")
    # Simulate API Gateway / SageMaker request
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    request_body = json.dumps({
        "text": text,
        "image": img_b64
    })
    request_content_type = 'application/json'
    
    try:
        input_data = input_fn(request_body, request_content_type)
        print("   Input processed successfully.")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # 3. Prediction
    print("3. Testing predict_fn...")
    try:
        prediction = predict_fn(input_data, model_artifacts)
        print(f"   Prediction shape: {prediction.shape}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    # 4. Output Processing
    print("4. Testing output_fn...")
    try:
        response = output_fn(prediction, 'application/json')
        print(f"   Response: {response}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    print("\n--- Local Inference Test PASSED ---")

if __name__ == "__main__":
    # Assuming run from repo root or deploy_aws/
    # Adjust paths as needed
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = BASE_DIR # model.onnx should be here
    IMAGE_PATH = os.path.join(BASE_DIR, "../data/hateful_memes/img/01235.png") # Example image
    
    if not os.path.exists(os.path.join(MODEL_DIR, "model.onnx")):
        print(f"Error: model.onnx not found in {MODEL_DIR}")
        exit(1)
        
    if not os.path.exists(IMAGE_PATH):
        # Fallback to creating a dummy image if dataset not present
        print("Image not found, creating dummy...")
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        IMAGE_PATH = os.path.join(BASE_DIR, "test_image.png")
        img.save(IMAGE_PATH)

    test_local_inference(MODEL_DIR, IMAGE_PATH)

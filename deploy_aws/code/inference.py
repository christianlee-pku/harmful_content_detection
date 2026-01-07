import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
from transformers import CLIPProcessor

# 1. Load Model
def model_fn(model_dir):
    """
    Load ONNX model and processor.
    """
    print(f"Loading model from {model_dir}")
    
    # Model path (SageMaker extracts model.tar.gz to model_dir)
    model_path = os.path.join(model_dir, "model.onnx")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create ONNX Runtime Session
    # Use 'CUDAExecutionProvider' if GPU is available
    providers = ['CPUExecutionProvider']
    try:
        import torch
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
    except ImportError:
        pass

    session = ort.InferenceSession(model_path, providers=providers)
    
    # Load HuggingFace Processor for preprocessing
    # Ensure internet access is available or pretrained files are in code/ directory
    # For now, we assume internet access or cache
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return {"session": session, "processor": processor}

# 2. Input Processing
def input_fn(request_body, request_content_type):
    """
    Deserialize inputs.
    Supported Content-Types: application/json
    Format: {"text": "some text", "image": "base64_string"}
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        text = data.get("text", "")
        img_b64 = data.get("image", "")
        
        if not img_b64:
            raise ValueError("No image provided in request")

        try:
            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
        
        return {"text": text, "image": image}
        
    raise ValueError(f"Unsupported content type: {request_content_type}")

# 3. Prediction
def predict_fn(input_data, model_dict):
    """
    Preprocess and Inference.
    """
    session = model_dict['session']
    processor = model_dict['processor']
    
    text = input_data['text']
    image = input_data['image']
    
    # Preprocess
    inputs = processor(text=[text], images=image, return_tensors="np", padding="max_length", max_length=77, truncation=True)
    
    # Prepare ONNX inputs
    onnx_inputs = {
        "pixel_values": inputs["pixel_values"].astype(np.float32),
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    # Run Inference
    # session.run returns list of outputs. Assuming logits is the first.
    logits = session.run(None, onnx_inputs)[0] 
    
    return logits

# 4. Output Processing
def output_fn(prediction, response_content_type):
    """
    Serialize output.
    """
    if response_content_type == 'application/json':
        # Softmax
        exp_preds = np.exp(prediction)
        probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        
        # Taking first item in batch
        probs = probs[0]
        pred_label = int(np.argmax(probs))
        score = float(probs[pred_label])
        
        result = {
            "label": pred_label,
            "score": score,
            "probabilities": probs.tolist()
        }
        return json.dumps(result)
        
    raise ValueError(f"Unsupported response content type: {response_content_type}")

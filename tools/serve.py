import os
import json
import torch
import logging
from PIL import Image
import io
import base64

from src.utils.config import Config
from src.core.registry import MODELS, TRANSFORMS

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

def model_fn(model_dir):
    """
    Load the model for inference
    """
    logger.info("Loading model...")
    
    # Locate config and checkpoint
    # Assuming standard SageMaker directory structure where artifacts are in model_dir
    # We expect a config file and a checkpoint file (e.g. model.pth)
    
    config_path = os.path.join(model_dir, "config.py")
    checkpoint_path = os.path.join(model_dir, "model.pth")
    
    if not os.path.exists(config_path):
        # Fallback to looking recursively if needed or raise error
        # For this example, we assume they are at root
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError("config.py not found in model_dir")
        
    cfg = Config.fromfile(config_path)
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS.build(cfg.model)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")

    model.to(device)
    model.eval()
    
    # Load transforms
    # We can cache transforms in the model object or a global if needed
    # For now, we rebuild them in input_fn or attach to model
    pipeline_cfg = cfg.data.get('test', cfg.data.get('val', {})).get('pipeline', [])
    transforms = []
    from src.core.registry import TRANSFORMS
    for transform_cfg in pipeline_cfg:
        if transform_cfg['type'] == 'LoadImageFromFile':
            continue
        transforms.append(TRANSFORMS.build(transform_cfg))
        
    model.transforms = transforms
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize the request body
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        text = input_data.get("text", "")
        
        # Handle image: expect base64 string or url (simple version: base64)
        img_data = input_data.get("image")
        if img_data:
            image = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
        else:
            raise ValueError("No image data provided")
            
        data = {'text': text, 'img': image}
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generate predictions
    """
    device = next(model.parameters()).device
    
    # Apply transforms
    for t in model.transforms:
        input_data = t(input_data)
        
    # Batchify
    inputs = {}
    for k, v in input_data.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.unsqueeze(0).to(device)
            
    with torch.no_grad():
        # Call model using keyword arguments
        logits = model(**inputs, mode='tensor')
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        score = probs[0, pred_label].item()
        
    return {
        'label': pred_label,
        'score': score,
        'probabilities': probs[0].tolist()
    }

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f"Unsupported response content type: {response_content_type}")

if __name__ == '__main__':
    # Local testing
    pass

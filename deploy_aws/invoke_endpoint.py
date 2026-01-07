import boto3
import json
import base64
import argparse
import os

def invoke_endpoint(endpoint_name, image_path, text, region_name='us-east-1'):
    """
    Invoke SageMaker endpoint with image and text.
    """
    # Create SageMaker runtime client
    try:
        client = boto3.client('sagemaker-runtime', region_name=region_name)
    except Exception as e:
        print(f"Error creating boto3 client: {e}")
        print("Ensure you have AWS credentials configured.")
        return

    # Read and encode image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    # Construct payload matches input_fn in inference.py
    payload = {
        "text": text,
        "image": img_b64
    }

    print(f"Invoking endpoint '{endpoint_name}'...")
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        response_body = response['Body'].read().decode('utf-8')
        result = json.loads(response_body)
        
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
        
    except Exception as e:
        print(f"Error invoking endpoint: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke SageMaker Endpoint")
    parser.add_argument("--endpoint", default="hcd-onnx-endpoint", help="SageMaker Endpoint Name")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--text", default="Look at this meme", help="Text input")
    parser.add_argument("--region", default="us-east-1", help="AWS Region")
    
    args = parser.parse_args()
    
    invoke_endpoint(args.endpoint, args.image, args.text, args.region)

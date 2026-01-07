import argparse
import os
import tarfile
import sagemaker
from sagemaker.pytorch import PyTorchModel

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy model to AWS SageMaker')
    parser.add_argument('model_dir', help='directory containing model artifacts (config.py, model.pth)')
    parser.add_argument('--role', help='AWS IAM role for SageMaker', default=None)
    parser.add_argument('--instance-type', help='EC2 instance type', default='ml.m5.xlarge')
    parser.add_argument('--endpoint-name', help='Endpoint name', default='hcd-endpoint')
    return parser.parse_args()

def package_model(model_dir, output_path='model.tar.gz'):
    """
    Package model artifacts into a tar.gz file.
    """
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")
    print(f"Model packaged to {output_path}")
    return output_path

def deploy(args):
    # check if sagemaker is installed
    try:
        import sagemaker
    except ImportError:
        print("Please install sagemaker sdk: pip install sagemaker")
        return

    # 1. Package model
    model_tar = package_model(args.model_dir)
    
    # 2. Upload to S3 (optional, or let SageMaker handle local file upload)
    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = f"hcd-models/{args.endpoint_name}"
    model_data = session.upload_data(path=model_tar, bucket=bucket, key_prefix=prefix)
    print(f"Model uploaded to {model_data}")

    # 3. Create SageMaker Model
    # We use the 'tools/serve.py' as the entry point
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=args.role or sagemaker.get_execution_role(),
        entry_point='serve.py',
        source_dir='tools', # directory containing serve.py and other deps if needed
        framework_version='2.0.0',
        py_version='py310',
        dependencies=['src'] # Upload src directory as well so serve.py can import it
    )

    # 4. Deploy
    print("Deploying endpoint...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name
    )
    print(f"Endpoint {args.endpoint_name} deployed!")

if __name__ == '__main__':
    args = parse_args()
    deploy(args)

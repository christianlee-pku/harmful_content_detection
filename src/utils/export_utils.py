import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Union

class ModelKwargsWrapper(nn.Module):
    """
    Wrapper to convert positional args from ONNX export to kwargs for the model.
    """
    def __init__(self, model: nn.Module, input_keys: List[str]):
        super().__init__()
        self.model = model
        self.input_keys = input_keys

    def forward(self, *args):
        # Map positional args back to kwargs
        kwargs = {k: v for k, v in zip(self.input_keys, args)}
        return self.model(**kwargs)

def export_to_onnx(model: nn.Module, 
                   output_file: str, 
                   input_sample: Union[torch.Tensor, Tuple, Dict], 
                   opset_version: int = 14,
                   input_names: List[str] = None,
                   output_names: List[str] = ['logits'],
                   dynamic_axes: Dict[str, Any] = None):
    """
    Export a PyTorch model to ONNX.
    """
    model.eval()
    
    # Handle dictionary inputs by wrapping the model to unpack positional args to kwargs
    if isinstance(input_sample, dict):
        if input_names is None:
            input_names = list(input_sample.keys())
        
        # Ensure iteration order is stable
        # Using the order from input_names which comes from keys() or passed explicitly
        # We must ensure values match input_names order
        args = tuple(input_sample[name] for name in input_names)
        
        # Wrap model
        model_to_export = ModelKwargsWrapper(model, input_names)
    else:
        args = input_sample
        model_to_export = model
        if input_names is None:
            input_names = ['input']

    if dynamic_axes is None:
        dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
        dynamic_axes[output_names[0]] = {0: 'batch_size'}

    torch.onnx.export(
        model_to_export,
        args,
        output_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"Model exported to {output_file} with input names: {input_names}")
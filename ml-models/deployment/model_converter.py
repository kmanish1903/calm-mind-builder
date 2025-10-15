import torch
import onnx
import numpy as np
from typing import Dict
import json

class ModelConverter:
    """Convert models to ONNX and TensorFlow.js formats"""
    
    def convert_to_onnx(self, model, input_shape: tuple, output_path: str):
        """Convert PyTorch model to ONNX"""
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return output_path
    
    def convert_to_tensorflowjs(self, onnx_path: str, output_dir: str):
        """Convert ONNX to TensorFlow.js (requires tensorflowjs_converter)"""
        import subprocess
        
        cmd = f"tensorflowjs_converter --input_format=onnx {onnx_path} {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        
        return output_dir
    
    def create_model_metadata(self, model_info: Dict) -> str:
        """Create model metadata JSON"""
        metadata = {
            'model_type': model_info.get('type', 'unknown'),
            'input_shape': model_info.get('input_shape', []),
            'output_classes': model_info.get('classes', 5),
            'preprocessing': model_info.get('preprocessing', {}),
            'performance': model_info.get('metrics', {})
        }
        
        return json.dumps(metadata, indent=2)
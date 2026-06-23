"""
Export to ONNX format.
"""
import sys
from pathlib import Path

ROOT = (Path(__file__).parent / "..").resolve()
sys.path.insert(0, str(ROOT / "src"))

from src.camera.lss import *

def export_to_onnx():
    """
    
    """
    up_model = Up()
    example_inputs = (torch.randn(1, 1, 32, 32),)
    onnx_program = torch.onnx.export(up_model, example_inputs, dynamo=True)
    onnx_program.save("image_classifier_model.onnx")
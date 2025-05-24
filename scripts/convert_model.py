import torch
from ultralytics import YOLO

def convert_to_onnx():
    # Load the PyTorch model
    model = YOLO('hyp_AdamW_mark003.pt')
    
    # Export to ONNX format
    model.export(format='onnx', imgsz=640)
    
    print("Model converted to ONNX format successfully!")

if __name__ == '__main__':
    convert_to_onnx() 
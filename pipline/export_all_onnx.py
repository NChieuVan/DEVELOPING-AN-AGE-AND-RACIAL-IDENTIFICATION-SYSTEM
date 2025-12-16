# Export YOLO (ultralytics) and MultiHeadClassifier to ONNX

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ultralytics import YOLO
from classification.multi_head import MultiHeadClassifier

def export_yolo_onnx(yolo_pt_path, onnx_path=None):
    model = YOLO(yolo_pt_path)
    result = model.export(format="onnx", dynamic=True, simplify=True)
    if onnx_path:
        # Move file to desired location
        os.rename(result, onnx_path)
    print(f"YOLO exported to: {onnx_path or result['file']}")

def export_classifier_onnx(classifier_pt_path, onnx_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name="mobilenet_v2").to(device)
    checkpoint = torch.load(classifier_pt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
    model,
    dummy,
    onnx_path,
    opset_version=17,        
    input_names=["input"],
    output_names=["age", "race"],  # đúng với multi-head của bạn
    dynamic_axes={"input": {0: "batch"}},
    do_constant_folding=True,
    dynamo=False,              # 🔴 QUAN TRỌNG
)
    print(f"Classifier exported to: {onnx_path}")

if __name__ == "__main__":
    # Export YOLO
    export_yolo_onnx("checkpoint/yolov11n-face.pt", "checkpoint/yolov11n-face.onnx")
    # Export classifier
    export_classifier_onnx("checkpoint/model_last.pth", "checkpoint/age_race_multihead.onnx")

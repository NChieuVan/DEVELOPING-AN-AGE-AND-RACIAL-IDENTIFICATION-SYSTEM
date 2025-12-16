import onnx
onnx_model = onnx.load("D:/Project_Git/DEVELOPING AN AGE AND RACIAL IDENTIFICATION SYSTEM/checkpoint/age_race_multihead.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
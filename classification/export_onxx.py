import torch
def export_onxx(model, dummy, onnx_file_path="age_race_multihead.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy,
        onnx_file_path,
        input_names=["input"],
        output_names=["age_out", "race_out"],
        opset_version=12
    )

import torch
from transformers import AutoModel

model_path = "./smolvlm2_model"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

vision_encoder = model.vision_model
vision_encoder.eval()

dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting ONNX model...")

torch.onnx.export(
    vision_encoder,
    dummy_input,
    "vision_encoder.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=18,
    dynamic_axes={
        "input": {0: "batch"},
        "features": {0: "batch"}
    }
)

print("✅ ONNX model exported successfully!")

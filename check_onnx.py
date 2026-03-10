import onnx

model = onnx.load("vision_encoder.onnx")
onnx.checker.check_model(model)

print("✅ ONNX model is valid.")

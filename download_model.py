from transformers import AutoModel, AutoProcessor

model_id = "HuggingFaceTB/SmolVLM2-256M-Instruct"

print("Downloading model...")

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="float32",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_id)

print("Saving locally...")

model.save_pretrained("./smolvlm2_model")
processor.save_pretrained("./smolvlm2_model")

print("✅ Model downloaded successfully!")

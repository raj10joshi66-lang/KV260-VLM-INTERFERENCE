import torch
from transformers import AutoModel

model_path = "./smolvlm2_model"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Extract vision encoder
vision_encoder = model.vision_model

# Save vision encoder
torch.save(vision_encoder.state_dict(), "vision_encoder.pth")

print("✅ Vision encoder extracted and saved.")

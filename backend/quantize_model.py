import torch
from torchvision import models

# Load the model
model = models.resnet50(pretrained=True)
model.eval()

# Apply dynamic quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(model.state_dict(), "model_quantized.pt")

print("Quantized model saved as model_quantized.pt")

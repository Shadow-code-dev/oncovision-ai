import torch
from src.models.model import get_efficientnet, get_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test EfficientNet
efficientnet = get_efficientnet().to(device)
x = torch.randn(1, 3, 224, 224).to(device)
outputs = efficientnet(x)

print("EfficientNet output shape:", outputs.shape)

# Test ResNet
resnet = get_resnet().to(device)
outputs2 = resnet(x)

print("ResNet output shape:", outputs2.shape)
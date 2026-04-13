import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.model import get_densenet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR.parent / "models" / "densenet_best.pth"

# Load model
model = get_densenet().to(device)
model.load_state_dict(torch.load(MODEL_DIR, map_location=device, weights_only=True))
model.eval()

# DenseNet target layer
target_layers = [model.features[-1]]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image_path = BASE_DIR.parent / "test_image.jpg"
image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# Grad-CAM
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=input_tensor)[0]

# Original image
rgb_img = np.array(image.resize((224, 224))) / 255.0

# Overlay heatmap
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save output
cv2.imwrite(
    "gradcam_densenet.jpg",
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
)

print("DenseNet Grad-CAM saved!")
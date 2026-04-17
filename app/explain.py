import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import io

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.model import get_efficientnet_v2
from src.utils.download import download_model
from src.utils.config import EFFICIENTNET_URL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

download_model(EFFICIENTNET_URL, MODEL_DIR / "best_efficientnet_v2.pth")

model = get_efficientnet_v2().to(device)
model.load_state_dict(
    torch.load(MODEL_DIR / "best_efficientnet_v2.pth", map_location=device, weights_only=True)
)
model.eval()

# Target Layer
target_layers = [model.features[-1]]

# Transform
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def generate_gradcam(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transforms(image).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Convert to bytes
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return buffer.tobytes()
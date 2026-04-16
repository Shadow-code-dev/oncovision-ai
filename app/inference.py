import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import io

from src.models.model import get_efficientnet_v2, get_densenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load models once
efficientnet = get_efficientnet_v2().to(device)
densenet = get_densenet().to(device)

efficientnet.load_state_dict(torch.load(MODEL_DIR / "best_efficientnet_v2.pth", map_location=device, weights_only=True))
densenet.load_state_dict(torch.load(MODEL_DIR / "densenet_best.pth", map_location=device, weights_only=True))

efficientnet.eval()
densenet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        out1 = efficientnet(input_tensor)
        out2 = densenet(input_tensor)

        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)

        avg_prob = 0.65 * prob1 + 0.35 * prob2

        pred = torch.argmax(avg_prob, dim=1).item()
        confidence = torch.max(avg_prob).item()

    return pred, confidence, avg_prob
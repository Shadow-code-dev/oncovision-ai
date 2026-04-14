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


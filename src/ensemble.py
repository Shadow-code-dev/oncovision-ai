import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.model import get_efficientnet_v2, get_resnet
from src.preprocessing.dataloader import get_dataloaders

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

EFFICIENTNET_PATH = MODEL_DIR / "best_efficientnet_v2.pth"
RESNET_PATH = MODEL_DIR / "best_resnet.pth"

def load_models():
    efficientnet = get_efficientnet_v2().to(device)
    resnet = get_resnet().to(device)

    efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device, weights_only=True))
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device, weights_only=True))

    efficientnet.eval()
    resnet.eval()

    return efficientnet, resnet

def evaluate_ensemble():
    _, _, test_loader = get_dataloaders()

    efficientnet, resnet = load_models()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            out1 = efficientnet(images)
            out2 = resnet(images)

            # Convert to probabilities
            prob1 = F.softmax(out1, dim=1)
            prob2 = F.softmax(out2, dim=1)

            # Ensemble
            avg_prob = 0.6 * prob1 + 0.4 * prob2

            preds = torch.argmax(avg_prob, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n Ensemble Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    evaluate_ensemble()
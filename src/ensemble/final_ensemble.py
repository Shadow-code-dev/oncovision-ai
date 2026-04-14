import torch
from pathlib import Path

from src.models.model import get_efficientnet_v2, get_densenet
from src.preprocessing.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR.parent / "models"

EFFICIENTNET_PATH = MODEL_DIR / "best_efficientnet_v2.pth"
DENSENET_PATH = MODEL_DIR / "densenet_best.pth"


def load_models():
    efficientnet = get_efficientnet_v2().to(device)
    efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device, weights_only=True))
    efficientnet.eval()

    densenet = get_densenet().to(device)
    densenet.load_state_dict(torch.load(DENSENET_PATH, map_location=device, weights_only=True))
    densenet.eval()

    return efficientnet, densenet


def evaluate():
    _, _, test_loader = get_dataloaders()

    efficientnet, densenet = load_models()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Predictions
            out1 = efficientnet(images)
            out2 = densenet(images)

            prob1 = torch.softmax(out1, dim=1)
            prob2 = torch.softmax(out2, dim=1)

            # Weighted Ensemble
            avg_prob = 0.65 * prob1 + 0.35 * prob2

            preds = torch.argmax(avg_prob, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n Final Ensemble Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
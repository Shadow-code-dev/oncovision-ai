import torch
from pathlib import Path

from src.models.model import get_efficientnet_v2
from src.preprocessing.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "best_efficientnet_v2.pth"


def evaluate():
    model = get_efficientnet_v2().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    _, _, test_loader = get_dataloaders()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n Best Model Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
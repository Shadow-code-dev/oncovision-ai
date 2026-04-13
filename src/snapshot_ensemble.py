import torch
from pathlib import Path

from src.models.model import get_efficientnet_v2
from src.preprocessing.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

SNAPSHOT_PATHS = [
    MODEL_DIR / "snapshot_epoch_8.pth",
    MODEL_DIR / "snapshot_epoch_12.pth",
    MODEL_DIR / "snapshot_epoch_14.pth",
    MODEL_DIR / "snapshot_epoch_15.pth"
]

def load_models():
    models = []

    for path in SNAPSHOT_PATHS:
        model = get_efficientnet_v2().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)

    return models

def evaluate():
    _, _, test_loader = get_dataloaders()

    models = load_models()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            probs = []

            for model in models:
                output = model(images)
                prob = torch.softmax(output, dim=1)
                probs.append(prob)

            # Average predictions
            avg_prob = torch.mean(torch.stack(probs), dim=0)

            preds = torch.argmax(avg_prob, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n Snapshot Ensemble Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate()

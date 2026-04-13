import torch
from pathlib import Path
from torchvision import transforms

from src.models.model import get_efficientnet_v2
from src.preprocessing.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

MODEL_PATH = MODEL_DIR / "snapshot_epoch_15.pth"

tta_transforms = [
    transforms.Lambda(lambda x: x)
]

def load_model():
    model = get_efficientnet_v2().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    return model

def evaluate():
    _, _, test_loader = get_dataloaders()

    model = load_model()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            tta_probs = []

            for tta in tta_transforms:
                # Apply TTA per image
                aug_images = torch.stack([tta(img.cpu()) for img in images]).to(device)

                output = model(aug_images)
                prob = torch.softmax(output, dim=1)

                tta_probs.append(prob)

            # Average TTA predictions
            avg_prob = torch.mean(torch.stack(tta_probs), dim=0)
            preds = torch.argmax(avg_prob, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n TTA Accuracy (Best Model): {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate()

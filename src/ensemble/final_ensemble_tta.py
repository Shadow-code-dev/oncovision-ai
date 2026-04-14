import torch
from pathlib import Path

from src.ensemble.final_ensemble import EFFICIENTNET_PATH
from src.models.model import get_efficientnet_v2, get_densenet
from src.preprocessing.dataloader import get_dataloaders
from src.stacking_ensemble import DENSENET_PATH

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
    _, _, test_dataloader = get_dataloaders()
    efficientnet, densenet = load_models()

    correct = 0
    total = 0

    # TTA transforms
    tta_transforms = [
        lambda x: x,
        lambda x: torch.clamp(x * 1.05, 0, 1),
    ]

    with torch.inference_mode():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            model_probs = []

            for model in [efficientnet, densenet]:
                tta_probs = []

                for tta in tta_transforms:
                    aug_images = tta(images)

                    output = model(aug_images)
                    prob = torch.softmax(output, dim=1)

                    tta_probs.append(prob)

                # Average TTA predictions
                avg_tta_prob = torch.mean(torch.stack(tta_probs), dim=0)

                model_probs.append(avg_tta_prob)

            # Weighted Ensemble
            weights = [0.65, 0.35]
            final_prob = sum(w * p for w, p in zip(weights, model_probs))

            preds = torch.argmax(final_prob, dim=1)

            correct = (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n Final TTA Ensemble Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
import torch
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)
import json

from src.preprocessing.dataloader import get_dataloaders
from src.models.model import get_efficientnet_v2, get_densenet
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_models():
    efficientnet = get_efficientnet_v2().to(device)
    densenet = get_densenet().to(device)

    efficientnet.load_state_dict(
        torch.load(MODELS_DIR / "best_efficientnet_v2.pth", map_location=device, weights_only=True)
    )

    densenet.load_state_dict(
        torch.load(MODELS_DIR / "densenet_best.pth", map_location=device, weights_only=True)
    )

    efficientnet.eval()
    densenet.eval()

    return efficientnet, densenet

def evaluate():
    _, _, test_loader = get_dataloaders()
    efficientnet, densenet = load_models()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            out1 = efficientnet(images)
            out2 = densenet(images)

            prob1 = torch.softmax(out1, dim=1)
            prob2 = torch.softmax(out2, dim=1)

            avg_prob = 0.65 * prob1 + 0.35 * prob2
            avg_prob[:, 1] *= 1.35
            preds = torch.argmax(avg_prob, dim=1)

            # Threshold Tuning
            malignant_mask = avg_prob[:, 1] > 0.35
            preds[malignant_mask] = 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("\n Evaluation Metrics: ")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n Confusion Matrix: ")
    print(confusion_matrix(all_labels, all_preds))

    print("\n Classification Report: ")
    print(classification_report(all_labels, all_preds))

    metrics_data = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    with open(BASE_DIR / "outputs" / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

if __name__ == "__main__":
    evaluate()
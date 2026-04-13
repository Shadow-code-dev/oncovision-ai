import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.models.model import get_efficientnet_v2, get_resnet
from src.preprocessing.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

EFFICIENTNET_PATH = MODEL_DIR / "best_efficientnet_v2.pth"
RESNET_PATH = MODEL_DIR / "best_resnet.pth"

# Meta Model
class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)

def load_models():
    efficientnet = get_efficientnet_v2().to(device)
    resnet = get_resnet().to(device)

    efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device, weights_only=True))
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device, weights_only=True))

    efficientnet.eval()
    resnet.eval()

    return efficientnet, resnet

# Create meta features from the VAL dataset (important!)
def create_meta_features(loader, efficientnet, resnet):
    features = []
    labels_list = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            out1 = efficientnet(images)
            out2 = resnet(images)

            prob1 = torch.softmax(out1, dim=1)
            prob2 = torch.softmax(out2, dim=1)

            combined = torch.cat([prob1, prob2], dim=1)

            features.append(combined.cpu())
            labels_list.append(labels)

    features = torch.cat(features)
    labels = torch.cat(labels_list)

    return features, labels

def train_meta_model():
    train_loader, val_loader, _ = get_dataloaders()

    efficientnet, resnet = load_models()

    # Important: Use VAL set for meta training
    X_meta, y_meta = create_meta_features(train_loader, efficientnet, resnet)

    meta_model = MetaModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        meta_model.parameters(),
        lr=0.001
    )

    epochs = 20

    X_meta = X_meta.to(device)
    y_meta = y_meta.to(device)

    for epoch in range(epochs):
        meta_model.train()

        outputs = meta_model(X_meta)
        loss = criterion(outputs, y_meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    return meta_model

# Evaluate on TEST dataset
def evaluate(meta_model):
    _, _, test_loader = get_dataloaders()

    efficientnet, resnet = load_models()

    correct, total = 0, 0

    meta_model.eval()

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            out1 = efficientnet(images)
            out2 = resnet(images)

            prob1 = torch.softmax(out1, dim=1)
            prob2 = torch.softmax(out2, dim=1)

            combined = torch.cat([prob1, prob2], dim=1)

            preds = meta_model(combined)
            preds = torch.argmax(preds, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n Stacking Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    meta_model = train_meta_model()
    evaluate(meta_model)
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.dataloader import get_dataloaders
from src.models.model import get_efficientnet, get_resnet, get_efficientnet_v2, get_densenet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        loop.set_postfix(
            loss=running_loss / (total + 1e-8),
            acc=correct / total
        )

    return running_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Validation", leave=False)

    with torch.inference_mode():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(
                loss=running_loss / (total + 1e-8),
                acc=correct / total
            )

    return running_loss / len(loader), correct / total

def train():
    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = get_densenet().to(device)

    # Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()

    lr = 1e-4
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    schedular = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    # Training
    epochs = 15
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("LR:", optimizer.param_groups[0]["lr"])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Save Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "densenet_best.pth")
            print("\nBest Model saved!")

        # Save snapshots
        # torch.save(model.state_dict(), MODEL_DIR / f"snapshot_epoch_{epoch + 1}.pth")

        schedular.step()

if __name__ == "__main__":
    train()
from torch.utils.data import DataLoader
from pathlib import Path

from src.preprocessing.dataset import BreastCancerDataset
from src.preprocessing.transforms import get_train_transforms, get_val_transforms

# Fixing base path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed"

def get_dataloaders(batch_size=32):
    train_dataset = BreastCancerDataset(
        root_dir=DATA_PATH / "train",
        transform=get_train_transforms()
    )

    val_dataset = BreastCancerDataset(
        root_dir=DATA_PATH / "val",
        transform=get_val_transforms()
    )

    test_dataset = BreastCancerDataset(
        root_dir=DATA_PATH / "test",
        transform=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

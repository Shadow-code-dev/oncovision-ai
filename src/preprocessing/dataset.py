from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Classes Mapping
CLASS_TO_IDX = {
    "benign": 0,
    "malignant": 1,
    "normal": 2
}

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        for cls in CLASS_TO_IDX:
            class_path = self.root_dir / cls

            for image_name in class_path.iterdir():
                self.image_paths.append(image_name)
                self.labels.append(CLASS_TO_IDX[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label
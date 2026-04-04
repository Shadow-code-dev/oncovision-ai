import os
import shutil
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Dataset_BUSI_with_GT"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

CLASSES = ["benign", "malignant", "normal"]
SPLIT_RATIO = (0.7, 0.15, 0.15) # train, val, test

def create_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (PROCESSED_DATA_PATH / split / cls).mkdir(parents=True, exist_ok=True)

def get_images(class_path):
    return [
        f for f in os.listdir(class_path)
        if f.endswith(".png") and "_mask" not in f
    ]

def split_data(files):
    random.shuffle(files)
    n = len(files)
    train_end = int(n * SPLIT_RATIO[0])
    val_end = int(n * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

    return (
        files[:train_end],
        files[train_end:val_end],
        files[val_end:]
    )

def copy_files(files, src_dir, dst_dir):
    for f in files:
        shutil.copy(src_dir / f, dst_dir / f)

def process():
    create_dirs()

    for cls in CLASSES:
        class_path = RAW_DATA_PATH / cls
        images = get_images(class_path)

        train, val, test = split_data(images)

        copy_files(train, class_path, PROCESSED_DATA_PATH / "train" / cls)
        copy_files(val, class_path, PROCESSED_DATA_PATH / "val" / cls)
        copy_files(test, class_path, PROCESSED_DATA_PATH / "test" / cls)

        print(f"{cls}: {len(images)} images processed")

if __name__ == "__main__":
    process()
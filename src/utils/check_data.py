import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
BASE_PATH = BASE_DIR / "data" / "processed"

def check_split(split):
    print(f"\nChecking {split} set:")
    split_path = BASE_PATH / split

    total = 0

    for cls in os.listdir(split_path):
        class_path = split_path / cls
        files = os.listdir(class_path)

        count = len(files)
        total += count

        print(f"{cls} : {count} files")

        mask_files = [f for f in files if "_mask" in f]
        if mask_files:
            print(f"WARNING: Found mask files in {cls}")

    print(f"Total images in {split} set: {total}")

def main():
    for split in ["train", "val", "test"]:
        check_split(split)

if __name__ == "__main__":
    main()
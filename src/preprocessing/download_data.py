import kagglehub
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw"
TARGET_FOLDER = RAW_DATA_PATH / "Dataset_BUSI_with_GT"

def dataset_exists():
    return TARGET_FOLDER.exists() and any(TARGET_FOLDER.iterdir())

def download_and_prepare():
    if dataset_exists():
        print("Dataset already downloaded. Skipping download.")
        return
    print("Downloading Dataset...")

    # Download dataset
    download_path = kagglehub.dataset_download("sabahesaraki/breast-ultrasound-images-dataset")

    print("Downloaded to:", download_path)

    source_folder = Path(download_path) / "Dataset_BUSI_with_GT"
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    print("Copying Dataset into:", RAW_DATA_PATH)

    # Copy everything
    shutil.copytree(
        source_folder,
        TARGET_FOLDER,
        dirs_exist_ok=True
    )
    print("Dataset copied to:", TARGET_FOLDER)

if __name__ == "__main__":
    download_and_prepare()
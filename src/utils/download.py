import urllib.request
from pathlib import Path

def download_model(url: str, path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)
        print(f"Successfully downloaded {path.name}.")
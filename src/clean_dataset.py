
from pathlib import Path

DATA_DIR = Path("data/raw/chest_xray/chest_xray")
removed = 0

for path in DATA_DIR.rglob(".DS_Store"):
    if path.is_file():
        path.unlink()
        print(f"Removed: {path}")
        removed += 1

print(f"\nTotal .DS_Store files removed: {removed}")

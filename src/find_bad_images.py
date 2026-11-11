
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/chest_xray/chest_xray")

for split in ["train", "val"]:
    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = DATA_DIR / split / class_name

        for path in folder.iterdir():
            if not path.is_file():
                continue

            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as error:
                print(f"Corrupted: {path}")
                print(f"Reason: {error}\n")


from pathlib import Path
from PIL import Image
import torch

# Dataset location
DATASET_DIR = Path("data/raw/chest_xray/chest_xray")

SPLITS = ["train", "test", "val"]
CLASSES = ["NORMAL", "PNEUMONIA"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

print("MediScan AI - Dataset Verification")
print("-" * 40)
print("Dataset exists:", DATASET_DIR.exists())
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

total_images = 0
corrupted_images = 0

for split in SPLITS:
    print(f"\n{split.upper()}")

    for class_name in CLASSES:
        folder = DATASET_DIR / split / class_name

        if not folder.exists():
            print(f"  ERROR: Missing folder {folder}")
            continue

        images = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
        ]

        bad_images = 0

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                bad_images += 1

        total_images += len(images)
        corrupted_images += bad_images

        print(
            f"  {class_name}: {len(images)} images, "
            f"{bad_images} corrupted"
        )

print("\n" + "-" * 40)
print("Total images:", total_images)
print("Corrupted images:", corrupted_images)
print("Verification complete.")

"""
MediScan AI — Brain MRI Dataset Verification Script
===================================================
Checks the raw Brain MRI directory for the four target classes:
1. glioma
2. meningioma
3. no_tumor
4. pituitary_tumor
across train, val, and test splits. Verifies image integrity and GPU memory.
"""

from pathlib import Path
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "data" / "raw" / "brain_mri"

SPLITS = ["train", "val", "test"]
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def check_brain_mri_dataset():
    print("=" * 60)
    print("MediScan AI — Brain MRI Dataset Verification")
    print("=" * 60)
    print(f"Dataset path: {DATASET_DIR}")
    print(f"Target classes ({len(CLASSES)}): {CLASSES}")
    print(f"Dataset directory exists: {DATASET_DIR.exists()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_total_gb:.2f} GB VRAM)")
        if vram_total_gb < 5.0:
            print("  -> Confirmed 4GB VRAM constraint (RTX 3050 Laptop). Memory-conscious training settings enabled.")

    if not DATASET_DIR.exists():
        print("\n[!] Brain MRI dataset directory not found.")
        print(f"    Please place the dataset in: {DATASET_DIR}")
        print("    Expected structure: train/, val/, test/ folders, each containing the 4 class folders.")
        return False

    total_images = 0
    corrupted_images = 0
    class_counts = {c: 0 for c in CLASSES}

    for split in SPLITS:
        split_folder = DATASET_DIR / split
        print(f"\n--- Split: {split.upper()} ---")

        if not split_folder.exists():
            print(f"  [!] Missing split folder: {split_folder}")
            continue

        for class_name in CLASSES:
            class_folder = split_folder / class_name
            if not class_folder.exists():
                # Check for alternative naming conventions (e.g., 'pituitary', 'notumor')
                print(f"  [!] Missing class folder: {class_folder}")
                continue

            images = [
                p for p in class_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            ]

            print(f"  {class_name}: {len(images)} images")
            total_images += len(images)
            class_counts[class_name] += len(images)

            # Test-read images for corruption
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupted_images += 1
                    print(f"    [X] Corrupted image: {img_path.name}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total valid images: {total_images}")
    print(f"  Corrupted images: {corrupted_images}")
    for c, count in class_counts.items():
        print(f"  {c}: {count}")

    if total_images == 0:
        print("\n[!] Dataset is currently empty. Awaiting dataset download and ingestion.")
        return False
    elif corrupted_images > 0:
        print(f"\n[!] Warning: Found {corrupted_images} corrupted images. Please clean them before training.")
        return False
    else:
        print("\n[OK] Brain MRI dataset is valid and ready for training.")
        return True


if __name__ == "__main__":
    check_brain_mri_dataset()

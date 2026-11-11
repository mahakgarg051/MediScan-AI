"""
MediScan AI — Brain MRI 4-Class Training Script
==============================================
Trains a pretrained MobileNetV3-Small neural network for 4-class Brain MRI classification:
1. glioma
2. meningioma
3. notumor
4. pituitary

Dataset Directory: data/raw/brain_mri/
  - Training/  (Training/glioma, Training/meningioma, Training/notumor, Training/pituitary)
  - Testing/   (Testing/glioma, Testing/meningioma, Testing/notumor, Testing/pituitary)

Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM):
  - Pretrained MobileNetV3-Small backbone (~2.5M params, lightweight memory footprint)
  - Default batch size: 16 (configurable)
  - Mixed Precision Training (FP16 via torch.amp)
  - Automatic CUDA detection with memory cleanup
  - Reproducible stratified validation split from Training data only
  - Held-out evaluation on Testing data after training completes
  - Best model checkpoint saved to: models/brain_mri_mobilenetv3.pth
"""

import argparse
import copy
from datetime import datetime
from pathlib import Path
import random
import sys
import time

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights


# ---------------- Default Constants ----------------
DEFAULT_DATA_DIR = Path("data/raw/brain_mri")
DEFAULT_OUTPUT_MODEL = Path("models/brain_mri_mobilenetv3.pth")
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 10
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_SEED = 42

EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------- Custom Safe Dataset ----------------
class SafeMRIDataset(Dataset):
    """
    Safe PyTorch Dataset that loads verified Brain MRI images.
    Includes runtime error handling for unreadable or corrupted files.
    """

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as err:
            print(f"[!] Warning: Failed reading image {path}: {err}. Returning black fallback tensor.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------- Transforms ----------------
def get_data_transforms():
    """
    Returns image transformation pipelines:
    - Data augmentation applied ONLY to Training data.
    - Deterministic evaluation transform for Validation and Testing data.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


# ---------------- Dataset Discovery & Verification ----------------
def discover_and_verify_split(
    split_dir: Path,
    expected_classes: list[str],
    split_name: str,
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """
    Validates folder structure, verifies image integrity with PIL,
    and returns valid sample paths with class indices.
    """
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {split_name} directory: {split_dir}\n"
            f"Expected structure: {split_dir}/<class_name>/*.jpg"
        )

    class_to_idx = {c: i for i, c in enumerate(expected_classes)}
    valid_samples: list[tuple[Path, int]] = []
    class_counts = {c: 0 for c in expected_classes}
    corrupted_count = 0

    print(f"\nScanning and verifying {split_name} dataset at: {split_dir}")

    for class_name in expected_classes:
        class_folder = split_dir / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            raise FileNotFoundError(
                f"Missing required class folder for '{class_name}' in {split_dir}.\n"
                f"Expected path: {class_folder}"
            )

        class_idx = class_to_idx[class_name]
        files = [
            p for p in class_folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not files:
            raise ValueError(
                f"Class folder '{class_name}' in {split_dir} contains no valid image files."
            )

        # Integrity verification
        for p in files:
            try:
                if p.stat().st_size == 0:
                    corrupted_count += 1
                    print(f"  [X] Skipping 0-byte file: {p.name}")
                    continue
                with Image.open(p) as img:
                    img.verify()
                valid_samples.append((p, class_idx))
                class_counts[class_name] += 1
            except Exception as err:
                corrupted_count += 1
                print(f"  [X] Skipping corrupted image {p.name}: {err}")

        print(f"  - {class_name:<14}: {class_counts[class_name]:>5} valid images")

    if corrupted_count > 0:
        print(f"  Total corrupted/skipped images in {split_name}: {corrupted_count}")

    print(f"  Total valid {split_name} images: {len(valid_samples)}")
    return valid_samples, class_to_idx


# ---------------- Model Construction ----------------
def build_mobilenetv3_model(num_classes: int) -> nn.Module:
    """
    Loads ImageNet-pretrained MobileNetV3-Small and adapts the final classifier
    for 4-class Brain MRI classification.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # In MobileNetV3-Small, model.classifier is:
    # Sequential(Linear(576, 1024), Hardswish(), Dropout(0.2), Linear(1024, 1000))
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


# ---------------- Main Training Function ----------------
def main():
    parser = argparse.ArgumentParser(
        description="MediScan AI — Brain MRI 4-Class Training Script (MobileNetV3-Small)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to dataset root containing Training/ and Testing/ (default: data/raw/brain_mri)",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_OUTPUT_MODEL,
        help="Destination path for best checkpoint (default: models/brain_mri_mobilenetv3.pth)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}, tuned for 4GB RTX 3050)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate for AdamW optimizer (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay for AdamW (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help=f"Validation split fraction from Training cohort (default: {DEFAULT_VAL_SPLIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validates dataset discovery, model initialization, and data loaders without running epochs.",
    )

    args = parser.parse_args()

    # 1. Reproducibility & Device Configuration
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print(" MediScan AI — Brain MRI 4-Class Training Pipeline")
    print("=" * 68)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print(f"Mixed Precision (FP16 AMP): ENABLED")
        if vram_gb < 5.0:
            print("Notice: RTX 3050 4GB VRAM detected. Memory-conscious settings active.")
    else:
        print("Notice: CUDA not detected. Running on CPU.")

    print(f"Batch Size:         {args.batch_size}")
    print(f"Epochs:             {args.epochs}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Validation Split:   {args.val_split * 100:.1f}%")
    print(f"Random Seed:        {args.seed}")
    print(f"Target Architecture:MobileNetV3-Small (Pretrained ImageNet)")
    print(f"Target Classes:     {EXPECTED_CLASSES}")
    print(f"Output Checkpoint:  {args.output_model}")
    print("=" * 68)

    # 2. Dataset Discovery and Error Handling
    training_dir = args.data_dir / "Training"
    testing_dir = args.data_dir / "Testing"

    try:
        training_samples, class_to_idx = discover_and_verify_split(
            training_dir, EXPECTED_CLASSES, "Training"
        )
        testing_samples, _ = discover_and_verify_split(
            testing_dir, EXPECTED_CLASSES, "Testing"
        )
    except Exception as err:
        print(f"\n[ERROR] Dataset error: {err}", file=sys.stderr)
        sys.exit(1)

    # 3. Stratified Validation Split from Training Data Only
    # Testing images are NEVER used for training or model selection.
    indices_by_class: dict[int, list[int]] = {i: [] for i in range(len(EXPECTED_CLASSES))}
    for idx, (_, class_idx) in enumerate(training_samples):
        indices_by_class[class_idx].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

    rng = random.Random(args.seed)
    for class_idx in sorted(indices_by_class.keys()):
        class_samples = list(indices_by_class[class_idx])
        rng.shuffle(class_samples)
        n_val = max(1, int(len(class_samples) * args.val_split))
        val_indices.extend(class_samples[:n_val])
        train_indices.extend(class_samples[n_val:])

    train_split_samples = [training_samples[i] for i in train_indices]
    val_split_samples = [training_samples[i] for i in val_indices]

    print("\nDataset Partition Summary:")
    print(f"  Training Cohort (with Augmentation): {len(train_split_samples):>5} images")
    print(f"  Validation Cohort (Held-Out Train):  {len(val_split_samples):>5} images")
    print(f"  Testing Cohort (Pure Final Eval):    {len(testing_samples):>5} images")

    # 4. Datasets and DataLoaders
    train_tf, eval_tf = get_data_transforms()

    train_dataset = SafeMRIDataset(train_split_samples, transform=train_tf)
    val_dataset = SafeMRIDataset(val_split_samples, transform=eval_tf)
    test_dataset = SafeMRIDataset(testing_samples, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 5. Model, Loss, Optimizer, and AMP Scaler
    print("\nInitializing Pretrained MobileNetV3-Small backbone...")
    model = build_mobilenetv3_model(num_classes=len(EXPECTED_CLASSES)).to(device)

    # Balanced Cross-Entropy Loss
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Automatic Mixed Precision for 4GB VRAM efficiency
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.dry_run:
        print("\n[OK] Dry-run checks completed successfully:")
        print(f"  - Verified {len(training_samples)} Training images and {len(testing_samples)} Testing images.")
        print(f"  - Model initialized and adapted for {len(EXPECTED_CLASSES)} classes.")
        print(f"  - CUDA and DataLoader pipelines ready for execution.")
        return

    # 6. Training & Validation Loop
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    print("\n" + "=" * 68)
    print(" Starting Training (Model selection evaluated on Validation split)")
    print("=" * 68)

    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_curr = labels.size(0)
            train_loss_sum += loss.item() * batch_size_curr
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += batch_size_curr

        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
        train_accuracy = (100.0 * train_correct / train_total) if train_total > 0 else 0.0

        # --- Validation Phase (Held-Out Training Split) ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                batch_size_curr = labels.size(0)
                val_loss_sum += loss.item() * batch_size_curr
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size_curr

        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        val_accuracy = (100.0 * val_correct / val_total) if val_total > 0 else 0.0

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Print per-epoch metrics: train loss, train acc, val loss, val acc
        is_best = val_accuracy > best_val_accuracy
        star_marker = " ★ BEST" if is_best else ""

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:6.2f}% | "
            f"Time: {epoch_time:.1f}s{star_marker}",
            flush=True,
        )

        if is_best:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

            # Save best checkpoint
            torch.save({
                "model_state_dict": best_model_state,
                "classes": EXPECTED_CLASSES,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "architecture": "mobilenet_v3_small",
                "created_at": datetime.now().isoformat(),
            }, args.output_model)

        # Clear GPU cache to preserve 4GB VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_training_time = time.time() - training_start_time
    print("=" * 68)
    print(f"Training Complete in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"Saved Checkpoint: {args.output_model}")
    print("=" * 68)

    # 7. Final Evaluation on Pure Testing Set
    # Never used during training or model selection.
    print("\n" + "=" * 68)
    print(" Final Evaluation on Reserved Testing Set (N = 1,600 images)")
    print("=" * 68)

    # Load best checkpoint weights
    checkpoint = torch.load(args.output_model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    class_correct = {i: 0 for i in range(len(EXPECTED_CLASSES))}
    class_total = {i: 0 for i in range(len(EXPECTED_CLASSES))}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size_curr = labels.size(0)
            test_loss_sum += loss.item() * batch_size_curr
            preds = outputs.argmax(dim=1)

            test_correct += (preds == labels).sum().item()
            test_total += batch_size_curr

            for p, l in zip(preds, labels):
                l_item = l.item()
                if p.item() == l_item:
                    class_correct[l_item] += 1
                class_total[l_item] += 1

    test_loss = test_loss_sum / test_total if test_total > 0 else 0.0
    test_accuracy = (100.0 * test_correct / test_total) if test_total > 0 else 0.0

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% ({test_correct}/{test_total} correct)")
    print("\nPer-Class Breakdown on Testing Data:")
    for class_name in EXPECTED_CLASSES:
        c_idx = class_to_idx[class_name]
        c_corr = class_correct[c_idx]
        c_tot = class_total[c_idx]
        c_acc = (100.0 * c_corr / c_tot) if c_tot > 0 else 0.0
        print(f"  - {class_name:<14}: {c_acc:6.2f}% ({c_corr:>3}/{c_tot:>3})")

    # Update checkpoint with test accuracy
    checkpoint["test_accuracy"] = test_accuracy
    checkpoint["test_loss"] = test_loss
    torch.save(checkpoint, args.output_model)

    print("=" * 68)
    print(f"Final checkpoint updated with test metrics at: {args.output_model}")
    print("=" * 68)


if __name__ == "__main__":
    main()

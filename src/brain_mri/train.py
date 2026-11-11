"""
MediScan AI — Brain MRI 4-Class Training Script
==============================================
Trains a neural network for 4-class Brain MRI classification:
1. glioma
2. meningioma
3. no_tumor (no tumour)
4. pituitary_tumor (pituitary tumour)

Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM):
- Mixed Precision (FP16 via torch.cuda.amp)
- Batch Size: 16
- Memory-efficient backbone: MobileNetV3-Large or ResNet-18
- Output checkpoint saved to: models/brain_mri/mediscan_brain_mri_best.pth
"""

from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------- Configuration & Paths ----------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "raw" / "brain_mri"
MODEL_DIR = ROOT / "models" / "brain_mri"
OUTPUT_DIR = ROOT / "outputs" / "brain_mri"

IMAGE_SIZE = 224
BATCH_SIZE = 16  # Tuned for RTX 3050 4GB VRAM
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Target 4 classes
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]


def get_transforms():
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

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def build_model(num_classes=4):
    """Builds a memory-efficient backbone suitable for 4GB VRAM."""
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_brain_mri(dry_run=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Safety checks
    if not (DATA_DIR / "train").exists():
        print(f"\n[!] Training dataset not found at: {DATA_DIR / 'train'}")
        print("    Please acquire and structure the dataset in data/raw/brain_mri/ before training.")
        return False

    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val" if (DATA_DIR / "val").exists() else DATA_DIR / "test",
        transform=val_tf,
    )

    print(f"Train samples: {len(train_dataset)} | Classes: {train_dataset.classes}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = build_model(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_checkpoint_path = MODEL_DIR / "mediscan_brain_mri_best.pth"

    if dry_run:
        print("[*] Dry run validation successful. Ready for training.")
        return True

    print("\n--- Starting Memory-Conscious Training (4GB VRAM) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0
        train_loss = running_loss / total if total > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / val_total if val_total > 0 else 0
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{EPOCHS} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "architecture": "mobilenet_v3_large",
                    "image_size": IMAGE_SIZE,
                    "best_val_accuracy": best_val_acc,
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )
            print(f"  -> Saved new best checkpoint: {best_checkpoint_path} (Acc: {best_val_acc:.4f})")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")
    return True


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    train_brain_mri(dry_run=dry)

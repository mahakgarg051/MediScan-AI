
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

# ---------------- CONFIG ----------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
VAL_FRACTION = 0.15

DATA_DIR = Path("data/raw/chest_xray/chest_xray")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
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

# Load the same training images twice, with different transforms.
train_base = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_base = datasets.ImageFolder(
    DATA_DIR / "train", transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test", transform=eval_transform
)

print("Classes:", train_base.classes, flush=True)
assert train_base.classes == ["NORMAL", "PNEUMONIA"], (
    f"Unexpected class order: {train_base.classes}"
)

# Stratified split: retain class balance in validation.
indices_by_class = {i: [] for i in range(len(train_base.classes))}
for idx, label in enumerate(train_base.targets):
    indices_by_class[label].append(idx)

train_indices, val_indices = [], []
for label, indices in indices_by_class.items():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_FRACTION))
    val_indices.extend(indices[:n_val])
    train_indices.extend(indices[n_val:])

train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)

print("Training images:", len(train_dataset), flush=True)
print("Validation images:", len(val_dataset), flush=True)
print("Test images (reserved):", len(test_dataset), flush=True)

# ---------------- MODEL ----------------
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Initially freeze the pretrained feature extractor.
for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 2)
model = model.to(device)

# Handle class imbalance using training-set class counts.
train_counts = torch.bincount(
    torch.tensor([train_base.targets[i] for i in train_indices]),
    minlength=2,
).float()

class_weights = train_counts.sum() / (2.0 * train_counts)
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

optimizer = torch.optim.AdamW(
    model.classifier.parameters(), lr=1e-3
)

scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

best_val_accuracy = 0.0
best_path = MODEL_DIR / "mediscan_mobilenetv3_best.pth"

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total
    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss_sum / train_total:.4f} | "
        f"Train accuracy: {train_accuracy:.2f}% | "
        f"Val loss: {val_loss_sum / val_total:.4f} | "
        f"Val accuracy: {val_accuracy:.2f}%",
        flush=True,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_base.classes,
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": best_val_accuracy,
        }, best_path)

print("\nTraining complete.", flush=True)
print("Best validation accuracy:", f"{best_val_accuracy:.2f}%", flush=True)
print("Best model saved to:", best_path, flush=True)
print("Original test set has NOT been evaluated yet.", flush=True)

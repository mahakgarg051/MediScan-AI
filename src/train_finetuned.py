
from pathlib import Path
import random
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
VAL_FRACTION = 0.15

DATA_DIR = Path("data/raw/chest_xray/chest_xray")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

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

train_base = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_base = datasets.ImageFolder(
    DATA_DIR / "train", transform=eval_transform
)

classes = train_base.classes
assert classes == ["NORMAL", "PNEUMONIA"], classes

# Reproduce the same stratified split used in the first training.
indices_by_class = {i: [] for i in range(len(classes))}
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

print("Training images:", len(train_dataset), flush=True)
print("Validation images:", len(val_dataset), flush=True)

# Start from ImageNet pretrained weights.
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Freeze all features first, then unfreeze the final feature block.
for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-2:].parameters():
    param.requires_grad = True

in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(classes))
model = model.to(device)

train_counts = torch.bincount(
    torch.tensor([train_base.targets[i] for i in train_indices]),
    minlength=len(classes),
).float()

class_weights = train_counts.sum() / (len(classes) * train_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

feature_params = [
    p for p in model.features.parameters() if p.requires_grad
]
classifier_params = list(model.classifier.parameters())

optimizer = torch.optim.AdamW([
    {"params": feature_params, "lr": 1e-5},
    {"params": classifier_params, "lr": 1e-4},
], weight_decay=1e-4)

best_val_accuracy = 0.0
best_state = None
best_path = MODEL_DIR / "mediscan_mobilenetv3_finetuned.pth"

for epoch in range(EPOCHS):
    model.train()

    # Keep frozen feature blocks in evaluation mode.
    model.features[:-2].eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    normal_correct = 0
    normal_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            normal_mask = labels == classes.index("NORMAL")
            normal_total += normal_mask.sum().item()
            normal_correct += (
                (preds[normal_mask] == labels[normal_mask]).sum().item()
            )

    train_acc = 100 * correct / total
    val_acc = 100 * val_correct / val_total
    normal_recall = (
        100 * normal_correct / normal_total if normal_total else 0
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {total_loss / total:.4f} | "
        f"Train acc: {train_acc:.2f}% | "
        f"Val loss: {val_loss_sum / val_total:.4f} | "
        f"Val acc: {val_acc:.2f}% | "
        f"NORMAL recall: {normal_recall:.2f}%",
        flush=True,
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": best_val_accuracy,
        }, best_path)

print("\nFine-tuning complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print("New checkpoint:", best_path)
print("Original checkpoint was not overwritten.")

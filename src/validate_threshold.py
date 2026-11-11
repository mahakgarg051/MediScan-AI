
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = Path("data/raw/chest_xray/chest_xray/train")
MODEL_PATH = Path("models/mediscan_mobilenetv3_best.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 16
VAL_FRACTION = 0.15
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
classes = checkpoint["classes"]

if dataset.classes != classes:
    raise ValueError("Dataset classes do not match checkpoint classes.")

# Recreate the same deterministic split used in train.py.
generator = torch.Generator().manual_seed(SEED)
indices_by_class = {i: [] for i in range(len(classes))}

for idx, label in enumerate(dataset.targets):
    indices_by_class[label].append(idx)

train_indices = []
val_indices = []

for label, indices in indices_by_class.items():
    indices = indices.copy()
    # Match train.py's Python random split exactly.
    import random
    random.seed(SEED)
    break

# Rebuild the exact split used by train.py.
random.seed(SEED)
indices_by_class = {i: [] for i in range(len(classes))}
for idx, label in enumerate(dataset.targets):
    indices_by_class[label].append(idx)

for label, indices in indices_by_class.items():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_FRACTION))
    val_indices.extend(indices[:n_val])
    train_indices.extend(indices[n_val:])

val_dataset = Subset(dataset, val_indices)
loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

normal_idx = classes.index("NORMAL")
pneumonia_idx = classes.index("PNEUMONIA")

labels_all = []
probs_all = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1).cpu()
        labels_all.extend(labels.tolist())
        probs_all.extend(probs[:, pneumonia_idx].tolist())

print("Validation images:", len(labels_all))
print("Threshold results (validation set only):")

for threshold in [0.3, 0.5, 0.7, 0.9]:
    tn = sum(
        1 for y, p in zip(labels_all, probs_all)
        if y == normal_idx and p < threshold
    )
    fp = sum(
        1 for y, p in zip(labels_all, probs_all)
        if y == normal_idx and p >= threshold
    )
    tp = sum(
        1 for y, p in zip(labels_all, probs_all)
        if y == pneumonia_idx and p >= threshold
    )
    fn = sum(
        1 for y, p in zip(labels_all, probs_all)
        if y == pneumonia_idx and p < threshold
    )

    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    accuracy = (tp + tn) / len(labels_all)

    print(
        f"Threshold {threshold:.1f} | "
        f"Accuracy {accuracy*100:.2f}% | "
        f"Sensitivity {sensitivity*100:.2f}% | "
        f"Specificity {specificity*100:.2f}%"
    )

print("Analysis complete. Model weights unchanged.")

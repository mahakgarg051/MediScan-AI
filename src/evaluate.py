
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# Paths
DATA_DIR = Path("data/raw/chest_xray/chest_xray")
MODEL_PATH = Path("models/mediscan_mobilenetv3_best.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# Check files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not (DATA_DIR / "test").exists():
    raise FileNotFoundError(f"Test dataset not found: {DATA_DIR / 'test'}")

# Same normalization used during training
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform,
)

# Load saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

classes = checkpoint["classes"]

if test_dataset.classes != classes:
    raise ValueError(
        f"Class mismatch! Model: {classes}, test data: {test_dataset.classes}"
    )

# Recreate the same model architecture
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

all_labels = []
all_predictions = []

print("Test images:", len(test_dataset), flush=True)

# Evaluate without updating model weights
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        
        pneumonia_idx = classes.index("PNEUMONIA")
        predictions = (torch.softmax(outputs, dim=1)[:, pneumonia_idx] >= 0.5).long().cpu()
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

# Metrics
accuracy = accuracy_score(all_labels, all_predictions)

print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=classes,
    digits=4,
    zero_division=0,
))

print("Best validation accuracy:", checkpoint["best_val_accuracy"])
print("\nEvaluation complete.")

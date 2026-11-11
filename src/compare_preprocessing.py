
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_DIR = Path("data/raw/chest_xray/chest_xray/test")
MODEL_PATH = Path("models/mediscan_mobilenetv3_best.pth")
BATCH_SIZE = 16
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
classes = checkpoint["classes"]
pneumonia_idx = classes.index("PNEUMONIA")

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# A: Original preprocessing used during training
transform_a = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# B: Preserve aspect ratio, then pad to a square
transform_b = transforms.Compose([
    transforms.Resize(
        IMAGE_SIZE,
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.Pad(
        (0, 0, IMAGE_SIZE, IMAGE_SIZE),
        fill=0
    ),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])

def evaluate(transform, name):
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    if dataset.classes != classes:
        raise ValueError(
            f"Class mismatch: {dataset.classes} vs {classes}"
        )

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    cm = confusion_matrix(
        all_labels, all_preds,
        labels=list(range(len(classes)))
    )

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print("Class order:", classes)
    print("Confusion matrix:\n", cm)

    return cm

cm_a = evaluate(transform_a, "A: Original resize")
cm_b = evaluate(transform_b, "B: Aspect ratio preserved")

print("\nComparison complete.")
print("No model weights changed.")

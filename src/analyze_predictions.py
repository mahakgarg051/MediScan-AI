
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = Path("data/raw/chest_xray/chest_xray/test")
MODEL_PATH = Path("models/mediscan_mobilenetv3_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

normal_idx = classes.index("NORMAL")
pneumonia_idx = classes.index("PNEUMONIA")

normal_probs = []
normal_actual = []
pneumonia_probs = []
pneumonia_actual = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)
        probabilities = torch.softmax(model(images), dim=1).cpu()

        for i, label in enumerate(labels.tolist()):
            p = probabilities[i, pneumonia_idx].item()

            if label == normal_idx:
                normal_probs.append(p)
                normal_actual.append(label)
            else:
                pneumonia_probs.append(p)
                pneumonia_actual.append(label)

print("Classes:", classes)
print("\nPNEUMONIA probability on NORMAL images:")
print("Mean:", round(sum(normal_probs) / len(normal_probs), 4))
print("Min:", round(min(normal_probs), 4))
print("Max:", round(max(normal_probs), 4))

print("\nPNEUMONIA probability on PNEUMONIA images:")
print("Mean:", round(sum(pneumonia_probs) / len(pneumonia_probs), 4))
print("Min:", round(min(pneumonia_probs), 4))
print("Max:", round(max(pneumonia_probs), 4))

print("\nThreshold analysis (diagnostic experiment only):")
for threshold in [0.3, 0.5, 0.7, 0.9]:
    tn = sum(p < threshold for p in normal_probs)
    fp = len(normal_probs) - tn
    tp = sum(p >= threshold for p in pneumonia_probs)
    fn = len(pneumonia_probs) - tp

    specificity = tn / len(normal_probs)
    sensitivity = tp / len(pneumonia_probs)
    accuracy = (tn + tp) / (len(normal_probs) + len(pneumonia_probs))

    print(
        f"Threshold {threshold:.1f}: "
        f"Accuracy={accuracy*100:.2f}%, "
        f"Sensitivity={sensitivity*100:.2f}%, "
        f"Specificity={specificity*100:.2f}%"
    )

print("\nAnalysis complete. No model weights changed.")


from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "raw" / "chest_xray" / "chest_xray" / "test"
MODEL_PATH = ROOT / "models" / "mediscan_mobilenetv3_finetuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=False
)
classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

pneumonia_idx = classes.index("PNEUMONIA")
all_probs = []
all_labels = []

with torch.inference_mode():
    for images, labels in loader:
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)[:, pneumonia_idx]
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())

# Actual labels: 1 means PNEUMONIA
actual = [
    1 if classes[label] == "PNEUMONIA" else 0
    for label in all_labels
]

print(f"Test images: {len(actual)}")
print("Rows = actual [NORMAL, PNEUMONIA]")
print("Columns = predicted [NORMAL, PNEUMONIA]")

for threshold in [0.50, 0.70]:
    predicted = [1 if p >= threshold else 0 for p in all_probs]
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(actual, predicted) * 100
    sensitivity = tp / (tp + fn) * 100 if tp + fn else 0
    specificity = tn / (tn + fp) * 100 if tn + fp else 0

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sensitivity (PNEUMONIA): {sensitivity:.2f}%")
    print(f"Specificity (NORMAL): {specificity:.2f}%")
    print("Confusion matrix:")
    print(matrix)

print("\nComparison complete. Model weights unchanged.")
print("Research/demo use only; not for medical diagnosis.")

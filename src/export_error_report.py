
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data" / "raw" / "chest_xray" / "chest_xray" / "test"
MODEL_PATH = ROOT / "models" / "mediscan_mobilenetv3_finetuned.pth"
OUTPUT_CSV = ROOT / "outputs" / "error_report.csv"

IMAGE_SIZE = 224
THRESHOLD = 0.70
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False
)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=False
)
classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

pneumonia_idx = classes.index("PNEUMONIA")
rows = []
image_index = 0

with torch.inference_mode():
    for images, labels in loader:
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1).cpu()

        for i, label in enumerate(labels):
            image_path, actual_index = dataset.samples[image_index]
            image_index += 1

            pneumonia_score = float(probabilities[i, pneumonia_idx])
            actual = classes[actual_index]

            predicted = (
                "PNEUMONIA"
                if pneumonia_score >= THRESHOLD
                else "NORMAL"
            )

            if actual != predicted:
                error_type = (
                    "False Positive"
                    if actual == "NORMAL"
                    else "False Negative"
                )

                rows.append({
                    "filename": Path(image_path).name,
                    "actual": actual,
                    "predicted": predicted,
                    "error_type": error_type,
                    "pneumonia_score_percent": round(
                        pneumonia_score * 100, 2
                    ),
                })

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    columns = [
        "filename",
        "actual",
        "predicted",
        "error_type",
        "pneumonia_score_percent",
    ]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("Test images:", len(dataset))
print("Misclassified images:", len(rows))
print("False positives:", sum(r["error_type"] == "False Positive" for r in rows))
print("False negatives:", sum(r["error_type"] == "False Negative" for r in rows))
print("CSV saved:", OUTPUT_CSV)

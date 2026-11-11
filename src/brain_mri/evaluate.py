"""
MediScan AI — Brain MRI 4-Class Evaluation Script
================================================
Evaluates a trained 4-class Brain MRI model checkpoint on the test set:
- glioma
- meningioma
- no_tumor (no tumour)
- pituitary_tumor (pituitary tumour)

Outputs:
- outputs/brain_mri/evaluation_metrics.json
- Classification report (precision, recall, f1-score, support)
- Confusion matrix
"""

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "raw" / "brain_mri" / "test"
MODEL_PATH = ROOT / "models" / "brain_mri" / "mediscan_brain_mri_best.pth"
OUTPUT_DIR = ROOT / "outputs" / "brain_mri"

IMAGE_SIZE = 224
BATCH_SIZE = 16


def evaluate_brain_mri():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    if not MODEL_PATH.exists():
        print(f"[!] Brain MRI model checkpoint not found: {MODEL_PATH}")
        print("    Please train the model first using: python src/brain_mri/train.py")
        return None

    if not DATA_DIR.exists():
        print(f"[!] Brain MRI test dataset not found: {DATA_DIR}")
        print("    Please ensure data/raw/brain_mri/test/ exists.")
        return None

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    classes = checkpoint["classes"]

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"

    metrics_data = {
        "accuracy": acc,
        "classes": classes,
        "confusion_matrix": cm,
        "classification_report": report,
        "test_images": len(y_true),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print("=" * 60)
    print("Brain MRI Evaluation Summary:")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 60)
    return metrics_data


if __name__ == "__main__":
    evaluate_brain_mri()

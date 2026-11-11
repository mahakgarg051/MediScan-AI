"""
MediScan AI — Brain MRI 4-Class Evaluation Script
================================================
Evaluates a trained MobileNetV3-Small Brain MRI model checkpoint on the untouched
test dataset in: data/raw/brain_mri/Testing

Features:
- Loads checkpoint: models/brain_mri_mobilenetv3.pth
- Preserves the exact 4-class order stored in the checkpoint:
    ['glioma', 'meningioma', 'notumor', 'pituitary']
- Uses identical 224x224 resolution and ImageNet normalization as training
- Uses model.eval() and torch.inference_mode()
- Automatically detects and uses CUDA (NVIDIA GPU) when available
- Computes overall test accuracy, per-class precision / recall / F1, and a 4x4 confusion matrix
- Displays total test images evaluated
- Robust error handling for checkpoint and dataset issues
- Strictly evaluation-only: does not modify checkpoints, train weights, or alter chest X-ray code
"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------- Configuration Defaults ----------------
DEFAULT_MODEL_PATH = Path("models/brain_mri_mobilenetv3.pth")
DEFAULT_TEST_DIR = Path("data/raw/brain_mri/Testing")
DEFAULT_METRICS_OUTPUT = Path("outputs/brain_mri/evaluation_metrics.json")

IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------- Custom Safe Evaluation Dataset ----------------
class SafeEvaluationDataset(Dataset):
    """
    Evaluation dataset wrapper that verifies images and applies
    the identical deterministic evaluation transform used during training.
    """

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as err:
            print(f"[!] Warning: Corrupted image at {path}: {err}. Returning black fallback.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------- Transform ----------------
def get_eval_transform():
    """
    Exact deterministic evaluation transform used during training:
    Resize(224, 224) -> ToTensor() -> Normalize(ImageNet mean & std).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------- Model Construction ----------------
def build_mobilenetv3_model(num_classes: int) -> nn.Module:
    """
    Recreates the exact MobileNetV3-Small architecture used during training.
    """
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ---------------- Dataset Discovery ----------------
def discover_test_samples(
    test_dir: Path,
    expected_classes: list[str],
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """
    Validates testing directory structure and collects all valid image samples.
    """
    if not test_dir.exists() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Testing dataset directory not found at: '{test_dir}'\n"
            f"Expected structure: {test_dir}/<class_name>/*.jpg"
        )

    class_to_idx = {c: i for i, c in enumerate(expected_classes)}
    samples: list[tuple[Path, int]] = []
    class_counts = {c: 0 for c in expected_classes}

    for class_name in expected_classes:
        class_folder = test_dir / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            raise FileNotFoundError(
                f"Missing required class folder for '{class_name}' in testing directory: '{test_dir}'\n"
                f"Expected path: {class_folder}"
            )

        files = [
            p for p in class_folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not files:
            raise ValueError(
                f"Class folder '{class_name}' in '{test_dir}' contains no valid image files."
            )

        class_idx = class_to_idx[class_name]
        for p in files:
            if p.stat().st_size > 0:
                samples.append((p, class_idx))
                class_counts[class_name] += 1

    return samples, class_counts


# ---------------- Pretty Printing Helpers ----------------
def print_confusion_matrix(cm, classes: list[str]):
    """
    Prints an aligned, human-readable 4x4 confusion matrix with labels.
    """
    col_width = max(len(c) for c in classes) + 4
    header = "True \\ Pred".ljust(15) + "".join(c.rjust(col_width) for c in classes) + "Total".rjust(col_width)
    print(header)
    print("-" * len(header))

    for idx, class_name in enumerate(classes):
        row_values = cm[idx]
        row_total = sum(row_values)
        row_str = class_name.ljust(15) + "".join(str(val).rjust(col_width) for val in row_values) + str(row_total).rjust(col_width)
        print(row_str)

    print("-" * len(header))
    pred_totals = [sum(cm[r][c] for r in range(len(classes))) for c in range(len(classes))]
    totals_row = "Total".ljust(15) + "".join(str(val).rjust(col_width) for val in pred_totals) + str(sum(pred_totals)).rjust(col_width)
    print(totals_row)


# ---------------- Main Evaluation Routine ----------------
def evaluate(checkpoint_path: Path, test_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE, save_metrics_path: Path = None):
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" MediScan AI — Brain MRI 4-Class Test Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU:    {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"Testing Directory: {test_dir}")
    print("=" * 70)

    # 2. Checkpoint Loading & Error Handling
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: '{checkpoint_path}'\n"
            f"Please verify that the model has been trained and saved to this path."
        )

    try:
        # Load checkpoint safely
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as err:
        try:
            # Fallback if non-tensor metadata requires weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as inner_err:
            raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {inner_err}") from err

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Invalid checkpoint format in '{checkpoint_path}': missing 'model_state_dict' key."
        )

    # Extract exact class ordering stored in checkpoint
    classes = checkpoint.get("classes", ["glioma", "meningioma", "notumor", "pituitary"])
    num_classes = len(classes)

    print(f"\nCheckpoint Metadata:")
    print(f"  Classes ({num_classes}):          {classes}")
    if "best_val_accuracy" in checkpoint:
        print(f"  Best Val Accuracy:     {checkpoint['best_val_accuracy']:.2f}%")
    if "best_epoch" in checkpoint:
        print(f"  Best Epoch:            {checkpoint['best_epoch']}")
    if "created_at" in checkpoint:
        print(f"  Checkpoint Timestamp:  {checkpoint['created_at']}")

    # 3. Discover and Verify Test Dataset
    test_samples, class_counts = discover_test_samples(test_dir, classes)
    total_test_images = len(test_samples)

    print(f"\nTest Dataset Overview (Untouched Test Cohort):")
    for c in classes:
        print(f"  - {c:<14}: {class_counts[c]:>5} images")
    print(f"  Total test images evaluated: {total_test_images}")

    # 4. Initialize Model and Load Weights
    model = build_mobilenetv3_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 5. Build DataLoader
    eval_transform = get_eval_transform()
    test_dataset = SafeEvaluationDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single-process loader for deterministic evaluation
        pin_memory=torch.cuda.is_available(),
    )

    # 6. Inference Loop (Strict inference mode, no gradients)
    all_targets = []
    all_predictions = []
    all_probabilities = []

    print("\nRunning inference on test dataset...")
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    # 7. Metrics Computation
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(num_classes)),
    )

    # 8. Print Results
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Images Evaluated: {total_test_images}")
    print(f"Overall Test Accuracy:       {accuracy * 100:.2f}%\n")

    print("Per-Class Metrics (Precision, Recall, F1-Score):")
    header = f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    print(header)
    print("-" * len(header))
    for i, class_name in enumerate(classes):
        print(
            f"{class_name:<16} "
            f"{precision[i] * 100:>9.2f}% "
            f"{recall[i] * 100:>9.2f}% "
            f"{f1[i] * 100:>9.2f}% "
            f"{support[i]:>10d}"
        )
    print("-" * len(header))

    # Macro & Weighted Averages
    macro_p = precision.mean() * 100
    macro_r = recall.mean() * 100
    macro_f1 = f1.mean() * 100
    print(
        f"{'Macro Average':<16} "
        f"{macro_p:>9.2f}% "
        f"{macro_r:>9.2f}% "
        f"{macro_f1:>9.2f}% "
        f"{total_test_images:>10d}"
    )

    print("\n" + "=" * 70)
    print(" 4×4 CONFUSION MATRIX")
    print("=" * 70)
    print_confusion_matrix(cm, classes)

    print("\nDetailed Scikit-Learn Classification Report:")
    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=list(range(num_classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    # 9. Optional: Save Metrics to outputs/brain_mri/evaluation_metrics.json
    if save_metrics_path:
        save_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_dict = {
            "model_path": str(checkpoint_path),
            "test_images": total_test_images,
            "overall_accuracy": float(accuracy),
            "classes": classes,
            "per_class": {
                classes[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(num_classes)
            },
            "macro_avg": {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1_score": float(f1.mean()),
            },
            "confusion_matrix": cm.tolist(),
        }
        with open(save_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics successfully exported to: {save_metrics_path}")

    print("=" * 70)
    print("Evaluation completed successfully.")
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classes": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MediScan AI — Evaluate MobileNetV3-Small on Brain MRI Test Set"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model checkpoint (default: models/brain_mri_mobilenetv3.pth)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Path to Testing directory (default: data/raw/brain_mri/Testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Destination path to export JSON metrics (default: outputs/brain_mri/evaluation_metrics.json)",
    )

    args = parser.parse_args()

    try:
        evaluate(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            save_metrics_path=args.save_metrics,
        )
    except Exception as err:
        print(f"\n[ERROR] Evaluation failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

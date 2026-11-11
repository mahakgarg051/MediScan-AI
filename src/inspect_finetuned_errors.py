
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "raw" / "chest_xray" / "chest_xray" / "test"
MODEL_PATH = ROOT / "models" / "mediscan_mobilenetv3_finetuned.pth"
OUTPUT_DIR = ROOT / "outputs" / "misclassified_finetuned"

THRESHOLD = 0.70
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

pneumonia_idx = classes.index("PNEUMONIA")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
saved = 0

for index, (image_path, actual_index) in enumerate(dataset.samples):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]

    pneumonia_score = probs[pneumonia_idx].item()
    actual = classes[actual_index]
    predicted = "PNEUMONIA" if pneumonia_score >= THRESHOLD else "NORMAL"

    if actual != predicted:
        label = (
            f"Actual: {actual} | Pred: {predicted} | "
            f"Pneumonia score: {pneumonia_score:.1%}"
        )
        preview = image.copy()
        preview.thumbnail((700, 700))

        canvas = Image.new("RGB", (720, preview.height + 70), "white")
        canvas.paste(preview, (10, 55))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), label, fill="black")

        output_name = f"{saved + 1:03d}_{actual}_as_{predicted}.jpg"
        canvas.save(OUTPUT_DIR / output_name)
        saved += 1

print(f"Misclassified images saved: {saved}")
print(f"Output folder: {OUTPUT_DIR}")
print("Model weights unchanged.")

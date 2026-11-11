
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

DATA_DIR = Path("data/raw/chest_xray/chest_xray/test")
MODEL_PATH = Path("models/mediscan_mobilenetv3_best.pth")
OUTPUT_DIR = Path("outputs/misclassified")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

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

saved = 0
image_index = 0

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)
        probabilities = torch.softmax(model(images), dim=1).cpu()
        predictions = probabilities.argmax(dim=1)

        for i, actual in enumerate(labels.tolist()):
            predicted = predictions[i].item()
            source_path, _ = dataset.samples[image_index]
            image_index += 1

            if actual == predicted:
                continue

            with Image.open(source_path) as img:
                preview = img.convert("RGB")
                preview.thumbnail((600, 600))

                canvas = Image.new("RGB", (620, 680), "white")
                canvas.paste(
                    preview,
                    ((620 - preview.width) // 2, 20)
                )

                draw = ImageDraw.Draw(canvas)
                confidence = probabilities[i, predicted].item() * 100

                draw.text(
                    (10, 630),
                    f"Actual: {classes[actual]}",
                    fill="black"
                )
                draw.text(
                    (10, 650),
                    f"Predicted: {classes[predicted]} "
                    f"({confidence:.1f}%)",
                    fill="black"
                )

                output_path = OUTPUT_DIR / (
                    f"error_{saved + 1:03d}_"
                    f"actual_{classes[actual]}_"
                    f"pred_{classes[predicted]}.jpg"
                )
                canvas.save(output_path)
                saved += 1

print(f"\nMisclassified images saved: {saved}")
print(f"Output folder: {OUTPUT_DIR.resolve()}")

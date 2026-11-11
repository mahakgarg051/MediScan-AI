
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Usage:
# python src/predict.py "path\to\xray.jpg"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mediscan_mobilenetv3_finetuned.pth"

IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 2:
    print('Usage: python src/predict.py "path\\to\\xray.jpg"')
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"Error: Image not found: {image_path}")
    sys.exit(1)

if not MODEL_PATH.is_file():
    print(f"Error: Model checkpoint not found: {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

try:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
except Exception as error:
    print(f"Could not read image: {error}")
    sys.exit(1)

with torch.inference_mode():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]


pneumonia_idx = classes.index("PNEUMONIA")
normal_idx = classes.index("NORMAL")

pneumonia_probability = float(probabilities[pneumonia_idx].item())

# Experimental threshold selected using validation data
PNEUMONIA_THRESHOLD = 0.70

if pneumonia_probability >= PNEUMONIA_THRESHOLD:
    predicted_class = "PNEUMONIA"
else:
    predicted_class = "NORMAL"

confidence = float(probabilities[classes.index(predicted_class)].item()) * 100


print("\n--- MediScan AI Prediction ---")
print(f"Device: {device}")
print(f"Image: {image_path.name}")
print(f"Prediction: {predicted_class}")
print(f"Model score: {confidence:.2f}%")

print("\nScores for each class:")
for index, class_name in enumerate(classes):
    score = float(probabilities[index].item()) * 100
    print(f"{class_name}: {score:.2f}%")

print("\nNote: This is an experimental AI prediction,")
print("not a medical diagnosis. Model score is not clinical certainty.")


from pathlib import Path
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mediscan_mobilenetv3_finetuned.pth"
OUTPUT_DIR = ROOT / "outputs" / "gradcam"

IMAGE_SIZE = 224
THRESHOLD = 0.70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/gradcam.py "path\\to\\xray.jpg"')
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

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

    # Last convolutional feature block
    target_layer = model.features[-1]
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    with Image.open(image_path) as opened:
        original = opened.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = transform(original).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    pneumonia_idx = classes.index("PNEUMONIA")
    pneumonia_score = probabilities[pneumonia_idx].item()
    predicted_class = (
        "PNEUMONIA" if pneumonia_score >= THRESHOLD else "NORMAL"
    )
    predicted_idx = classes.index(predicted_class)

    # Explain the score for the predicted class
    model.zero_grad(set_to_none=True)
    logits[0, predicted_idx].backward()

    feature_maps = activations[0].detach()
    grads = gradients[0].detach()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    cam = cam.cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    forward_handle.remove()
    backward_handle.remove()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"

    image_array = np.asarray(original)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Input X-ray")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(image_array)
    axes[2].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title("Heatmap overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {predicted_class} | "
        f"NORMAL: {probabilities[classes.index('NORMAL')].item():.1%} | "
        f"PNEUMONIA: {pneumonia_score:.1%}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Prediction:", predicted_class)
    print(f"NORMAL score: {probabilities[classes.index('NORMAL')].item():.2%}")
    print(f"PNEUMONIA score: {pneumonia_score:.2%}")
    print("Grad-CAM saved:", output_path)
    print("Note: Heatmap is an experimental explanation, not a diagnosis.")


if __name__ == "__main__":
    main()

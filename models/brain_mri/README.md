# Brain MRI Model Documentation

Documentation for the trained 4-class Brain MRI classification model used in MediScan AI.

## Model Overview

- **Actual Checkpoint**: `models/brain_mri_mobilenetv3.pth`
- **Architecture**: MobileNetV3-Small
- **Image Size**: 224×224
- **Classes**: `glioma`, `meningioma`, `notumor`, `pituitary`
- **Training Images**: 5,600
- **Testing Images**: 1,600
- **Best Validation Accuracy**: 99.05%
- **Test Accuracy**: 94.69%

## Git Exclusion & Checkpoint Generation

The model checkpoint (`models/brain_mri_mobilenetv3.pth`) is intentionally excluded from Git tracking via `.gitignore` (`*.pth`) due to file size constraints. It must be generated separately by running the training script.

### Running the Training Script

To train the model and generate the checkpoint:

```bash
python src/train_brain_mri.py
```

By default, the training script:
- Reads datasets from `data/raw/brain_mri/` (`Training/` and `Testing/` subdirectories).
- Uses a pretrained MobileNetV3-Small backbone adapted for 4 classes.
- Automatically saves the best-performing checkpoint to `models/brain_mri_mobilenetv3.pth`.

### Evaluating the Model

To evaluate the checkpoint on the held-out test dataset:

```bash
python src/evaluate_brain_mri.py
```

Evaluation metrics will be saved to `outputs/brain_mri/evaluation_metrics.json`.

## Checkpoint Structure

The saved checkpoint (`models/brain_mri_mobilenetv3.pth`) contains a dictionary with the model state and metadata:

```python
{
    "model_state_dict": model.state_dict(),
    "classes": ["glioma", "meningioma", "notumor", "pituitary"],
    "class_to_idx": {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3,
    },
    "architecture": "mobilenet_v3_small",
    "image_size": 224,
    "best_val_accuracy": 99.0476,
    "best_val_loss": 0.0443,
    "best_epoch": 8,
}
```

*Note: Brain MRI model checkpoints and pipelines are maintained independently from Chest X-ray models.*

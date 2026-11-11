
from pathlib import Path
from collections import Counter

folder = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "misclassified_finetuned"
)

files = list(folder.glob("*.jpg"))

counts = Counter()

for file in files:
    name = file.name

    if "_NORMAL_as_PNEUMONIA.jpg" in name:
        counts["NORMAL predicted as PNEUMONIA (False Positive)"] += 1
    elif "_PNEUMONIA_as_NORMAL.jpg" in name:
        counts["PNEUMONIA predicted as NORMAL (False Negative)"] += 1
    else:
        counts["Other filename"] += 1

print(f"Total misclassified images: {len(files)}")

for label, count in counts.items():
    print(f"{label}: {count}")

print("\nSummary complete.")

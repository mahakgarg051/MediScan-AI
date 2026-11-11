
from pathlib import Path
from collections import defaultdict
import hashlib

DATA_DIR = Path("data/raw/chest_xray/chest_xray")
splits = ["train", "test", "val"]

image_hashes = defaultdict(list)

for split in splits:
    split_dir = DATA_DIR / split

    for path in split_dir.rglob("*"):
        if not path.is_file():
            continue

        try:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            image_hashes[digest].append(
                (split, path.parent.name, str(path))
            )
        except OSError as error:
            print(f"Could not read {path}: {error}")

cross_split_duplicates = []

for entries in image_hashes.values():
    entry_splits = {entry[0] for entry in entries}

    if len(entry_splits) > 1:
        cross_split_duplicates.append(entries)

print("Exact duplicate groups across splits:",
      len(cross_split_duplicates))

for group in cross_split_duplicates[:20]:
    print("\nDuplicate group:")
    for split, class_name, path in group:
        print(f"  {split} | {class_name} | {path}")

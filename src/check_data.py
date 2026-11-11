
from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib

DATA_DIR = Path("data/raw/chest_xray/chest_xray")

for split in ["train", "test", "val"]:
    split_dir = DATA_DIR / split

    if not split_dir.exists():
        print(f"\n{split}: folder missing")
        continue

    print(f"\n--- {split.upper()} ---")

    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = split_dir / class_name

        if not folder.exists():
            print(f"{class_name}: folder missing")
            continue

        files = [
            p for p in folder.iterdir()
            if p.is_file()
        ]

        sizes = []
        modes = Counter()
        hashes = Counter()
        bad_images = 0

        for path in files:
            try:
                with Image.open(path) as img:
                    sizes.append(img.size)
                    modes[img.mode] += 1
                    img.verify()

                digest = hashlib.md5(path.read_bytes()).hexdigest()
                hashes[digest] += 1

            except Exception:
                bad_images += 1

        duplicate_groups = sum(
            count > 1 for count in hashes.values()
        )

        print(f"{class_name}: {len(files)} images")
        print(f"  Image sizes (most common): {Counter(sizes).most_common(5)}")
        print(f"  Image modes: {modes}")
        print(f"  Corrupted images: {bad_images}")
        print(f"  Duplicate groups within class: {duplicate_groups}")

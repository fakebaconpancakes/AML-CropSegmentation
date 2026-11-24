
from pathlib import Path
from PIL import Image
import numpy as np
import json

DATA_DIR = Path("data/raw/data2017_miniscale")
RGB_DIR = DATA_DIR / "field_images" / "rgb"
LABEL_DIR = DATA_DIR / "field_labels"
SPLIT_FILE = Path("data/raw/data2017_splits.json")

PATCH_SIZE = 64
IMAGE_SIZE = 512

# Load official field splits
with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

# Prepare annotation categories
categories = sorted([
    d for d in LABEL_DIR.iterdir()
    if d.is_dir()
])

# Create field-to-split mapping
field_to_split = {}

for split_name, field_ids in splits.items():
    for field_id in field_ids:
        field_to_split[field_id] = split_name

# Statistics
stats = {
    name: {
        "images": 0,
        "normal_patches": 0,
        "anomaly_patches": 0
    }
    for name in splits
}

for image_path in RGB_DIR.glob("*.jpg"):

    image_id = image_path.stem
    field_id = image_id.split("_")[0]

    # Ignore fields not found in official splits
    if field_id not in field_to_split:
        continue

    split_name = field_to_split[field_id]
    stats[split_name]["images"] += 1

    # Combine all 9 annotation masks
    combined_mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=bool
    )

    for category in categories:
        mask_path = category / f"{image_id}.png"

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        combined_mask |= (mask > 0)

    # Divide into 64x64 patches
    for y in range(0, IMAGE_SIZE, PATCH_SIZE):
        for x in range(0, IMAGE_SIZE, PATCH_SIZE):

            patch = combined_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            if patch.any():
                stats[split_name]["anomaly_patches"] += 1
            else:
                stats[split_name]["normal_patches"] += 1

# Display results
print("=" * 55)
print("PATCH CANDIDATE SUMMARY")
print("=" * 55)

for split_name, result in stats.items():

    normal = result["normal_patches"]
    anomaly = result["anomaly_patches"]
    total = normal + anomaly

    percentage = normal / total * 100 if total else 0

    print(f"\n{split_name.upper()}")
    print(f"Images: {result['images']}")
    print(f"Normal candidate patches: {normal:,}")
    print(f"Anomaly-containing patches: {anomaly:,}")
    print(f"Total patches: {total:,}")
    print(f"Normal candidate percentage: {percentage:.2f}%")


from pathlib import Path
from PIL import Image
import json

DATA_DIR = Path("data/raw/data2017_miniscale")
LABEL_DIR = DATA_DIR / "field_labels"
SPLIT_FILE = Path("data/raw/data2017_splits.json")

categories = sorted([
    d for d in LABEL_DIR.iterdir()
    if d.is_dir()
])

# Load official field splits
with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

# Map each image to its field ID
rgb_dir = DATA_DIR / "field_images" / "rgb"
images = list(rgb_dir.glob("*.jpg"))

normal_candidates = []
anomalous_images = []
category_counts = {category.name: 0 for category in categories}

for image_path in images:

    image_id = image_path.stem
    field_id = image_id.split("_")[0]

    has_anomaly = False

    # Check all 9 anomaly masks
    for category in categories:

        mask_path = category / f"{image_id}.png"

        with Image.open(mask_path) as mask:
            pixels = mask.convert("L")

            if any(pixel > 0 for pixel in pixels.getdata()):
                has_anomaly = True
                category_counts[category.name] += 1

    if has_anomaly:
        anomalous_images.append(image_id)
    else:
        normal_candidates.append(image_id)

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"Total images: {len(images)}")
print(f"Normal candidates: {len(normal_candidates)}")
print(f"Images with annotations: {len(anomalous_images)}")

print("\nCategory counts:")
for category, count in category_counts.items():
    print(f"{category:25}: {count}")

print("\nNormal candidate percentage:")
print(f"{len(normal_candidates) / len(images) * 100:.2f}%")

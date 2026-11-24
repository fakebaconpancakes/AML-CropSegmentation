
import json
from pathlib import Path

# Dataset paths
DATA_DIR = Path("data/raw/data2017_miniscale")
RGB_DIR = DATA_DIR / "field_images" / "rgb"
SPLIT_FILE = Path("data/raw/data2017_splits.json")

# Load official field splits
with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

# Extract field IDs from RGB filenames
rgb_files = list(RGB_DIR.glob("*.jpg"))

field_to_images = {}

for image_path in rgb_files:
    # Example: 16RQA6NB1_2362-523-2874-1035.jpg
    field_id = image_path.stem.split("_")[0]

    field_to_images.setdefault(field_id, []).append(image_path.name)

print(f"Total RGB images: {len(rgb_files)}")
print(f"Unique fields in RGB: {len(field_to_images)}")

print("\nField mapping:")

for split_name, field_ids in splits.items():
    matched_fields = set(field_ids) & set(field_to_images.keys())
    missing_fields = set(field_ids) - set(field_to_images.keys())

    tile_count = sum(
        len(field_to_images[field_id])
        for field_id in matched_fields
    )

    print(f"\n{split_name.upper()}")
    print(f"Fields in split: {len(field_ids)}")
    print(f"Fields matched: {len(matched_fields)}")
    print(f"Missing fields: {len(missing_fields)}")
    print(f"RGB tiles: {tile_count}")

    if missing_fields:
        print("Missing examples:", list(missing_fields)[:10])

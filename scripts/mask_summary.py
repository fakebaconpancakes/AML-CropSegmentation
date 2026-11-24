
from pathlib import Path
from PIL import Image

# Dataset location
DATA_DIR = Path("data/raw/data2017_miniscale/field_labels")

# Go through every anomaly category
for category_dir in sorted(DATA_DIR.iterdir()):

    if not category_dir.is_dir():
        continue

    total_images = 0
    annotated_images = 0
    total_anomaly_pixels = 0

    for mask_path in category_dir.glob("*.png"):
        total_images += 1

        with Image.open(mask_path) as mask:
            # Convert mask to grayscale
            mask = mask.convert("L")

            # Count pixels marked as anomaly (white)
            anomaly_pixels = sum(pixel > 0 for pixel in mask.getdata())

            if anomaly_pixels > 0:
                annotated_images += 1
                total_anomaly_pixels += anomaly_pixels

    print(
        f"{category_dir.name:25} "
        f"Annotated: {annotated_images:5} / {total_images:5} | "
        f"Anomaly pixels: {total_anomaly_pixels:,}"
    )


from pathlib import Path
from PIL import Image
import numpy as np

DATA_DIR = Path("data/raw/data2017_miniscale")
RGB_DIR = DATA_DIR / "field_images" / "rgb"
LABEL_DIR = DATA_DIR / "field_labels"

categories = sorted([
    d for d in LABEL_DIR.iterdir()
    if d.is_dir()
])

unannotated_percentages = []

for image_path in RGB_DIR.glob("*.jpg"):

    image_id = image_path.stem

    # Combined mask: True means annotated anomaly
    combined_mask = np.zeros((512, 512), dtype=bool)

    for category in categories:
        mask_path = category / f"{image_id}.png"

        mask = np.array(Image.open(mask_path).convert("L"))
        combined_mask |= (mask > 0)

    # Calculate percentage of pixels without annotations
    unannotated_ratio = 1.0 - combined_mask.mean()
    unannotated_percentages.append(unannotated_ratio * 100)

values = np.array(unannotated_percentages)

print("TOTAL IMAGES:", len(values))
print("\nUNANNOTATED PIXEL COVERAGE")
print(f"Mean:   {values.mean():.2f}%")
print(f"Median: {np.median(values):.2f}%")
print(f"Minimum: {values.min():.2f}%")
print(f"Maximum: {values.max():.2f}%")

print("\nIMAGES WITH SUFFICIENT UNANNOTATED AREA")

for threshold in [50, 70, 80, 90]:
    count = np.sum(values >= threshold)
    print(f"At least {threshold}% unannotated: {count}")

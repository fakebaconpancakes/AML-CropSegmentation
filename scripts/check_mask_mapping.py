from pathlib import Path

DATA_DIR = Path("data/raw/data2017_miniscale")

RGB_DIR = DATA_DIR / "field_images" / "rgb"
LABEL_DIR = DATA_DIR / "field_labels"

# Collect RGB image filenames without extensions
rgb_files = {p.stem for p in RGB_DIR.glob("*.jpg")}

print(f"Total RGB images: {len(rgb_files)}")

print("\nChecking annotation categories...\n")

for category_dir in sorted(LABEL_DIR.iterdir()):
    if not category_dir.is_dir():
        continue

    mask_files = {p.stem for p in category_dir.glob("*.png")}

    matched = rgb_files & mask_files
    missing_masks = rgb_files - mask_files
    extra_masks = mask_files - rgb_files

    print(f"{category_dir.name}")
    print(f"  Matching masks: {len(matched)}")
    print(f"  Missing masks:  {len(missing_masks)}")
    print(f"  Extra masks:    {len(extra_masks)}")
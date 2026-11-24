"""Load RGB reconstruction pairs using the official field-level splits.

Anomaly annotations, field bounds, validity masks, and NIR are not loaded.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AgricultureVisionDataset(Dataset):
    """Return an RGB image and an identical reconstruction target.

    Args:
        split: Official split name: "train", "val", or "test".
        data_dir: Extracted dataset folder containing ``field_images/rgb``.
        split_file: Official JSON file mapping split names to field ID lists.

    Default paths are relative to the current working directory. Run from the
    project root, or pass explicit paths when using Colab or Kaggle.

    Each sample is ``(image, target)``. Both are float32 tensors with shape
    ``(3, 128, 128)`` and values in [0, 1]. The target is a separate copy so
    changing the input later cannot accidentally change the target.
    """

    def __init__(
        self,
        split: str,
        data_dir: str | Path = "data/raw/data2017_miniscale",
        split_file: str | Path = "data/raw/data2017_splits.json",
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError('split must be "train", "val", or "test".')

        rgb_dir = Path(data_dir) / "field_images" / "rgb"
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

        with Path(split_file).open("r", encoding="utf-8") as file:
            splits = json.load(file)

        # Validate every split before assigning any image. A field must belong
        # to exactly one split, including when the JSON file is changed later.
        field_to_split = {}
        for split_name in ("train", "val", "test"):
            field_ids = splits.get(split_name)
            if not isinstance(field_ids, list):
                raise ValueError(f"Split {split_name!r} must contain a field ID list.")
            for field_id in field_ids:
                if not isinstance(field_id, str) or not field_id:
                    raise ValueError("Field IDs must be nonempty strings.")
                if field_id in field_to_split:
                    raise ValueError(f"Repeated field ID in official splits: {field_id}")
                field_to_split[field_id] = split_name

        self.split = split
        self.image_paths: list[Path] = []
        for image_path in sorted(rgb_dir.glob("*.jpg")):
            # Actual filenames look like 16RQA6NB1_2362-523-2874-1035.jpg.
            field_id, separator, _ = image_path.stem.partition("_")
            if not separator or field_id not in field_to_split:
                raise ValueError(f"Image has no official field assignment: {image_path.name}")
            if field_to_split[field_id] == split:
                self.image_paths.append(image_path)

        # Fields listed in the official file may be absent from this miniscale
        # subset. That is allowed; images are never reassigned to another split.
        if not self.image_paths:
            raise ValueError(f"No RGB images found for split {split!r} in {rgb_dir}")

    def __len__(self) -> int:
        """Number of RGB tiles in this split."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one RGB tile and prepare its reconstruction pair."""
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            image = image.resize((128, 128), resample=Image.Resampling.BILINEAR)
            pixels = np.array(image, dtype=np.float32) / 255.0

        # Pillow uses (height, width, channels); PyTorch uses (channels, height, width).
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        target_tensor = image_tensor.clone()
        return image_tensor, target_tensor

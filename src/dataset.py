"""Official field splits and priority-safe segmentation targets."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ("Normal farmland", "Drydown", "Double plant", "Weed cluster", "Planter skip")
SELECTED = ("drydown", "double_plant", "weed_cluster", "planter_skip")
EXCLUDED = ("endrow", "storm_damage", "water", "nutrient_deficiency", "waterway")
IGNORE_INDEX = 255
EXPECTED_COUNTS = {"train": 4505, "val": 2048, "test": 1792}


def index_splits(data_dir, split_file, expected_counts=None):
    """Inspect filenames only; never read test pixels during training setup."""
    root = Path(data_dir)
    with Path(split_file).open(encoding="utf-8") as handle:
        splits = json.load(handle)
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Official splits must have exactly train, val, test keys.")
    assignments = {}
    for split, fields in splits.items():
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{split}: expected a nonempty field ID list.")
        for field in fields:
            if not isinstance(field, str) or not field or "_" in field:
                raise ValueError(f"Invalid field ID: {field!r}")
            if field in assignments:
                raise ValueError(f"Repeated field ID across/within official splits: {field}")
            assignments[field] = split
    rgb_dir = root / "field_images" / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    paths = {name: [] for name in splits}
    seen = set()
    for path in sorted(rgb_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        field, separator, _ = path.stem.partition("_")
        if not separator or field not in assignments:
            raise ValueError(f"Image has no official field assignment: {path.name}")
        if path.stem in seen:
            raise ValueError(f"Duplicate RGB sample ID: {path.stem}")
        seen.add(path.stem)
        paths[assignments[field]].append(path)
    counts = {name: len(items) for name, items in paths.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Empty image split: {counts}")
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Expected image counts {expected_counts}, found {counts}")
    # Include all official fields and sample IDs; relocation does not change identity.
    identity = {"fields": {k: sorted(v) for k, v in splits.items()},
                "images": {k: [p.name for p in v] for k, v in paths.items()}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return paths, fingerprint


def construct_target(selected, excluded, valid):
    """Native-resolution target: invalid/excluded/ambiguous always win."""
    selected = np.asarray(selected, dtype=bool)
    excluded = np.asarray(excluded, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if selected.shape != (4, *valid.shape) or excluded.shape != (5, *valid.shape):
        raise ValueError("Expected four selected and five excluded masks aligned with validity.")
    count = selected.sum(axis=0)
    ignored = ~valid | excluded.any(axis=0) | (count > 1)
    target = np.zeros(valid.shape, dtype=np.uint8)
    for class_id, mask in enumerate(selected, start=1):
        target[mask & (count == 1) & ~ignored] = class_id
    target[ignored] = IGNORE_INDEX
    return target


def augment_pair(image, target):
    """Apply identical integer grid transforms without interpolating labels."""
    if random.random() < 0.5:
        image, target = image[:, ::-1], target[:, ::-1]
    if random.random() < 0.5:
        image, target = image[::-1], target[::-1]
    turns = random.randrange(4)
    return np.rot90(image, turns), np.rot90(target, turns)


class AgricultureVisionDataset(Dataset):
    def __init__(self, data_dir, image_paths, split, image_size=256,
                 native_size=512, augment=False):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if augment and split != "train":
            raise ValueError("Only training data may be augmented.")
        self.root = Path(data_dir)
        self.image_paths = list(image_paths)
        self.split = split
        self.image_size = image_size
        self.native_size = native_size
        self.augment = augment
        folders = [self.root / "field_labels" / c for c in SELECTED + EXCLUDED]
        folders += [self.root / "field_masks", self.root / "field_bounds"]
        for folder in folders:
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing annotation directory: {folder}")
            # All nine classes are densely stored, even when a mask is empty.
            present = {p.stem for p in folder.glob("*.png")}
            missing = [p.stem for p in self.image_paths if p.stem not in present]
            if missing:
                raise FileNotFoundError(f"{folder}: missing masks, examples: {missing[:3]}")

    def __len__(self):
        return len(self.image_paths)

    def _binary(self, path):
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
        if array.shape != (self.native_size, self.native_size):
            raise ValueError(f"Wrong mask dimensions {array.shape}: {path}")
        if not np.isin(array, (0, 255)).all():
            raise ValueError(f"Mask must be binary 0/255: {path}")
        return array == 255

    def load_target(self, index):
        """Deterministic resized target for both frequency counting and training."""
        name = self.image_paths[index].stem + ".png"
        selected = [self._binary(self.root / "field_labels" / c / name) for c in SELECTED]
        excluded = [self._binary(self.root / "field_labels" / c / name) for c in EXCLUDED]
        valid = (self._binary(self.root / "field_masks" / name)
                 & self._binary(self.root / "field_bounds" / name))
        target = construct_target(selected, excluded, valid)
        target = Image.fromarray(target).resize((self.image_size, self.image_size),
                                                Image.Resampling.NEAREST)
        return np.array(target, dtype=np.int64)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            if image.size != (self.native_size, self.native_size):
                raise ValueError(f"Wrong RGB dimensions {image.size}: {path}")
            image = image.convert("RGB").resize((self.image_size, self.image_size),
                                                 Image.Resampling.BILINEAR)
            pixels = np.array(image, dtype=np.float32) / 255.0
        target = self.load_target(index)
        if self.augment:
            pixels, target = augment_pair(pixels, target)
        return (torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).contiguous(),
                torch.from_numpy(np.ascontiguousarray(target)))


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, num_workers, generator, cuda=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.augment,
                      num_workers=num_workers, pin_memory=cuda, drop_last=False,
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=generator)


def check_batch(images, targets, image_size):
    if images.shape != (len(images), 3, image_size, image_size):
        raise ValueError(f"Wrong image batch shape: {tuple(images.shape)}")
    if targets.shape != (len(images), image_size, image_size):
        raise ValueError(f"Wrong target batch shape: {tuple(targets.shape)}")
    if images.dtype != torch.float32 or targets.dtype != torch.int64:
        raise ValueError("Expected float32 images and int64 targets.")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("RGB pixels must be finite and in [0, 1].")
    if not torch.isin(targets, torch.tensor([0, 1, 2, 3, 4, 255])).all():
        raise ValueError("Unexpected target labels.")

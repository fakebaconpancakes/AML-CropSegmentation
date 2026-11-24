"""Train the baseline CAE: python -m src.train --help.

Use --smoke-test for a two-image train/validation check without saving files.
No anomaly annotations or official test images are loaded.
"""

import argparse
import csv
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import AgricultureVisionDataset
from .losses import masked_reconstruction_loss
from .masking import random_patch_mask
from .model import ConvolutionalAutoencoder


def set_seed(seed: int) -> None:
    """Seed model initialization, masking, and other random operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a reproducible NumPy/Python seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_valid_mask(data_dir: Path, image_path: Path) -> torch.Tensor:
    """Read matching PNGs and return a binary (1, 128, 128) float32 tensor."""
    filename = image_path.with_suffix(".png").name
    bounds_path = data_dir / "field_bounds" / filename
    mask_path = data_dir / "field_masks" / filename
    with Image.open(bounds_path) as bounds, Image.open(mask_path) as validity:
        with Image.open(image_path) as rgb:
            if bounds.size != validity.size or bounds.size != rgb.size:
                raise ValueError(f"RGB/bounds/validity sizes differ for {image_path.name}")
        if bounds.mode != "L" or validity.mode != "L":
            raise ValueError(f"Expected single-channel bounds/validity for {filename}")
        bounds_pixels = np.array(bounds)
        validity_pixels = np.array(validity)

    # Keep only pixels where BOTH original source masks equal 255.
    valid = (bounds_pixels == 255) & (validity_pixels == 255)
    binary_image = Image.fromarray(valid.astype(np.uint8))
    binary_image = binary_image.resize((128, 128), resample=Image.Resampling.NEAREST)
    return torch.from_numpy(np.array(binary_image, dtype=np.float32)).unsqueeze(0)


class DatasetWithValidMasks(Dataset):
    """Add lazily loaded valid masks to the existing official-split dataset."""

    def __init__(self, split: str, data_dir: Path, split_file: Path) -> None:
        if split not in ("train", "val"):
            raise ValueError("Training supports only the official train and val splits.")
        self.images = AgricultureVisionDataset(split, data_dir, split_file)
        self.data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, target = self.images[index]
        # Use the exact path at this index; never glob or reorder masks separately.
        valid_mask = load_valid_mask(self.data_dir, self.images.image_paths[index])
        return image, target, valid_mask


def make_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device,
                training: bool) -> DataLoader:
    """Shuffle training images only; seed workers independently of masking."""
    generator = torch.Generator().manual_seed(args.seed + (0 if training else 1))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device,
              patch_size: int, mask_ratio: float,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, int]:
    """Train or validate once; return selected-pixel mean loss and skipped batches."""
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_selected_values = 0
    skipped_batches = 0
    non_blocking = device.type == "cuda"

    # Validation explicitly disables autograd and never calls the optimizer.
    with nullcontext() if training else torch.no_grad():
        for images, targets, valid_mask in loader:
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            valid_mask = valid_mask.to(device, non_blocking=non_blocking)
            masked_images, mask = random_patch_mask(images, patch_size, mask_ratio)

            # A random mask can miss all valid pixels in a batch. Skip it openly
            # rather than inventing a loss value or changing the random selection.
            selected_pixels = int((mask.bool() & valid_mask.bool()).sum().item())
            if selected_pixels == 0:
                skipped_batches += 1
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(masked_images)
            loss = masked_reconstruction_loss(reconstruction, targets, mask, valid_mask)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite reconstruction loss; stopping this run.")
            if training:
                loss.backward()
                optimizer.step()

            # Weight by the number of selected RGB values, not the batch size.
            selected_values = selected_pixels * targets.shape[1]
            total_squared_error += loss.item() * selected_values
            total_selected_values += selected_values

    if total_selected_values == 0:
        raise ValueError("Entire epoch has no masked valid pixels; check masks/settings.")
    return total_squared_error / total_selected_values, skipped_batches


def parse_args() -> argparse.Namespace:
    """Read settings without requiring code edits or absolute platform paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/data2017_miniscale"))
    parser.add_argument("--split-file", type=Path, default=Path("data/raw/data2017_splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Check two real images per split, then exit without saving.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.num_workers < 0:
        parser.error("batch-size/epochs must be positive; num-workers must be nonnegative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative.")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1.")
    if args.patch_size <= 0 or 128 % args.patch_size != 0:
        parser.error("patch-size must be positive and divide 128 evenly.")
    if not 0 < args.mask_ratio <= 1:
        parser.error("mask-ratio must be greater than 0 and at most 1.")
    if int(args.mask_ratio * (128 // args.patch_size)**2) == 0:
        parser.error("These settings mask zero patches; increase mask-ratio.")
    return args


def smoke_test(model: torch.nn.Module, datasets: dict, args: argparse.Namespace,
               device: torch.device, optimizer: torch.optim.Optimizer) -> None:
    """Exercise one real training batch and one validation batch without saving."""
    args.batch_size = 2
    for split, dataset in datasets.items():
        image, target, valid_mask = dataset[0]
        assert image.shape == target.shape == (3, 128, 128)
        assert valid_mask.shape == (1, 128, 128)
        assert valid_mask.dtype == torch.float32
        assert ((valid_mask == 0) | (valid_mask == 1)).all()
        print(f"{split}: {len(dataset)} images; valid mask {tuple(valid_mask.shape)}, "
              f"binary float32; valid fraction {valid_mask.mean().item():.4f}")

    loaders = {
        split: make_loader(Subset(dataset, range(min(2, len(dataset)))), args, device,
                           training=split == "train")
        for split, dataset in datasets.items()
    }
    before = [parameter.detach().clone() for parameter in model.parameters()]
    train_loss, _ = run_epoch(model, loaders["train"], device, args.patch_size,
                              args.mask_ratio, optimizer)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before_validation = [p.detach().clone() for p in model.parameters()]
    val_loss, _ = run_epoch(model, loaders["val"], device, args.patch_size, args.mask_ratio)
    assert all(torch.equal(old, new) for old, new in zip(before_validation, model.parameters()))
    print(f"Smoke test PASS: train MSE={train_loss:.6f}, val MSE={val_loss:.6f}; "
          "forward/loss/backward/AdamW step succeeded; validation parameters unchanged. "
          "No files saved.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seed: {args.seed}")
    datasets = {
        split: DatasetWithValidMasks(split, args.data_dir, args.split_file)
        for split in ("train", "val")
    }
    model = ConvolutionalAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    if args.smoke_test:
        smoke_test(model, datasets, args, device, optimizer)
        return

    checkpoint_path = args.output_dir / "best.pt"
    history_path = args.output_dir / "history.csv"
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError("Run outputs already exist. Choose a new --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(image_size=128, device=str(device), optimizer="AdamW")
    loaders = {split: make_loader(dataset, args, device, training=split == "train")
               for split, dataset in datasets.items()}
    print(f"Train: {len(datasets['train'])}; validation: {len(datasets['val'])}; "
          f"outputs: {args.output_dir}")
    best_val_loss = float("inf")

    # Exclusive creation protects previous history. Flush each completed epoch.
    with history_path.open("x", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_skipped_batches",
                         "val_skipped_batches", "seconds"])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss, train_skipped = run_epoch(
                model, loaders["train"], device, args.patch_size, args.mask_ratio, optimizer)
            val_loss, val_skipped = run_epoch(
                model, loaders["val"], device, args.patch_size, args.mask_ratio)
            improved = val_loss < best_val_loss
            if improved:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": config,
                    "seed": args.seed,
                }, checkpoint_path)
                best_val_loss = val_loss
            elapsed = time.perf_counter() - started
            writer.writerow([epoch, train_loss, val_loss, train_skipped, val_skipped, elapsed])
            history_file.flush()
            print(f"Epoch {epoch:02d}/{args.epochs}: train={train_loss:.6f}, "
                  f"val={val_loss:.6f}, best={best_val_loss:.6f}, {elapsed:.1f}s; "
                  f"skipped batches train/val={train_skipped}/{val_skipped}"
                  + ("; saved best.pt" if improved else ""), flush=True)


if __name__ == "__main__":
    main()

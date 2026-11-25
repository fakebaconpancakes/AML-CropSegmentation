"""Validate the real miniscale training pipeline without loading test pixels.

Run from the repository root: python -m scripts.validate_dataset
"""

import argparse
from pathlib import Path

import torch

from src.dataset import AgricultureVisionDataset, EXPECTED_COUNTS, check_batch, index_splits, make_loader
from src.model import UNet
from src.train import class_weights
from src.utils import plot_sample, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/data2017_miniscale")
    parser.add_argument("--split-file", default="data/raw/data2017_splits.json")
    parser.add_argument("--output-dir", default="outputs/data_validation")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    seed_everything(42)
    torch.set_num_threads(2)
    paths, fingerprint = index_splits(args.data_dir, args.split_file, EXPECTED_COUNTS)
    print("Official counts:", {k: len(v) for k, v in paths.items()})
    dataset = AgricultureVisionDataset(args.data_dir, paths["train"], "train", augment=True)
    loader = make_loader(dataset, 8, args.num_workers, torch.Generator().manual_seed(42))
    images, targets = next(iter(loader))
    check_batch(images, targets, 256)
    print("Batch:", tuple(images.shape), tuple(targets.shape), "labels", targets.unique().tolist())
    model = UNet()
    logits = model(images[:1])
    assert logits.shape == (1, 5, 256, 256)
    torch.nn.functional.cross_entropy(logits, targets[:1], ignore_index=255).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print("Standard U-Net forward/backward passed; parameters:", sum(p.numel() for p in model.parameters()))
    weights, distribution = class_weights(dataset)
    # class_weights calls load_target directly, bypassing all stochastic augmentation.
    output = Path(args.output_dir)
    root = Path(args.data_dir).resolve()
    if output.resolve() == root or root in output.resolve().parents:
        raise ValueError("Validation outputs must be outside the input dataset.")
    output.mkdir(parents=True, exist_ok=True)
    plot_sample(images[0].numpy(), targets[0].numpy(), targets[0].numpy(),
                output / "training_alignment.png", "Real training sample; prediction panels repeat target")
    save_json(output / "report.json", {
        "status": "data and forward/backward validation only; no model training/evaluation",
        "image_counts": {k: len(v) for k, v in paths.items()},
        "split_fingerprint": fingerprint, "image_batch_shape": list(images.shape),
        "target_batch_shape": list(targets.shape), "labels": targets.unique().tolist(),
        "class_distribution": distribution, "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "workers": args.num_workers,
        "test_pixels_loaded": False})
    print("Saved report:", output / "report.json", "weights:", weights.tolist())


if __name__ == "__main__":
    main()

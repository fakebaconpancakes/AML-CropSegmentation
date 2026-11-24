"""Evaluate a CAE checkpoint on the official test split without training.

Protocol: one random masking pass per image. Only masked AND valid pixels
are scored; visible pixels are unscored, not assigned zero anomaly scores.
Metrics are pooled over scored pixels at 128 x 128. No threshold is selected.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import AgricultureVisionDataset
from .masking import random_patch_mask
from .model import ConvolutionalAutoencoder
from .train import load_valid_mask, seed_worker, set_seed


CATEGORIES = (
    "double_plant", "drydown", "endrow", "nutrient_deficiency", "planter_skip",
    "storm_damage", "water", "waterway", "weed_cluster",
)


class TestDataset(Dataset):
    """Attach evaluation annotations to exact paths from the existing test loader."""

    def __init__(self, data_dir: Path, split_file: Path) -> None:
        self.data_dir = Path(data_dir)
        self.images = AgricultureVisionDataset("test", data_dir, split_file)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image, target = self.images[index]
        path = self.images.image_paths[index]
        valid = load_valid_mask(self.data_dir, path).bool()
        with Image.open(path) as original:
            source_size = original.size
        annotations = []
        for category in CATEGORIES:
            label_path = self.data_dir / "field_labels" / category / f"{path.stem}.png"
            with Image.open(label_path) as label:
                if label.mode != "L" or label.size != source_size:
                    raise ValueError(f"Annotation is not aligned single-channel data: {label_path}")
                # Resize binary annotations using the same pixel-center mapping
                # as the bounds/validity masks. Missing files are errors, not zeros.
                binary = Image.fromarray((np.array(label) == 255).astype(np.uint8))
                resized = binary.resize((128, 128), resample=Image.Resampling.NEAREST)
                annotations.append(torch.from_numpy(np.array(resized, dtype=bool)))
        return image, target, valid, torch.stack(annotations), path.stem


def load_checkpoint(path: Path, device: torch.device):
    """Load the train.py dictionary strictly; never infer missing mask settings."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a train.py checkpoint containing model_state_dict.")
    config = checkpoint.get("config", {})
    if not all(key in config for key in ("patch_size", "mask_ratio")):
        raise ValueError("Checkpoint config must record patch_size and mask_ratio.")
    patch_size, mask_ratio = config["patch_size"], config["mask_ratio"]
    if not isinstance(patch_size, int) or patch_size <= 0 or 128 % patch_size:
        raise ValueError("Checkpoint patch_size must be a positive divisor of 128.")
    if not 0 < mask_ratio <= 1 or int(mask_ratio * (128 // patch_size)**2) == 0:
        raise ValueError("Checkpoint mask_ratio must select at least one patch.")
    if config.get("image_size", 128) != 128:
        raise ValueError("This evaluation supports only 128 x 128 checkpoints.")
    model = ConvolutionalAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def reconstruct_scores(model, images, targets, patch_size, mask_ratio, device):
    """Return raw per-pixel RGB MSE and input mask; no annotations enter here.

    Generate masks on CPU before transfer so a seed gives the same masking
    pattern on CPU and CUDA. The masking strategy and ratio match training.
    """
    model.eval()
    masked, mask = random_patch_mask(images, patch_size, mask_ratio)
    non_blocking = device.type == "cuda"
    reconstruction = model(masked.to(device, non_blocking=non_blocking))
    target = targets.to(device, non_blocking=non_blocking)
    if reconstruction.shape != target.shape:
        raise ValueError("Reconstruction shape does not match the RGB target.")
    scores = (reconstruction - target).square().mean(dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite reconstruction scores; metrics cannot be computed.")
    return scores.cpu().numpy(), mask[:, 0].bool().numpy()


def ranking_metrics(sorted_labels: np.ndarray, group_ends: np.ndarray) -> dict:
    """Calculate exact AUROC and non-interpolated AP from descending-score labels.

    Equal scores are grouped before computing the ROC/precision-recall points,
    so ties do not gain an arbitrary advantage from pixel ordering. AP is the
    sum of precision * recall increase; AUROC uses trapezoids on the ROC curve.
    Both metrics are unavailable if either class is absent.
    """
    count = len(sorted_labels)
    positives = int(np.count_nonzero(sorted_labels))
    negatives = count - positives
    result = {"evaluated_pixels": count, "evaluated_positive_pixels": positives,
              "evaluated_negative_pixels": negatives,
              "pixel_auroc": None, "average_precision": None, "status": "ok"}
    if count == 0 or positives == 0 or negatives == 0:
        result["status"] = ("no_evaluated_pixels" if count == 0 else
                            "no_positive_pixels" if positives == 0 else "no_negative_pixels")
        return result

    true_positives = np.cumsum(sorted_labels, dtype=np.int64)[group_ends]
    false_positives = group_ends + 1 - true_positives
    recall = true_positives / positives
    false_positive_rate = false_positives / negatives
    precision = true_positives / (group_ends + 1)
    recall_increase = np.diff(np.r_[0.0, recall])
    result["average_precision"] = float(np.sum(recall_increase * precision))
    result["pixel_auroc"] = float(np.sum(
        np.diff(np.r_[0.0, false_positive_rate]) *
        (recall + np.r_[0.0, recall[:-1]]) / 2
    ))
    return result


def metric_rows(scores, label_bits, valid_pixels, all_positive_counts):
    """Sort scores once for overall and nine one-versus-rest category metrics.

    Each uint16 in label_bits stores nine binary labels, reducing RAM usage.
    Other anomaly categories count as negatives for a category unless they
    overlap that category. Overall positives are the union of all categories.
    """
    if len(scores) != len(label_bits) or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must align and scores must be finite.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_bits = label_bits[order]
    group_ends = (np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
                  if len(scores) else np.array([], dtype=np.int64))
    rows = []
    for index, category in enumerate(("overall",) + CATEGORIES):
        labels = sorted_bits != 0 if index == 0 else (sorted_bits & (1 << (index - 1))) != 0
        row = {"category": category, "valid_pixels": int(valid_pixels),
               "positive_pixels": int(all_positive_counts[index])}
        row.update(ranking_metrics(labels, group_ends))
        rows.append(row)
    return rows


def save_visualization(example: dict, destination: Path, heatmap_max: float) -> None:
    """Save four panels with fixed color limits; gray heatmap pixels are unscored."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = example["image"].transpose(1, 2, 0)
    scored = example["scored"]
    ground_truth = np.ma.array(example["ground_truth"], mask=~example["valid"])
    heatmap = np.ma.array(example["scores"], mask=~scored)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("lightgray")
    truth_cmap = plt.get_cmap("gray").copy()
    truth_cmap.set_bad("lightgray")
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[1].imshow(ground_truth, cmap=truth_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Ground truth (category union)")
    plot = axes[2].imshow(heatmap, cmap=cmap, vmin=0, vmax=heatmap_max,
                          interpolation="nearest")
    axes[2].set_title("RGB MSE (gray = unscored)")
    axes[3].imshow(rgb)
    overlay_cmap = plt.get_cmap("inferno").copy()
    overlay_cmap.set_bad((0, 0, 0, 0))
    axes[3].imshow(heatmap, cmap=overlay_cmap, vmin=0, vmax=heatmap_max,
                   alpha=0.55, interpolation="nearest")
    axes[3].set_title("Heatmap overlay")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(plot, ax=list(axes[2:]), shrink=0.7, label="Mean squared RGB error")
    figure.suptitle(f"{example['name']} | one masking pass; valid & masked pixels only")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/data2017_miniscale"))
    parser.add_argument("--split-file", type=Path, default=Path("data/raw/data2017_splits.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_test"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None,
                        help="Mask seed; defaults to the checkpoint's saved seed.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--heatmap-max", type=float, default=0.01,
                        help="Fixed visualization color limit; does not change metric scores.")
    args = parser.parse_args()
    if args.num_workers < 0 or args.batch_size <= 0 or args.num_examples < 0:
        parser.error("Workers/examples must be nonnegative; batch size must be positive.")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1.")
    if not np.isfinite(args.heatmap_max) or args.heatmap_max <= 0:
        parser.error("heatmap-max must be finite and positive.")
    return args


def main() -> None:
    args = parse_args()
    # Never reuse the checkpoint directory or replace existing evaluation files.
    if args.output_dir.resolve() == args.checkpoint.resolve().parent:
        raise ValueError("Evaluation output must be separate from the checkpoint directory.")
    if args.output_dir.exists():
        raise FileExistsError("Evaluation output already exists; choose a new --output-dir.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    seed = args.seed if args.seed is not None else checkpoint.get("seed", config.get("seed"))
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("No valid saved seed; supply --seed explicitly.")
    # Seed after model initialization/loading; masks do not depend on its RNG use.
    set_seed(seed)
    dataset = TestDataset(args.data_dir, args.split_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(seed))
    example_indices = set(np.linspace(0, len(dataset) - 1,
                                      min(args.num_examples, len(dataset)), dtype=int).tolist())
    examples, score_chunks, label_chunks = [], [], []
    valid_pixels = 0
    positive_counts = np.zeros(10, dtype=np.int64)
    offset = 0
    print(f"Official test split: {len(dataset)} images; device={device}; seed={seed}; "
          f"patch_size={config['patch_size']}; mask_ratio={config['mask_ratio']}", flush=True)
    print("Protocol: one pass; metrics only on masked AND valid pixels. No threshold fitting.")

    for images, targets, valid, annotations, names in loader:
        # Only RGB tensors enter reconstruction. Labels are used below, on CPU.
        scores, masked = reconstruct_scores(model, images, targets, config["patch_size"],
                                             config["mask_ratio"], device)
        valid = valid[:, 0].numpy()
        annotations = annotations.numpy()
        union = annotations.any(axis=1)
        selected = masked & valid
        valid_pixels += int(valid.sum())
        positive_counts[0] += int((union & valid).sum())
        packed = np.zeros(valid.shape, dtype=np.uint16)
        for index in range(len(CATEGORIES)):
            positive_counts[index + 1] += int((annotations[:, index] & valid).sum())
            packed |= annotations[:, index].astype(np.uint16) << index
        score_chunks.append(scores[selected])
        label_chunks.append(packed[selected])
        for local_index, name in enumerate(names):
            if offset + local_index in example_indices:
                examples.append({"name": name, "image": images[local_index].numpy(),
                                 "valid": valid[local_index], "scored": selected[local_index],
                                 "ground_truth": union[local_index], "scores": scores[local_index]})
        offset += len(names)
        if offset % (args.batch_size * 10) == 0 or offset == len(dataset):
            print(f"Reconstructed {offset}/{len(dataset)} test images", flush=True)

    scores = np.concatenate(score_chunks)
    label_bits = np.concatenate(label_chunks)
    del score_chunks, label_chunks
    rows = metric_rows(scores, label_bits, valid_pixels, positive_counts)
    report = {
        "protocol": "single_pass_masked_valid_pixels",
        "score": "mean RGB squared reconstruction error; higher means more anomalous",
        "split": "test", "images": len(dataset), "resolution": [128, 128],
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_seed": checkpoint.get("seed"), "evaluation_seed": seed,
        "patch_size": config["patch_size"], "mask_ratio": config["mask_ratio"],
        "device": str(device), "batch_size": args.batch_size,
        "data_dir": str(args.data_dir), "split_file": str(args.split_file),
        "valid_pixels": valid_pixels, "evaluated_pixels": len(scores),
        "valid_pixel_coverage": len(scores) / valid_pixels if valid_pixels else 0.0,
        "count_definitions": {
            "valid_pixels": "All pixels in bounds AND field validity at 128 x 128",
            "positive_pixels": "Category anomaly pixels among all valid pixels",
            "evaluated_pixels": "Valid pixels hidden in this single masking pass",
            "evaluated_positive_pixels": "Category positives within evaluated pixels",
        },
        "category_protocol": "one-versus-rest over evaluated pixels, including other anomalies as negatives",
        "heatmap_color_limits": [0, args.heatmap_max],
        "visualization_selection": "Evenly spaced test indices; independent of labels and scores",
        "limitations": [
            "Visible pixels are not evaluated; this is not full-image localization coverage.",
            "One random mask gives seed-dependent results and may miss small anomalies.",
            "Nearest-neighbor resizing to 128 x 128 may remove small annotations.",
            "Category scores share one class-agnostic reconstruction-error map.",
            "Missing either positive or negative evaluated pixels makes both metrics unavailable.",
        ],
        "metrics": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    heatmap_dir = args.output_dir / "heatmaps"
    heatmap_dir.mkdir()
    with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    with (args.output_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for example in examples:
        save_visualization(example, heatmap_dir / f"{example['name']}.png", args.heatmap_max)
    for row in rows:
        print(f"{row['category']}: AUROC={row['pixel_auroc']}, AP={row['average_precision']} "
              f"({row['status']}); scored positives={row['evaluated_positive_pixels']}")
    print(f"Saved evaluation to {args.output_dir}; valid-pixel coverage="
          f"{report['valid_pixel_coverage']:.2%}")


if __name__ == "__main__":
    main()

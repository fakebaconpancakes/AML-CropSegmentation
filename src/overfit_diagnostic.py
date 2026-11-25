"""Memorize a few fixed training crops with the unchanged baseline U-Net.

This is a training-only sanity check, not patch-sampling Experiment 2.
Run from the repository root; no packages are installed by this script.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

from .dataset import (AgricultureVisionDataset, CLASSES, EXPECTED_COUNTS, IGNORE_INDEX,
                      check_batch, index_splits)
from .model import UNet
from .train import SCHEMA, class_weights
from .utils import (COLORS, IGNORE_COLOR, atomic_checkpoint, capture_rng, choose_device,
                    colorize, confusion_matrix, metrics_from_confusion, plot_sample,
                    save_csv, save_json, seed_everything)


class FixedTrainingSubset(Dataset):
    """Cache only the chosen crops; no augmentation, resampling, or external split."""
    split = "train"
    augment = False

    def __init__(self, images, targets):
        self.images, self.targets = images, targets
        self.image_size = targets.shape[-1]
        check_batch(images, targets, self.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def load_target(self, index):
        return self.targets[index].numpy()


def crop_for_class(target, class_id, size, minimum):
    """Choose one fixed window with substantial anomaly AND normal supervision."""
    coordinates = np.argwhere(target == class_id)
    if len(coordinates) < minimum:
        return None
    best, best_score = None, -1
    # Deterministic bounded search. No RNG or model performance enters selection.
    for index in np.linspace(0, len(coordinates) - 1, min(64, len(coordinates)), dtype=int):
        y, x = coordinates[index]
        top = int(np.clip(y - size // 2, 0, target.shape[0] - size))
        left = int(np.clip(x - size // 2, 0, target.shape[1] - size))
        patch = target[top:top + size, left:left + size]
        valid = patch != IGNORE_INDEX
        counts = np.bincount(patch[valid], minlength=5)
        if valid.mean() < 0.8 or min(counts[0], counts[class_id]) < minimum:
            continue
        score = min(counts[0], counts[class_id])
        if score > best_score:
            best_score = score
            best = (top, left, counts)
    return best


def select_subset(dataset, crop_size=64, minimum=128, max_scan=256):
    if dataset.split != "train" or dataset.augment:
        raise ValueError("Selection requires unaugmented official training data.")
    if crop_size > dataset.image_size or crop_size % 16 or crop_size < 16:
        raise ValueError("Crop must fit the preprocessed image and be divisible by 16.")
    images, targets, manifest = [], [], []
    missing = set(range(1, 5))
    for index in range(min(max_scan, len(dataset))):
        target = dataset.load_target(index)
        for class_id in sorted(missing):
            location = crop_for_class(target, class_id, crop_size, minimum)
            if location is None:
                continue
            top, left, counts = location
            image, exact_target = dataset[index]
            if not np.array_equal(target, exact_target.numpy()):
                raise ValueError("Selection and Dataset supervision disagree.")
            images.append(image[:, top:top + crop_size, left:left + crop_size].clone())
            targets.append(exact_target[top:top + crop_size, left:left + crop_size].clone())
            path = dataset.image_paths[index]
            manifest.append({"sample_id": path.stem, "split": "train",
                             "field_id": path.stem.split("_", 1)[0],
                             "training_index": index, "targeted_class": CLASSES[class_id],
                             "crop_xywh_after_resize": [left, top, crop_size, crop_size],
                             "pre_crop_image_size": dataset.image_size,
                             "class_pixel_counts": counts.tolist(),
                             "ignored_pixels": int((targets[-1] == IGNORE_INDEX).sum())})
            missing.remove(class_id)
        if not missing:
            break
    if missing:
        raise ValueError(f"No sufficiently visible crop found for {[CLASSES[i] for i in missing]} "
                         f"within {max_scan} training images. No training was started.")
    subset = FixedTrainingSubset(torch.stack(images), torch.stack(targets))
    total = np.bincount(subset.targets[subset.targets != IGNORE_INDEX].numpy(), minlength=5)
    if (total < minimum).any():
        raise ValueError("Selected subset does not contain enough supervision for all five classes.")
    return subset, manifest, index + 1


def hardware_report():
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
              "cuda_available": torch.cuda.is_available(), "gpus": []}
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        report["gpus"].append({"name": properties.name, "vram_bytes": properties.total_memory})
    try:
        import psutil  # Optional, already installed locally; never install automatically.
        memory = psutil.virtual_memory()
        report.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available)
    except ImportError:
        report["memory_note"] = "psutil unavailable; query system memory externally."
    return report


def plot_selection(subset, manifest, path):
    fig, axes = plt.subplots(len(subset), 2, figsize=(7, 3 * len(subset)), squeeze=False)
    for i, (image, target) in enumerate(subset):
        axes[i, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[i, 1].imshow(colorize(target.numpy()))
        axes[i, 0].set_title(manifest[i]["sample_id"], fontsize=9)
        axes[i, 1].set_title(manifest[i]["targeted_class"] + " crop target", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    handles = [Patch(color=c / 255, label=n) for c, n in zip(COLORS, CLASSES)]
    handles.append(Patch(color=IGNORE_COLOR / 255, label="Ignored"))
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty. Use --train-prepared for an existing "
                              "diagnostic or choose a new directory; never overwrite baseline artifacts.")
    root = Path(config["data_dir"]).resolve()
    if output.resolve() == root or root in output.resolve().parents:
        raise ValueError("Diagnostic output must be outside the dataset.")
    seed_everything(config["seed"])
    torch.set_num_threads(config["threads"])
    # This indexes train/val/test FILENAMES for leakage checks; only train pixels are opened.
    paths, fingerprint = index_splits(config["data_dir"], config["split_file"], EXPECTED_COUNTS)
    dataset = AgricultureVisionDataset(config["data_dir"], paths["train"], "train", augment=False)
    subset, manifest, scanned = select_subset(dataset, config["crop_size"],
                                               config["minimum_class_pixels"], config["max_scan"])
    weights, distribution = class_weights(subset)
    config.update(schema=SCHEMA, split_fingerprint=fingerprint,
                  training_targets_scanned=scanned, augmentation=False,
                  image_preprocessing="Baseline bilinear RGB / nearest target resize to 256, then FIXED crop",
                  class_weights_source="Selected training subset, existing inverse-sqrt/cap implementation",
                  initialization="Random seed 42; no baseline checkpoint or pretrained weights",
                  success_definition="Every class IoU >= threshold for consecutive measured checks",
                  purpose="Memorization diagnostic only; not a generalization experiment")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "configuration.json", config)
    save_json(output / "hardware.json", hardware_report())
    save_json(output / "selected_samples.json", manifest)
    save_json(output / "class_distribution.json", distribution)
    atomic_checkpoint(output / "selected_tensors.pt", {"images": subset.images, "targets": subset.targets,
                                                       "weights": weights})
    config["selected_tensors_sha256"] = hashlib.sha256((output / "selected_tensors.pt").read_bytes()).hexdigest()
    save_json(output / "configuration.json", config)
    samples = output / "selected_samples"
    samples.mkdir(exist_ok=True)
    for i, (image, target) in enumerate(subset):
        Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(samples / f"{i:02d}_rgb.png")
        Image.fromarray(target.numpy().astype(np.uint8)).save(samples / f"{i:02d}_target.png")
    plot_selection(subset, manifest, output / "selected_samples.png")
    print(f"Selected {len(subset)} fixed training crops after inspecting {scanned} training targets.", flush=True)
    print("All five classes present; selection figure:", output / "selected_samples.png", flush=True)


def train_prepared(output):
    output = Path(output)
    if (output / "checkpoint.pt").exists() or (output / "history.csv").exists():
        raise FileExistsError("Diagnostic training already exists; preserve it rather than overwrite.")
    config = json.loads((output / "configuration.json").read_text())
    if hashlib.sha256((output / "selected_tensors.pt").read_bytes()).hexdigest() != config["selected_tensors_sha256"]:
        raise ValueError("Prepared training tensors changed.")
    seed_everything(config["seed"])
    torch.set_num_threads(config["threads"])
    cached = torch.load(output / "selected_tensors.pt", map_location="cpu", weights_only=True)
    subset = FixedTrainingSubset(cached["images"], cached["targets"])
    device = choose_device(config["device"])
    images, targets = subset.images.to(device), subset.targets.to(device)
    weights = cached["weights"].to(device)
    # Exactly the existing baseline architecture; only spatial input size changes.
    model = UNet(base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # Identical weighted Cross-Entropy numerator/denominator to src.train.run_epoch.
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX, reduction="sum")
    valid = targets != IGNORE_INDEX
    denominator = weights[targets[valid]].sum()
    history, consecutive, best_worst_iou = [], 0, -1.0
    start = time.perf_counter()
    stop_reason = "step_budget_reached"
    best_metrics, best_step = None, 0

    def measure(step, elapsed, gradient_norm=None):
        model.eval()
        with torch.no_grad():
            logits = model(images)
            loss = float((criterion(logits, targets) / denominator).item())
            predictions = logits.argmax(1)
            metrics = metrics_from_confusion(confusion_matrix(predictions, targets))
        row = {"step": step, "elapsed_seconds": elapsed, "train_loss": loss,
               "mean_iou": metrics["mean_iou"], "min_class_iou": min(r["iou"] for r in metrics["per_class"]),
               "gradient_norm": gradient_norm}
        for r in metrics["per_class"]:
            row[f"class_{r['id']}_iou"] = r["iou"]
            row[f"class_{r['id']}_dice"] = r["dice"]
        history.append(row)
        save_csv(output / "history.csv", history)
        return metrics, row, predictions.cpu()

    initial, _, predictions = measure(0, 0.0)
    save_json(output / "initial_metrics.json", initial)
    manifest = json.loads((output / "selected_samples.json").read_text())
    for i in range(len(subset)):
        plot_sample(subset.images[i].numpy(), subset.targets[i].numpy(), predictions[i].numpy(),
                    output / f"initial_{i:02d}.png", "Before training: " + manifest[i]["sample_id"])
    step = 0
    for step in range(1, config["max_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets) / denominator
        if not torch.isfinite(loss):
            stop_reason = "nonfinite_loss"
            break
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = all(torch.isfinite(g).all() for g in gradients)
        gradient_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)).item())
        if not finite:
            stop_reason = "nonfinite_gradients"
            break
        optimizer.step()
        elapsed = time.perf_counter() - start
        deadline = elapsed >= config["max_seconds"]
        if step % config["check_every"] == 0 or step == 1 or step == config["max_steps"] or deadline:
            metrics, row, predictions = measure(step, elapsed, gradient_norm)
            worst = row["min_class_iou"]
            consecutive = consecutive + 1 if worst >= config["target_iou"] else 0
            print(f"Step {step}: loss={row['train_loss']:.5f}, mIoU={row['mean_iou']:.4f}, "
                  f"class IoU={[round(r['iou'], 4) for r in metrics['per_class']]}, "
                  f"elapsed={elapsed:.1f}s", flush=True)
            if worst > best_worst_iou:
                best_worst_iou, best_step, best_metrics = worst, step, metrics
                atomic_checkpoint(output / "checkpoint.pt", {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": config, "schema": SCHEMA,
                    "weights": weights.cpu(), "metrics": metrics, "rng": capture_rng({}),
                    "selection": "Highest minimum per-class IoU on the SAME training subset"})
                for i in range(len(subset)):
                    plot_sample(subset.images[i].numpy(), subset.targets[i].numpy(), predictions[i].numpy(),
                                output / f"prediction_{i:02d}.png", "Diagnostic training subset: " + manifest[i]["sample_id"])
                save_json(output / "per_class_metrics.json", metrics)
                save_csv(output / "per_class_metrics.csv", metrics["per_class"])
            if consecutive >= config["success_checks"]:
                stop_reason = "all_classes_memorized"
                break
            if deadline:
                stop_reason = "time_budget_reached"
                break
    if best_metrics is None:
        raise RuntimeError(f"No measured training result was produced: {stop_reason}")
    elapsed = time.perf_counter() - start
    passed = stop_reason == "all_classes_memorized"
    summary = {"passed": passed, "stop_reason": stop_reason, "steps_completed": step,
               "best_step": best_step, "elapsed_seconds": elapsed,
               "best_metrics": best_metrics, "initial_metrics": initial,
               "training_samples": len(subset), "device": str(device),
               "model_parameters": sum(p.numel() for p in model.parameters()),
               "validation_or_test_pixels_loaded": False,
               "threshold_iou": config["target_iou"], "required_consecutive_checks": config["success_checks"]}
    save_json(output / "summary.json", summary)
    lines = ["Small training-subset overfit diagnostic", "",
             f"Result: {'PASS' if passed else 'NOT PASSED within the configured budget'}",
             f"Stop reason: {stop_reason}; steps: {step}; selected checkpoint step: {best_step}",
             f"Unchanged U-Net: {summary['model_parameters']:,} parameters; device: {device}",
             "No baseline checkpoint, augmentation, validation, or test content was used.", ""]
    for row in best_metrics["per_class"]:
        lines.append(f"{row['class']}: IoU={row['iou']:.4f}, Dice={row['dice']:.4f}, pixels={row['support_pixels']}")
    lines += ["", "Interpretation: " + (
        "All five classes can be memorized on these fixed crops. This argues against a gross "
        "class-channel/loss/gradient failure, but does not establish full-image generalization or "
        "rule out every annotation issue. Validation-based patch-sampling investigation is reasonable."
        if passed else
        "This bounded run did not meet the all-class threshold. Do not infer a mask bug from this alone; "
        "inspect per-class learning curves, selected masks, gradients, and time/step budget before Experiment 2."),
        "These are training memorization metrics, not validation/test performance."]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["step"] for r in history], [r["train_loss"] for r in history])
    axes[0].set(xlabel="Optimizer step", ylabel="Same-subset weighted CE loss")
    for class_id, name in enumerate(CLASSES):
        axes[1].plot([r["step"] for r in history], [r[f"class_{class_id}_iou"] for r in history],
                     label=name, color=COLORS[class_id] / 255)
    axes[1].axhline(config["target_iou"], color="black", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="Optimizer step", ylabel="Same-subset IoU", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/data2017_miniscale")
    parser.add_argument("--split-file", default="data/raw/data2017_splits.json")
    parser.add_argument("--output-dir", default="results/overfit_diagnostic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--minimum-class-pixels", type=int, default=128)
    parser.add_argument("--max-scan", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--target-iou", type=float, default=0.9)
    parser.add_argument("--success-checks", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train-prepared", action="store_true")
    args = parser.parse_args()
    if args.train_prepared:
        train_prepared(args.output_dir)
        return
    config = vars(args).copy()
    for key in ("prepare_only", "train_prepared", "output_dir"):
        config.pop(key)
    config.update(seed=42, learning_rate=0.001, base_channels=32)
    if any(config[k] <= 0 for k in ("threads", "minimum_class_pixels", "max_scan", "max_steps",
                                     "max_seconds", "check_every", "success_checks")):
        parser.error("Diagnostic budgets and sizes must be positive.")
    if not 0 < config["target_iou"] <= 1:
        parser.error("target-iou must be in (0, 1].")
    prepare(config, args.output_dir)
    if not args.prepare_only:
        train_prepared(args.output_dir)


if __name__ == "__main__":
    main()

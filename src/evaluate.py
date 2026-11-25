"""One final test pass with metrics and qualitative examples from the same pass."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .dataset import AgricultureVisionDataset, check_batch, index_splits, make_loader
from .model import UNet
from .train import SCHEMA
from .utils import (choose_device, confusion_matrix, load_checkpoint, metrics_from_confusion,
                    plot_confusion, plot_sample, save_csv, save_json, seed_everything)


def evaluate(checkpoint, output_dir=None, data_dir=None, split_file=None,
             device="auto", num_workers=None, num_visualizations=6):
    checkpoint = Path(checkpoint)
    output = Path(output_dir) if output_dir else checkpoint.parent / "test"
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Test results already exist: {output}. Reuse them; do not retune on test.")
    state = load_checkpoint(checkpoint)
    if state.get("resume_capable", True):
        raise ValueError("Final evaluation requires the validation-selected best.pt, not last.pt.")
    if state["schema"] != SCHEMA:
        raise ValueError("Checkpoint class/target schema is incompatible.")
    config = state["config"]
    data_dir = data_dir or config["data_dir"]
    split_file = split_file or config["split_file"]
    resolved_output, root = output.resolve(), Path(data_dir).resolve()
    if resolved_output == root or root in resolved_output.parents:
        raise ValueError("Evaluation outputs must be outside the input dataset.")
    if str(resolved_output).replace("\\", "/").startswith("/kaggle/input/"):
        raise ValueError("Save Kaggle outputs under /kaggle/working/.")
    if num_visualizations < 0:
        raise ValueError("num_visualizations must be nonnegative.")
    seed_everything(config["seed"])
    device = choose_device(device)
    paths, fingerprint = index_splits(data_dir, split_file, config["expected_counts"])
    if fingerprint != state["split_fingerprint"]:
        raise ValueError("Official splits/sample inventory differ from the selected checkpoint.")
    dataset = AgricultureVisionDataset(data_dir, paths["test"], "test", config["image_size"],
                                       config["native_size"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = make_loader(dataset, config["batch_size"],
                         config["num_workers"] if num_workers is None else num_workers,
                         generator, device.type == "cuda")
    model = UNet(config["base_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    matrix = torch.zeros((5, 5), dtype=torch.int64)
    samples, planter_sample, offset = [], None, 0
    # Cache evenly spaced examples plus the first Planter Skip example without a second pass.
    indices = set(np.linspace(0, len(dataset) - 1, min(num_visualizations, len(dataset)), dtype=int))
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Final official test evaluation"):
            check_batch(images, targets, config["image_size"])
            predictions = model(images.to(device, non_blocking=True)).argmax(1).cpu()
            matrix += confusion_matrix(predictions, targets)
            for i in range(len(images)):
                sample_id = dataset.image_paths[offset + i].stem
                if offset + i in indices:
                    samples.append((sample_id, images[i].numpy(), targets[i].numpy(), predictions[i].numpy()))
                if planter_sample is None and (targets[i] == 4).any():
                    planter_sample = (sample_id, images[i].numpy(), targets[i].numpy(), predictions[i].numpy())
            offset += len(images)
    metrics = metrics_from_confusion(matrix)
    if not metrics["valid_pixels"]:
        raise ValueError("Test split has no supervised pixels.")
    metrics.update({"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "selected_epoch": state["epoch"], "best_validation_mean_iou": state["best_score"],
                    "split": "test", "split_fingerprint": fingerprint,
                    "image_size": config["image_size"], "ignore_index": 255,
                    "device": str(device), "torch_version": torch.__version__})
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "per_class_metrics.csv", metrics["per_class"])
    np.savetxt(output / "confusion_matrix.csv", matrix.numpy(), fmt="%d", delimiter=",")
    plot_confusion(matrix.numpy(), output / "confusion_matrix.png")
    manifest = []
    if planter_sample is not None and planter_sample[0] not in {sample[0] for sample in samples}:
        samples.append(planter_sample)
    for sample_id, image, target, prediction in samples:
        path = output / f"sample_{sample_id}.png"
        plot_sample(image, target, prediction, path, sample_id)
        manifest.append({"sample_id": sample_id, "figure": path.name,
                         "contains_planter_skip": bool((target == 4).any())})
    save_json(output / "visualizations.json", manifest)
    # Written last so an interrupted artifact export can be retried explicitly.
    save_json(output / "metrics.json", metrics)
    print("Class                 IoU       Dice       Support")
    for row in metrics["per_class"]:
        iou = "undefined" if row["iou"] is None else f"{row['iou']:.5f}"
        dice = "undefined" if row["dice"] is None else f"{row['dice']:.5f}"
        print(f"{row['class']:20s} {iou:10s} {dice:10s} {row['support_pixels']:,}")
    print(f"Mean IoU: {metrics['mean_iou']:.5f}; pixel accuracy: {metrics['pixel_accuracy']:.5f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir, args.data_dir, args.split_file,
             args.device, args.num_workers)


if __name__ == "__main__":
    main()

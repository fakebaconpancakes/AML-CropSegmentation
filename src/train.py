"""Plain PyTorch training. Test pixels are never accessed here."""

import argparse
import copy
import json
import platform
from pathlib import Path

import numpy as np
import PIL
import torch
from torch import nn
from tqdm.auto import tqdm

from .dataset import (AgricultureVisionDataset, CLASSES, SELECTED, EXCLUDED,
                      IGNORE_INDEX, check_batch, index_splits, make_loader)
from .model import UNet
from .utils import (atomic_checkpoint, capture_rng, choose_device, confusion_matrix,
                    load_checkpoint, metrics_from_confusion, plot_history, plot_sample,
                    restore_rng, save_csv, save_json, seed_everything)

SCHEMA = {"classes": list(CLASSES), "selected": list(SELECTED), "excluded": list(EXCLUDED),
          "ignore_index": IGNORE_INDEX, "target_version": 1,
          "validity": "field_masks==255 AND field_bounds==255"}


def prepare_datasets(config):
    paths, fingerprint = index_splits(config["data_dir"], config["split_file"],
                                       config["expected_counts"])
    print("Official image counts:", {k: len(v) for k, v in paths.items()})
    datasets = {
        split: AgricultureVisionDataset(config["data_dir"], paths[split], split,
                                        config["image_size"], config["native_size"],
                                        augment=(split == "train"))
        for split in ("train", "val")
    }
    return datasets, fingerprint


def class_weights(dataset, cap=3.0):
    if dataset.split != "train":
        raise ValueError("Class frequencies must come exclusively from the training split.")
    if cap <= 0:
        raise ValueError("Weight cap must be positive.")
    counts = np.zeros(5, dtype=np.int64)
    ignored = 0
    for index in tqdm(range(len(dataset)), desc="Training pixel distribution"):
        target = dataset.load_target(index)
        valid = target != IGNORE_INDEX
        counts += np.bincount(target[valid], minlength=5)
        ignored += int((~valid).sum())
    if (counts == 0).any():
        missing = [CLASSES[i] for i in np.where(counts == 0)[0]]
        raise ValueError(f"No supervised training pixels for {missing}. Check masks and resizing.")
    frequencies = counts / counts.sum()
    inverse = 1.0 / np.sqrt(frequencies)
    weights = np.minimum(inverse / inverse.mean(), cap).astype(np.float32)
    distribution = {"counts": counts.tolist(), "frequencies": frequencies.tolist(),
                    "ignored_pixels": ignored, "weights": weights.tolist(),
                    "source_split": "train", "image_size": dataset.image_size,
                    "weight_cap": cap}
    print("Class | pixels | frequency | weight")
    for name, count, freq, weight in zip(CLASSES, counts, frequencies, weights):
        print(f"{name:16s} {count:12,d} {freq:.6f} {weight:.4f}")
    print(f"Ignored training pixels: {ignored:,}")
    return torch.tensor(weights), distribution


def run_epoch(model, loader, weights, device, optimizer=None, collect_metrics=True):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX, reduction="sum")
    total_loss, total_weight, skipped = 0.0, 0.0, 0
    matrix = torch.zeros((5, 5), dtype=torch.int64)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, desc="Train" if training else "Validate", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = targets != IGNORE_INDEX
            if not valid.any():
                skipped += 1
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss_sum = criterion(logits, targets)
            denominator = weights[targets[valid]].sum()
            loss = loss_sum / denominator
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; inspect inputs, logits, and class weights.")
            if training:
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError("Nonfinite gradients.")
                optimizer.step()
            total_loss += loss_sum.detach().item()
            total_weight += denominator.item()
            if collect_metrics:
                matrix += confusion_matrix(logits.detach().argmax(1), targets)
    if total_weight == 0:
        raise ValueError("Epoch has no supervised pixels; all targets are ignored.")
    return total_loss / total_weight, matrix, skipped


def compatible_config(current, saved):
    # Paths may change when attaching a saved run to a fresh Kaggle session.
    mutable = {"data_dir", "split_file", "output_dir", "epochs", "device"}
    if {k: v for k, v in current.items() if k not in mutable} != {
            k: v for k, v in saved.items() if k not in mutable}:
        raise ValueError("Resume configuration changed training/preprocessing settings.")


def validate_config(config):
    for key in ("image_size", "native_size", "base_channels", "batch_size", "epochs",
                "early_stopping_patience"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if config["image_size"] % 16:
        raise ValueError("image_size must be divisible by 16.")
    if config["num_workers"] < 0 or config["learning_rate"] <= 0 or config["weight_cap"] <= 0:
        raise ValueError("Invalid worker count, learning rate, or weight cap.")
    if not 0 < config["scheduler_factor"] < 1 or config["scheduler_patience"] < 0:
        raise ValueError("Invalid scheduler factor or patience.")
    output = Path(config["output_dir"]).resolve()
    root = Path(config["data_dir"]).resolve()
    if output == root or root in output.parents:
        raise ValueError("Outputs must not be written inside the input dataset.")
    if str(output).replace("\\", "/").startswith("/kaggle/input/"):
        raise ValueError("Kaggle input is read-only; save under /kaggle/working/.")


def train(config, resume=None):
    """Run or resume at epoch boundaries; return paths to best and latest checkpoints."""
    config = copy.deepcopy(config)
    validate_config(config)
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output / name).exists() for name in ("last.pt", "best.pt", "history.csv")):
        raise FileExistsError("Run already exists. Resume last.pt or choose a new output directory.")
    datasets, fingerprint = prepare_datasets(config)
    generators = {split: torch.Generator().manual_seed(config["seed"] + i)
                  for i, split in enumerate(("train", "val"))}
    loaders = {split: make_loader(ds, config["batch_size"], config["num_workers"],
                                 generators[split], device.type == "cuda")
               for split, ds in datasets.items()}
    # Inspect a separate deterministic sample, without advancing loader or augmentation RNGs.
    sanity = AgricultureVisionDataset(config["data_dir"], datasets["train"].image_paths[:1],
                                      "train", config["image_size"], config["native_size"])
    image, target = sanity[0]
    check_batch(image[None], target[None], config["image_size"])
    print("Sanity shapes:", tuple(image[None].shape), tuple(target[None].shape),
          "labels:", target.unique().tolist())
    plot_sample(image.numpy(), target.numpy(), target.numpy(), output / "training_sanity.png",
                "Training sample alignment (prediction panels repeat target)")
    model = UNet(config["base_channels"]).to(device)
    print(f"Device: {device}; model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["scheduler_factor"], patience=config["scheduler_patience"])
    history, start_epoch, best_score, bad_epochs = [], 0, -1.0, 0
    best_model, best_epoch = None, 0
    if resume:
        state = load_checkpoint(resume)
        if not state.get("resume_capable", False):
            raise ValueError("Resume training from last.pt; best.pt is for final evaluation.")
        compatible_config(config, state["config"])
        if state["schema"] != SCHEMA or state["split_fingerprint"] != fingerprint:
            raise ValueError("Resume class semantics or official split identity changed.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer.load_state_dict moves parameter state to the parameter device.
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start_epoch, best_score = state["epoch"], state["best_score"]
        bad_epochs = state["bad_epochs"]
        best_model, best_epoch = state["best_model"], state["best_epoch"]
        weights = state["weights"].to(device)
        distribution = state["distribution"]
        if config["epochs"] < start_epoch:
            raise ValueError("epochs is a total budget and cannot precede the resumed epoch.")
        restore_rng(state["rng"], generators)
        print(f"Resuming after epoch {start_epoch}; best validation mIoU={best_score:.6f}")
    else:
        weights, distribution = class_weights(datasets["train"], config["weight_cap"])
        weights = weights.to(device)
    save_json(output / "config.json", config)
    save_json(output / "class_distribution.json", distribution)
    import matplotlib as mpl
    import tqdm as tqdm_package
    save_json(output / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "pillow": PIL.__version__, "matplotlib": mpl.__version__, "tqdm": tqdm_package.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda, "deterministic_algorithms": True,
        "split_fingerprint": fingerprint, "schema": SCHEMA})

    def checkpoint(epoch):
        return {"format_version": 1, "resume_capable": True,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_score": best_score, "bad_epochs": bad_epochs,
                "best_model": best_model, "best_epoch": best_epoch,
                "weights": weights.detach().cpu(), "distribution": distribution,
                "config": config, "schema": SCHEMA, "split_fingerprint": fingerprint,
                "history": history, "rng": capture_rng(generators)}

    def save_best():
        # Embed the selected weights in last.pt so resume into a new directory needs only one file.
        atomic_checkpoint(output / "best.pt", {
            "format_version": 1, "resume_capable": False,
            "model": best_model, "epoch": best_epoch, "best_score": best_score,
            "config": config, "schema": SCHEMA, "split_fingerprint": fingerprint,
            "weights": weights.detach().cpu(), "distribution": distribution})

    if best_model is not None:
        save_best()
    if bad_epochs >= config["early_stopping_patience"]:
        print("Early stopping was already reached in this checkpoint; no additional training.")
    else:
        for epoch in range(start_epoch + 1, config["epochs"] + 1):
            used_lr = optimizer.param_groups[0]["lr"]
            train_loss, _, skipped_train = run_epoch(model, loaders["train"], weights, device,
                                                    optimizer, collect_metrics=False)
            val_loss, matrix, skipped_val = run_epoch(model, loaders["val"], weights, device)
            score = metrics_from_confusion(matrix)["mean_iou"]
            improved = score > best_score
            if improved:
                best_score, bad_epochs, best_epoch = score, 0, epoch
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
            scheduler.step(score)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_mean_iou": score, "learning_rate": used_lr,
                   "next_learning_rate": optimizer.param_groups[0]["lr"],
                   "skipped_train_batches": skipped_train, "skipped_val_batches": skipped_val}
            history.append(row)
            atomic_checkpoint(output / "last.pt", checkpoint(epoch))
            if improved:
                save_best()
            save_csv(output / "history.csv", history)
            plot_history(history, output / "learning_curves.png")
            print(f"Epoch {epoch}/{config['epochs']}: train={train_loss:.5f}, val={val_loss:.5f}, "
                  f"val mIoU={score:.5f}, lr={used_lr:.6g}")
            if bad_epochs >= config["early_stopping_patience"]:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break
    # Also materialize the full resumed state if the requested epoch budget was already complete.
    if start_epoch and len(history) == start_epoch:
        atomic_checkpoint(output / "last.pt", checkpoint(start_epoch))
        save_csv(output / "history.csv", history)
        plot_history(history, output / "learning_curves.png")
    if not (output / "best.pt").exists():
        raise RuntimeError("No best checkpoint was produced.")
    return {"best": output / "best.pt", "last": output / "last.pt", "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    train(config, args.resume)


if __name__ == "__main__":
    main()

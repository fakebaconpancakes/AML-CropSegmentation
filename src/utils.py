"""Small reproducibility, metric, checkpoint, and plotting helpers."""

import csv
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from .dataset import CLASSES, IGNORE_INDEX

COLORS = np.array([[65, 140, 65], [225, 170, 45], [60, 145, 215],
                   [180, 80, 195], [235, 75, 65]], dtype=np.uint8)
IGNORE_COLOR = np.array([125, 125, 125], dtype=np.uint8)


def seed_everything(seed):
    # Set before initializing CUDA. Repeatability is scoped to hardware/software versions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Enable a Kaggle GPU accelerator.")
    return device


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # These project-generated checkpoints contain Python/NumPy RNG states.
    # Only load checkpoints you created or otherwise trust.
    return torch.load(path, map_location="cpu", weights_only=False)


def capture_rng(generators):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loaders": {k: g.get_state() for k, g in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


def confusion_matrix(predictions, targets):
    """Return CPU int64 matrix: rows=true, columns=predicted; ignore target 255."""
    valid = targets != IGNORE_INDEX
    true, pred = targets[valid].long(), predictions[valid].long()
    if ((true < 0) | (true >= 5) | (pred < 0) | (pred >= 5)).any():
        raise ValueError("Metrics received labels outside 0..4/ignore.")
    return torch.bincount(5 * true + pred, minlength=25).reshape(5, 5).cpu()


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = support + predicted - tp
    dice_denominator = support + predicted
    iou = [float(t / d) if d else None for t, d in zip(tp, union)]
    dice = [float(2 * t / d) if d else None for t, d in zip(tp, dice_denominator)]
    defined = [x for x in iou if x is not None]
    anomalies = [x for x in iou[1:] if x is not None]
    return {
        "per_class": [{"id": i, "class": name, "iou": iou[i], "dice": dice[i],
                       "support_pixels": int(support[i])} for i, name in enumerate(CLASSES)],
        "mean_iou": float(np.mean(defined)) if defined else None,
        "anomaly_mean_iou": float(np.mean(anomalies)) if anomalies else None,
        "mean_iou_classes": [CLASSES[i] for i, value in enumerate(iou) if value is not None],
        "pixel_accuracy": float(tp.sum() / support.sum()) if support.sum() else None,
        "valid_pixels": int(support.sum()), "confusion_matrix": matrix.tolist(),
    }


def colorize(target):
    target = np.asarray(target)
    result = np.empty((*target.shape, 3), dtype=np.uint8)
    result[:] = IGNORE_COLOR
    for i, color in enumerate(COLORS):
        result[target == i] = color
    return result


def plot_sample(image, target, prediction, path, title=""):
    image = np.asarray(image)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    target, prediction = np.asarray(target), np.asarray(prediction)
    shown_prediction = prediction.copy()
    shown_prediction[target == IGNORE_INDEX] = IGNORE_INDEX
    overlay = image.copy()
    valid = target != IGNORE_INDEX
    overlay[valid] = 0.6 * image[valid] + 0.4 * colorize(prediction)[valid] / 255.0
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, pixels, label in zip(axes, [image, colorize(target), colorize(shown_prediction), overlay],
                                 ["RGB", "Ground truth", "Prediction (argmax)", "Prediction overlay"]):
        ax.imshow(pixels)
        ax.set_title(label)
        ax.axis("off")
    handles = [Patch(color=color / 255, label=name) for color, name in zip(COLORS, CLASSES)]
    handles.append(Patch(color=IGNORE_COLOR / 255, label="Ignored"))
    fig.legend(handles=handles, loc="lower center", ncol=6)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    for name in ("train_loss", "val_loss"):
        axes[0].plot(epochs, [r[name] for r in history], label=name)
    axes[0].set_ylabel("Weighted cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_mean_iou"] for r in history])
    axes[1].set_ylabel("Validation mean IoU")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(matrix, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="Valid pixels")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

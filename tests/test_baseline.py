"""Critical semantic checks and a small CPU run; no real test pixels are read."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.dataset import (AgricultureVisionDataset, SELECTED, EXCLUDED, augment_pair,
                         check_batch, construct_target, index_splits)
from src.evaluate import evaluate
from src.model import UNet
from src.train import class_weights, run_epoch, train
from src.utils import (capture_rng, confusion_matrix, load_checkpoint,
                       metrics_from_confusion, restore_rng, seed_everything)


def synthetic_dataset(folder):
    """Dense binary masks with all five classes, a shared class overlap, and invalid pixels."""
    root = Path(folder) / "data"
    for category in SELECTED + EXCLUDED:
        (root / "field_labels" / category).mkdir(parents=True)
    for directory in ("field_images/rgb", "field_masks", "field_bounds"):
        (root / directory).mkdir(parents=True)
    ids = ["T_0", "T_1", "V_0", "E_0"]
    for image_id in ids:
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 8
        pixels[:, :, 1] = 100
        Image.fromarray(pixels).save(root / "field_images/rgb" / (image_id + ".jpg"))
        valid = np.full((32, 32), 255, dtype=np.uint8)
        valid[:2] = 0
        Image.fromarray(valid).save(root / "field_masks" / (image_id + ".png"))
        bounds = np.full((32, 32), 255, dtype=np.uint8)
        bounds[:, :2] = 0
        Image.fromarray(bounds).save(root / "field_bounds" / (image_id + ".png"))
        for i, category in enumerate(SELECTED + EXCLUDED):
            mask = np.zeros((32, 32), dtype=np.uint8)
            if i < 4:
                mask[:, 6 * (i + 1):6 * (i + 2)] = 255
            if category == "endrow":
                mask[2:4, :] = 255
            Image.fromarray(mask).save(root / "field_labels" / category / (image_id + ".png"))
    split_file = Path(folder) / "splits.json"
    split_file.write_text(json.dumps({"train": ["T"], "val": ["V"], "test": ["E"]}))
    return root, split_file


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ignore_priority(self):
        selected = np.zeros((4, 2, 5), dtype=bool)
        excluded = np.zeros((5, 2, 5), dtype=bool)
        valid = np.ones((2, 5), dtype=bool)
        for i in range(4):
            selected[i, 0, i + 1] = True
        selected[0, 1, 0:3] = True
        selected[1, 1, 1] = True  # selected/selected overlap
        excluded[0, 1, 0] = True  # selected/excluded overlap
        valid[1, 2] = False
        excluded[4, 1, 3] = True  # excluded-only
        result = construct_target(selected, excluded, valid)
        np.testing.assert_array_equal(result, [[0, 1, 2, 3, 4], [255, 255, 255, 255, 0]])

    def test_synchronized_augmentation_all_grid_transforms(self):
        target = np.array([[0, 1, 2], [3, 4, 255], [2, 1, 0]], dtype=np.int64)
        image = np.repeat(target[:, :, None], 3, axis=2)
        for horizontal in (False, True):
            for vertical in (False, True):
                for turns in range(4):
                    with patch("src.dataset.random.random", side_effect=[0.0 if horizontal else 1.0,
                                                                           0.0 if vertical else 1.0]), \
                         patch("src.dataset.random.randrange", return_value=turns):
                        transformed_image, transformed_target = augment_pair(image, target)
                    np.testing.assert_array_equal(transformed_image[:, :, 0], transformed_target)
                    self.assertEqual(set(np.unique(transformed_target)), {0, 1, 2, 3, 4, 255})

    def test_splits_and_missing_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            root, split_file = synthetic_dataset(folder)
            paths, _ = index_splits(root, split_file, {"train": 2, "val": 1, "test": 1})
            with self.assertRaisesRegex(ValueError, "Expected image counts"):
                index_splits(root, split_file, {"train": 3, "val": 1, "test": 1})
            split_file.write_text(json.dumps({"train": ["T"], "val": ["T"], "test": ["E"]}))
            with self.assertRaisesRegex(ValueError, "Repeated field"):
                index_splits(root, split_file)
            split_file.write_text(json.dumps({"train": ["T", "T"], "val": ["V"], "test": ["E"]}))
            with self.assertRaisesRegex(ValueError, "Repeated field"):
                index_splits(root, split_file)
            split_file.write_text(json.dumps({"train": ["OTHER"], "val": ["V"], "test": ["E"]}))
            with self.assertRaisesRegex(ValueError, "no official field"):
                index_splits(root, split_file)
            (root / "field_labels/drydown/T_0.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing masks"):
                AgricultureVisionDataset(root, paths["train"], "train", 16, 32)

    def test_dataset_resizing_weights_and_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            root, split_file = synthetic_dataset(folder)
            paths, _ = index_splits(root, split_file)
            dataset = AgricultureVisionDataset(root, paths["train"], "train", 16, 32)
            image, target = dataset[0]
            check_batch(image[None], target[None], 16)
            self.assertEqual(set(target.unique().tolist()), {0, 1, 2, 3, 4, 255})
            # Top and left pixels remain ignored after nearest-neighbor resizing.
            self.assertTrue((target[0] == 255).all())
            self.assertTrue((target[:, 0] == 255).all())
            weights, info = class_weights(dataset, cap=1.1)
            expected = sum(np.bincount(dataset.load_target(i)[dataset.load_target(i) != 255],
                                       minlength=5) for i in range(len(dataset)))
            self.assertEqual(info["counts"], expected.tolist())
            self.assertLessEqual(float(weights.max()), 1.1 + 1e-6)
            self.assertTrue((weights > 0).all())
            val = AgricultureVisionDataset(root, paths["val"], "val", 16, 32)
            with self.assertRaisesRegex(ValueError, "exclusively"):
                class_weights(val)
            with self.assertRaisesRegex(ValueError, "Only training"):
                AgricultureVisionDataset(root, paths["val"], "val", 16, 32, augment=True)
            # A class erased from training must produce a clear error.
            for path in (root / "field_labels/planter_skip").glob("T_*.png"):
                Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "No supervised training pixels"):
                class_weights(dataset)

    def test_metrics_known_values_and_ignore(self):
        target = torch.tensor([[0, 0, 1, 1, 255]])
        prediction = torch.tensor([[0, 1, 1, 1, 4]])
        matrix = confusion_matrix(prediction, target)
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(int(matrix.sum()), 4)
        stats = metrics_from_confusion(matrix)
        self.assertAlmostEqual(stats["per_class"][0]["iou"], 1 / 2)
        self.assertAlmostEqual(stats["per_class"][1]["iou"], 2 / 3)
        self.assertAlmostEqual(stats["per_class"][1]["dice"], 4 / 5)
        self.assertAlmostEqual(stats["pixel_accuracy"], 3 / 4)
        self.assertIsNone(stats["per_class"][4]["iou"])
        self.assertEqual(stats["mean_iou_classes"], ["Normal farmland", "Drydown"])
        split_matrix = (confusion_matrix(prediction[:, :2], target[:, :2])
                        + confusion_matrix(prediction[:, 2:], target[:, 2:]))
        torch.testing.assert_close(matrix, split_matrix)
        false_positive = metrics_from_confusion(confusion_matrix(torch.tensor([4]), torch.tensor([0])))
        self.assertEqual(false_positive["per_class"][4]["iou"], 0.0)
        self.assertIsNone(metrics_from_confusion(torch.zeros(5, 5))["mean_iou"])

    def test_model_loss_and_all_ignored(self):
        model = UNet(base_channels=2)
        images = torch.rand(2, 3, 32, 32)
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 0] = 255
        output = model(images)
        self.assertEqual(output.shape, (2, 5, 32, 32))
        output.retain_grad()
        loss = torch.nn.functional.cross_entropy(output, target, ignore_index=255,
                                                 weight=torch.arange(1, 6).float())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(output.grad[:, :, 0].abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "no supervised pixels"):
            run_epoch(model, [(images, torch.full_like(target, 255))], torch.ones(5), "cpu")

    def test_rng_roundtrip(self):
        seed_everything(42)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        generators = {"train": torch.Generator().manual_seed(123)}
        state = capture_rng(generators)
        expected = (random.random(), np.random.rand(), torch.rand(1),
                    torch.rand(1, generator=generators["train"]))
        restore_rng(state, generators)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2])
        torch.testing.assert_close(torch.rand(1, generator=generators["train"]), expected[3])

    def test_scheduler_early_stop_and_resume_stays_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            root, split_file = synthetic_dataset(folder)
            config = json.loads(Path("configs/baseline.json").read_text())
            config.update(data_dir=str(root), split_file=str(split_file), image_size=32,
                          native_size=32, base_channels=2, batch_size=1, num_workers=0,
                          epochs=5, device="cpu", expected_counts={"train": 2, "val": 1, "test": 1},
                          early_stopping_patience=1, scheduler_patience=0,
                          output_dir=str(Path(folder) / "stopping"))
            fixed_matrix = torch.eye(5, dtype=torch.int64)
            with patch("src.train.run_epoch", return_value=(1.0, fixed_matrix, 0)):
                result = train(config)
            state = load_checkpoint(result["last"])
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["best_epoch"], 1)
            self.assertEqual(state["bad_epochs"], 1)
            self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], 0.0005)
            config["output_dir"] = str(Path(folder) / "stopped_resume")
            with patch("src.train.run_epoch") as epoch:
                continued = train(config, resume=result["last"])
                epoch.assert_not_called()
            self.assertEqual(load_checkpoint(continued["last"])["epoch"], 2)
            self.assertEqual(load_checkpoint(continued["best"])["epoch"], 1)

    def test_training_resume_and_final_evaluation(self):
        self._check_training_resume(num_workers=0)

    def test_training_resume_with_two_workers(self):
        self._check_training_resume(num_workers=2)

    def _check_training_resume(self, num_workers):
        with tempfile.TemporaryDirectory() as folder:
            root, split_file = synthetic_dataset(folder)
            config = json.loads(Path("configs/baseline.json").read_text())
            config.update(data_dir=str(root), split_file=str(split_file), image_size=32,
                          native_size=32, base_channels=2, batch_size=1, num_workers=num_workers,
                          epochs=2, device="cpu", expected_counts={"train": 2, "val": 1, "test": 1},
                          output_dir=str(Path(folder) / "continuous"))
            continuous = train(config)
            first = copy.deepcopy(config)
            first.update(epochs=1, output_dir=str(Path(folder) / "first"))
            initial = train(first)
            resumed_config = copy.deepcopy(config)
            resumed_config["output_dir"] = str(Path(folder) / "resumed")
            resumed = train(resumed_config, resume=initial["last"])
            full_state, resumed_state = load_checkpoint(continuous["last"]), load_checkpoint(resumed["last"])
            self.assertEqual(full_state["history"], resumed_state["history"])
            for key in full_state["model"]:
                torch.testing.assert_close(full_state["model"][key], resumed_state["model"][key], rtol=0, atol=0)
            for key in ("best_score", "best_epoch", "bad_epochs", "scheduler"):
                self.assertEqual(full_state[key], resumed_state[key])
            for param_id, value in full_state["optimizer"]["state"].items():
                for key, tensor in value.items():
                    torch.testing.assert_close(tensor, resumed_state["optimizer"]["state"][param_id][key],
                                               rtol=0, atol=0)
            metrics = evaluate(resumed["best"], device="cpu", num_visualizations=1)
            self.assertGreater(metrics["valid_pixels"], 0)
            out = resumed["best"].parent / "test"
            for name in ("metrics.json", "per_class_metrics.csv", "confusion_matrix.png",
                         "confusion_matrix.csv", "visualizations.json"):
                self.assertTrue((out / name).exists())
            self.assertTrue(list(out.glob("sample_*.png")))
            with self.assertRaises(FileExistsError):
                evaluate(resumed["best"], device="cpu")
            with self.assertRaisesRegex(ValueError, "best.pt"):
                evaluate(resumed["last"], output_dir=Path(folder) / "wrong_checkpoint")
            changed = copy.deepcopy(resumed_config)
            changed["image_size"] = 16
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                train(changed, resume=resumed["last"])
            with self.assertRaises(FileExistsError):
                train(config)


if __name__ == "__main__":
    unittest.main()

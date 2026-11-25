"""Selection and loss checks only; do not launch another diagnostic training run."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.overfit_diagnostic import crop_for_class, prepare, select_subset


class ToyTrainingData:
    split = "train"
    augment = False
    image_size = 64
    image_paths = [Path(f"TRAIN{i}_tile.jpg") for i in range(4)]

    def __len__(self):
        return 4

    def load_target(self, index):
        target = np.zeros((64, 64), dtype=np.int64)
        target[16:48, 24:56] = index + 1
        target[:8] = 255
        return target

    def __getitem__(self, index):
        target = self.load_target(index)
        image = np.where(target == 255, 0, target).astype(np.float32) / 4
        return torch.from_numpy(np.repeat(image[None], 3, axis=0)), torch.from_numpy(target)


class DiagnosticTests(unittest.TestCase):
    def test_existing_baseline_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "config.json"
            sentinel.write_text('{"baseline": true}')
            with self.assertRaisesRegex(FileExistsError, "never overwrite baseline"):
                prepare({}, folder)
            self.assertEqual(sentinel.read_text(), '{"baseline": true}')

    def test_fixed_crops_retain_exact_supervision_and_all_classes(self):
        data = ToyTrainingData()
        subset, manifest, scanned = select_subset(data, crop_size=32, minimum=32, max_scan=4)
        self.assertEqual(len(subset), 4)
        self.assertEqual(scanned, 4)
        self.assertFalse(subset.augment)
        counts = np.bincount(subset.targets[subset.targets != 255].numpy(), minlength=5)
        self.assertTrue((counts >= 32).all())
        for i, item in enumerate(manifest):
            x, y, w, h = item["crop_xywh_after_resize"]
            image, target = data[item["training_index"]]
            torch.testing.assert_close(subset.targets[i], target[y:y+h, x:x+w])
            torch.testing.assert_close(subset.images[i], image[:, y:y+h, x:x+w])
            self.assertEqual(item["split"], "train")
        again, second_manifest, _ = select_subset(data, crop_size=32, minimum=32, max_scan=4)
        self.assertEqual(manifest, second_manifest)
        torch.testing.assert_close(subset.images, again.images)

    def test_selection_rejects_validation_augmentation_and_missing_class(self):
        data = ToyTrainingData()
        data.split = "val"
        with self.assertRaisesRegex(ValueError, "official training"):
            select_subset(data)
        data.split, data.augment = "train", True
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            select_subset(data)
        data.augment = False
        with self.assertRaisesRegex(ValueError, "Planter skip"):
            select_subset(data, crop_size=32, minimum=32, max_scan=3)

    def test_ignore_regions_cannot_supply_anomaly_support(self):
        target = np.full((64, 64), 255, dtype=np.int64)
        target[:8] = 4
        self.assertIsNone(crop_for_class(target, 4, 32, 32))

    def test_diagnostic_loss_matches_baseline_and_ignores_255(self):
        generator = torch.Generator().manual_seed(42)
        logits = torch.rand(1, 5, 4, 4, generator=generator, requires_grad=True)
        target = torch.tensor([[[0, 1, 2, 3], [4, 255, 0, 1], [2, 3, 4, 255], [0, 1, 2, 3]]])
        weights = torch.tensor([0.4, 0.8, 1.0, 1.8, 0.9])
        valid = target != 255
        diagnostic = torch.nn.functional.cross_entropy(logits, target, weight=weights,
                                                       ignore_index=255, reduction="sum")
        diagnostic = diagnostic / weights[target[valid]].sum()
        baseline = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=255)
        torch.testing.assert_close(diagnostic, baseline)
        diagnostic.backward()
        self.assertTrue((logits.grad.permute(0, 2, 3, 1)[~valid] == 0).all())
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()

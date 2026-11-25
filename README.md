# Multi-Class Agricultural Anomaly Segmentation

Supervised RGB semantic segmentation for Agriculture-Vision 2017 miniscale using a standard randomly initialized U-Net. No reconstruction components or pretrained weights.

## Project layout

- `src/dataset.py`: official field splits, priority-safe targets, synchronized augmentation.
- `src/model.py`: U-Net with double convolutions, max pooling, transposed convolutions, and skip connections; base width 32.
- `src/train.py`: training-only class weights, training/validation, resumable checkpoints.
- `src/evaluate.py`: final test metrics and visualizations in one inference pass.
- `src/utils.py`: reproducibility, metrics, checkpoint and plot helpers.
- `configs/baseline.json`: explicit baseline settings.
- `notebooks/kaggle_baseline.ipynb`: sequential fresh-session Kaggle workflow.
- `scripts/`: preserved legacy split/filename diagnostics and `validate_dataset.py` for training-only data validation. No dependency on `old_project/`.
- `tests/test_baseline.py`: synthetic correctness and integration checks.

## Kaggle Run All

1. Upload `src/`, `configs/`, and `requirements.txt` together as a Kaggle code dataset, retaining their directory structure. Attach it alongside Agriculture-Vision 2017 miniscale and the official split JSON.
2. Import `notebooks/kaggle_baseline.ipynb`, enable a GPU accelerator, and select Run All. The notebook locates the code dataset and copies source to `/kaggle/working/agriculture_code`. It uses Kaggle's installed PyTorch and needs no internet or previous-session variables.
3. Default input paths are `/kaggle/input/datasets/abelnathanaelh/agriculture-vision-2017/data2017_miniscale` and `/kaggle/input/datasets/abelnathanaelh/data2017-splits/data2017_splits.json`. Edit the explicit notebook settings if the mounts differ.
4. Outputs are saved under `/kaggle/working/agriculture_segmentation`. The baseline uses seed 42, Adam at 0.001, batch size 8, and at most 30 epochs. Reduce batch size explicitly before starting a run if necessary.

## Local setup and commands

Python 3.10+ is required. Create/activate a virtual environment, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.validate_dataset
```

Install `requirements-dev.txt` instead to include the optional fresh-kernel notebook test dependencies. The dataset validator uses local `data/raw/` paths by default and saves its report under `outputs/data_validation/`; it never reads test pixels.

Copy `configs/baseline.json` to `configs/local.json` and set `data_dir`, `split_file`, and `output_dir` to local paths:

```bash
python -m src.train --config configs/local.json
python -m src.train --config configs/local.json --resume outputs/my_run/last.pt
python -m src.evaluate --checkpoint outputs/my_run/best.pt --device auto
```

Expected official counts are 4,505 train / 2,048 validation / 1,792 test. Fields use the filename prefix before the first underscore and must be disjoint. Configurable sizes/counts support explicit synthetic tests; retain baseline settings for the actual experiment. The pipeline never falls back to random splits.

## Target and preprocessing rules

| ID | Class | Color |
|---|---|---|
| 0 | Normal farmland | Green |
| 1 | Drydown | Gold |
| 2 | Double plant | Blue |
| 3 | Weed cluster | Purple |
| 4 | Planter skip | Red |
| 255 | Ignored | Gray |

Both `field_masks` and `field_bounds` must equal 255 for valid supervision, following the [dataset team's explanation](https://groups.google.com/g/agriculture-vision/c/fhQ726FQSUs). The inspected miniscale data densely stores all nine class masks, including empty masks. Missing masks raise errors.

Targets are constructed at 512 x 512. Invalid pixels, Endrow, Storm damage, Water, Nutrient deficiency, Waterway, and overlaps among selected classes are assigned 255 last. Excluded masks always win over selected labels. Unannotated valid pixels are normal farmland. Resize targets with nearest-neighbor interpolation, RGB with bilinear interpolation, and scale RGB to [0,1]. Small regions may disappear at 256 x 256; consider this limitation for Planter Skip.

Training alone uses synchronized flips and multiples of 90-degree rotation. Images have shape `[B,3,256,256]`, targets `[B,256,256]`, and logits `[B,5,256,256]`. Cross-Entropy receives raw logits; prediction uses argmax.

Training-only unaugmented resized targets determine frequencies. Weights are inverse square roots of frequencies, divided by their mean, then capped at 3.0 without another rescaling. Counts and weights are printed/saved. Missing supervised training classes produce an error. Weighted Cross-Entropy ignores 255. All-ignored batches are skipped/counted; an all-ignored epoch fails. Epoch loss divides total weighted loss by total valid target weight.

## Selection and evaluation

Validation mean IoU chooses the best model. ReduceLROnPlateau halves LR after three non-improving validation epochs; early stopping uses patience seven. Validation/test are never augmented. Test content is not loaded during training or weight estimation.

Whole-split confusion matrices use true labels as rows and predictions as columns, excluding target 255. IoU is `TP/(TP+FP+FN)` and Dice is `2TP/(2TP+FP+FN)`. Zero denominators produce `null`; mean IoU averages defined classes and reports their names. Absent classes with false positives receive zero. Main mean IoU includes normal farmland; anomaly-only mean IoU is separate. Planter Skip has an explicit IoU, Dice, and valid-pixel support row. This project's exclusive-label overlap-ignore policy differs from the challenge's overlapping-label scoring convention.

Test evaluation loads `best.pt`, accumulates metrics, and caches evenly spaced examples plus a Planter Skip example if present in one pass. Figures show RGB, ground truth, argmax prediction, and prediction overlay with consistent colors/legend. Existing final metrics are reused by the notebook; CLI evaluation refuses to overwrite them. Do not tune against test results.

## Resume and artifacts

- `last.pt`: model, Adam/scheduler state, epoch, best score/model/epoch, early-stop counter, config, weights/counts, split identity, history, Python/NumPy/PyTorch/CUDA RNG and loader-generator states. Resume from this file.
- `best.pt`: validation-selected inference weights and class/preprocessing/split metadata. Use this file for final evaluation.
- `history.csv`, `learning_curves.png`, `class_distribution.json`, `config.json`, `environment.json`, `training_sanity.png`.
- `test/metrics.json`, `per_class_metrics.csv`, raw/labeled confusion matrices, four-panel sample PNGs, `visualizations.json`.

Checkpoint writes use atomic replacement. `last.pt` embeds the best model so resumption into a new output folder requires only one file. `epochs` is the total budget, including completed epochs. Resume validates split identity and training/preprocessing settings; paths, device, and total epochs may change. An already early-stopped run stays stopped. Fresh training refuses to overwrite existing runs. Only load trusted checkpoints because resume restores Python/NumPy state.

Save a Kaggle notebook version with outputs, or download artifacts before ending the runtime. Discarded sessions do not preserve `/kaggle/working/`. Attach the saved `last.pt` as Kaggle input and set `RESUME_FROM` in the notebook. If the saved run was already finally evaluated, preserve its `test/` results too. Resume at epoch boundaries; unfinished epochs restart from the last completed checkpoint.

Worker seeding is enabled and deterministic algorithms are preferred with `warn_only=True`. Some CUDA builds lack a deterministic NLL/Cross-Entropy kernel, so those operations emit a warning instead of stopping training. Exact CUDA numerical reproducibility is not guaranteed. Exact continuation depends on matching hardware, library versions, and worker settings. Actual determinism settings and software/device metadata are saved in `environment.json`. Kaggle's preinstalled CUDA-enabled PyTorch is retained rather than replaced.

## Verification and status

Synthetic tests cover target priority, resizing, augmentation, split validation, metrics, ignored-pixel gradients, all-ignored batches, RNG restoration, continuous versus resumed training, and evaluation/figure export. Real-data validation and actual GPU training are separate checks. No segmentation performance is claimed before the full Kaggle run.

NIR, anomaly-aware patch sampling, pretrained encoders, attention, and alternative losses are deferred. Use validation evidence to plan improvements and reserve the official test set for final reporting.

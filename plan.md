# Implementation Plan: Multi-Class Agricultural Anomaly Segmentation

## Scope and current repository

Build a simple supervised RGB semantic segmentation baseline for Agriculture-Vision 2017 miniscale. The implementation is now present; full Kaggle GPU training and final official test evaluation remain pending. No trained segmentation performance is claimed.

Repository inspection on 2026-11-25:

- `README.md` is currently a project title; `plan.md` previously held the project requirements.
- Previous source and baseline outputs were inspected in `old_project/`. The useful inspection scripts have been copied into root-level `scripts/`; the implementation has no dependency on the archive, which can be deleted. Preserve existing unrelated working-tree changes.
- Preserve the inspected dataset design ideas in the new implementation: validate official field assignments, reject repeated field IDs, use the filename prefix before the first underscore, and load RGB with Pillow. No archived source import is required.
- Replace reconstruction targets, the convolutional autoencoder, reconstruction losses, random patch masking, reconstruction heatmaps, and reconstruction evaluation. Their outputs are not segmentation baseline results.
- Implement seed, logging, and checkpoint utilities directly in the new pipeline according to this plan; no further archive access is needed.

Preserved scripts in `scripts/`: `check_splits.py` (split overlap report), `check_field_mapping.py` (RGB filename/field inventory), and `check_mask_mapping.py` (annotation filename coverage). These are legacy read-only diagnostics with local paths relative to the repository root, not the final strict validators. They inspect metadata and filenames without reading test label pixels. Reconstruction candidate/patch scripts and whole-dataset label summaries are not needed for this baseline and were not copied.

## 1. Establish the minimal project structure

- [x] Create `src/__init__.py`, `src/dataset.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, and `src/utils.py`.
- [x] Add `configs/baseline.json`, `notebooks/kaggle_baseline.ipynb`, and a small `tests/` suite for critical correctness checks.
- [x] Add minimal dependency declarations for PyTorch, NumPy, Pillow, Matplotlib, and tqdm; use the existing Kaggle PyTorch installation where suitable.
- [x] Update README with setup, dataset paths, training, resume, evaluation, output locations, and metric definitions.
- [x] Ignore generated checkpoints, metrics, plots, caches, and notebook runtime artifacts. Do not commit dataset files or legacy outputs as new results.

Default configuration:

| Setting | Value |
|---|---|
| Dataset root | `/kaggle/input/datasets/abelnathanaelh/agriculture-vision-2017/data2017_miniscale` |
| Official splits | `/kaggle/input/datasets/abelnathanaelh/data2017-splits/data2017_splits.json` |
| Output root | `/kaggle/working/agriculture_segmentation` |
| Input | RGB only, 256 x 256, float32 in [0, 1] |
| Classes | 5 |
| Ignore index | 255 |
| Seed | 42 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 8; reduce explicitly if GPU memory requires it |
| Maximum epochs | 30 total, including completed epochs on resume |
| Early stopping | Validation mean IoU, patience 7 |
| Scheduler | ReduceLROnPlateau on validation mean IoU, mode=max, factor=0.5, patience=3 |

## 2. Validate dataset layout and official splits

- [x] Inspect the actual split JSON and image/mask naming conventions before implementing path resolution. Local data can support inspection; Kaggle paths remain configurable defaults.
- [x] Map each RGB tile to its field using the filename portion before the first underscore.
- [x] Validate split keys, nonempty field IDs, duplicates within splits, and disjoint field sets across train/validation/test.
- [x] Index images deterministically. Reject unknown image field IDs, duplicate sample IDs, and empty image splits. Official fields absent from the miniscale subset may remain unused.
- [x] Require 4,505 training, 2,048 validation, and 1,792 test images for the full miniscale baseline. Fail clearly on mismatches rather than falling back to a random split.
- [x] Inspect the meanings and binary conventions of `field_masks/` and `field_bounds/`. Use verified validity/boundary semantics; do not assume that a boundary outline is a filled valid-field mask.
- [x] Check required files and mask dimensions. If per-class masks are sparse, verify the dataset convention before treating an absent mask as empty; otherwise fail clearly on missing annotations.
- [x] Limit pretraining test access to split metadata and file inventory checks. Do not load test image content or test label pixels for exploratory analysis, weight estimation, or model decisions.

## 3. Construct targets with explicit ignore priority

Class mapping:

| ID | Class |
|---|---|
| 0 | Normal farmland |
| 1 | Drydown |
| 2 | Double plant |
| 3 | Weed cluster |
| 4 | Planter skip |
| 255 | Ignored pixel |

Excluded classes: Endrow, Storm damage, Water, Nutrient deficiency, and Waterway. Resolve their exact directory spellings from the dataset.

- [x] Load binary annotations at their original 512 x 512 resolution and validate alignment.
- [x] Compute `excluded_union` over all excluded class masks.
- [x] Compute `selected_count` over the four selected anomaly masks.
- [x] Define `ignored = outside_valid_field OR excluded_union OR (selected_count > 1)`, incorporating any verified boundary exclusion rule.
- [x] Initialize normal farmland as 0. Assign selected class IDs only where exactly one selected mask is positive and `ignored` is false. Set all `ignored` pixels to 255 as the final operation.
- [x] Resize the completed target to 256 x 256 using nearest-neighbor interpolation. This keeps the priority decision at native resolution and preserves discrete IDs.
- [x] Assert target dtype int64, shape `[256, 256]`, and values contained in `{0, 1, 2, 3, 4, 255}`.

Normal farmland means valid field pixels with no selected or excluded anomaly annotation. Excluded annotations always override selected classes, including pixels shared by both.

## 4. Implement datasets and DataLoaders

- [x] Load RGB, resize with bilinear interpolation, divide by 255, and return contiguous float32 tensors in channel-first format.
- [x] Apply synchronized horizontal flips, vertical flips, and random multiples of 90-degree rotation to training images and targets only. Use independent flip probability 0.5 and rotation sampled from 0, 90, 180, and 270 degrees.
- [x] Keep validation/test preprocessing deterministic and augmentation disabled.
- [x] Use shuffle only for training. Start with two workers, CUDA pinning, and seeded worker initialization and DataLoader generators. Support `num_workers=0` for debugging.
- [x] Avoid persistent workers initially so epoch-boundary resume can restore worker-seeding behavior reliably. Retain all validation/test samples with `drop_last=False`.
- [x] Inspect a training batch before optimization: images `[B, 3, 256, 256]`, targets `[B, 256, 256]`, correct dtypes, image range, and allowed label values. Show aligned image/target examples.

## 5. Implement a standard U-Net

- [x] Use double 3 x 3 convolution/ReLU blocks, four max-pooling stages, a bottleneck, transposed convolutions, concatenated skip connections, and a final 1 x 1 convolution.
- [x] Start with encoder widths 32, 64, 128, 256 and bottleneck width 512; record widths in configuration.
- [x] Accept `[B, 3, 256, 256]` and return raw logits `[B, 5, 256, 256]`.
- [x] Use random initialization without pretrained weights, attention, or extra architectural modules.
- [x] Verify forward shape and finite backward gradients before full training. Apply softmax neither before Cross-Entropy nor before argmax prediction.

## 6. Estimate training-only class weights

- [x] Count valid pixels in deterministic, resized, unaugmented training targets only. Exclude 255.
- [x] Print and save counts, valid-pixel frequencies, ignored-pixel count, and final weights.
- [x] Define `f_c = count_c / sum(counts)`, `r_c = 1 / sqrt(f_c)`, `s_c = r_c / mean(r)`, and `w_c = min(s_c, 3.0)`. Keep the final cap intact; do not renormalize afterward. Cross-Entropy's weighted mean handles the overall scale.
- [x] Fail clearly if any selected class has zero supervised training pixels after resizing; investigate mapping or data before proceeding.
- [x] Use weighted `CrossEntropyLoss(ignore_index=255)`.
- [x] Skip all-ignored batches safely, with a recorded count, and fail if an entire epoch has no supervised pixels. Aggregate epoch loss using summed weighted loss divided by the summed target weights for valid pixels.

## 7. Train, validate, and resume

- [x] Seed Python, NumPy, PyTorch, CUDA, and worker generators with 42. Record software versions, device, configuration, and deterministic settings; document hardware/version limits on exact reproducibility.
- [x] Train with Adam and progress bars. Validate after every epoch using eval mode and disabled gradients.
- [x] Build validation metrics from one accumulated confusion matrix over the full validation split.
- [x] Log epoch, training loss, validation loss, validation mean IoU, and learning rate to CSV; save a learning-curve figure.
- [x] Select `best.pt` only by validation mean IoU. Apply the scheduler and early stopping using validation data only.
- [x] Save `last.pt` after each completed epoch and `best.pt` on improvement using atomic file replacement.
- [x] Include model, optimizer, scheduler, completed epoch, best score, early-stopping counter, class weights/counts, full configuration, history, Python/NumPy/PyTorch/CUDA RNG states, and DataLoader generator states in resumable checkpoints.
- [x] Validate class mapping, preprocessing, model configuration, and split identity when resuming. Restore optimizer tensors to the selected device and continue at the next epoch without duplicating history.
- [x] Resume from an explicit checkpoint path. Explain how to preserve/download outputs and attach a saved run as Kaggle input after a runtime reset; `/kaggle/working/` alone is not durable across discarded sessions.

## 8. Evaluate the official test set once

- [x] Keep test evaluation separate from training. After validation-based selection is complete, explicitly load `best.pt` and evaluate the official test loader once.
- [x] Predict discrete class IDs with `argmax(dim=1)`. Remove targets equal to 255 before updating a 5 x 5 confusion matrix, with true classes as rows and predictions as columns.
- [x] Compute per-class IoU, per-class Dice, mean IoU, and overall pixel accuracy from aggregated counts, never by averaging per-batch IoUs.
- [x] Define IoU as `TP / (TP + FP + FN)` and Dice as `2TP / (2TP + FP + FN)`.
- [x] Report classes with zero denominator as undefined (`null` in JSON). Average only defined IoUs and report which classes contributed; a class with false positives but no ground-truth pixels still has IoU zero.
- [x] Use mean IoU across the five classes as the primary validation/test summary. Also report anomaly-only mean IoU for context, labeled separately.
- [x] Save metrics JSON/CSV, raw and labeled confusion matrices, and valid ground-truth support per class. Give Planter Skip its own visible row with IoU, Dice, and pixel support.
- [x] Cache visualization samples and predictions during this evaluation pass. Record checkpoint identity and evaluation configuration. Do not tune the baseline against test results.

## 9. Produce consistent visualizations

- [x] Use a fixed palette for all five classes and a separate neutral ignore color, with a readable legend.
- [x] Render several deterministic test examples as four panels: RGB, ground truth, argmax prediction, and predicted-mask overlay.
- [x] Show ignored regions explicitly and keep out-of-field/ignored regions from obscuring the image in overlays.
- [x] Save figures under the run output directory. Include a Planter Skip example if present, identifying it as a qualitative example rather than a representative performance estimate.

## 10. Build a fresh-session Kaggle notebook

- [x] Provide sequential sections: setup/seed, paths and validation, official splits, training distribution, datasets/loaders and batch checks, model, class weights, training/resume, best-checkpoint test evaluation, and saved results/visualizations.
- [x] Document one explicit source delivery method: upload the versioned project code as a Kaggle input dataset, copy it to `/kaggle/working/`, and import its `src` modules. No dependency on this Windows repository path, hidden variables, or internet-only Git cloning.
- [x] Use the same modules and baseline config as CLI execution to avoid duplicate implementations. Keep all writable paths under `/kaggle/working/`.
- [x] Run training or resume in the same logical flow, then gate final evaluation on successful best-checkpoint creation. A fresh Run All performs training followed by one final test pass.
- [x] Validate notebook structure and execute a small synthetic smoke run locally if full Kaggle data/GPU is unavailable. Clearly distinguish synthetic checks from full dataset validation and actual GPU training.

## 11. Verify critical behavior before the full run

- [x] Test synthetic target cases: normal, each selected class, outside field, excluded-only, selected/excluded overlap, multiple selected overlap, and ignored pixels after resizing and augmentation.
- [x] Test duplicate/disjoint field handling, filename mapping, unknown fields, and image-count failures using fixtures.
- [x] Check model output shape, weighted loss/backward, ignored-pixel contribution, and all-ignored handling.
- [x] Test confusion-matrix orientation, known IoU/Dice values, ignore filtering, absent classes, and equality between batched and whole-set accumulation.
- [x] Test checkpoint round-trip and epoch-boundary continuation, including optimizer/scheduler state, best score, early-stop counter, history, and RNG restoration.
- [x] Run a small CPU integration smoke run covering training, checkpoint reload, evaluation, and figure saving; follow with Kaggle GPU sanity checks before the 30-epoch run (still pending on Kaggle).

## Completion criteria and deferred work

The baseline implementation is ready when correctness checks pass, the documented notebook works from a fresh Kaggle session, training can resume from saved outputs, and evaluation/visualization use the selected best checkpoint. The experiment is complete only after actual GPU training and final official test evaluation produce saved artifacts; do not claim performance before then.

After the baseline, analyze validation results and rare-class failures to decide whether a later experiment is justified. Anomaly-aware patch sampling, NIR input, pretrained encoders, attention, and other architectural/loss changes are outside this baseline and must not be implemented now.



## Implementation verification status

- Source, baseline configuration, README, and sequential Kaggle notebook are implemented. The notebook uses only the new modules and has no archive dependency.
- Official local split inventories match 4,505 / 2,048 / 1,792; all nine annotation folders have complete file coverage. Validity is the intersection of white field-mask and white boundary pixels, per the dataset team.
- All 11 tests passed, covering priority rules, transformations, metrics, gradients, scheduler/early stopping, checkpoint continuation with zero and two workers, and fresh-kernel notebook execution. The verification log is `outputs/verification-tests.log`.
- Every notebook code cell has executed in a fresh kernel against synthetic CPU data with only mount paths, accelerator, and experiment scale substituted. This verifies the workflow, not actual Kaggle GPU behavior.
- [ ] Complete the actual fresh-session Kaggle GPU run and save training artifacts.
- [ ] Evaluate the validation-selected checkpoint once on the official test set and record measured results.

Checked implementation items above describe delivered code and local checks; they do not claim completion of the two actual experiment steps listed here.

The upload-ready shared code bundle is `outputs/kaggle_source.zip`; attach its extracted contents as a Kaggle dataset.

## Real training-data validation

The full scan of 4,505 training targets passed at 256 x 256. The batch was `[8, 3, 256, 256]` / `[8, 256, 256]` with labels `{0,1,2,3,4,255}`. The baseline U-Net has 7,760,229 parameters and passed CPU forward/backward with finite gradients. No official test image or label pixels were loaded.

| Class | Supervised pixels | Frequency among valid targets | Final weight |
|---|---:|---:|---:|
| Normal farmland | 195,847,541 | 76.3846% | 0.113613 |
| Drydown | 43,653,281 | 17.0257% | 0.240646 |
| Double plant | 5,484,128 | 2.1389% | 0.678942 |
| Weed cluster | 11,204,336 | 4.3699% | 0.475000 |
| Planter skip | 207,336 | 0.0809% | 3.000000 |

Ignored training pixels: 38,843,058. Report: `outputs/data_validation/report.json`. The training alignment figure was inspected. These are dataset statistics and pipeline checks, not trained model results. Local PyTorch is CPU-only; GPU checks and actual training remain pending in Kaggle.

CUDA compatibility fix: deterministic algorithms now use `warn_only=True`, allowing CUDA NLL/Cross-Entropy kernels without deterministic implementations. Seeds and cuDNN deterministic settings remain enabled; environment metadata records warning mode. Exact CUDA numerical reproducibility is not guaranteed.

# Experiment 2 handover: agricultural anomaly patch sampling

Prepared on 2026-11-25 for a teammate using their own Kaggle account.

## Quick Start

1. Open your Kaggle account and create a GPU notebook for Experiment 2.
2. Attach the already-shared Agriculture-Vision dataset and official split JSON.
3. Obtain the correct project code revision and record it.
4. Verify the GPU, dataset paths, official split counts, and class mapping.
5. Implement the uniform patch sampler and connect it to the existing training components.
6. Preview and validate its crops, masks, and shared validity/retry rules.
7. Train Run B, then save/download its results and checkpoints.
8. Implement anomaly-aware sampling using the agreed 50:50 policy and fallback rules.
9. Preview and validate its crops, targeted-class support, and fallback logging.
10. Train Run C separately, then save/download its results and checkpoints.
11. Compare validation results and return the files using the reporting template below. Keep official test evaluation separate until the final model is selected.

These steps are the roadmap; the detailed reference below explains implementation requirements, checks, and outputs. No Experiment 2 sampler or training notebook has been implemented yet.

**Access update from the project owner:** the teammate has already been invited to the project data, Kaggle resources, and GitHub repository. Shared access is arranged and is not an outstanding handover decision. Use the invited accounts and existing shared resources; no new invitation is required.

**Status: Experiment 1 and the small overfit diagnostic are complete. Experiment 2 is not implemented or trained. This guide specifies the remaining work; it does not provide a finished patch-training notebook or runner.**

**Use source version `experiment-2-handover-v1`.** This Git tag contains the baseline code/evidence, completed diagnostic code/tests/evidence, and this guide. The matching Kaggle archive is `outputs/experiment_2_handover_v1.zip`; `outputs/kaggle_source.zip` is an identical complete upload alias. The archive's `HANDOVER_VERSION.json` records its exact commit and per-file SHA-256 checksums. Use one of these complete snapshots rather than an older baseline-only upload.

## 1. Understand the project and protect completed work

The task is supervised, multi-class semantic segmentation of Agriculture-Vision 2017 miniscale aerial RGB images. Each output pixel identifies normal farmland or one agricultural anomaly. RGB is the only input; NIR is outside this experiment.

| Target ID | Class | Figure color |
|---|---|---|
| 0 | Normal farmland | Green |
| 1 | Drydown | Gold |
| 2 | Double plant | Blue |
| 3 | Weed cluster | Purple |
| 4 | Planter skip | Red |
| 255 | Ignored; excluded from loss and metrics | Gray |

The unchanged model is `src.model.UNet(base_channels=32)`: double convolution/ReLU blocks, four max-pooling stages, a bottleneck, transposed-convolution upsampling, skip connections, and a five-channel output head. It has **7,760,229 parameters**, no pretrained weights, and no attention. It produces logits; use `argmax(dim=1)` for a discrete class map.

Do not retrain the baseline, rerun its test evaluation, overwrite its outputs, or repeat the diagnostic without a specific reason. Do not use `old_project/`; it contains abandoned reconstruction work.

### Completed Experiment 1

The baseline trained on full 512 x 512 images resized to 256 x 256, using official field-level splits and weighted Cross-Entropy. The best checkpoint was selected at **epoch 21**, with **22.3383% validation mean IoU**. Early stopping ended training at epoch 28.

| Class/summary | Baseline test IoU | Baseline test Dice |
|---|---:|---:|
| Normal farmland | 73.43% | 84.68% |
| Drydown | 41.17% | 58.33% |
| Double plant | 0.12% | 0.24% |
| Weed cluster | 1.38% | 2.73% |
| Planter skip | 0.00% | 0.00% |
| Mean IoU, five classes | 23.22% | Not reported as a summary |
| Anomaly-only mean IoU | 10.67% | Not reported as a summary |

Pixel accuracy was 76.06%, but an always-normal predictor would score approximately 76.13% accuracy on that same test set. Its mean IoU would be only 15.23%. This demonstrates why accuracy alone is unsuitable. The baseline predicted no Planter skip pixels on valid test regions, although 194,504 such target pixels existed.

Use the saved evidence in `experiments/baseline/`. The original local download is `results/agriculture_segmentation/`; its checkpoints are deliberately outside Git. Baseline per-class **validation** metrics were not saved in the provided evidence; do not substitute test values into a validation comparison.

### Completed diagnostic

Four fixed 64 x 64 crops were taken **after the baseline resize to 256 x 256**, using training data only. They contained all five classes. With augmentation disabled, the same randomly initialized U-Net memorized them on CPU. Subset weights were recomputed using the existing weighting formula.

| Class | Diagnostic training-subset IoU |
|---|---:|
| Normal farmland | 99.00% |
| Drydown | 99.52% |
| Double plant | 98.06% |
| Weed cluster | 94.32% |
| Planter skip | 98.65% |
| Mean IoU | 97.91% |

It stopped at step 700 after three consecutive checks with every class above 90% IoU. The selected checkpoint is step 660. All five classes can be learned in this controlled setting; Weed cluster was slowest. Temporary loss spikes recovered without intervention. The checkpoint's metrics were reproduced after reload.

These are **memorization scores, not validation/test scores**. They argue against a gross class-channel, loss, or gradient failure, but do not prove the absence of all annotation issues or explain the baseline's generalization failure. The diagnostic used an easier input distribution than the baseline and is not Experiment 2.

The source and tests are `src/overfit_diagnostic.py` and `tests/test_overfit_diagnostic.py`. Small diagnostic artifacts are versioned in `experiments/overfit_diagnostic/`: summaries, configuration/hardware report, exact crop manifest, small selected input tensors, RGB/target PNGs, history, per-class metrics, initial/final predictions, learning curves, and pipeline audit. They are included in the tagged clone and complete Kaggle archive; you do not need the owner's local filesystem to inspect them. The original raw run remains in ignored `results/overfit_diagnostic/`. Its approximately 93 MB model checkpoint is not needed for Experiment 2, which must start from fresh weights, and stays outside Git.

## 2. Know what already exists and what you must implement

| Existing component | Reuse | Important limitation |
|---|---|---|
| `src/dataset.py` | `index_splits`, `AgricultureVisionDataset`, `construct_target`, `augment_pair`, `check_batch` | No random-patch or anomaly-aware sampler exists |
| `src/model.py` | Exact `UNet(base_channels=32)` | Start both new runs from seed 42, not baseline/diagnostic weights |
| `src/train.py` | `run_epoch`, `SCHEMA`, training/checkpoint design | `train()` always constructs the baseline datasets and recomputes weights |
| `src/utils.py` | Seeds, RNG capture/restore, metrics, plotting, atomic checkpoints | Precision and recall are not currently exported |
| `src/evaluate.py` | Final selected-model test evaluation | Always reads the official test split; not a validation command |
| `configs/baseline.json` | Reference hyperparameters | Contains no functional patch-sampling configuration |
| `notebooks/kaggle_baseline.ipynb` | Reference setup and checks | Run All trains the baseline and then evaluates test; do not use it unchanged for Experiment 2 |
| `src/overfit_diagnostic.py` | Inspect the diagnostic if needed | Fixed cached crops, not a production patch sampler |

The complete handover revision is pinned by Git tag `experiment-2-handover-v1`. It includes the diagnostic source/tests and curated evidence as well as the baseline. Do not use the earlier baseline-only commit `300287c` as the handover version. Record the tag's resolved commit from your clone or the archive manifest before adding Experiment 2 implementation files.

The remaining implementation is a small patch dataset/sampler and a separate Experiment 2 entry point that reuses the existing model, epoch function, target construction, and checkpoint helpers. Add experiment-specific files/configuration rather than rewriting or changing the baseline. Names for those future files/commands have not been assigned. **Adding a `sampling` key and calling the current baseline `train()` will not implement this experiment.**

## 3. Agree on the two new runs before coding

Research question: Does native-resolution patch training help, and does anomaly-aware sampling add a further benefit?

| Run | Training input | Status and purpose |
|---|---|---|
| A: full-image baseline | Full 512 x 512 RGB/target resized to 256 x 256 | Completed reference; do not retrain |
| B: uniform patches | Uniform random 256 x 256 windows from native 512 x 512 training pairs | New control: effect of crop training, retained detail, and changed context |
| C: anomaly-aware patches | Native 256 x 256 crops, approximately 50% uniform and 50% anomaly-focused | New treatment: additional effect of targeted sampling |

Run B is necessary: A versus C alone would confound targeted sampling with the change in scale and field context. B versus C isolates the sampling policy more closely. Do not crop the already resized 256 x 256 baseline images for B/C; that would reproduce the diagnostic's scale rather than this design.

### Agreed starting defaults

| Setting | Decision |
|---|---|
| Crop size | 256 x 256, taken from native 512 x 512 training pairs |
| Minimum valid pixels | At least 50% of the crop: 32,768 out of 65,536 pixels |
| Maximum retries | 10; the retry budget is bounded as described below |
| If the targeted class cannot be found | Select another eligible training image containing the same class |
| If targeted sampling remains unsuccessful | Fall back to a random valid training crop |
| Invalid/ignored pixels | Keep label 255; never convert them to Normal farmland |
| Sampling source | Official training data only |

These are practical starting values agreed by the project owner, **not experimentally proven optimal values**. Use the same crop-validity threshold and retry rules in Runs B and C. Do not tune them separately for either arm.

For B, visit/shuffle training image slots and choose integer top/left coordinates uniformly from 0 through 256 inclusive. Select windows without deliberately favoring an anomaly class. Accept only windows with at least 50% valid target pixels; normal-only windows satisfying that rule are allowed. Validity means `target != 255` after all invalid-field, excluded-class, and overlap rules have been applied. Rejected windows must not have their ignored labels changed to make them valid. Uniform sampling is therefore conditional on the validity rule and documented fallback behavior.

For C, choose a branch with probability 0.5 for each training slot:

1. Uniform branch: the same policy as B.
2. Focused branch: choose one of the four anomaly classes with equal probability, then choose an eligible training image uniformly, then a pixel uniformly from that class's **final valid target**. Draw a crop containing it.

Thus each anomaly class receives approximately 12.5% of all training slots as the **requested** targeted class, not 12.5% of all training pixels. Crops may include several classes. Failed focused requests become random valid crops, so the actual emitted uniform/focused mix can differ from 50:50. Log requested branches/classes, successful focused crops, and random fallbacks separately; do not report a fallback as a successful targeted crop or claim the resulting pixel distribution is balanced.

Do not choose anchors from raw binary anomaly masks: excluded/invalid/overlapping pixels must already have been removed from eligibility. Build the class-to-image inventory from training targets only, sort it deterministically, and save its fingerprint. Decode a chosen image's coordinates as needed rather than holding all nine masks for every image in GPU memory.

For a pixel at coordinate `x`, an inclusive crop of width 256 in an image of width 512 must have its left coordinate in:

```text
max(0, x - 255) <= left <= min(x, 256)
```

Apply the analogous range to `y`. This ensures the selected pixel is actually inside the crop, including at image edges. Random placement within the permitted interval avoids always putting anomalies at the center.

Apply the following bounded retry/fallback procedure consistently:

1. A crop must retain at least 32,768 valid pixels. A focused crop must additionally retain at least one valid pixel of its requested anomaly class after cropping and augmentation.
2. Allow one initial candidate and **at most 10 retries** per sampling stage (at most 11 candidates). Record this counting convention in `sampling_config.json`; do not reset the focused retry budget every time the image changes.
3. For a failed focused request, choose another eligible **training image containing the same class** and retry its anchor/crop selection. Keep the requested class fixed while retrying; never use validation/test images or select anchors from ignored pixels. If no eligible image is available, proceed to the fallback stage and record the reason.
4. After focused retries are exhausted, emit a random valid training crop instead. The fallback can be normal-only and need not contain the originally requested class, but must still satisfy the same 50% validity threshold. Record the failed request, retry count, and actual fallback class counts.
5. Uniform requests that exhaust retries also use this same random-valid fallback. For fallback candidates, choose from training images without anomaly-class preference. Bound this stage to an initial candidate plus at most 10 retries as well; do not recursively restart it indefinitely.
6. If even the bounded random-valid fallback cannot produce a crop meeting the threshold, stop with a clear error and save the failure details. Never return an invalid crop, silently drop the training slot, lower the threshold, or relabel ignored pixels as normal. This final guard handles failure to find *any valid crop*, rather than blocking the agreed fallback from a failed targeted request.

Log repeated missing-class/eligibility failures and contact the owner before an expensive run if the preview cannot demonstrate all five classes. This policy defines the two agreed runs; it is not a request to add alternatives or search for an optimal ratio/threshold.

## 4. Freeze the shared training and validation protocol

Use these settings for both B and C unless an explicit pretraining agreement changes them:

| Setting | Shared value |
|---|---|
| Model | Existing U-Net, base width 32, random initialization |
| Input/output | RGB float32 [0,1], `[B,3,256,256]`; five logits per pixel |
| Optimizer | Adam, initial learning rate 0.001 |
| Loss | Existing weighted Cross-Entropy, ignore index 255 |
| Class weights | Reuse the baseline training-only weights, identical in both arms |
| Crop validity | At least 50% non-255 target pixels; no relabeling of ignored pixels |
| Retry/fallback rules | At most 10 retries per stage; failed focused requests use a random valid training crop |
| Training slots per epoch | 4,505 crops, including the final partial batch |
| Batch size | 8, if GPU memory permits; same in both arms |
| Epoch budget | At most 30, including completed epochs on resume |
| Augmentation | Synchronized 0.5 horizontal flip, 0.5 vertical flip, random 0/90/180/270 rotation |
| Validation | All 2,048 official validation images, full images resized to 256 x 256, no augmentation |
| Checkpoint selection | Highest five-class validation mean IoU |
| Scheduler | ReduceLROnPlateau, mode=max, factor=0.5, patience=3 |
| Early stopping | Seven epochs without validation mean IoU improvement |
| Seed | 42, including workers and loader/sampler generators |
| Workers | Baseline used 2; keep the same chosen setting in both new runs |

With batch size 8 and no discarded/failed batches, 4,505 crops yield 564 optimizer updates per epoch. The same maximum budget and stopping protocol do not guarantee equal actual updates if one arm stops earlier. Record actual epochs, updates, and elapsed time.

The baseline weights in `experiments/baseline/class_distribution.json` are, in ID order:

```text
[0.1136128753, 0.2406457365, 0.6789422631, 0.4749999344, 3.0]
```

They were computed only from baseline unaugmented training targets at 256 x 256, using inverse-square-root frequencies, mean normalization, and a final cap of 3.0. Holding them fixed isolates sampling more clearly. Do not recompute different weights from B's and C's sampled distributions or reuse diagnostic weights. Confirm this decision before implementation; the current baseline trainer automatically recomputes weights, so the new runner must explicitly load the agreed vector.

Validation remains downscaled full-image evaluation to match Experiment 1. Native training crops and resized validation images have different scales/context; this is a limitation of the chosen comparison. Do not silently switch to native-resolution, sliding-window, or patch-selected validation. Both new arms must use the same protocol.

Reset seeds before initializing each new model, and use a separately seeded sampler/loader generator so previews do not change training initialization or sample order. Sampling-policy randomness will make the two streams differ even with the same seed; record the seed without promising identical random sequences across policies or hardware.

## 5. Preserve target semantics at native resolution

Official counts are **4,505 train / 2,048 validation / 1,792 test**. Fields are the filename prefix before the first underscore. `index_splits` rejects duplicate/overlapping field assignments, unknown image fields, and count mismatches. Official fields absent from miniscale may remain unused. Never substitute a random image-level split.

The dataset densely stores a 0/255 PNG for every image in each of the nine class folders. A missing file is an error, not an empty annotation.

The existing target rules are:

1. A pixel is valid only where **both** `field_masks` and `field_bounds` equal 255.
2. Excluded categories are `endrow`, `storm_damage`, `water`, `nutrient_deficiency`, and `waterway`.
3. Count overlaps among `drydown`, `double_plant`, `weed_cluster`, and `planter_skip`.
4. Outside-valid, excluded, and selected-overlap pixels are assigned **255 last**. Excluded pixels override selected anomalies.
5. Valid pixels with no anomaly annotation are normal farmland.

Reuse this logic instead of reconstructing targets differently for B and C. The existing Dataset can return native pairs with `image_size=512, native_size=512, augment=False`; its target-resize step then preserves the native grid. The patch wrapper should take exactly the same `top:left` window from RGB and target, then apply the existing synchronized augmentation. No crop resizing is necessary for native 256 x 256 windows. If a resize is ever explicitly approved, use bilinear for RGB and nearest-neighbor for categorical targets.

Targets remain int64 `[B,256,256]` with values in `{0,1,2,3,4,255}`. Do not convert ignored pixels to class 0, interpolate masks bilinearly, or add a sixth model output for ignore.

## 6. Prepare your own Kaggle account and obtain the code

1. Sign in to your own Kaggle account and create a new Python notebook for Experiment 2. Enable an available GPU under Settings > Accelerator before starting the run. Complete any account verification Kaggle requires. Internet is needed only if choosing Git cloning; an attached source dataset works without it. See [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).
2. Attach the already-shared Agriculture-Vision 2017 miniscale data and official split JSON using Add Input while signed in to your invited Kaggle account. No separate dataset upload or access request is needed. Your notebook still cannot access the owner's Windows folders or existing runtime variables.
3. Obtain a fixed source revision using one of the methods below. Do not install a different PyTorch build over Kaggle's GPU environment merely to match the local CPU environment.

### Recommended portable method: an attached code dataset

Use the generated `outputs/experiment_2_handover_v1.zip` or its identical `outputs/kaggle_source.zip` alias. It contains all versioned source, configurations, tests, scripts, the existing baseline notebook, `experiments/baseline/`, `experiments/overfit_diagnostic/`, requirements, README, and this handover. The diagnostic source/tests are included, not optional local-only additions. Raw datasets, `.venv`, `.git`, large model checkpoints, and duplicate repositories are excluded. The small diagnostic `selected_tensors.pt` is input evidence, not a model checkpoint, and is deliberately included.

Create/update the shared private Kaggle code dataset from that archive and attach the correct dataset version. Kaggle supports ZIP uploads and preserves their extracted folder layout. See [Kaggle dataset documentation](https://www.kaggle.com/docs/datasets). The old baseline-only contents of `outputs/kaggle_source.zip` have been replaced by this complete snapshot. An older version already uploaded to Kaggle is not updated automatically: select the newly uploaded version and verify its manifest.

Copy the attached code to a writable working directory before implementation. This example discovers the archive root; set an explicit source path if more than one version is attached:

```python
from pathlib import Path
import hashlib
import json
import shutil
import sys

candidates = sorted({p.parent.parent for p in Path('/kaggle/input').rglob('src/train.py')
                     if (p.parent.parent / 'configs/baseline.json').is_file()
                     and (p.parent.parent / 'experiments/baseline/class_distribution.json').is_file()
                     and (p.parent.parent / 'HANDOVER_VERSION.json').is_file()})
assert len(candidates) == 1, f'Attach one complete source snapshot; found {candidates}'
SOURCE_DIR = candidates[0]
version = json.loads((SOURCE_DIR / 'HANDOVER_VERSION.json').read_text())
assert version['version'] == 'experiment-2-handover-v1', version
for name, expected in version['files_sha256'].items():
    assert hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() == expected, name
print('Verified source version:', version['version'], 'commit:', version['commit'])
PROJECT_ROOT = Path('/kaggle/working/AML-CropSegmentation')
assert not PROJECT_ROOT.exists(), 'Use a fresh working directory to avoid mixing source versions'
shutil.copytree(SOURCE_DIR, PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
```

Record the manifest's commit, archive SHA-256 sidecar supplied by the owner, and your eventual implementation revision. The packaging script checks archive integrity and each file against the tagged Git content. It can be rebuilt with `python scripts/package_handover.py --ref experiment-2-handover-v1 --update-kaggle-alias`. Do not change attached code versions midway through a resumable run.

### Alternative: clone GitHub using your existing repository access

The real configured remote is **https://github.com/fakebaconpancakes/AML-CropSegmentation.git**. The owner confirms that repository access has already been shared with the teammate. With notebook internet enabled and authenticated Git access where required, the following Python cell clones the complete handover tag directly:

```python
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path('/kaggle/working/AML-CropSegmentation')
subprocess.run(['git', 'clone', '--branch', 'experiment-2-handover-v1', '--single-branch',
                'https://github.com/fakebaconpancakes/AML-CropSegmentation.git',
                str(PROJECT_ROOT)], check=True)
print('Verified handover commit:', subprocess.check_output(
    ['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip())
sys.path.insert(0, str(PROJECT_ROOT))
```

This tag includes the required completed-work source/evidence, but Experiment 2 still needs implementation. Keep the tag as an unchanged reference; record your implementation revision separately. Repository membership does not automatically authenticate Git inside a fresh Kaggle runtime. Use your own authenticated Git setup or the code-dataset method if cloning requires authentication; do not put credentials in notebook code or copy the owner's tokens. Run only one of the two acquisition methods in a fresh working directory.

## 7. Verify the environment and training data before implementation

First check the installed environment:

```python
import platform
import torch
from src.dataset import CLASSES, EXPECTED_COUNTS, IGNORE_INDEX
from src.utils import seed_everything

print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Enable a Kaggle GPU before full training'
gpu = torch.cuda.get_device_properties(0)
print('GPU:', gpu.name, 'VRAM GiB:', gpu.total_memory / 2**30)
print('Mapping:', dict(enumerate(CLASSES)), 'ignore:', IGNORE_INDEX)
seed_everything(42)
```

The baseline used a Tesla T4 and PyTorch `2.11.0+cu130`. Different GPU/software versions may change numerical results; record your actual environment rather than claiming an exact reproduction. The local diagnostic used CPU-only PyTorch on an Intel Core Ultra 7 258V with an Intel Arc 140V integrated GPU and about 32 GB RAM; those local resources are not needed for your Kaggle run.

Verify the actual input mount. There is a known spelling difference: `configs/baseline.json` contains `agriculture-vision-2017`, while the successful baseline's saved `config.json` contains `agriculturevision-2017`. Copy the real path shown in your Input panel. The following uses the successful run's path as a starting point, not a guarantee about your mount:

```python
from pathlib import Path
from src.dataset import index_splits

DATA_DIR = Path('/kaggle/input/datasets/abelnathanaelh/agriculturevision-2017/data2017_miniscale')
SPLIT_FILE = Path('/kaggle/input/datasets/abelnathanaelh/data2017-splits/data2017_splits.json')
assert (DATA_DIR / 'field_images/rgb').is_dir(), 'Correct DATA_DIR from the Input panel'
assert SPLIT_FILE.is_file(), 'Correct SPLIT_FILE from the Input panel'
paths, split_fingerprint = index_splits(DATA_DIR, SPLIT_FILE, EXPECTED_COUNTS)
print({name: len(items) for name, items in paths.items()})
print('Split fingerprint:', split_fingerprint)
```

Expected counts are train=4505, val=2048, test=1792. The inspected dataset/split fingerprint is `3ee33e649122cb4d8caa49d690de6e56dcb4b3f9fab90ad7f96b35dc6e2a8d67`. This function reads filenames and split metadata, not test pixels. Do not construct a test loader during development.

Inspect one **training** pair using the existing loader, at native resolution:

```python
import matplotlib.pyplot as plt
from src.dataset import AgricultureVisionDataset, check_batch
from src.utils import colorize

preview = AgricultureVisionDataset(DATA_DIR, paths['train'][:1], 'train',
                                   image_size=512, native_size=512, augment=False)
image, target = preview[0]
check_batch(image[None], target[None], 512)
print('Shapes:', tuple(image.shape), tuple(target.shape))
print('Target values:', target.unique().tolist())
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(image.permute(1, 2, 0).numpy())
axes[0].set_title('Training RGB')
axes[1].imshow(colorize(target.numpy()))
axes[1].set_title('Priority-safe native target')
for ax in axes:
    ax.axis('off')
plt.show()
```

Do not expect every class in this one image. Verify class coverage across training-only sampler checks. The original dataset has all nine annotation directories, including excluded categories with empty masks.

## 8. Implement only the necessary patch integration, then sanity-check it

Before a full run, the teammate must implement and review the following missing pieces:

- Native-pair crop wrapper with uniform and requested-50:50 anomaly-aware modes; one shared target implementation, the same post-crop augmentation, and the agreed 50%-valid/10-retry/fallback rules.
- Training-only eligibility inventory and seeded coordinate selection. Save its construction rules and sample IDs.
- Separate patch-training entry point accepting the agreed fixed class weights and run-specific paths. Reuse `run_epoch`, the U-Net, and checkpoint helpers; do not invoke the unchanged baseline `train()` and assume it samples patches.
- Same full-image validation loader as the baseline, plus saved best-validation per-class metrics, precision/recall, and actual training crop/pixel distributions.
- Checkpoints that preserve sampler RNG/configuration and split/cache identity, along with the optimizer, scheduler, completed epoch, early-stop counter, best model/score, weights, history, and loader RNG states.

Document the actual new entry point and configuration paths once implemented. This guide intentionally gives no executable training command for a module that does not yet exist.

Before training, generate and save a small RGB/target grid with the existing palette and readable legend. Show several uniform crops and targeted examples for every anomaly class. Record each source training ID, crop coordinates, branch, targeted class, valid-pixel count, and per-class counts. Assert:

- RGB is float32 `[3,256,256]` in [0,1]; targets are int64 `[256,256]` with only the five IDs or 255.
- Cropping uses identical coordinates for RGB and masks. Transforms remain synchronized.
- Each focused crop contains the class it targeted **after** priority handling and augmentation.
- Every emitted crop has at least 32,768 valid supervised pixels; ordinary random/fallback crops need not contain anomalies.
- Failed focused requests retry another eligible training image for the same class, then use the bounded random-valid fallback. A fallback is explicitly identified rather than falsely marked as containing the requested class.
- Invalid, excluded, and ambiguous pixels remain 255. Include synthetic edge/overlap cases, not only attractive example images.
- All five classes have positive support across the preview set; explicitly show Planter skip.
- Every source field belongs to the official training split. No validation/test content enters the eligibility inventory or sampler.
- Training class/branch counts and sampler seeds are reproducible after checkpoint restoration. Resuming one run must not load the other arm's checkpoint/cache.

Run the relevant existing correctness tests and add small selection/alignment/resume checks for the new sampler. A brief training-only smoke check must disable official test evaluation. Do not run the baseline notebook for one epoch as a smoke test: its later cells automatically evaluate test.

## 9. Train B first, preserve it, then train C

Once the missing integration and sampler checks pass:

1. Freeze and save both configurations. Record code revision, batch size, seed, class weights, crop rule, rejection thresholds/retries, and the 50:50 policy.
2. Create a patch-specific notebook using the reviewed integration. Do not overwrite the existing baseline notebook. Include only training and validation during development; leave official test execution as a separate deliberate final step.
3. Run setup and crop/model checks interactively. Reset the seed inside the training entry point so previews do not affect the actual run.
4. Select **Save Version > Save & Run All** to execute B in the background. It starts at the first cell, so all paths/configuration/imports must be self-contained. Do not also start an interactive training run.
5. Wait for completion, verify B's best/last checkpoints, history, validation metrics, and output figures, and save/download its outputs.
6. Run C in a separate saved version or separate notebook, with its own output directory and a fresh seed-42 model. Keep the same GPU type/settings where practical. Change only the sampling policy and necessary run name/path.
7. Save/download C, compare B/C on validation data, and document the selection before opening test pixels.

Use these writable Kaggle locations:

```text
/kaggle/working/experiment_2/uniform_patch/
/kaggle/working/experiment_2/anomaly_aware_patch/
```

Do not use `/kaggle/input/` for outputs or reuse `/kaggle/working/agriculture_segmentation/`. Run one arm at a time; this implementation targets one GPU, even if Kaggle offers two.

Checkpoints should use atomic writes. `last.pt` resumes training; `best.pt` is the validation-selected inference model. Preserve the existing convention that the resumable checkpoint also carries the best weights so a new session can recreate `best.pt`. If a runtime is interrupted, resume from the last **completed** epoch, not from a guessed batch number. `epochs=30` means 30 total epochs, not 30 additional epochs.

Save a version with output files and download essential artifacts before ending the runtime. After an interactive run, use Quick Save with output saving enabled instead of accidentally launching another Save & Run All. Kaggle documents these options in its [advanced save announcement](https://www.kaggle.com/discussions/product-feedback/161328). Reattach a saved notebook output or checkpoint dataset in a new session and copy its real input path into the patch-specific resume setting. Do not assume `/kaggle/working/` survives a discarded runtime.

## 10. Evaluate fairly and reserve test for the selected model

After each training epoch, validate on **all official validation images**, deterministically resized to 256 x 256, without random crops or augmentation. Use `model.eval()` and disabled gradients. Accumulate one confusion matrix for the whole split, with true classes as rows and predictions as columns, after removing target 255.

Save per-class IoU, Dice, precision, recall, support, five-class mean IoU, anomaly-only mean IoU, pixel accuracy, and the confusion matrix for the best-validation checkpoint. Compute ratios from aggregate counts, not by averaging per-batch ratios.

```text
IoU       = TP / (TP + FP + FN)
Dice      = 2*TP / (2*TP + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

`metrics_from_confusion` already computes IoU, Dice, support, mean IoU, anomaly mean IoU, and accuracy. Add an experiment-specific reporting helper for precision/recall rather than changing the baseline's archived results. For example, this helper can process an already accumulated matrix without any new inference:

```python
import numpy as np
from src.utils import metrics_from_confusion

def extended_metrics(matrix):
    cm = np.asarray(matrix, dtype=np.int64)
    result = metrics_from_confusion(cm)
    true_counts, predicted_counts = cm.sum(axis=1), cm.sum(axis=0)
    for i, row in enumerate(result['per_class']):
        tp = int(cm[i, i])
        row['precision'] = float(tp / predicted_counts[i]) if predicted_counts[i] else None
        row['recall'] = float(tp / true_counts[i]) if true_counts[i] else None
    return result
```

Zero denominators should be undefined (`null` in JSON), not invented perfect scores. An absent target class with false-positive predictions still has IoU zero. Report which defined classes contribute to mean IoU. When Planter skip has target pixels but no predictions, its IoU/Dice/recall are zero and precision is undefined. Accuracy is supplementary.

Choose the final model by the pre-agreed **five-class validation mean IoU**; report rare-class tradeoffs alongside it rather than changing the selection metric after seeing results. Use the same checkpoint rule for B and C. A single seed is descriptive evidence, not proof of statistical significance.

Do not load official test images/labels for crop selection, debugging, failure analysis, stopping, or tuning. The baseline test has already been evaluated and discussed, so describe it honestly as a previously reported benchmark; it is not a newly unseen dataset. No further test-driven choices are permitted.

After implementation, configurations, and the B/C choice are frozen, evaluate **only the selected new model** once on the official test set, using the original full-image resize/ignore/argmax protocol. Do not test both arms repeatedly to decide which looks better. Reuse the archived baseline test results for the final comparison; no baseline rerun is needed.

The existing `src.evaluate.evaluate()` can be reused if the patch runner emits compatible `best.pt` keys (`model`, `config`, `schema`, `split_fingerprint`, `epoch`, `best_score`, `resume_capable=False`) and the same target schema. It does not produce precision/recall itself; extend the saved matrix report using the helper above, without a second model-inference pass. Verify compatibility during implementation, before accessing test content.

Keep comparisons on the same split: compare B validation against C validation; compare the final selected model's test results against A's test results. Do not claim improvement by comparing B/C validation values to A test values. A's best validation mean IoU is available, but its per-class validation values are unavailable in the current archive.

## 11. Save and return these artifacts

Recommended downloaded layout (proposed directories; not created by this document):

```text
results/experiment_2/
  uniform_patch/
    config.json
    sampling_config.json
    environment.json
    source_revision.txt
    class_weights.json
    training_history.csv
    sampling_history.csv
    sampler_sanity.png
    best.pt
    last.pt
    best_validation/
      metrics.json
      per_class_metrics.csv
      confusion_matrix.csv
      confusion_matrix.png
      visualizations.json
      sample_*.png
    run_notes.txt
  anomaly_aware_patch/
    ...same files...
  comparison.json
  selected_model.txt
  selected_test/
    metrics.json
    per_class_metrics.csv
    confusion_matrix.csv
    confusion_matrix.png
    visualizations.json
    sample_*.png
```

The new runner may retain the existing `history.csv` naming convention; choose and document one consistent name rather than saving competing histories. `selected_model.txt` must identify the run, checkpoint hash/epoch, validation selection score, and frozen configuration/revision.

For each arm, confirm you saved:

- [ ] Full training/sampling configuration, including seed 42, input/crop sizes, branch/class probabilities, image/pixel selection rule, validity/retry policy, sample budget, and augmentation.
- [ ] Actual class weights, their training-only provenance, source revision/archive hash, dataset/split fingerprint, and sampler inventory identity.
- [ ] Epoch training/validation loss, validation metrics, learning rate, completed updates, and early-stop outcome.
- [ ] `best.pt` and resumable `last.pt`, with correct run identity and RNG states.
- [ ] Best-validation per-class IoU, Dice, precision, recall, support, both mean IoUs, accuracy, and raw/labeled confusion matrix.
- [ ] Actual class pixels seen per epoch, ignored pixels, requested versus emitted uniform/focused counts, requested/successful targeted-class counts, retries, random-fallback counts/reasons, terminal failures, and total emitted samples. Count augmented/resampled training crops actually consumed, not the original dataset once. Report the actual focused proportion after fallbacks.
- [ ] Fixed validation visual examples, representative failure cases, class-color legend, and sample-selection manifest. Include rare-class examples when present; label them qualitative selections.
- [ ] Wall-clock time, epoch/update counts, GPU, Python/PyTorch/CUDA/library versions, and determinism settings.
- [ ] Notes on errors, interruptions, deviations from the agreement, and resume paths.
- [ ] Final frozen B/C validation comparison and selected model; one final selected-model test report when authorized by completion of the protocol.

Keep large checkpoints and raw notebook downloads outside Git. This repository ignores `/results/`, `outputs/`, `/results.zip`, `*.pt`, and `*.pth`, with one explicit exception for `experiments/overfit_diagnostic/selected_tensors.pt` (the small diagnostic inputs). Necessary source, documentation, tests, baseline evidence, and curated diagnostic evidence are versioned and included in the handover. Return/download full future run archives privately, then curate small metrics/configurations/figures into `experiments/experiment_2/` if requested. Do not push duplicate source repositories, `.git` folders, `.venv`, or notebook-rendered image copies.

## 12. Interpret outcomes without overstating them

| Observed validation outcome | Defensible interpretation |
|---|---|
| B improves over A's available validation summary | Crop/native-detail training is promising; it does not establish an anomaly-aware benefit |
| C improves over B | Targeted sampling adds benefit under the shared protocol |
| B improves, but C is similar | Cropping may explain most of the gain; targeted sampling added little |
| Rare-class recall rises but precision falls | The model finds more anomalies but creates more false positives |
| Some rare classes improve while others/normal decline | A tradeoff, not uniform improvement; show all classes |
| Neither new arm improves meaningfully | Sampling alone did not solve the weakness under this budget |

Inspect both metrics and the location/type of mistakes in validation figures. Do not select only attractive examples or omit classes that worsened. A model predicting more red pixels is not automatically better at Planter skip. A higher pixel accuracy can reflect increased normal-class dominance.

The diagnostic supports model capacity on a tiny subset; it does not guarantee patch sampling will improve full-field generalization. Report scale/context limitations and the absence of repeated seeds. Do not add a new architecture, loss, resolution protocol, or extra sampling strategy to rescue an unfavorable result.

## 13. Troubleshooting and scope limits

| Symptom | Action |
|---|---|
| Input path missing | Use the invited Kaggle account, attach the existing shared input, and copy its actual mount; check the `agriculture-vision-2017` / `agriculturevision-2017` spelling. Access is already arranged. Do not edit official splits to hide missing data. |
| Source archive not detected | Use the complete `experiment-2-handover-v1` archive with `HANDOVER_VERSION.json`, diagnostic source/evidence, and baseline evidence. Attach one code snapshot or set the source path explicitly. Update an older Kaggle dataset version and restart the kernel to clear cached imports. |
| CUDA unavailable | Enable a GPU accelerator and restart before training. Check account access/quota; do not launch a full CPU run accidentally. |
| CUDA deterministic NLL/Cross-Entropy error | The source already uses `torch.use_deterministic_algorithms(True, warn_only=True)`. Check that your uploaded revision contains the fix. A separate cell may be overridden by `seed_everything()`. Keep the seed and record warning mode; exact CUDA identity is not guaranteed. |
| `can only test a child process` during DataLoader cleanup | If epochs continue, preserve the log and let the run finish. If the run fails, inspect the first earlier error. For a fresh paired run, agree on `num_workers=0` for both arms if needed. Existing resume config checks reject changing workers mid-run; do not bypass them silently. |
| GPU out of memory | Before the full pair, choose a batch size fitting both arms. Start fresh run directories if changing it. Do not reduce U-Net width or input dimensions to make one arm fit. |
| Requested focused crop lacks its targeted class | Check final target eligibility, row/column ordering, inclusive ranges, and post-transform counts. Retry another eligible training image for the same class, up to the retry limit, then use/log a random valid fallback. If a crop is marked as a successful focused sample but lacks its class, fix that bug before training. Never relabel the missing class or ignored pixels as normal. |
| NaN loss, insufficient valid area, all-ignored batch, or missing class | Inspect crop validity, weights, target IDs, and denominator. Every emitted crop must be at least 50% valid. Use the agreed bounded retries/fallback for sampling failures; stop if no valid fallback is found or the numerical loss is nonfinite. Existing loss skips all-ignored batches, but the patch sampler must not emit them or silently reduce the update budget. |
| Run directory/checkpoint mismatch | Use a new run-specific output path or resume the matching `last.pt`. Preserve existing results. Changing the sampling mode is a new run, not a resume. |
| Runtime disconnected | Resume from a preserved completed-epoch checkpoint. If nothing was saved outside the discarded runtime, do not claim it can be recovered. |
| Evaluation refuses existing metrics | Reuse saved final results. Do not delete the guard just to repeat test inference. |

There are exactly two new main runs in scope: B and C. No transformers, attention, pretrained encoders, NIR, new datasets, complex losses, large sweeps, or additional sampling alternatives. Use the agreed stopping rule and available coursework budget. If implementation or training remains problematic, document the issue and contact the project owner before expanding scope.

## 14. Start-to-finish checklist and decisions to confirm

- [x] Project data, Kaggle resources, and GitHub repository invitations/shared access are arranged, as confirmed by the owner.
- [ ] Use Git tag `experiment-2-handover-v1` or the matching complete ZIP; record its exact commit/archive hash and attach the shared data sources in the Experiment 2 notebook.
- [ ] Read the baseline and diagnostic findings; preserve them and do not rerun either experiment.
- [ ] Confirm native 512-to-256 crop semantics and unchanged full-image 256 validation/test evaluation.
- [ ] Confirm the proposed class-then-image-then-pixel selection rule and 50:50 branch policy.
- [ ] Confirm fixed baseline class weights, 4,505 slots per epoch, shared batch/worker settings, validation-mIoU selection, scheduler, and stopping rule.
- [x] Owner-approved starting defaults: 256 x 256 crops, at least 50% valid pixels, at most 10 retries, another eligible same-class training image for targeted retries, then a random valid training fallback; preserve 255 and use training data only. These are practical defaults, not proven optima.
- [ ] Verify both implementations use those same defaults, bounded retry counting, and explicit requested-versus-emitted/fallback logging before training.
- [ ] Implement/review only the missing sampler, patch runner, reporting, and future patch-specific notebook. Record their real commands once they exist.
- [ ] Verify GPU, paths, official counts/fingerprint, mapping, and native training pair.
- [ ] Pass training-only sampler previews and correctness/resume checks before full training.
- [ ] Run B in the background, save/download all outputs, and verify them.
- [ ] Run C separately with identical shared settings, save/download all outputs, and verify them.
- [ ] Compare B/C on validation only and freeze the selected model/revision/configuration.
- [ ] Evaluate only that selected model once on official test; reuse A's archived test results.
- [ ] Complete the report below and return the full artifacts plus small comparison evidence.

Shared access is confirmed and the required handover source/evidence are included in `experiment-2-handover-v1`. Record its resolved commit or archive manifest, and update the attached Kaggle code dataset to that version rather than using a stale upload. No new training notebook or Experiment 2 sampling code was created as part of the handover.

## 15. Results template to return to the owner

Fill actual measurements; leave unavailable values blank and explain them. Percentages below must use the same scale throughout.

```text
Experiment 2 report
Teammate / Kaggle notebook links:
Source commit/archive hash:
Dataset and split fingerprint:
GPU / software versions:
Agreed sampling, weights, validity/retry, batch/worker deviations:

Run B (uniform): best epoch / completed epochs / updates / duration:
Run C (anomaly-aware): best epoch / completed epochs / updates / duration:
Output archive links:
Checkpoint hashes and best-validation metrics paths:
Requested versus emitted uniform/focused counts and class frequencies:
Targeted retry counts, same-class image reselections, random fallbacks/reasons:
Actual focused proportion after fallbacks and crop-validity checks:
Interruptions, errors, or deviations:
```

| Validation metric | B: uniform | C: anomaly-aware | C minus B (percentage points) |
|---|---:|---:|---:|
| Normal farmland IoU | | | |
| Drydown IoU | | | |
| Double plant IoU | | | |
| Weed cluster IoU | | | |
| Planter skip IoU | | | |
| Five-class mean IoU | | | |
| Anomaly-only mean IoU | | | |
| Pixel accuracy (supplementary) | | | |

Attach the complete per-class Dice/precision/recall/support tables and confusion matrices for B and C. Note the existing A best-validation mean IoU of 22.3383% separately; its per-class validation table is unavailable in the provided archive.

```text
Selected run / epoch / checkpoint hash:
Selection rationale using the agreed validation criterion:
Important rare-class precision/recall tradeoffs:
Representative validation successes/failures and their sample IDs:
Was all implementation/configuration frozen before the selected test pass?
Final selected test metrics and output path (if completed):
```

| Final test metric | Existing A baseline | Selected new model | Difference (percentage points) |
|---|---:|---:|---:|
| Normal farmland IoU | 73.43% | | |
| Drydown IoU | 41.17% | | |
| Double plant IoU | 0.12% | | |
| Weed cluster IoU | 1.38% | | |
| Planter skip IoU | 0.00% | | |
| Five-class mean IoU | 23.22% | | |
| Anomaly-only mean IoU | 10.67% | | |
| Pixel accuracy (supplementary) | 76.06% | | |

Conclude with: Did crop training help? Did anomaly-aware sampling add benefit over uniform crops? Which classes improved or worsened? What limitations remain? Do not describe diagnostic memorization scores as generalization results or claim significance from one seed.

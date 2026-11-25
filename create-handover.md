I want you to create a detailed but easy-to-follow handover guide for Experiment 2 of my Advanced Machine Learning coursework.

Create a new Markdown file:

`EXPERIMENT_2_HANDOVER.md`

This document is intended for another teammate who will take over the implementation and execution of Experiment 2 using their own Kaggle account and notebook.

Do not implement Experiment 2 yet. Your task is to inspect the existing repository, understand the current baseline and diagnostic, and write an accurate step-by-step handover guide.

# 1. Project Background

Our project focuses on agricultural anomaly segmentation using Agriculture-Vision 2017 Miniscale.

We use aerial RGB images to classify farmland pixels into five classes:

| Label | Class |
|---|---|
| 0 | Normal farmland |
| 1 | Drydown |
| 2 | Double plant |
| 3 | Weed cluster |
| 4 | Planter skip |
| 255 | Ignored pixels |

The task is multi-class semantic segmentation.

Our model is U-Net trained from scratch using RGB images. We use weighted Cross-Entropy Loss to address class imbalance.

Our main challenge is that normal farmland occupies most of the supervised pixels, while some anomaly classes are extremely rare.

# 2. Existing Experiment 1: Baseline

The baseline is already completed and must not be modified or retrained.

The baseline:

- Uses full images resized to 256 × 256.
- Uses RGB input.
- Uses a standard U-Net trained from scratch.
- Uses weighted Cross-Entropy Loss.
- Uses official field-level train, validation, and test splits.
- Uses validation performance to select the best checkpoint.
- Evaluates on the official test set.

Baseline test IoU:

| Class | IoU |
|---|---:|
| Normal farmland | 73.43% |
| Drydown | 41.17% |
| Double plant | 0.12% |
| Weed cluster | 1.38% |
| Planter skip | 0.00% |
| Mean IoU | 23.22% |
| Anomaly-only mean IoU | 10.67% |

The model recognizes Normal farmland and Drydown reasonably better than the remaining anomaly classes.

Do not overwrite, delete, or modify the baseline checkpoints, metrics, visualizations, or training history.

# 3. Completed Diagnostic

A small overfitting diagnostic has already been completed.

The unchanged U-Net trained on four fixed 64 × 64 crops containing all five classes.

It successfully memorized the selected crops:

| Class | Training-subset IoU |
|---|---:|
| Normal farmland | 99.00% |
| Drydown | 99.52% |
| Double plant | 98.06% |
| Weed cluster | 94.32% |
| Planter skip | 98.65% |
| Mean IoU | 97.91% |

This was only a memorization test. These scores are NOT validation or test performance.

The diagnostic suggests the model is capable of learning all five classes on a small controlled subset. It does not establish why the full-image baseline failed.

The diagnostic files are stored in:

`results/overfit_diagnostic/`

Do not repeat this diagnostic unless there is a specific reason.

# 4. Objective of Experiment 2

We want to investigate whether showing the model more anomaly-containing image crops during training can improve segmentation of rare anomaly classes.

Instead of always feeding the entire farmland image to the model, we sample smaller crops from the original images.

The crops should provide more opportunities for the model to learn anomalies that occupy only a small portion of farmland.

The model architecture remains U-Net. We are investigating a change in training data sampling, not a new architecture.

The research question is:

"Does patch-based training, particularly anomaly-aware sampling, improve agricultural anomaly segmentation compared with standard full-image training?"

# 5. Experiments to Perform

The handover should explain the following three runs clearly.

## Run A: Existing Full-Image Baseline

Already completed.

- Full image resized to 256 × 256.
- Existing results will serve as our reference.
- Do not retrain.

## Run B: Uniform Random Patch Sampling

Train the same U-Net using randomly selected 256 × 256 crops from the original 512 × 512 training images.

Purpose:

Determine whether training on native-resolution crops instead of downscaled full images improves segmentation.

The crops should be sampled without deliberately prioritizing anomaly-containing regions.

This run serves as a control for Experiment 2.

## Run C: Anomaly-Aware Patch Sampling

Train the same U-Net using a combination of ordinary random crops and anomaly-focused crops.

Proposed sampling ratio:

- 50% ordinary random crops.
- 50% anomaly-aware crops.

For anomaly-aware crops:

- Select an anomaly class.
- Select a valid labeled pixel belonging to that class.
- Sample a crop containing that pixel.
- Balance anomaly-focused selections across the four anomaly classes where possible.
- Avoid allowing Drydown to dominate anomaly-focused sampling.
- Retain ordinary random crops so the model continues learning normal farmland and broader field context.

The sampling must use training labels only.

The validation and test sets must never be used to select crop locations or guide training.

If this proposed sampling method has technical problems, document them and suggest a reasonable alternative rather than silently changing the experiment.

# 6. Fair Comparison

Explain to the teammate that the comparison should be as controlled as reasonably possible.

Keep the following the same across Runs B and C:

- U-Net architecture
- Input channels
- Output classes
- Optimizer
- Learning rate
- Loss function
- Class weights, if applicable
- Number of training samples or updates per epoch
- Epoch budget
- Checkpointing procedure
- Augmentations
- Validation set
- Evaluation metrics
- Random seed where practical

The main difference should be how training crops are selected.

Explain why Run B is important: without it, we would not know whether improvement comes from anomaly-aware sampling or simply training on native-resolution crops.

# 7. Dataset and Mask Handling

Inspect the repository to identify the actual dataset paths and mask preprocessing implementation.

The original dataset is available in Kaggle at:

`/kaggle/input/datasets/abelnathanaelh/agriculture-vision-2017/data2017_miniscale`

The split JSON is:

`/kaggle/input/datasets/abelnathanaelh/data2017-splits/data2017_splits.json`

The dataset contains RGB images and anomaly annotations.

The segmentation pipeline must continue to respect:

- Official field-level splits.
- The five selected classes.
- Ignore index 255.
- Invalid field regions.
- Excluded anomaly classes.
- Ambiguous overlapping labels.

Do not accidentally treat excluded anomalies as Normal farmland.

Do not create data leakage between training, validation, and test fields.

The handover must explain how crop coordinates are applied consistently to both RGB images and their segmentation masks.

Mask crops must use appropriate resizing/interpolation if resizing is necessary. Never use bilinear interpolation for categorical segmentation masks.

# 8. Kaggle Setup Guide

Write a beginner-friendly, step-by-step guide for the teammate to run Experiment 2 using their own Kaggle account.

Include:

## Step 1: Prepare Kaggle

Explain how to:

- Sign in to Kaggle.
- Create a new notebook.
- Select GPU accelerator.
- Attach the Agriculture-Vision dataset.
- Attach the official split JSON dataset.
- Confirm the dataset paths are accessible.
- Confirm that GPU is available.

Do not assume their notebook has access to my local computer or my existing Kaggle session.

## Step 2: Get the Project Code

Inspect the repository and recommend the easiest reliable way to obtain the code inside their Kaggle notebook.

For example, if the repository is public and GitHub cloning is suitable, provide the appropriate workflow.

If it is private, explain alternatives such as uploading the necessary notebook or project files.

Do not invent a repository URL. Use the real URL if it exists in the repository configuration, or leave a clearly marked placeholder if it cannot be determined.

## Step 3: Verify the Environment

Provide commands or notebook checks for:

- Python version.
- PyTorch version.
- CUDA availability.
- GPU name.
- Dataset paths.
- Split counts.
- Class mapping.
- A sample image and mask.

## Step 4: Run a Small Sampler Sanity Check

Before training, instruct the teammate to verify that:

- Crops have the expected dimensions.
- RGB and masks are aligned.
- Anomaly-aware crops actually contain their targeted class.
- Normal random crops still contain valid training pixels.
- Ignore pixels remain correctly ignored.
- All five classes can appear in the training samples.
- Sampling does not accidentally use validation or test labels.

The sanity check should produce a small visualization grid of RGB crops and corresponding masks.

## Step 5: Run Training

Explain the order in which the teammate should run the experiments:

1. Uniform random patch sampling.
2. Save its results and checkpoint.
3. Anomaly-aware patch sampling.
4. Save its results and checkpoint.

The teammate should not run both simultaneously if GPU memory or Kaggle session limits make it unsafe.

Include guidance on saving checkpoints and outputs to `/kaggle/working/` and downloading important results before the Kaggle session ends.

# 9. Evaluation

Both patch experiments must be evaluated using the same validation protocol.

Required metrics:

- Per-class IoU.
- Mean IoU.
- Anomaly-only mean IoU.
- Per-class Dice.
- Precision and recall.
- Pixel accuracy.
- Confusion matrix.

Emphasize that overall pixel accuracy is insufficient because Normal farmland dominates the dataset.

The official test set must remain untouched during Experiment 2 development and checkpoint selection.

Only after the experiment is finalized should the chosen model be evaluated on the official test set, following the original project evaluation protocol.

If the existing baseline test results are already available, do not rerun baseline test evaluation just for comparison.

# 10. Results to Save

Create a checklist of outputs that the teammate must save for each run:

- Training configuration.
- Sampling configuration.
- Random seed.
- Training history.
- Best checkpoint.
- Best validation metrics.
- Per-class IoU and Dice.
- Precision and recall.
- Confusion matrix.
- Visual prediction examples.
- Representative failure cases.
- Class distribution actually seen during training.
- Time taken.
- GPU used.
- Notes on errors or deviations.

Recommend a clear directory structure that does not overwrite Experiment 1 or diagnostic results.

For example:

`results/experiment_2/uniform_patch/`

`results/experiment_2/anomaly_aware_patch/`

Adapt this if the existing project already has a better structure.

# 11. How to Interpret Results

Explain that Experiment 2 is not guaranteed to improve every class.

Possible outcomes include:

- Rare-class IoU improves.
- Rare-class recall improves but false positives increase.
- Some anomaly classes improve while others worsen.
- Normal farmland segmentation decreases.
- Patch sampling produces no meaningful improvement.
- Anomaly-aware sampling performs better than uniform patches.
- Uniform patches improve over the baseline, but anomaly-aware sampling adds little.

The teammate must not cherry-pick metrics.

Explain that improvement should be assessed using validation results first and then confirmed using the official test set only after final selection.

The key comparison is:

- Full-image baseline vs. uniform patches: Does cropping/native-resolution patch training help?
- Uniform patches vs. anomaly-aware patches: Does targeted sampling add further benefit?

# 12. Stop Conditions and Scope Control

This is a coursework project with limited time.

Do not allow the implementation to expand into:

- New architectures.
- Transformers.
- Additional channels such as NIR.
- New datasets.
- Complex losses.
- Multiple sampling strategies beyond the two agreed patch runs.
- Large hyperparameter searches.
- Excessive tuning.

If training or implementation becomes problematic, document the issue and contact me before expanding scope.

The objective is to complete a fair, reproducible comparison within the available coursework time.

# 13. Final Handover Document Requirements

The final `EXPERIMENT_2_HANDOVER.md` should:

- Be written for a teammate unfamiliar with the project.
- Use plain language.
- Be organized in chronological steps.
- Explain what to do in their own Kaggle notebook.
- Explain what files to download, save, and return.
- Clearly distinguish completed work from work that remains.
- Include practical code snippets or shell commands where appropriate.
- Include troubleshooting for common Kaggle problems.
- Include a checklist they can follow from start to finish.
- Include a final results template for reporting their findings to me.
- Be accurate to the actual repository and code.

Do not implement Experiment 2.
Do not modify the baseline.
Do not create a new training notebook as part of this task.
Do not fabricate results, paths, or repository URLs.

Inspect the repository first, then write the handover Markdown file and summarize what you included and any unresolved decisions that the teammate must confirm before training.
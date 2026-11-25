# Experiment 2: practical guide for Kaggle and Gemini

Follow one stage at a time. Use [EXPERIMENT_2_HANDOVER.md](EXPERIMENT_2_HANDOVER.md) as the technical reference whenever a rule or setting is unclear. This shorter guide does not replace it.

## 1. What are we testing?

We want to know whether training on image patches helps the model recognize agricultural anomalies, especially rare ones.

- **Run B: uniform patches.** Choose random patches without favoring anomalies.
- **Run C: anomaly-aware patches.** Mix random patches with patches chosen around labeled anomalies, following the agreed policy.

Both use the same U-Net and shared training settings. Comparing B with C tells us whether deliberately showing more anomalies adds a benefit beyond cropping alone.

## 2. What is already finished?

**Experiment 1 is complete.** Its results are in `experiments/baseline/`. **The overfit diagnostic is also complete** and showed that the model can memorize all five classes on a tiny training subset. Its evidence is in `experiments/overfit_diagnostic/`; these are not validation/test scores.

**Experiment 2 still needs implementation and execution.** Do not retrain the baseline, repeat the diagnostic, or change the model. The existing baseline trainer does not become a patch trainer just by adding a configuration setting.

## 3. Prepare Kaggle and your Gemini chat

1. Sign in to your invited Kaggle account. Create a new **GPU notebook for Experiment 2** and attach the shared Agriculture-Vision data and official split JSON.
2. Obtain project tag **`experiment-2-handover-v1`**, or the matching complete source ZIP from the owner. Keep this guide and the handover available. Follow handover sections 6–7 for source setup, GPU checks, actual dataset paths, and official split counts.
3. Open a Gemini chat. Give it the handover and the actual files requested at each stage. Do not assume it can see your repository, previous chat, or Kaggle session.
4. Ask for **complete new file contents**, filenames, and Kaggle cells that write those files under `/kaggle/working/` before importing them. Copy and run those cells yourself. A background Run All starts from the first cell, so code must not depend on manually created files from a previous session.

Keep your latest working files and outputs. When asking for a fix, send Gemini the current code and full error message, not just the original source or a description of the error.

## 4. Work through these stages

### Stage 1 — Understand the code; do not implement yet

Provide the handover, `src/dataset.py`, `src/model.py`, `src/train.py`, `src/utils.py`, `src/evaluate.py`, and `configs/baseline.json`. Send files in separate messages if needed.

**Gemini prompt:**

```text
Read these actual project files and EXPERIMENT_2_HANDOVER.md. Explain what can be reused and what is missing for Run B. Identify the dataset, training, checkpoint, and validation integration points. Do not write implementation code yet. Preserve the baseline and follow the handover's agreed methodology.
```

**Finish when:** you understand which new files/cells are needed and which existing components stay unchanged.

### Stage 2 — Implement Run B's uniform sampler

Provide `dataset.py` and the handover's sampling requirements again, along with any current implementation files. Keep this stage limited to preparing crops.

**Gemini prompt:**

```text
Implement only Run B's uniform native-resolution patch dataset, reusing the existing target construction and synchronized transforms. Follow the handover's crop, validity, retry, and fallback rules exactly. Return complete new file contents and Kaggle cells to write/import them. Do not modify baseline files, start training, or access validation/test pixels for sampling.
```

**Finish when:** the new sampler can produce batches in Kaggle without errors.

### Stage 3 — Check Run B's sampler before training

Give Gemini your implemented sampler and its current configuration.

**Gemini prompt:**

```text
Write a small, training-only sanity-check cell for this sampler. Check shapes, image/mask alignment, allowed labels, valid-area requirements, retries/fallbacks, and training-field membership against the handover. Display RGB crops beside their masks with the project legend. Do not train or load official test content. Explain what passing output should look like.
```

Run the cell and inspect the pictures yourself. Fix failed checks before continuing. See handover section 8 for the full checklist.

### Stage 4 — Train Run B and evaluate on validation

Provide the checked sampler, current training utilities, baseline class-weight JSON, and shared configuration. First obtain the separate patch-training integration; then run it.

**Gemini prompt:**

```text
Using this validated sampler, provide a separate Run B training entry point that reuses the existing U-Net and training/checkpoint helpers. Follow the handover's shared settings, fixed baseline class weights, resume behavior, and unchanged full-image validation protocol. Return ordered Kaggle cells for setup, preflight checks, training, and best-checkpoint validation reports. Include all required metrics and artifacts. Do not call the unchanged baseline train() as the patch runner or run official test evaluation.
```

Check setup interactively, then use **Save Version → Save & Run All** for background training. Run one training job at a time. Save/download Run B's outputs before changing code or ending the session.

“Evaluate” here means **validation**, not the official test set. IoU measures overlap between predicted and correct regions; higher is better. Use the handover's validation metric to select the checkpoint.

### Stage 5 — Implement Run C's anomaly-aware sampler

Give Gemini the working Run B sampler and runner, plus the handover's anomaly-aware policy. Preserve Run B's files and outputs.

**Gemini prompt:**

```text
Add Run C's anomaly-aware sampling using the existing Run B implementation. Follow the handover's agreed mixture, class selection, same-class retries, and random-valid fallback. Keep shared training settings and target handling unchanged. Log requested versus successful focused crops and fallbacks. Return only the necessary new files/cells. Do not train yet or replace Run B's saved results.
```

**Finish when:** Run C can produce crops and report which sampling branch actually produced them.

### Stage 6 — Check Run C, then train and validate it

Start with a separate check request:

```text
Create training-only checks and a visualization grid for Run C. Show targeted examples for every anomaly class, including Planter skip. Verify alignment, targeted-class presence for successful focused crops, shared validity rules, and correctly labeled fallbacks. Compare the shared settings against Run B. Do not start training or inspect test pixels.
```

After the checks pass, use this prompt:

```text
Reuse the reviewed Run B runner to train and validate Run C with this checked sampler. Start a fresh seed-42 model, not Run B's checkpoint. Keep the agreed shared settings and use a separate output directory. Provide the exact cells to run and save the same reports/artifacts as B. Keep official test evaluation disabled.
```

Train C separately, then save/download its outputs. Use the handover's paths under `/kaggle/working/experiment_2/` for both runs.

### Stage 7 — Compare results; keep final test separate

Provide both best-validation reports, histories, sampling logs, and representative prediction images.

**Gemini prompt:**

```text
Compare Run B and Run C using their validation results and the handover's selection rule. Report every class, mean IoU, anomaly-only mean IoU, Dice, precision, recall, and relevant false positives. Check actual sampling/fallback counts and explain tradeoffs without cherry-picking. Do not compare validation scores directly with baseline test scores, tune new settings, or run test inference.
```

Once implementation, settings, and the selected model are final, follow **handover section 10** for one official test evaluation of that selected new model. Do not test both runs to choose a winner or rerun baseline test evaluation. Return unsuccessful outcomes honestly; do not add experiments to rescue the results.

## 5. What to send back to the project owner

Return the following for **both B and C**, using the directory/checklist in handover section 11:

- Final source files and Kaggle notebook/version links, including the code revision used.
- Training and sampling configurations, class weights, seed, and environment/GPU details.
- Training history and learning curves.
- `best.pt` and resumable `last.pt` checkpoints, shared privately rather than committed to Git.
- Best-validation metrics: per-class IoU, Dice, precision, recall, support, both mean IoUs, accuracy, and confusion matrix.
- Sampler checks, actual class/branch/fallback counts, prediction examples, and representative failures.
- Time taken and notes about errors, interruptions, fixes, or deviations.
- A short B-versus-C validation comparison and selected-model explanation. Include the selected model's final test artifacts if that last step has been completed.

Use the results template in **handover section 15**. Download outputs before discarding a Kaggle runtime. If Gemini proposes a method change or a problem cannot be resolved within the agreed scope, document it and contact the owner before expanding the work.

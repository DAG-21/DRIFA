## DRIFA-Net Project — Current Implementation Status (Update)

Supersedes `DRIFA_Project_Current_Status_and_Implementation_Guide.pdf` (dated 2026-08-13), which stops at "pending: verify T1000, run smoke test." Everything in that PDF up to and including section 8 (dataset loading, augmentation, architecture retained as-is, Dense(4)+Dense(7) classifier heads) is still accurate and not repeated in full here — only what changed or was learned since is documented below. This file is markdown source (editable); the PDF has no source file, so this supersedes it going forward rather than editing it in place.

Working notebook: `Code_v2.ipynb`. Executed via `papermill Code_v2.ipynb Code_v2.ipynb --log-output --kernel python3` (see `fix_and_run.py`), logging to `execution_log.txt.out` / `.err`.

---

### 1. T1000 migration — resolved

TensorFlow sees the GPU via the `tensorflow-directml` plugin (TF 2.10.1, not CUDA). Two DirectML devices are detected (`GPU:0`, `GPU:1`, `name: DML`). The memory-duplication issue that crashed Colab did not recur locally.

**Important, previously unmeasured finding**: forward-only throughput (`.evaluate()`) runs at ~245ms/step, but real training (forward + backward pass) runs at **~3.4s/step** on this hardware — roughly 14x slower than the evaluate-loop rate, not the ~2x that's typical on CUDA GPUs. With 201 steps/epoch (8,012 aligned training samples, batch size 32, 20% validation split), that's **~11.5 minutes/epoch measured**, not the ~49s/epoch that evaluate-loop timing would suggest. Any future epoch-budget planning on this machine should use the ~11.5 min/epoch figure, not evaluate-loop throughput.

### 2. Baseline training — complete

Full training ran to completion. A second, further training run was deliberately disabled by rolling back to the **epoch-26 checkpoint** (`best_DRIFA_brain_HAM.keras`, also backed up as `best_DRIFA_brain_HAM_epoch26_backup.keras`) as the final baseline model, saved as `best_model_ever.keras`. `model.compile` uses fixed `loss_weights=[0.5, 0.5]` (unchanged from the original design), plain `categorical_crossentropy` for both heads, no `sample_weight` in this baseline.

**Final baseline evaluation** (`X_test_s` + `X_test_h1`, both test sets index-aligned to 1,600/2,003 samples respectively per the original pairing scheme):

| Branch | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Brain MRI (4-class) | 92.4% | 92.7% | 92.4% | 92.2% |
| HAM10000 (7-class) | 77.7% | 55.9% | 51.4% | 53.2% |

Evaluation now correctly uses `np.argmax` on both predictions and one-hot labels (the original binary-thresholding bug from the handoff PDF's pending list is resolved). A confusion-matrix cell (side-by-side, both branches, using `ConfusionMatrixDisplay`) was added as the notebook's new final cell, reusing the already-computed `y_pred_labels`/`y_true_labels` from the preceding prediction cell — no re-run of `model.predict` needed.

The notebook contains some redundant leftover cells from earlier iterative fixes (duplicate `evaluate()`/`predict()` calls at different cell positions) — harmless, not yet cleaned up, out of scope unless requested.

### 3. Diagnosed problem: HAM10000 class imbalance

The Brain MRI branch is solid (92% F1). HAM10000's accuracy/F1 gap (77.7% vs. 53.2%) traces to severe class imbalance in the training data (`Dataset/HAM10000/HAM10000_metadata.csv`, `dx` column):

| Class | Count | % of data |
|---|---|---|
| nv | 6,705 | 66.95% |
| mel | 1,113 | 11.11% |
| bkl | 1,099 | 10.97% |
| bcc | 514 | 5.13% |
| akiec | 327 | 3.27% |
| vasc | 142 | 1.42% |
| df | 115 | 1.15% |

A 58:1 ratio between the largest (`nv`) and smallest (`df`) class. The model was defaulting toward `nv`/`mel`/`bkl`, explaining high accuracy but low recall/F1 on rare classes.

### 4. Class-balanced sample weighting — implemented, training in progress

Added (new cell, inserted right after the train-set alignment cell, before the compile/callbacks cell):
```python
from sklearn.utils.class_weight import compute_class_weight
y_train_h_labels = np.argmax(y_train_h, axis=1)
class_weights_ham = compute_class_weight('balanced', classes=np.arange(7), y=y_train_h_labels)
sample_weight_ham = class_weights_ham[y_train_h_labels]
sample_weight_brain = np.ones(len(y_train_s))
```
Re-enabled the training cell (previously disabled/skipped) to call `model.fit(..., sample_weight=[sample_weight_brain, sample_weight_ham], epochs=20, ...)`, resuming from the existing `best_DRIFA_brain_HAM.keras` checkpoint rather than training from scratch. Final save target changed to `best_model_class_weighted.keras` so the original `best_model_ever.keras` baseline artifact is preserved on disk for comparison, not overwritten.

Computed class weights (via `sklearn`, `'balanced'` strategy): `nv≈0.21x`, `mel≈1.29x`, `bkl≈1.30x`, `bcc≈2.78x`, `akiec≈4.37x`, `vasc≈10.0x`, `df≈12.4x`.

**Status as of this writing: running in the background, epoch 3/20.** Epoch 1 val_loss: 1.200 → Epoch 2 val_loss: 0.688 (both checkpointed). At ~11.5 min/epoch, full 20-epoch completion is expected to take **~3.5-4 hours**, not the ~20 min originally estimated from evaluate-loop throughput — see the throughput note in section 1.

**Expected result** (estimate, not yet measured): accuracy likely *drops* somewhat (e.g. into the high-60s/low-70s%) as the model stops defaulting to `nv`; macro recall/precision/F1 expected to rise into the roughly 60-70% range. This trade (lower accuracy, higher recall/F1) is the intended, correct outcome of class balancing, not a regression.

### 5. Methodological "value addition" — architecture-level candidate, queued

Separate from the class-imbalance fix (a data/loss-level change), a second, architecture-level addition was scoped for presentation purposes, under hard constraints: must be dataset-agnostic (not tuned to HAM10000/Brain-MRI specifically), implementable in a small time budget, and must not risk decreasing accuracy or efficiency versus baseline.

**Key architecture-review finding** (corrects an assumption in an externally-drafted proposal that hadn't inspected the actual code): `MIFA` (`DeeperAttentionLayer`) is not called once after two `MFA` branches — it fires **8 times**, interleaved with `RGSA` blocks at every filter stage (64→64→128→128→256→256→512→512) inside `residual_GLC_branch1`. Its `alpha1`/`alpha2` scale factors are **global learned parameters** (`shape=(1,1,1,C)`), identical for every sample — not sample-adaptive. The actual final fusion point — `Concatenate(axis=-1)([x1, x2])` in the model-build cell, immediately before the shared `GlobalAveragePooling2D` and the two classification heads — has **no attention or weighting at all** today. That is the real, safe insertion point for a new module, not "before MIFA" as originally assumed.

**Candidates evaluated**: Adaptive Fusion Gate (AFG — per-sample competitive softmax weighting of the two branches, `alpha+beta=1`, at the final concat point), Squeeze-and-Excitation channel recalibration on the fused vector, learned attention pooling (replacing plain `GlobalAveragePooling2D`), and Dynamic Weight Averaging (DWA — adjusts the two task-loss weights `[w_brain, w_HAM]` per epoch based on relative loss-improvement rate, `w_brain(t)+w_HAM(t)=1`, converges to standard uncertainty-weighting form).

**Safety design (applies to any of the above)**: naive versions risk an initialization-time regression — e.g. AFG's softmax gives `alpha=beta≈0.5` at init, halving both branches' feature magnitude versus the current unweighted (effectively `1+1`) concat. Fix: wrap the module in a **zero-initialized residual gate**, `output = baseline_output + scale * NewModule(...)`, with `scale` a trainable scalar initialized to 0 — guarantees the model is mathematically identical to the current baseline at the moment of insertion, and can only diverge if doing so lowers the loss during training.

**Known limitation, to state honestly in any presentation**: Brain MRI and HAM10000 are *not* paired samples from the same patient — they're index-aligned only to satisfy Keras's multi-input batching. Use "branch contribution" or "sample-specific feature contribution" in write-ups, not "modality reliability," which implies a clinical pairing that doesn't exist. Additionally, the shared fused vector feeds *both* classification heads, so any per-sample reweighting between branches is a potential trade-off between the two tasks, not a pure win — this is a live training-time dynamic that the zero-init trick does not eliminate (it only guarantees a safe starting point).

**Decision**: implement **AFG**, not DWA, specifically *because* the class-imbalance fix (section 4) is already a loss-reweighting technique — pairing it with DWA (also loss-reweighting) would read as one idea done twice. AFG operates at a different level (feature fusion, not loss weighting), giving a package that spans two distinct layers of the pipeline: a diagnosed-and-measured data-level fix, plus a hypothesis-driven, safety-constrained architecture-level addition. For a fair comparison, AFG will be evaluated **on top of** the class-weighted model from section 4 (not the original unweighted baseline), so its own contribution is isolated rather than conflated with the class-weighting gain.

**Status: not yet implemented.** Queued to start once the section 4 training run finishes and produces the new baseline checkpoint.

### 6. Updated pending work

1. Finish class-weighted training run (section 4), record real before/after metrics (accuracy/precision/recall/F1, both branches) against the section 2 baseline.
2. Implement AFG (zero-init residual-gated) at the `Concatenate([x1, x2])` insertion point in the model-build cell.
3. Smoke-test AFG (shape/dtype check under the `mixed_float16` policy; verify `scale=0` reproduces baseline logits exactly).
4. Train AFG on top of the class-weighted checkpoint from step 1; compare metrics.
5. Presentation write-up: report both contributions separately and honestly hedged (no guaranteed-improvement claims), using "branch contribution" terminology for AFG, and noting the ~11.5 min/epoch T1000 throughput as the reason training scope was limited (genuine constraint, not an excuse).

### 7. Claude Code handoff prompt (updated)

```
Continue the DRIFA-Net local implementation in C:\Dhiren\CV Project\DRIFA-Net.

State: baseline DRIFA (epoch-26 checkpoint) is fully trained and evaluated —
Brain MRI 92.4% acc / 92.2% F1, HAM10000 77.7% acc / 53.2% F1 (macro).
HAM10000's gap is diagnosed as class imbalance (nv 67% vs df 1.15%, 58:1).

In progress: class-balanced sample_weight retraining (20 epochs, resumed from
the existing checkpoint, saves to best_model_class_weighted.keras, does NOT
overwrite best_model_ever.keras). Real throughput on this T1000 (DirectML,
not CUDA) is ~11.5 min/epoch for training (forward+backward) — much slower
than the ~245ms/step evaluate-only throughput, so budget accordingly.

Queued next: Adaptive Fusion Gate (AFG) — a small, zero-init residual-gated,
per-sample competitive softmax module inserted at the Concatenate([x1, x2])
line in the model-build cell (the one true unweighted fusion point; MIFA
itself already fires 8 times inside residual_GLC_branch1 and should not be
modified). Must be evaluated against the class-weighted checkpoint above,
not the original baseline, to isolate its own effect. Do not describe it as
"modality reliability" in any write-up — Brain MRI and HAM10000 samples are
index-paired, not from the same patient; use "branch contribution" instead.

Do not modify MFA, MIFA, RGSA, or residual_GLC_branch1 internals.
Do not overwrite best_model_ever.keras or best_DRIFA_brain_HAM_epoch26_backup.keras.
This remains an adaptation of the DRIFA-Net paper (arXiv 2412.01248), not a
validated exact reproduction — SIPaKMeD was replaced with Brain Tumor MRI.
```

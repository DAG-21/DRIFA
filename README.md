# DRIFA-Net Adaptation — Brain Tumor MRI + HAM10000

This repository adapts **DRIFA-Net** (Dhar et al., WACV 2025 — *"Multimodal Fusion Learning with Dual Attention for Medical Imaging"*, [arXiv:2412.01248](https://arxiv.org/abs/2412.01248)) to a two-dataset dual-branch classification setup: **Brain Tumor MRI** (4 classes) + **HAM10000** skin lesion images (7 classes), trained locally on an NVIDIA T1000.

This is an **adaptation, not a reproduction** of the original paper's protocol — the original work spans five datasets across multiple modality pairs; this project narrows scope to two datasets that were obtainable, replacing the original SIPaKMeD branch with Brain Tumor MRI. See `DRIFA_Project_Current_Status_and_Implementation_Guide.md` for the full rationale and change log.

## What's in this repo

| File | What it is |
|---|---|
| `Code_v2.ipynb` | Main adapted pipeline: dataset loading, preprocessing/augmentation, model build, training, evaluation |
| `AFG_Value_Addition.ipynb` | Adaptive Fusion Gate experiment — an architecture-level addition tested on top of the baseline (see Results below) |
| `DRIFA_Project_Current_Status_and_Implementation_Guide.md` | Full engineering log: dataset substitution, architecture review, training config, hardware notes, decisions and their reasoning |
| `doc_for_report_creation.pdf` / `.html` | Researcher-facing reference document — citation, methodology, results at each stage, honest limitations |
| `requirements.txt` | Python dependencies |

**Model weights are not included** — the trained checkpoints are 200MB-650MB each, well past what's practical to version in git. Results below are the actual measured numbers from local runs; the notebooks show the executed outputs (training logs, printed metrics) directly.

## Datasets

| Branch | Dataset | Classes | Split |
|---|---|---|---|
| Input 1 | Brain Tumor MRI | glioma, meningioma, notumor, pituitary | 5,600 train (→10,373 after augmentation) / 1,600 test |
| Input 2 | HAM10000 | akiec, bcc, bkl, df, mel, nv, vasc | 8,012 train / 2,003 test |

HAM10000 is severely imbalanced — `nv` is 67% of the data, `df` is 1.15% (a 58:1 ratio) — which turned out to be the central problem this project ended up diagnosing and addressing (see below).

## Architecture

Retained from the original repository: **MFA** (multi-branch fusion attention), **MIFA** (multimodal information fusion attention, applied 8 times throughout the dual-branch backbone), and **RGSA** residual attention blocks. Only the final classifier heads were changed (`Dense(5)` → `Dense(4)` for the new Brain MRI branch; HAM10000's `Dense(7)` unchanged). Full architecture data-flow and parameter counts are documented in the status guide.

## Results

**Baseline** (trained backbone, unchanged `loss_weights=[0.5,0.5]`, no class weighting):

| Branch | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Brain MRI (4-class) | 92.4% | 92.7% | 92.4% | 92.2% |
| HAM10000 (7-class) | 77.7% | 55.9% | 51.4% | 53.2% |

HAM10000's gap between accuracy and macro recall/F1 traces directly to the class imbalance above — the model was defaulting toward the majority class rather than learning rare classes well.

**Class-balanced training** (`sklearn` balanced class weights applied to the HAM10000 loss, sqrt-dampened and combined with a lower fine-tuning learning rate for training stability): in progress — this document will be updated with final numbers once the run completes.

**Adaptive Fusion Gate (AFG)**: an architecture-level addition inserted at the one point in the network with no learned weighting today — the final feature concatenation immediately before the classification heads — designed to let the model learn a per-sample, competitive weighting between the two branches, with a zero-initialization guarantee that the model starts mathematically identical to baseline. Tested over a short training run; results did not show an improvement over baseline on either branch. Documented honestly as a negative result — see `doc_for_report_creation.pdf` for the full design, the safety argument, and analysis of why it may not have helped (most likely: too few epochs for the newly-added parameters to specialize, and the two branches being index-paired rather than genuinely paired samples, limiting how much real "branch trust" signal exists to learn from).

## Setup

```
pip install -r requirements.txt
```
Trained and evaluated locally on an NVIDIA T1000 via the `tensorflow-directml` plugin (TensorFlow 2.10.1).

## Original Paper

**DRIFA** (Dual Robust Information Fusion Attention) proposes a multi-branch fusion attention module and a multimodal information fusion attention module, combined with Monte Carlo Dropout for uncertainty estimation, evaluated across five publicly available datasets spanning dermoscopy, pap smear, MRI, and CT-scan modalities.

```bibtex
@inproceedings{dhar2025multimodal,
  title={Multimodal Fusion Learning with Dual Attention for Medical Imaging},
  author={Dhar, Joy and Zaidi, N. and Haghighat, M. and Goyal, P. and Roy, S. and Alavi, A. and Kumar, V.},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025},
  url={https://arxiv.org/abs/2412.01248}
}
```

![image](https://github.com/user-attachments/assets/183e6cfa-c351-4fac-a2ee-5058c5a3a883)

*Figure: DRIFA-Net architecture from the original paper — (A) target-specific multimodal fusion learning phase using MFA/MIFA/RGSA, (B) uncertainty quantification phase.*

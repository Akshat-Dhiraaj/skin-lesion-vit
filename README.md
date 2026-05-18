# Skin Lesion Classification — HAM10000 (v1 22-phase benchmark + v2 honest re-do)

This repository contains the full project:

| Path | What it is |
|---|---|
| `./` (repo root) | **v1** — 22-phase HAM10000 benchmark on image-level splits (this README) |
| [`v2/`](v2/) | **v2** — the honest re-do: lesion-level splits, dermatology foundation models, cross-dataset fairness audit, TENT. Start at [`v2/README.md`](v2/README.md) |
| [`info_bank/`](info_bank/) | All cross-cutting knowledge docs — see [`info_bank/README.md`](info_bank/README.md) |

**Where to start:**
- New here / want the story → [`info_bank/03_COMPLETE_GUIDE.md`](info_bank/03_COMPLETE_GUIDE.md) (32-part teach-yourself guide)
- Final results + figures → [`info_bank/01_RESULTS.md`](info_bank/01_RESULTS.md)
- Every model/technique explained + papers → [`info_bank/02_MODELS_AND_TECHNIQUES.md`](info_bank/02_MODELS_AND_TECHNIQUES.md)
- Findings / publishability / resume bullets → [`info_bank/04_FINDINGS_AND_RESUME.md`](info_bank/04_FINDINGS_AND_RESUME.md)

---

## v1 — 22-Phase Benchmark

A 22-phase progression on HAM10000 dermoscopy classification, from frozen ResNet50 baseline to a 6-model stacked ensemble. Each phase is a numbered experiment under `experiments/`, each model has a saved checkpoint plus a `results.txt`, and the engine is one unified training loop driven by a phase config.

> **Honesty note.** All numbers in this README come straight from `experiments/<phase>/results.txt` files written by the training scripts. The splits used here are **image-level** `sklearn.train_test_split` stratified on `dx` (see [src/dataset.py:87](src/dataset.py:87)). This is known to leak between near-duplicate images of the same lesion. The successor project [`v2/`](v2/) re-runs the strongest models with **lesion-level** splits and is the version to compare against published HAM10000 numbers.

---

## Leaderboard (image-level splits, test set N=1503)

Sorted strictly by test macro OvR AUC, descending. **Note**: Phase 18 (ConvNeXt-V2 standalone) has the highest raw test AUC; Phase 21 (the 6-model stack) has the highest *val* AUC (0.9956) and was the headline result selected by val. Phase 22 Check 3 (`ConvNeXt(0.05) + LogReg(0.95)` at 0.9887 test) is even higher but is a post-hoc sanity-check variant and is not included here — see the Phase 22 table below.

| Rank | Phase | Model / Step | Val AUC | Test AUC | Test Acc | Source |
|------|-------|--------------|---------|----------|----------|--------|
| 1 | 18 | **ConvNeXt-V2 Large @ 384** (single model) | 0.9893 | **0.9880** | 0.8862 | [phase18](experiments/phase18_convnextv2_large/results.txt) |
| 2 | 21 | 6-model stacked ensemble (LogReg meta) — val-selected headline | 0.9956 | 0.9840 | 0.8969 | [phase21](experiments/phase21_final_ensemble/results.txt) |
| 3 | 16 | Unified eff+dino+multimodal (50/50 tune/eval§) | 0.9805 | 0.9819 | — | [phase16](experiments/phase16_unified_pipeline/unified_results.txt) |
| 4 | 14 | 3-model ensemble (eff+TTA / dino-FT+TTA / multimodal) | — | 0.9818 | 0.8909 | [phase14](experiments/phase14_three_model_ensemble/results.txt) |
| 5 | 9  | Ensemble (EfficientNetV2 0.60 / DINOv2+Reg 0.40) | — | 0.9808 | 0.8836 | [phase9](experiments/phase9_ensemble/results.txt) |
| 6 | 11 | Re-ensemble (eff 0.55 / dino+reg full-FT 0.45) | — | 0.9768 | 0.8776 | [phase11](experiments/phase11_ensemble_v2/results.txt) |
| 7 | 8  | EfficientNetV2-M + TTA (8 views) | — | 0.9766 | 0.8696 | [phase8](experiments/phase8_tta/results.txt) |
| 8 | 15 | DINOv2+Reg full-FT + TTA | — | 0.9697 | 0.8470 | [phase15](experiments/phase15_tta_dinov2_ft/results.txt) |
| 9 | 10 | DINOv2+Reg (full fine-tune, lr_bb=5e-6, 25 epochs) | 0.9750 | 0.9692 | 0.8463 | [phase10](experiments/phase10_dinov2_registers_finetune/results.txt) |
| 10 | 2  | EfficientNetV2-M (full fine-tune, 384) | 0.9762 | 0.9685 | 0.8490 | [phase2](experiments/phase2_efficientnetv2/results.txt) |
| 11 | 8  | DINOv2+Reg (frozen) + TTA | — | 0.9663 | 0.8510 | [phase8](experiments/phase8_tta/results.txt) |
| 12 | 5  | DINOv2+Registers (full fine-tune, lr_bb=1e-5, 20 epochs) | 0.9724 | 0.9653 | 0.8390 | [phase5](experiments/phase5_dinov2_registers/results.txt) |
| 13 | 12 | Multimodal (DINOv2+Reg + age/sex/localization) | 0.9750 | 0.9627 | 0.8616 | [phase12](experiments/phase12_multimodal/results.txt) |
| 14 | 13 | Focal loss (γ=2, class-α) on Phase 10 ckpt | 0.9682 | 0.9562 | 0.7731 | [phase13](experiments/phase13_focal_loss/results.txt) |
| 15 | 3  | ViT-Base, last 4 blocks unfrozen | 0.9665 | 0.9538 | 0.7638 | [phase3](experiments/phase3_vit_base/results.txt) |
| 16 | 4  | DINOv2-Base (full fine-tune) | 0.9559 | 0.9458 | 0.7498 | [phase4](experiments/phase4_dinov2/results.txt) |
| 17 | 19 | Mel-binary specialist (DINOv2+Reg, focal binary) | 0.9351† | 0.9402† | — | [phase19](experiments/phase19_mel_binary/results.txt) |
| 18 | 6  | AIMv2-Large (linear probe) | 0.9550 | 0.9376 | 0.6800 | [phase6](experiments/phase6_aimv2/results.txt) |
| 19 | 7  | SigLIP-SO400M (linear probe) | 0.9189 | 0.9069 | 0.5842 | [phase7](experiments/phase7_siglip/results.txt) |
| 20 | 1  | ResNet50 (frozen, head-only) | 0.9114 | 0.8940 | 0.6041 | [phase1](experiments/phase1_resnet50_frozen/results.txt) |

† Phase 19 reports **mel binary AUC** (mel vs rest), not multi-class macro AUC; not directly comparable.  
§ Phase 16 numbers are on a 50/50 split of the test set (`Tune AUC` and `Eval AUC (OOS)` in its unified_results.txt), not a v1-standard 70/15/15 split — listed here only for ranking parity.

### Phase 21 per-class test AUC (best run)

| Class | Test AUC |
|------|----------|
| mel  | 0.9660 |
| nv   | 0.9824 |
| bcc  | 0.9970 |
| akiec| 0.9940 |
| bkl  | 0.9794 |
| df   | 0.9700 |
| vasc | 0.9993 |

Melanoma — AUC 0.9660, recall 0.8323, precision 0.6557. See [phase21 results.txt](experiments/phase21_final_ensemble/results.txt) for the full classification report.

### Phase 22 sanity checks (post-Phase-21)

Three quick checks on the Phase 20 `.npy` artifacts (CPU only, see [phase22](experiments/phase22_quick_checks/results.txt)):

| Variant | Val AUC | Test AUC | Mel AUC | Mel Rcl | Mel Prc |
|---|---|---|---|---|---|
| Baseline: LogReg (Phase 21) | 0.9956 | 0.9840 | 0.9660 | 0.8323 | 0.6557 |
| Calibrated LogReg (per-class isotonic) | **0.9963** | 0.9525 | 0.9622 | 0.8323 | 0.6526 |
| LogReg with cleaner mel-binary expansion | 0.9955 | 0.9838 | 0.9652 | 0.8563 | 0.6190 |
| ConvNeXt-V2 alone (Phase 18) | 0.9896 | **0.9883** | 0.9674 | 0.8563 | 0.5789 |
| ConvNeXt(0.05) + LogReg(0.95) | 0.9956 | 0.9887 | 0.9723 | 0.8323 | 0.6588 |

> The honest choice (best by **val** AUC) is the calibrated LogReg, which underperforms on test — a known overfitting failure on small rare-class val populations.

---

## Dataset

**HAM10000** — 10,015 dermatoscopic images, 7 classes, severe class imbalance.

| Class | Description | Count |
|------|-------------|-------|
| nv    | Melanocytic nevi | 6705 |
| mel   | Melanoma | 1113 |
| bkl   | Benign keratosis-like lesions | 1099 |
| bcc   | Basal cell carcinoma | 514 |
| akiec | Actinic keratoses / Bowen's | 327 |
| vasc  | Vascular lesions | 142 |
| df    | Dermatofibroma | 115 |

`src/dataset.py:load_metadata` reads `data/raw/HAM10000_metadata.csv` and walks both `HAM10000_images_part_1` and `HAM10000_images_part_2` for `.jpg` files. Source: [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

Splits are produced by `split_dataset()` in [src/dataset.py:87](src/dataset.py:87) — image-level stratified, fixed `random_state=42`, default 70/15/15. **No grouping by lesion_id** — this is the leakage caveat at the top of the file.

---

## Repository layout

```
skin-lesion-vit/
├── data/
│   ├── raw/                       HAM10000 metadata + two image folders
│   └── processed/metadata_cleaned.csv
├── src/
│   ├── dataset.py                 image-level splits + HAM10000Dataset + WeightedRandomSampler
│   ├── transforms.py              train (heavy aug) + val transforms
│   ├── model.py                   ViT-Base partial fine-tune + ResNet50 baseline (legacy)
│   ├── train.py                   legacy ViT-only loop (superseded by engine.py)
│   ├── engine.py                  unified phase runner — phases 1-7 and 10
│   ├── advanced_phases.py         phases 11, 12, 13 (re-ensemble / multimodal / focal)
│   ├── tta_ensemble.py            phases 8 (TTA) and 9 (eff + dino+reg ensemble)
│   ├── mega_tta.py                shared TTA inference dumper (writes mega_tta_sync/)
│   ├── opt_slsqp_weights.py       per-class SLSQP weight search (uses mega_tta_sync)
│   ├── opt_learned_thresholds.py  F2-optimal mel threshold (uses mega_tta_sync)
│   ├── opt_xgboost_stack.py       XGBoost meta-learner (uses mega_tta_sync)
│   ├── phase14_three_model_ensemble.py
│   ├── Phase17 maximum push.py    20-view TTA + 5-fold SLSQP overnight run
│   ├── phase16_unified_pipeline.py  unified ensemble + thresholds (50/50 split)
│   ├── phase18_convnextv2_train.py  ConvNeXt-V2 Large @ 384px
│   ├── phase19_mel_binary_train.py  binary mel-vs-rest specialist
│   ├── phase20_val_set_inference.py TTA inference on val+test for 6 models
│   ├── phase21_final_ensemble.py    stacked meta-learner (LogReg / SLSQP / meta-meta)
│   ├── phase22_quick_checks.py      post-21 sanity checks
│   ├── evaluate.py                metrics + GradCAM helpers
│   └── transforms.py
├── experiments/                   one folder per phase, contains best_model.pth + results.txt
│   ├── phase1_resnet50_frozen/  … phase22_quick_checks/    (NO phase17 folder — see Caveats)
│   ├── phase7_medsiglip/          empty stub from an aborted MedSigLIP attempt
│   ├── phase16_optimizations/     legacy slsqp_probs.npy + xgb_probs.npy from opt_*.py
│   ├── mega_tta_sync/             shared eff + dino TTA probs + labels (used by opt_*.py)
│   └── phase20_val_inference/     val+test logits/probs for 6 models (used by phase21/22)
├── checkpoints/                   legacy from train.py (unused now)
├── notebooks/01_eda.ipynb
├── logs/                          run_all.bat captured stdout
├── mlflow.db                      MLflow tracking store (file:./mlruns equivalent)
├── run_all.bat                    runs phases 1-7, 10, 11, 12, 13 in sequence
└── requirements.txt
```

---

## Methodology

### Class imbalance
`get_weighted_sampler()` in [src/dataset.py:104](src/dataset.py:104) builds a `WeightedRandomSampler` with weights `1 / count(class)` for every sample. Used on the training loader; val and test loaders use sequential order.

### Training (engine.py)
- **AMP** via `torch.cuda.amp.autocast` + `GradScaler` (engine.py uses the legacy `torch.cuda.amp` API).
- **Gradient accumulation** — effective batch = `batch_size * accum_steps`. Per-phase values in `PHASE_CONFIGS`.
- **Gradient checkpointing** — explicitly set on ViT-Base in `run_phase` ([engine.py:360](src/engine.py:360)). Other backbones rely on framework defaults.
- **Discriminative LRs** — `lr_head` and `lr_backbone` per phase config. Frozen-backbone phases set `lr_backbone=0.0`.
- **OneCycleLR** — only when `lr_backbone > 0`. 10% warmup.
- **Optimizer** — AdamW, `weight_decay=1e-4`.

### Augmentation (transforms.py)
HFlip, VFlip, RandomRotate90, ShiftScaleRotate (rotate 30°), `OneOf(GaussNoise / GaussianBlur / MotionBlur)`, `OneOf(OpticalDistortion / GridDistortion)`, ColorJitter, then Normalize (ImageNet stats) + ToTensorV2. Val and test: Resize + Normalize only.

### TTA — 8 views
Defined in `get_tta_transforms` ([src/tta_ensemble.py](src/tta_ensemble.py)): original, H-flip, V-flip, rotate 90 / 180 / 270, HV-flip, slight zoom. Softmax probabilities averaged across views.

Phase 17 (the overnight run) uses 20 views, adding colour jitter, multi-scale crops, and elastic distortion. Phase 20 uses per-model view budgets (4–20).

### Ensembling
- **Grid search** (Phase 9): two-model weights stepping by 0.05, summing to 1.
- **Per-class SLSQP** (Phases 16, 17, 21): one weight vector per class, constrained to sum to 1, `bounds=(0, 1)`, multi-restart from Dirichlet inits.
- **Stacking** (Phase 21): a `LogisticRegression` trained on stacked logits from 6 models (`eff`, `dino_ft`, `p13` focal, `p12_mm` multimodal, `convnext`, `mel_binary`). Mel binary's `(N, 2)` output is expanded to `(N, 7)` by diffusing `(1-P(mel))/6` over non-mel classes.
- **Meta-meta** (Phase 21): averages SLSQP-logit / LogReg / (XGBoost — disabled) outputs.
- **F2-optimal per-class thresholds** for `vasc`, `df`, `mel`, applied via priority override (`vasc > df > mel > argmax`).

### Pipeline rule
The pipeline corrected over time. Earlier scripts (`opt_*.py`) optimized weights on the test set, which is leakage; the docstrings of those files document the fix. **Phase 20+ is the honest pipeline**: weights and thresholds are fit on the val set (1502 samples), test (1503 samples) is reserved for the final number.

---

## Models evaluated

Specs come from [src/engine.py:PHASE_CONFIGS](src/engine.py) and the individual phase scripts.

| Phase | Model ID | Backbone | Mode | Input | Batch × Accum | LR head / backbone | Epochs |
|------|----------|----------|------|------|---------------|--------------------|--------|
| 1 | `resnet50` (timm) | CNN | frozen | 224 | 64 × 1 | 1e-3 / 0 | 15 |
| 2 | `tf_efficientnetv2_m` (timm) | CNN | full FT | 384 | 16 × 4 | 1e-3 / 1e-5 | 20 |
| 3 | `google/vit-base-patch16-224` (HF) | ViT | last 4 blocks + head | 224 | 16 × 4 | 1e-3 / 1e-5 | 20 |
| 4 | `facebook/dinov2-base` (HF) | SSL ViT | full FT | 224 | 16 × 4 | 1e-3 / 1e-5 | 20 |
| 5 | `facebook/dinov2-with-registers-base` (HF) | SSL ViT | full FT | 224 | 16 × 4 | 1e-3 / 1e-5 | 20 |
| 6 | `apple/aimv2-large-patch14-224` (via timm `aimv2_large_patch14_224`) | Autoregressive SSL | linear probe | 224 | 16 × 4 | 1e-3 / 0 | 20 |
| 7 | `google/siglip-so400m-patch14-224` (HF, vision tower) | VLM vision | linear probe | 224 | 8 × 8 | 1e-3 / 0 | 20 |
| 10 | `facebook/dinov2-with-registers-base` (HF) | SSL ViT | full FT (low backbone LR) | 224 | 8 × 8 | 1e-3 / 5e-6 | 25 |
| 18 | `convnextv2_large.fcmae_ft_in22k_in1k_384` (timm) | CNN, FCMAE pretrain | full FT | 384 | 4 × 8 | 1e-3 / 3e-6 | 25 |
| 19 | `facebook/dinov2-with-registers-base` (HF, binary head) | SSL ViT | last 4 blocks + binary head, focal loss (α=0.75, γ=2) | 224 | — | — | — |

Phases 11–17, 20–22 do not train new backbones; they reuse the checkpoints above for ensembling, TTA, and meta-learning.

---

## Key findings (from these experiments)

1. **Full fine-tuning ≫ linear probing on HAM10000.** Phase 5 DINOv2+Registers (full FT) lands at 0.9653 test AUC; Phase 4 DINOv2-Base (also full FT) at 0.9458. The Phase 6/7 frozen probes plateau at 0.94/0.91 despite being far larger models.
2. **DINOv2+Registers > DINOv2-Base on the same recipe** — 0.9653 vs 0.9458 — replicating Darcet et al. (2024)'s register-token benefit.
3. **The lowered-LR full FT (Phase 10) beats Phase 5** — 0.9692 vs 0.9653 — at the cost of 5 extra epochs and a 5× smaller backbone LR.
4. **ConvNeXt-V2 Large at 384 is the strongest model in the repo, full stop** — 0.9880 test AUC, 0.8862 acc as a single model. It **beats** the 6-model Phase 21 LogReg ensemble (0.9840) on raw test AUC. The Phase 22 sanity check confirms this: `ConvNeXt + LogReg(0.95)` weighting only nudges to 0.9887 — i.e., LogReg adds essentially nothing over ConvNeXt alone.
5. **Ensembling gives diminishing returns on top of weaker singles, but does NOT outperform Phase 18.** Phase 9 (2-model grid) → 0.9808; Phase 14 (3-model) → 0.9818; Phase 21 (6-model LogReg stack) → 0.9840 — all below Phase 18 standalone. The "Phase 21 as headline" choice was driven by val AUC (0.9956 for Phase 21 LogReg vs 0.9893 for Phase 18 single), not test AUC.
6. **Test-set leakage in early ensembles** — `opt_slsqp_weights.py`, `opt_xgboost_stack.py`, and `opt_learned_thresholds.py` originally tuned on the test set; their docstrings document the fix to a held-out val split. **Phase 20+ uses honest val for tuning.**
7. **Melanoma recall is the binding constraint.** Even the Phase 21 ensemble — best macro AUC — reaches mel recall 0.83 / precision 0.66 only with F2-tuned per-class thresholds.
8. **Calibration helps val but hurts test** here (Phase 22 check 1). The honest-by-val-AUC pick (`Calibrated LogReg`) tanks to 0.9525 test, while uncalibrated LogReg holds 0.9840 — a textbook small-rare-class overfit on isotonic regression.

---

## Setup

```powershell
conda create -n skin-lesion python=3.11 -y
conda activate skin-lesion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```

The data folder is hardcoded in `engine.py` (`DATA_DIR = D:\ML\skin-lesion-vit\data\raw`). Adjust for your install.

---

## Run

```powershell
# Phases 1-7 and 10 — single-model training
python src/engine.py --phase 1     # ResNet50 frozen
python src/engine.py --phase 2     # EfficientNetV2-M
python src/engine.py --phase 3     # ViT-Base partial FT
python src/engine.py --phase 4     # DINOv2-Base
python src/engine.py --phase 5     # DINOv2+Registers
python src/engine.py --phase 6     # AIMv2-Large (probe)
python src/engine.py --phase 7     # SigLIP-SO400M (probe)
python src/engine.py --phase 10    # DINOv2+Registers full FT (refined)

# Phases 11-13 — re-ensemble, multimodal, focal loss
python src/advanced_phases.py --phase 11
python src/advanced_phases.py --phase 12
python src/advanced_phases.py --phase 13

# Phases 8-9 — TTA + 2-model ensemble
python src/tta_ensemble.py --phase 8
python src/tta_ensemble.py --phase 9

# Phase 14 — 3-model ensemble
python src/phase14_three_model_ensemble.py

# Phase 16 — unified pipeline (50/50 split)
python src/phase16_unified_pipeline.py

# Phase 17 — 20-view TTA + 5-fold SLSQP (overnight)
# NOTE: never been completed — no experiments/phase17* folder exists.
python "src/Phase17 maximum push.py"

# Phase 18 — ConvNeXt-V2 Large @ 384
python src/phase18_convnextv2_train.py

# Phase 19 — mel-binary specialist
python src/phase19_mel_binary_train.py

# Phase 20 — TTA inference on val+test (writes phase20_val_inference/*.npy)
python src/phase20_val_set_inference.py

# Phase 21 — final stacked ensemble (depends on Phase 20)
python src/phase21_final_ensemble.py

# Phase 22 — post-21 quick checks
python src/phase22_quick_checks.py

# Bulk run for phases 1-7, 10-13
.\run_all.bat

# MLflow dashboard (reads ./mlflow.db)
mlflow ui
```

---

## Hardware

- GPU: NVIDIA RTX 4060 Laptop, 8 GB VRAM
- RAM: 32 GB
- OS: Windows 11 + PowerShell

All phases were sized to fit 8 GB via mixed precision, gradient accumulation, and (where supported) gradient checkpointing.

---

## Caveats

- **Image-level splits leak across near-duplicate lesion images.** Numbers here overestimate generalization. The v2 project uses `StratifiedGroupKFold(group='lesion_id')` and CI tests to enforce no group overlap.
- **The 50/50 split used in Phase 16** halves the val set used for tuning, making per-class threshold tuning noisy on rare classes (`df=8`, `vasc=11`). Phase 20+ uses the full val set (1502 samples).
- **Phase 19 reports `mel binary AUC`, not multi-class macro AUC** — see results.txt.
- **Phase 17 (`src/Phase17 maximum push.py`) was never completed** — no `experiments/phase17*` folder. The script exists and is runnable; nothing depends on its output.
- **`experiments/phase7_medsiglip/`** is an empty folder from an aborted MedSigLIP attempt.
- **`experiments/phase16_optimizations/`** holds artifacts from the older `opt_*.py` scripts (which have leakage caveats — see their docstrings).
- **`1.0.0` in this directory is a captured pip-install transcript**, not a release note. Ignore.

---

## Successor

[`v2/`](v2/) ports the strongest models to **lesion-level splits** with pytest leakage tests, adds DermLIP-PanDerm and SigLIP 2, and extends to **cross-dataset evaluation** (ISIC 2019, DDI, Fitzpatrick17k) with Fitzpatrick-stratified fairness reporting.

---

## References

- Tschandl, Rosendahl, Kittler (2018). [*The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*](https://www.nature.com/articles/sdata2018161). Scientific Data 5:180161.
- Dosovitskiy et al. (2021). [*An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929). ICLR 2021.
- Tan & Le (2021). [*EfficientNetV2: Smaller Models and Faster Training*](http://proceedings.mlr.press/v139/tan21a/tan21a.pdf). ICML 2021.
- Oquab et al. (2023). [*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193). arXiv 2304.07193.
- Darcet, Oquab, Mairal, Bojanowski (2024). [*Vision Transformers Need Registers*](https://arxiv.org/abs/2309.16588). ICLR 2024.
- Fini et al. (2024). [*Multimodal Autoregressive Pre-training of Large Vision Encoders*](https://arxiv.org/abs/2411.14402) (the AIMv2 paper). arXiv 2411.14402; CVPR 2025 Highlight.
- Zhai, Mustafa, Kolesnikov, Beyer (2023). [*Sigmoid Loss for Language Image Pre-Training (SigLIP)*](https://arxiv.org/abs/2303.15343). arXiv 2303.15343.
- Woo et al. (2023). [*ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*](https://arxiv.org/abs/2301.00808). CVPR 2023.
- Lin, Goyal, Girshick, He, Dollár (2017). [*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002). ICCV 2017.

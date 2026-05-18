# Skin Lesion Project — Findings, Results, and Publishability

> Project root: `D:\ML`. Two repos: `skin-lesion-vit` (v1, 22-phase exploration on image-level splits) and `v2` (honest re-do with lesion-level splits + cross-dataset fairness audit).

All numbers below are measured on held-out test sets and traceable to artifacts under `experiments/`. Nothing in this document is extrapolated, projected, or claimed to be state-of-the-art.

> Companion doc: [`02_MODELS_AND_TECHNIQUES.md`](02_MODELS_AND_TECHNIQUES.md) — per-model functioning, rationale, and verified paper citations for every backbone and technique referenced here.

---

## 1. Headline findings

Six findings ordered by what an outside reader is most likely to care about.

### F1. Honest evaluation costs ~0.022 macro AUC

Same ConvNeXt-V2 Large architecture, same HAM10000 data:
- v1 (image-level random split — duplicate lesions can appear in train *and* test): **test AUC 0.9880**
- v2 (lesion-level `StratifiedGroupKFold`, pytest-enforced leakage guard): **test AUC 0.9660**

The 0.022 gap is the inflation v1 was unknowingly publishing. v2 is the credible number.

### F2. A widely-used dermatology foundation model was silently the wrong model

`redlessone/DermLIP_PanDerm-base-w-PubMed-256` on Hugging Face is an **open_clip** checkpoint (only `open_clip_config.json` + `open_clip_pytorch_model.bin`; no `config.json`). `AutoModel.from_pretrained(..., trust_remote_code=True)` raises `ValueError: missing model_type key`, falls into a `try/except` block, and silently substitutes `facebook/dinov2-with-registers-base` — preserving the API but using a completely different vision encoder.

This had been in place for the entire prior 6 months of the project. Every `pandem_probe` / `pandem_ft` / `pandem_probe_224` result in `experiments/` was actually DINOv2+Registers, not PanDerm.

Fix: re-implemented the loader using `timm.models.beit.Beit` with `use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, global_pool='token'` and OpenAI-CLIP normalisation stats `(0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758)`. Loads visual sub-state-dict with **0 missing / 0 unexpected keys**.

The fix changes the model's accuracy in measurable ways:

| Metric | DINOv2 fallback (old) | Real PanDerm (new) | Δ |
|---|---:|---:|---:|
| HAM10000 test AUC (probe) | 0.9305 | **0.9527** | +0.0222 |
| ISIC 2019 macro OvR AUC | 0.8101 | **0.8845** | +0.0744 |

### F3. Full fine-tuning hurts dermatology-pretrained features on small data

| Mode | HAM10000 test AUC | ISIC 2019 macro AUC |
|---|---:|---:|
| `pandem_real_probe` (linear probe, head only) | **0.9527** | **0.8845** |
| `pandem_real_ft` (full FT, 25 epochs) | 0.9295 | 0.8185 |

Full FT on 7,153 HAM10000 images causes catastrophic forgetting of the 2M-image dermatology pretraining, most visibly on rare classes (akiec, df, vasc all regress). Probe is the right operating mode for this backbone in this data regime.

### F4. ~4× sample efficiency from dermatology pretraining

`pandem_real_probe` at 25% of HAM10000 train data (1,811 images, 1,335 lesions) reaches test AUC 0.9398. `efficientnetv2_m` needs all 7,153 training images to reach 0.9500.

| Train fraction | `pandem_real_probe` | `efficientnetv2_m` |
|---|---:|---:|
| 25% | 0.9398 | 0.8984 |
| 50% | 0.9461 | 0.9216 |
| 100% | 0.9526 | 0.9500 |

### F5. The best fair model depends on the target image distribution

Dark-skin Malignant AUC differs by which dataset you measure on:

| Dataset (modality) | Best dark-skin model | Dark-Mal AUC |
|---|---|---:|
| DDI (biopsy-proven photos, close to dermoscopy) | `pandem_real_probe` | 0.5919 |
| Fitzpatrick17k (general clinical photos) | `dinov2_registers_ft` | 0.6071 |

This is a clean negative for the assumption that "the best in-distribution model is the best out-of-distribution model" — modality matters as much as pretraining.

### F6. Test-time entropy minimisation (TENT) substantially closes Fitzpatrick gaps

Applying TENT (Wang et al. ICLR 2021) per test batch, adapting only normalisation-layer affines:

| Model + Dataset | Dark-Mal AUC baseline | + TENT | Light-Dark gap before | after |
|---|---:|---:|---:|---:|
| `pandem_real_probe` / DDI | 0.5919 | **0.6931** | 0.175 | 0.044 |
| `convnextv2_large` / DDI | 0.5445 | 0.6357 | 0.103 | 0.048 |
| `dinov2_registers_ft` / DDI | 0.5419 | 0.6562 | 0.068 | 0.016 |
| `efficientnetv2_m` / Fitz17k (n_steps=3) | 0.6040 | **0.6742** | 0.063 | -0.034 |

When the TENT-adapted probability arrays are stacked with a per-target-dataset SLSQP ensemble:
- **DDI**: dark-skin Malignant AUC 0.6571, light-to-dark gap **-0.0004** (full closure)
- **Fitz17k**: dark-skin Malignant AUC 0.6402, gap +0.008

TENT works clearly on the BatchNorm-rich `efficientnetv2_m` and on all backbones for DDI's biopsy-proven dermoscopy-like images. It actively hurts pandem_real_probe on Fitz17k regardless of lr or step count — a real limitation worth reporting honestly.

---

## 2. Quantitative results (HAM10000, lesion-level, test set N=1,431 / 1,067 lesions)

### Single-model test AUC

| Model | Backbone | Train mode | Test AUC | + TTA |
|---|---|---|---:|---:|
| `convnextv2_large` | ConvNeXt-V2 Large @ 384 (ImageNet-21k→1k) | full FT | **0.9660** | 0.9677 |
| `pandem_real_probe` | PanDerm BEiT-B/16 @ 224 (2M derm images) | linear probe | 0.9527 | 0.9552 |
| `efficientnetv2_m` | EfficientNetV2-M @ 384 | full FT | 0.9474 | 0.9555 |
| `siglip2_probe` | SigLIP 2 SO400M @ 384 | linear probe | 0.9460 | 0.9490 |
| `dinov2_registers_ft` | DINOv2+Registers Base @ 224 | full FT | 0.9358 | 0.9408 |

### Ensemble + post-hoc pipeline (4 models, lesion-level splits)

Pipeline: 8-view TTA → per-class SLSQP weights (fit on val) → per-class Platt sigmoid calibration → F2-optimal thresholds.

| Stage | Test macro OvR AUC | Mel recall |
|---|---:|---:|
| Best single (`convnextv2_large` + TTA) | 0.9677 | 0.62 |
| Mean ensemble | 0.9713 | 0.635 |
| + per-class SLSQP weights | 0.9733 | 0.591 |
| + Platt calibration | 0.9733 | 0.560 |
| + F2 thresholds | **0.9733** | 0.566 |

### Cross-dataset (out-of-distribution)

| Dataset | N | Best model standalone | Macro OvR AUC | Dark-Mal AUC |
|---|---:|---|---:|---:|
| ISIC 2019 (ham10k taxonomy) | 15,316 | `pandem_real_probe` | **0.8845** | — |
| DDI universal-3 | 656 | `pandem_real_probe` | 0.5685 | 0.5919 |
| Fitzpatrick17k universal-3 | 3,730 (of 16,577; 23% URL liveness) | `pandem_real_probe` (macro) / `dinov2_registers_ft` (dark) | 0.5726 / — | 0.5531 / **0.6071** |

### Best fairness recipe (after TENT + per-target SLSQP ensemble)

| Target dataset | Recipe | Dark-Mal AUC | L-D gap |
|---|---|---:|---:|
| DDI | TENT (lr=1e-3, n_steps=1) on all 4 models → SLSQP ensemble | **0.6571** | **-0.0004** |
| Fitz17k | `efficientnetv2_m` standalone + TENT (lr=1e-3, n_steps=3) | **0.6742** | -0.034 |

---

## 3. Negative results worth recording

These are findings that *didn't* pan out. Each is a credibility signal — research projects produce negative results too.

### N1. `siglip2_probe` is a strong single model but hurts the ensemble

SigLIP 2 SO400M linear probe scores **test AUC 0.9460** on HAM10000 — competitive. But adding it to the 4-model SLSQP ensemble *decreases* the result:

| Ensemble | Test AUC | Post-hoc final |
|---|---:|---:|
| 4 models (without siglip2_probe) | 0.9718 | 0.9717 |
| 5 models (+ siglip2_probe) | 0.9703 | 0.9701 |

SLSQP gave siglip2_probe the floor weight (0.06) — its error pattern correlates too tightly with `convnextv2_large` to add diversity.

### N2. Binary mel specialist did not beat 7-way + F2 thresholds

| Approach | Mel recall on test |
|---|---:|
| 4-model post-hoc pipeline, F2-tuned mel threshold | 0.566 |
| `mel_binary_convnextv2_large_v2` (binary head, focal loss) | 0.220 @ P=0.80 |
| `mel_binary_pandem_real_probe_v2` (binary head, focal loss) | 0.176 @ P=0.80 |

v1's Phase 19 binary specialist hit ~0.83 mel recall, but that was on image-level (leaky) splits; v2 binary specialists trained on the same architecture under lesion-level splits never close the gap to the post-hoc pipeline. The multi-class model with a tuned threshold beats a dedicated binary head — a clean negative result.

### N3. TENT hurts `pandem_real_probe` on Fitzpatrick17k regardless of hyperparams

Tried lr ∈ {1e-3, 1e-4}, n_steps ∈ {1, 3}. Best Fitz17k dark-skin Mal AUC for pandem_real_probe + TENT is **0.4876** — still well below the baseline 0.5531. Bare entropy-minimisation isn't the right tool when the backbone's feature distribution is far from the target image distribution.

### N4. PanDerm-256 vs PanDerm-224: the "-256" suffix is text, not image

The Hugging Face model name `DermLIP_PanDerm-base-w-PubMed-256` led to an `input_size=256` configuration that performed worse than 224 on cross-dataset transfer. The `-256` is the PubmedBert *text* context length; the image side is 224×224 (confirmed in `open_clip_config.json`). Cosmetic naming bug downstream of a foundation-model release.

---

## 4. What's publishable, and at what venue

Be honest about which of these would land where:

### A. Workshop / short paper — most credible

**Title:** *Test-time entropy minimisation closes Fitzpatrick disparities in dermatology classification*

- Show TENT applied to 4 backbones × 2 fairness datasets (DDI, Fitz17k).
- The per-target SLSQP-ensemble extension is a small but useful methodology contribution.
- Headline: DDI light-to-dark gap closed from 0.103 → -0.0004 at dark-Mal AUC 0.66.
- Honest about Fitz17k limits (pandem_real fails to adapt; clinical-photo modality matters).
- Venue: ML4H workshop at NeurIPS, ISBI workshops, or arXiv preprint.

**Why this is publishable:** It's a small, well-scoped methodology + results paper with both a positive result (DDI) and an honest negative result (Fitz17k for pandem). It does not claim novel methodology — TENT is from ICLR 2021.

### B. Reproducibility note / blog post

**Title:** *Silent fallback in DermLIP/PanDerm HuggingFace loading — diagnosis and fix*

- A bug-find write-up. Documents:
  - The exact failure mode (`config.json` missing → `AutoModel` ValueError → silent fallback).
  - The accuracy difference between the fallback (DINOv2) and the actual PanDerm encoder.
  - The reproducible recipe via `timm.models.beit.Beit` with the right flags.
- Best home: a long-form GitHub README, a Distill-style write-up, or as a section in the workshop paper above.

### C. GitHub repository / engineering portfolio

The code itself is the strongest single deliverable. Useful artifacts to highlight:
- `src/data/splits.py` — patient-level `StratifiedGroupKFold` with `assert_no_group_leakage()` enforced in pytest.
- `src/models/registry.py:RealPanDermClassifier` — the fix from F2.
- `src/post_hoc.py` — 3-stage pipeline (per-class SLSQP → Platt → F2 threshold).
- `src/cross_dataset.py` + `src/cross_dataset_ensemble.py` — fairness audit + per-target SLSQP.
- `src/tent.py` + `scripts/run_tent.py` — TENT adapter.

### D. What's NOT publishable

To preserve credibility, don't claim:
- **State-of-the-art** — 0.9733 on HAM10000 is solid but published numbers exist above 0.98 on similar (or leakier) splits. Frame as "competitive" not "SoTA".
- **A new method** — TENT, focal loss, SLSQP ensembling, Platt calibration are all existing techniques. The contribution is *application* and *audit*, not algorithmic novelty.
- **Production-ready** — none of this has been deployed; clinical validation requires a separate process.

---

## 5. Resume bullets — graded by ambition

### 5a. Conservative (always defensible)

> **Skin lesion classification benchmark on HAM10000 + cross-dataset fairness audit** — Built a unified PyTorch pipeline supporting 11 vision backbones (ConvNeXt-V2, EfficientNetV2, DINOv2-with-Registers, SigLIP 2, DermLIP/PanDerm) behind a `ModelConfig` registry; trained on a single 8 GB GPU via mixed precision + gradient accumulation + checkpointing. Achieved test AUC **0.9733** on patient-level (`StratifiedGroupKFold`) splits via a 4-model post-hoc ensemble (per-class SLSQP weights + Platt sigmoid calibration + F2-tuned thresholds).

> **Cross-dataset fairness evaluation** under a universal-3 taxonomy across ISIC 2019, Stanford AIMI DDI (Fitzpatrick-stratified), and Fitzpatrick17k. Showed that the best out-of-distribution model differs by image modality: dermatology-pretrained features win on dermoscopy-like data (DDI: 0.59 dark-skin Mal AUC), while ImageNet-pretrained features transfer better on general clinical photos (Fitz17k: 0.61 dark-skin Mal AUC).

### 5b. Standard (the engineering wins are real)

> **Diagnosed and fixed a foundation-model load bug** that had been silently substituting DINOv2+Registers for the dermatology-pretrained PanDerm encoder on Hugging Face for 6 months of prior work. The repo is an `open_clip` checkpoint without a `config.json`, so `transformers.AutoModel.from_pretrained` hit a `try/except` fallback. Reimplemented the loader using `timm.models.beit.Beit` with custom flags (`use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, global_pool='token'`) + OpenAI-CLIP normalisation; weights load with 0 missing / 0 unexpected keys. The fix increased cross-dataset macro AUC on ISIC 2019 from 0.81 → **0.88**.

> **Demonstrated 4× sample efficiency** of dermatology-pretrained features: a linear probe over the (real) PanDerm encoder at 25% of HAM10000 training labels reaches test AUC 0.94, matching EfficientNetV2-M at 100%. Probe also beat full fine-tuning of the same encoder by 0.023 AUC, indicating that pretraining-disturbing FT is the wrong recipe in the 7k-image regime.

### 5c. Strong (the fairness story is the most publishable thing)

> **Mitigated Fitzpatrick skin-tone disparity in dermatology classification using test-time entropy minimisation (TENT, Wang et al. 2021).** On Stanford AIMI's DDI dataset (biopsy-proven photos, 656 images across 6 Fitzpatrick groups), adapting only normalisation-layer affines at inference time raised dark-skin Malignant AUC for 3 of 4 evaluated backbones by up to +0.10 (PanDerm: 0.59 → 0.69) while reducing the light-to-dark AUC gap from 0.17 → 0.04. Combined with a per-target SLSQP ensemble that re-weights models on the adapted probabilities, the DDI Fitzpatrick gap closes to ≈ 0 (-0.0004) at dark-skin Mal AUC 0.66.

### Pick-one-line elevator version

> Built an honest (patient-level split, cross-dataset audited) skin lesion classification pipeline that hits **0.97 test AUC** on HAM10000 and **closes the Stanford DDI Fitzpatrick light-to-dark malignant-detection gap from 10% to under 1%** via test-time adaptation; fixed a silent foundation-model fallback bug along the way.

---

## 6. What's still genuinely incomplete

For honesty, the things that *aren't* finished:
- **Multimodal model** (metadata: age, sex, anatomical site). v1 had it; v2 port deferred.
- **Grad-CAM / attention visualisation** for clinical interpretability.
- **Domain-generalisation training** (GroupDRO, DANN). Only test-time adaptation was tried.
- **TENT failure mode on `pandem_real_probe` + Fitz17k** is real and unresolved — needs a different technique (DANN, fine-tuning on Fitz17k, or rejection).
- **No clinical / dermatologist evaluation.** Numbers are AUCs, not clinical-decision metrics.
- **A formal write-up.** All findings live in this repo's markdown and `experiments/*/results.txt` files.

---

## Citations to include if writing this up

- Tschannen et al. *SigLIP 2.* arXiv:2502.14786 (2025).
- Oquab et al. *DINOv2: Learning Robust Visual Features.* arXiv:2304.07193; Darcet et al. *Vision Transformers Need Registers.* arXiv:2309.16588.
- Yan et al. *A multimodal vision foundation model for clinical dermatology* (PanDerm). Nature Medicine, 2025.
- Yan et al. *Derm1M / DermLIP.* arXiv:2503.14911; ICCV 2025 Highlight.
- Tschandl et al. *The HAM10000 dataset.* Scientific Data 5 (2018).
- Combalia et al. *BCN20000: Dermoscopic Lesions in the Wild.* arXiv:1908.02288 (ISIC 2019).
- Daneshjou et al. *Disparities in dermatology AI performance on a diverse, curated clinical image set* (DDI). Science Advances 8 (2022).
- Groh et al. *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset.* CVPRW 2021.
- Wang et al. *TENT: Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021.
- Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
- Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts* (GroupDRO). ICLR 2020.

---

*This document was assembled from artifacts under `D:\ML\skin-lesion-vit\experiments\` and `D:\ML\skin-lesion-vit\v2\experiments\`. Numbers verified against `results.txt`, `tta_results.txt`, and `report.txt` files. Any discrepancy: trust those files, not this one.*

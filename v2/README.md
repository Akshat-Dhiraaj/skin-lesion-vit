# Skin Lesion v2 — Patient-Level HAM10000 + Cross-Dataset Robustness

Successor to the [v1 22-phase benchmark](../README.md) (this repository's root). Knowledge base in [`../info_bank/`](../info_bank/). Same models and infra, three changes:

1. **Lesion-level splits** via `StratifiedGroupKFold(group='lesion_id')` — no near-duplicate leakage. Enforced by `pytest tests/test_splits.py` (`assert_no_group_leakage`).
2. **Refactored model registry** — every backbone is one `ModelConfig` dataclass, one `forward(x) → logits` signature, four `model_type` dispatch keys (`timm` / `dinov2` / `pandem` / `siglip2`).
3. **Cross-dataset evaluation** plumbing for ISIC 2019, DDI, Fitzpatrick17k — universal 3-class and HAM10K-7 label taxonomies, Fitzpatrick-stratified malignant AUC reporting.

> Numbers below come directly from `experiments/<model>/results.txt`, `experiments/<model>/tta_results.txt`, and `experiments/cross_dataset/*/report.txt`. See [PHASES.md](PHASES.md) for full phase status.

---

## HAM10000 results (lesion-level splits, test N=1431)

### Single-model (best checkpoint, val-selected)

| Model | Best Val AUC | Test AUC | Test Acc | Results |
|------|--------------|----------|----------|---------|
| `convnextv2_large`     | 0.9803 | **0.9660** | 0.8553 | [link](experiments/convnextv2_large/results.txt) |
| `pandem_real_probe`    | 0.9682 | 0.9527 | 0.7757 | [link](experiments/pandem_real_probe/results.txt) |
| `efficientnetv2_m`     | 0.9637 | 0.9474 | 0.8141 | [link](experiments/efficientnetv2_m/results.txt) |
| `siglip2_probe`        | 0.9570 | 0.9460 | 0.7771 | [link](experiments/siglip2_probe/results.txt) |
| `dinov2_registers_ft`  | 0.9570 | 0.9358 | 0.7987 | [link](experiments/dinov2_registers_ft/results.txt) |
| `pandem_ft` ✗          | 0.9593 | 0.9330 | 0.7324 | [link](experiments/pandem_ft/results.txt) |
| `pandem_probe` ✗       | 0.9478 | 0.9305 | 0.7645 | [link](experiments/pandem_probe/results.txt) |
| `pandem_real_ft`       | 0.9656 | 0.9295 | 0.7806 | [link](experiments/pandem_real_ft/results.txt) |
| `pandem_probe_224` ✗   | 0.9439 | 0.9242 | — | [link](experiments/pandem_probe_224/results.txt) |
| `dinov2_registers_probe` | 0.9425 | 0.9232 | 0.7205 | [link](experiments/dinov2_registers_probe/results.txt) |

> ✗ = silently uses DINOv2+Registers, not PanDerm. See *PanDerm loader fix* in **Methodology** for why and how `pandem_real_*` is different.
>
> The v1 image-level ConvNeXt-V2 hit 0.9880 test AUC. The 0.022 drop on the v2 lesion-level split is the realistic generalization cost of the leak fix.
>
> **Counter-intuitive result on `pandem_real_*`:** the linear probe (0.9527) beats the full fine-tune (0.9295) by 0.023 AUC. The 2M-image dermatology pretraining produces features too task-relevant to disturb on 7k HAM10000 images; full FT causes catastrophic forgetting, most visibly on rare classes (akiec, df, vasc all regress). Probe is the right operating mode for this backbone.

### Single-model + 8-view TTA

`tta.py` writes both val and test arrays. All `tta_results.txt` files contain both `[VAL]` and `[TEST]` sections (re-run with `--split both` after T-FIX-3).

| Model | TTA Val AUC | TTA Val Acc | TTA Test AUC | Results |
|------|-------------|-------------|--------------|---------|
| `convnextv2_large`     | 0.9817 | 0.8763 | **0.9677** | [results](experiments/convnextv2_large/tta_results.txt) |
| `efficientnetv2_m`     | 0.9684 | 0.8435 | 0.9555 | [results](experiments/efficientnetv2_m/tta_results.txt) |
| `pandem_real_probe`    | 0.9692 | 0.7973 | 0.9552 | [results](experiments/pandem_real_probe/tta_results.txt) |
| `siglip2_probe`        | 0.9596 | 0.8001 | 0.9490 | [results](experiments/siglip2_probe/tta_results.txt) |
| `dinov2_registers_ft`  | 0.9635 | 0.8085 | 0.9408 | [results](experiments/dinov2_registers_ft/tta_results.txt) |
| `pandem_real_ft`       | 0.9678 | 0.8162 | 0.9368 | [results](experiments/pandem_real_ft/tta_results.txt) |
| `pandem_ft` ✗          | 0.9603 | 0.7666 | 0.9359 | [results](experiments/pandem_ft/tta_results.txt) |

### 4-model SLSQP ensemble (test) — canonical = pandem_real_probe variant

`src/ensemble.py` with `--method slsqp --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe`:

```
weights : convnextv2_large=0.775  efficientnetv2_m=0.075
          dinov2_registers_ft=0.075  pandem_real_probe=0.075
Test AUC : 0.9731
Test Acc : 0.8581
Lift vs best single : +0.0054
```

Per-class TTA AUCs feeding this:
`convnextv2_large=0.9677 / efficientnetv2_m=0.9555 / dinov2_registers_ft=0.9408 / pandem_real_probe=0.9552`. See [experiments/ensemble/results.txt](experiments/ensemble/results.txt).

For traceability, side-by-side variants are kept under `experiments/ensemble/`:
- `results.txt` — canonical (pandem_real_probe), 0.9731
- `results_with_pandem_real_probe.txt` — same canonical, preserved
- `results_with_pandem_real_ft.txt` — 0.9728 (pandem_real_ft variant, slightly worse)
- `results_5model.txt` — 0.9703 (added `siglip2_probe` — net negative; see Caveats)

### Post-hoc pipeline (`src/post_hoc.py`)

Per-class SLSQP weights → Platt sigmoid calibration → F2-optimal per-class thresholds. Fit on val, evaluated on test.

**Canonical (`pandem_real_probe` in the 4-model set):**

| Stage | Test AUC | Test Acc | Mel Recall |
|-------|----------|----------|-----------|
| mean ensemble (baseline) | 0.9713 | — | 0.635 |
| + per-class weighted ensemble | 0.9733 | 0.8581 | 0.591 |
| + Platt sigmoid calibration | 0.9733 | 0.8574 | 0.560 |
| + F2 threshold tuning | **0.9733** | 0.8532 | 0.566 |

**Previous (pandem_ft DINOv2-fallback variant), for comparison:**

| Stage | Test AUC | Mel Recall |
|-------|----------|-----------|
| mean ensemble | 0.9690 | 0.616 |
| final (F2-tuned) | 0.9717 | 0.566 |

See [experiments/post_hoc/results.txt](experiments/post_hoc/results.txt) for per-class weights, thresholds, and per-class classification report. The Platt-sigmoid calibration was a deliberate switch from isotonic — isotonic collapses to <30 unique values per class on rare-class val populations (`df=16`, `vasc=20`) and tanks macro OvR AUC by ~0.04; see the FIX notes at the top of `src/post_hoc.py`.

---

## Cross-dataset 1 — ISIC 2019 (HAM10K-7 mapping)

Inference only — no ISIC training. Models are evaluated on the 15,316 ISIC 2019 training images that **do not** appear in HAM10000, mapped to the 7 HAM10000 classes (SCC → akiec). See [src/cross_dataset.py](src/cross_dataset.py) + [src/data/external/taxonomy.py](src/data/external/taxonomy.py).

| Model | Macro OvR AUC | Malignant AUC | Mel AUC |
|------|--------------|---------------|---------|
| `pandem_real_probe`    | **0.8845** | — | — |
| `convnextv2_large`     | 0.8546 | 0.7890 | 0.7889 |
| `pandem_probe_224` ✗   | 0.8337 | 0.7460 | 0.7597 |
| `efficientnetv2_m`     | 0.8294 | 0.7569 | 0.7405 |
| `pandem_real_ft`       | 0.8185 | — | — |
| `pandem_ft` ✗          | 0.8101 | 0.7120 | 0.7594 |
| `dinov2_registers_ft`  | 0.8077 | 0.7272 | 0.7481 |
| `siglip2_probe`        | 0.6610 | 0.5708 | 0.6782 |

✗ = silently DINOv2+Registers. **`pandem_real_probe` (the actual PanDerm encoder, linear probe) leads ISIC 2019 by +0.030 macro AUC** over the strongest fine-tuned model — dermatology pretraining transfers exceptionally well to other dermoscopy-like data. Full per-class breakdown in `experiments/cross_dataset/isic2019/<model>/report.txt`.

## Cross-dataset 2 — DDI (Stanford AIMI, Fitzpatrick-stratified)

656 biopsy-proven images, ~one third per Fitzpatrick group (208 light / 241 medium / 207 dark). Universal-3 taxonomy (malignant / benign / non-neoplastic). HAM10000-trained 7-way probs are projected to 3-way.

| Model | Macro OvR | Mal AUC | Light Mal | Medium Mal | Dark Mal | L→D Gap |
|---|---:|---:|---:|---:|---:|---:|
| `pandem_real_probe` | **0.5685** | **0.6821** | 0.6796 | **0.7666** | **0.5919** | 0.175 |
| `dinov2_registers_ft` | 0.5433 | 0.5784 | 0.5787 | 0.6101 | 0.5419 | **0.068** |
| `convnextv2_large` | 0.5426 | 0.6022 | 0.6022 | 0.6478 | 0.5445 | 0.103 |
| `siglip2_probe` | 0.5400 | 0.5814 | 0.6101 | 0.6261 | 0.5033 | 0.123 |
| `efficientnetv2_m` | 0.5321 | 0.5708 | 0.5491 | 0.6329 | 0.4869 | 0.146 |
| `pandem_ft` ✗ | 0.4982 | 0.5159 | 0.5710 | 0.5239 | 0.4396 | 0.131 |

DDI is meaningfully harder than HAM10000 — clinical photographs rather than dermoscopy. `pandem_real_probe` wins absolute Mal AUC and dark-skin Mal AUC; `dinov2_registers_ft` has the smallest disparity. `pandem_ft` (DINOv2 fallback) drops to **worse than random** (0.4396) on dark skin — a strong argument for retiring the fallback checkpoints.

## Cross-dataset 3 — Fitzpatrick17k (general clinical photos)

3,730 of 16,577 images (~23% URL liveness after Atlas Dermatologico and DermaAmin link rot). Buckets: 818 light / 2,076 medium / 836 dark.

| Model | Macro OvR | Mal AUC | Light Mal | Medium Mal | Dark Mal | L→D Gap |
|---|---:|---:|---:|---:|---:|---:|
| `pandem_real_probe` | **0.5726** | 0.6282 | **0.7202** | 0.6158 | 0.5531 | 0.167 |
| `convnextv2_large` | 0.5660 | 0.6282 | 0.6884 | 0.5996 | 0.5784 | 0.110 |
| `efficientnetv2_m` | 0.5592 | 0.6132 | 0.6567 | 0.5936 | 0.6040 | 0.063 |
| `siglip2_probe` | 0.5561 | 0.5887 | 0.5603 | 0.5847 | 0.5684 | -0.008 |
| `dinov2_registers_ft` | 0.5288 | 0.6011 | 0.5796 | 0.5838 | **0.6071** | **-0.027** |
| `pandem_ft` ✗ | 0.5223 | 0.6117 | 0.6046 | 0.5820 | 0.5886 | -0.016 |

**The best-on-dark-skin model FLIPS between DDI and Fitz17k.** On DDI, dermatology-pretrained PanDerm wins. On Fitz17k clinical photos, ImageNet-pretrained models do better on dark skin — the image-modality mismatch outweighs the domain pretraining advantage.

## Cross-dataset 4 — Test-time entropy minimisation (TENT)

Following [Wang et al. ICLR 2021](https://arxiv.org/abs/2006.10726), at inference time only normalisation-layer affines are unfrozen; for each test batch we take one gradient step on softmax entropy and then read off the adapted prediction. See `src/tent.py`.

**DDI dark-skin Malignant AUC** (baseline → + TENT lr=1e-3, n_steps=1):

| Model | Baseline | + TENT | Δ | Gap before | Gap after |
|---|---:|---:|---:|---:|---:|
| `pandem_real_probe` | 0.5919 | **0.6931** | +0.101 | 0.175 | 0.044 |
| `dinov2_registers_ft` | 0.5419 | 0.6562 | +0.114 | 0.068 | 0.016 |
| `convnextv2_large` | 0.5445 | 0.6357 | +0.091 | 0.103 | 0.048 |
| `efficientnetv2_m` | 0.4869 | 0.4374 | -0.050 | 0.146 | 0.092 |

**Fitz17k dark-skin Malignant AUC** (best per-model TENT config; LN models use lr=1e-4, BN model uses n_steps=3):

| Model | Baseline | + TENT | Δ |
|---|---:|---:|---:|
| `efficientnetv2_m` (n_steps=3) | 0.6040 | **0.6742** | +0.070 |
| `dinov2_registers_ft` (lr=1e-4) | 0.6071 | 0.6214 | +0.014 |
| `convnextv2_large` (lr=1e-4) | 0.5784 | 0.5906 | +0.012 |
| `pandem_real_probe` (any) | 0.5531 | 0.4876 | -0.066 |

### Best fairness recipe per target (TENT + per-target SLSQP)

`src/cross_dataset_ensemble.py` fits dataset-specific ensemble weights on the TENT-adapted probability arrays. See `experiments/cross_dataset_ensemble_tent_lr1e-3/report.txt`.

| Target | Recipe | Dark-Mal AUC | L→D Gap |
|---|---|---:|---:|
| DDI | TENT (lr=1e-3) on all 4 models → SLSQP weights | **0.6571** | **-0.0004** |
| Fitz17k | `efficientnetv2_m` + TENT (n_steps=3) standalone | **0.6742** | -0.034 |

**The DDI Fitzpatrick light-to-dark gap is essentially closed** (gap ≈ 0) at dark-skin Malignant AUC 0.66, up from baseline 0.54. Fitz17k benefits the most from a CNN/BN model under TENT rather than the dermatology-pretrained ViT.

## Sample efficiency (E12)

Stratified-per-class subsampling of HAM10000 train lesions; val/test fixed.

| Train fraction | Lesions | Images | `pandem_real_probe` Test AUC | `efficientnetv2_m` Test AUC |
|---|---:|---:|---:|---:|
| 25% | 1,335 | 1,811 | **0.9398** | 0.8984 |
| 50% | 2,668 | 3,579 | 0.9461 | 0.9216 |
| 100% | 5,336 | 7,153 | 0.9526 | 0.9500 |

PanDerm linear probe at 25% data matches EfficientNetV2-M full FT at 100%. Reports in `experiments/sample_efficiency/<model>_fracNNN/results.txt`.

## Binary melanoma specialist (OA1)

Binary mel vs non-mel head (sigmoid logit) + BinaryFocalLoss + 1:1 balanced sampler. Compare two backbones, with v2 hyperparams (α=0.5, longer schedule, halved lrs for convnext):

| Specialist | Val mel AUC | Test mel AUC | R @ P=0.95 | R @ P=0.80 |
|---|---:|---:|---:|---:|
| `mel_binary_convnextv2_large_v2` | 0.8934 | **0.8942** | 0.063 | 0.220 |
| `mel_binary_pandem_real_probe_v2` | 0.9047 | 0.8794 | 0.069 | 0.176 |

**The 7-way model with F2-tuned mel threshold still beats both** (mel recall 0.566 from the post-hoc pipeline). Recorded as a negative methodology result; v1 Phase 19's image-level-split mel recall of 0.83 does not survive lesion-level splits with this recipe.

---

## Model registry

> **For what each backbone is, how it works, why it was chosen, and the paper behind it, see the [Model & Technique Compendium](../info_bank/MODELS.md)** — a per-model deep-dive with verified citations. The table below is the operational config only.

`src/models/registry.py` exposes one `ModelConfig` per backbone. `build_model()` dispatches on `model_type`. All backbones implement `forward(x) -> logits`.

| Registry key | HF / timm ID | Type | Mode | Input | Batch × Accum | LR head / bb | Epochs |
|--------------|--------------|------|------|------|---------------|--------------|--------|
| `convnextv2_large`        | `convnextv2_large.fcmae_ft_in22k_in1k_384` | timm | full FT | 384 | 8 × 8 | 1e-3 / 5e-6 | 25 |
| `dinov2_registers_probe`  | `facebook/dinov2-with-registers-base` | dinov2 | linear probe | 224 | 64 × 1 | 1e-3 / 0 | 20 |
| `dinov2_registers_ft`     | `facebook/dinov2-with-registers-base` | dinov2 | full FT | 224 | 8 × 8 | 1e-3 / 5e-6 | 25 |
| `pandem_probe` ✗          | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | pandem (silent DINOv2 fallback) | linear probe | 256 | 64 × 1 | 1e-3 / 0 | 20 |
| `pandem_ft` ✗             | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | pandem (silent DINOv2 fallback) | full FT | 256 | 8 × 8 | 1e-3 / 5e-6 | 25 |
| `pandem_probe_224` ✗      | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | pandem (silent DINOv2 fallback) | linear probe | 224 | 64 × 1 | 1e-3 / 0 | 20 |
| `pandem_ft_224` ✗         | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | pandem (silent DINOv2 fallback) | full FT (not trained) | 224 | 8 × 8 | 1e-3 / 5e-6 | 25 |
| `pandem_real_probe`       | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | **pandem_real** (BEiT via open_clip) | linear probe | 224 | 64 × 1 | 1e-3 / 0 | 20 |
| `pandem_real_ft`          | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | **pandem_real** (BEiT via open_clip) | full FT | 224 | 8 × 8 | 1e-3 / 5e-6 | 25 |
| `siglip2_probe`           | `google/siglip2-so400m-patch14-384` | siglip2 | linear probe | 384 | 8 × 8 | 1e-3 / 0 | 20 |
| `efficientnetv2_m`        | `tf_efficientnetv2_m` | timm | full FT | 384 | 8 × 4 | 1e-3 / 1e-5 | 20 |

The `pandem` model_type uses `transformers.AutoModel.from_pretrained`; the repo lacks a `config.json` (it's an open_clip checkpoint), so the load raises and the `except` falls back to `facebook/dinov2-with-registers-base`. Every `pandem_*` row marked ✗ is therefore DINOv2+Registers, not PanDerm. The **`pandem_real`** model_type (`RealPanDermClassifier` in [src/models/registry.py](src/models/registry.py)) is the actual fix — see *PanDerm loader fix* under **Methodology**. References: PanDerm [Yan et al., Nature Medicine 2025](https://www.nature.com/articles/s41591-025-03747-y); DermLIP / Derm1M [arXiv 2503.14911, ICCV 2025](https://arxiv.org/abs/2503.14911).

---

## Repository layout

```
skin-lesion-vit/v2/          (this subtree; v1 is the repo root, docs in ../info_bank/)
├── configs/
│   └── data/ham10000.yaml      legacy / unused — no code reads it; data path is hardcoded as the default of --data_dir in scripts/train.py and scripts/verify_data.py
├── src/
│   ├── data/
│   │   ├── splits.py                StratifiedGroupKFold + leakage asserter
│   │   ├── ham10000.py              loader, Dataset, WeightedRandomSampler, get_dataloaders
│   │   ├── transforms.py            train / val / TTA pipelines (Albumentations)
│   │   └── external/
│   │       ├── ddi.py               DDI loader (manual download)
│   │       ├── fitzpatrick17k.py    Fitz17k loader (manual download)
│   │       ├── isic2019.py          ISIC 2019 loader (manual download)
│   │       └── taxonomy.py          HAM10K / universal / Fitzpatrick bucket mappings
│   ├── models/registry.py           9-entry ModelConfig registry + build_model
│   ├── engine.py                    train_one_epoch / evaluate / run_phase
│   ├── tta.py                       8-view TTA inference → tta_{val,test}_probs.npy
│   ├── ensemble.py                  global SLSQP / grid weighted ensemble
│   ├── post_hoc.py                  per-class SLSQP + Platt + F2 thresholds pipeline
│   └── cross_dataset.py             HAM10K-trained → DDI / Fitz17k / ISIC2019 eval
├── scripts/
│   ├── train.py                     CLI: --model <registry_key>
│   ├── verify_data.py               smoke test the data pipeline
│   ├── leaderboard.py               scrape experiments/*/results.txt → markdown table
│   ├── run_option1.ps1              overnight script (training subset)
│   └── test_option1.ps1
├── tests/test_splits.py             pytest — split leakage, registry, evaluate(), train_one_epoch()
├── experiments/                     one folder per registry key + cross_dataset/ + ensemble/ + post_hoc/
├── data/external/ISIC2019/          15,316 ISIC images + ground truth CSV
├── pyproject.toml                   hatchling, py311, ruff + pytest
├── mlflow.db                        MLflow store
└── PHASES.md                        phase status table (E1–E14, B1–B5, OA1–OA4)
```

---

## Why this exists (delta from v1)

| | v1 (skin-lesion-vit) | v2 (this repo) |
|---|---|---|
| **Splits** | `sklearn.train_test_split` on image rows, stratified by `dx` | `StratifiedGroupKFold(group='lesion_id')` — disjoint lesions across train/val/test |
| **Leakage guard** | None | `assert_no_group_leakage` called after every split + CI tests |
| **Model config** | Dicts in `engine.py:PHASE_CONFIGS` | `ModelConfig` dataclass registry, one `forward(x) → logits` |
| **Forward dispatch** | If/elif on `model_type` inside `engine.py:get_logits` | Each backbone class handles its own forward |
| **Backbones added** | — | PanDerm vision encoder via DermLIP checkpoint (2025), SigLIP 2 (2025) |
| **Calibration** | Per-class isotonic (Phase 22) — tanked val→test transfer | Per-class Platt sigmoid — strictly monotonic, preserves AUC |
| **Cross-dataset eval** | — | ISIC 2019 / DDI / Fitzpatrick17k with universal-3 + ham10k-7 taxonomies |
| **Fairness** | — | Fitzpatrick light/medium/dark malignant-AUC reporting in `src/cross_dataset.py` |
| **Packaging** | `requirements.txt` | `pyproject.toml` + hatchling, ruff |
| **CI** | None | `pytest tests/` — splits, registry, evaluate, train_one_epoch |

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pytest -v   # split + registry + engine smoke tests
```

The HAM10000 data path is hardcoded as the default of `--data_dir` in `scripts/train.py` ([line 28](scripts/train.py)) and `scripts/verify_data.py` — pointing at v1's `D:\ML\skin-lesion-vit\data\raw`. Override via `--data_dir <path>` or edit the default. `configs/data/ham10000.yaml` exists with the same values but **no code currently reads it** (hydra-core is in `pyproject.toml` deps but never imported).

External datasets must be downloaded manually (gated):

- **ISIC 2019** — `data/external/ISIC2019/ISIC_2019_Training_Input/`, `ISIC_2019_Training_GroundTruth.csv`. Already present in this repo.
- **DDI** — register at Stanford AIMI, place under `data/external/DDI/`. Not yet present.
- **Fitzpatrick17k** — clone `mattgroh/fitzpatrick17k`, place under `data/external/Fitzpatrick17k/`. Not yet present.

---

## Run

```powershell
# Single-model training (writes experiments/<model>/best_model.pth + results.txt)
python scripts/train.py --model convnextv2_large
python scripts/train.py --model dinov2_registers_ft
python scripts/train.py --model pandem_ft --num_workers 0      # 0 workers on Windows fork issues
python scripts/train.py --model efficientnetv2_m
python scripts/train.py --model pandem_probe
python scripts/train.py --model dinov2_registers_probe

# 8-view TTA on val+test (writes tta_{val,test}_{probs,labels}.npy + tta_results.txt)
python -m src.tta --model convnextv2_large --data_dir D:\ML\skin-lesion-vit\data\raw

# Weighted ensemble (writes experiments/ensemble/{ensemble_probs,labels}.npy + results.txt)
python -m src.ensemble --method slsqp --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_ft

# Post-hoc pipeline (per-class weights + Platt + F2 thresholds)
python -m src.post_hoc --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_ft

# Cross-dataset eval (HAM10000-trained → other datasets)
python -m src.cross_dataset --model convnextv2_large --dataset isic2019 --data_root data/external --label_mode ham10k
python -m src.cross_dataset --model convnextv2_large --dataset ddi      --data_root data/external --label_mode universal
python -m src.cross_dataset --model convnextv2_large --dataset fitz17k  --data_root data/external --label_mode universal

# Data sanity / leaderboard
python scripts/verify_data.py
python scripts/leaderboard.py

# Tests
pytest -v

# MLflow
mlflow ui     # reads ./mlflow.db
```

---

## Methodology

### Lesion-level splits
`split_ham10000()` in [src/data/splits.py:212](src/data/splits.py:212) is a two-pass `StratifiedGroupKFold`. Pass 1 separates test from train+val; pass 2 separates val from train. Splits are reproducible (`random_state=42`) and disjoint by `lesion_id`. `assert_no_group_leakage()` runs immediately after every split (negligible cost) and is asserted again in `pytest`.

HAM10000 doesn't expose `patient_id`, so `lesion_id` is the tightest available grouping and what the official HAM10000 paper recommends.

### Training engine
- One `forward(x) → logits` signature for every backbone — no per-model_type dispatch in the loop.
- Gradient clipping at `max_norm=1.0`.
- Gradient checkpointing: `_enable_grad_checkpointing()` tries the HF API (`gradient_checkpointing_enable`) for `dinov2`/`pandem`, then the timm API (`set_grad_checkpointing`) for the timm backbones, then silently no-ops. ConvNeXt-V2 supports it; whether timm's `tf_efficientnetv2_m` exposes `set_grad_checkpointing` depends on the installed timm version, so coverage isn't guaranteed.
- AMP via `torch.amp.autocast(device_type)` — works on CPU for debugging.
- NaN guard in `evaluate()` — EfficientNetV2 can overflow under AMP; NaN rows are replaced with uniform probabilities to keep the run from crashing.
- End-of-epoch flushes any remaining accumulated gradients (legacy v1's `(step + 1) % accum_steps == 0` check dropped the partial accumulation when the loader exhausted mid-bucket).

### Normalization
`src/data/transforms.py` exposes a `get_norm_stats(model_type)` helper (added in T-FIX-1) that returns SigLIP stats `mean=std=(0.5,0.5,0.5)` when `model_type="siglip2"`, CLIP stats `(0.4815,0.4578,0.4082)`/`(0.2686,0.2613,0.2758)` when `model_type="pandem_real"`, and ImageNet stats `(0.485,0.456,0.406)`/`(0.229,0.224,0.225)` otherwise. `get_train_transform` and `get_val_transform` accept an optional `model_type` argument and default to ImageNet stats, preserving backward compatibility. The 8-view TTA pipeline is `src/tta.py:get_tta_transforms` (single source of truth — a dead duplicate in `transforms.py` was removed in the 2026-05-16 refactor; the two were not behaviorally identical so were never merged). `src/engine.py` and `src/tta.py` pass `cfg.model_type` through. The DINOv2, PanDerm-fallback, ConvNeXt-V2, and EfficientNetV2-M checkpoints were pretrained with ImageNet normalization; SigLIP 2 uses `(0.5,0.5,0.5)`; the real PanDerm BEiT uses OpenAI-CLIP stats per `open_clip_config.json`.

### TTA
8 deterministic views: original, H-flip, V-flip, rot 90/180/270, HV-flip, slight zoom (via `ShiftScaleRotate(scale_limit=0.1)`). Softmax probabilities averaged across views. `tta.py` writes `tta_val_*.npy` and `tta_test_*.npy` — both splits are needed for `post_hoc.py`.

### Post-hoc analysis
Fit-on-val, apply-on-test, in three stages:

1. **Per-class SLSQP weights** — independent weight vector per class, OvR-AUC objective, multi-restart from one-hot + uniform inits, bounds `(0.01, 0.99)`, sum-to-1 constraint.
2. **Platt sigmoid calibration** — one `LogisticRegression` per class on the OvR scores. Strictly monotonic, preserves AUC, gives meaningful probabilities for thresholding. *Does NOT renormalise rows* — that would destroy per-class ranking.
3. **F2-optimal per-class thresholds** — grid search over `[0.01, 0.99]` in steps of 0.01.

### PanDerm loader fix (`pandem_real`)
The Hugging Face repo `redlessone/DermLIP_PanDerm-base-w-PubMed-256` is an **open_clip** checkpoint — only `open_clip_config.json` + `open_clip_pytorch_model.bin`, no `config.json`. `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` raises `ValueError: missing model_type key`, falls into the `except`, and silently substitutes `facebook/dinov2-with-registers-base`. Every `pandem_*` checkpoint in `experiments/` (probe/ft, 256/224) was therefore trained on DINOv2+Registers, not PanDerm.

`RealPanDermClassifier` in [src/models/registry.py](src/models/registry.py) is the fix. The visual sub-state-dict of the open_clip weights matches a `timm.models.beit.Beit` configured as:

```python
Beit(img_size=224, patch_size=16, in_chans=3,
     num_classes=0, embed_dim=768, depth=12, num_heads=12,
     mlp_ratio=4, qkv_bias=True,
     use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
     init_values=0.1,                  # LayerScale (gamma_1, gamma_2)
     global_pool='token',              # 'norm' over CLS, NOT 'fc_norm' over pool
)
```

Loading the filtered visual state_dict yields **0 missing / 0 unexpected keys**. Normalisation uses OpenAI-CLIP stats `(0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758)` per `open_clip_config.json`, NOT ImageNet or SigLIP — see `get_norm_stats("pandem_real")` in [src/data/transforms.py](src/data/transforms.py).

### TENT — test-time entropy minimisation
[src/tent.py](src/tent.py) implements [Wang et al., ICLR 2021](https://arxiv.org/abs/2006.10726). `wrap_for_tent(model)` freezes everything except normalisation-layer affines (BatchNorm: also disables `track_running_stats` so each test batch estimates its own statistics; LayerNorm: only the affine params adapt). `tent_evaluate(model, loader, lr, n_steps)` for each test batch:
1. Forward → entropy of softmax.
2. Backward → one Adam step on the norm-layer affines.
3. Final forward (no grad) → emit adapted softmax probabilities.

Best per-architecture hyperparams from the sweep:
- BatchNorm-rich CNNs (`efficientnetv2_m`): `lr=1e-3, n_steps=3` for Fitz17k; `lr=1e-3, n_steps=1` for DDI.
- LayerNorm-only models (`convnextv2_large`, `dinov2_registers_ft`, `pandem_real_probe`): `lr=1e-3, n_steps=1` for DDI; `lr=1e-4, n_steps=1` for Fitz17k.

`src/cross_dataset_ensemble.py` then fits per-target SLSQP weights *on the TENT-adapted probabilities*, exposing the dataset-specific accuracy-fairness frontier. Output: `experiments/cross_dataset_ensemble_tent_lr1e-3/report.txt`.

### Cross-dataset taxonomies
[src/data/external/taxonomy.py](src/data/external/taxonomy.py) defines:

- **HAM10K-7**: `[mel, nv, bcc, akiec, bkl, df, vasc]`, malignants `{mel, bcc, akiec}`.
- **Universal-3**: `[malignant, benign, non_neoplastic]` for fundamentally different label schemas (DDI, Fitz17k).
- **ISIC2019 → HAM10K**: 8 classes → 7, with `SCC → akiec`.
- **DDI / Fitz17k → universal**: malignant flag / `three_partition_label` (with a curated HAM10K mapping for the diseases that overlap clearly).
- **Fitzpatrick bucketing**: `1-2 → light`, `3-4 → medium`, `5-6 → dark`.

A HAM10000-trained model never predicts non-neoplastic; that column gets 0 in the universal projection and is skipped in macro AUC.

---

## Hardware

- GPU: NVIDIA RTX 4060 Laptop, 8 GB VRAM
- RAM: 32 GB
- OS: Windows 11 + PowerShell

All configs sized for 8 GB via AMP, gradient accumulation, and gradient checkpointing. ConvNeXt-V2 Large at 384 is the largest fit (`batch=8 × accum=8 = effective 64`).

---

## Caveats / open work

- **PHASES.md is current.** See [PHASES.md](PHASES.md) for the live phase ledger.
- **`pandem_*` (without `_real`) silently uses DINOv2+Registers.** Resolved by the `pandem_real` model_type (`RealPanDermClassifier` — see *PanDerm loader fix* under **Methodology**). The DINOv2-fallback checkpoints are kept on disk for traceability but should not be cited as PanDerm results.
- **`pandem_real_ft` is worse than `pandem_real_probe`** by 0.023 HAM10000 test AUC and 0.066 ISIC macro AUC. Full FT on 7k images destroys the 2M-image dermatology pretraining; use the probe. `pandem_real_ft` artifacts are preserved at `experiments/pandem_real_ft/` for reproducibility.
- **Adding `siglip2_probe` to the 4-model ensemble hurts** (0.9718 → 0.9703). Its error pattern correlates too tightly with `convnextv2_large`; SLSQP assigns it the floor weight. Side-by-side artifact at `experiments/ensemble/results_5model.txt`.
- **The binary mel specialist (OA1) does not beat the 7-way + F2 threshold approach** on lesion-level splits. v1 Phase 19's 0.83 mel recall came from image-level (leaky) splits. v2 numbers at `experiments/mel_binary/mel_binary_*_v2/results.txt`.
- **TENT hurts `pandem_real_probe` on Fitz17k.** Tried `lr ∈ {1e-3, 1e-4}` × `n_steps ∈ {1, 3}`; every config drops dark-skin Mal AUC. Use the unadapted baseline (0.5531) on Fitz17k for this model, or switch to `efficientnetv2_m + TENT(n_steps=3)` which reaches 0.6742.
- **Fitzpatrick17k images are partially missing.** 3,887 of 16,577 URLs downloaded (~23% liveness). After loader filtering, 3,730 usable images (818 light / 2,076 medium / 836 dark). The remaining 12k+ are dead Atlas Dermatologico / DermaAmin links. Numbers are honest within the recovered subset; absolute AUCs may shift slightly on a fuller dataset.
- **No clinical / dermatologist evaluation.** All numbers are model AUCs on annotation-aligned ground truth; no human-in-the-loop validation. This is a methodology study, not a clinical-deployment claim.

---

## References

- Tschandl, Rosendahl, Kittler (2018). [*The HAM10000 dataset*](https://www.nature.com/articles/sdata2018161). Scientific Data 5:180161.
- Oquab et al. (2023). [*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193). arXiv 2304.07193.
- Darcet, Oquab, Mairal, Bojanowski (2024). [*Vision Transformers Need Registers*](https://arxiv.org/abs/2309.16588). ICLR 2024.
- **PanDerm** — Yan et al. (2025). [*A multimodal vision foundation model for clinical dermatology*](https://www.nature.com/articles/s41591-025-03747-y). Nature Medicine. (The PanDerm vision encoder. The HF checkpoint used here, `redlessone/DermLIP_PanDerm-base-w-PubMed-256`, packages this encoder with the DermLIP text head from the Derm1M paper below.)
- **DermLIP / Derm1M** — Yan et al. (2025). [*Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology*](https://arxiv.org/abs/2503.14911). arXiv 2503.14911; ICCV 2025 Highlight.
- Zhai, Mustafa, Kolesnikov, Beyer (2023). [*Sigmoid Loss for Language Image Pre-Training (SigLIP)*](https://arxiv.org/abs/2303.15343). ICCV 2023.
- Tschannen et al. (2025). [*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*](https://arxiv.org/abs/2502.14786). arXiv 2502.14786 (Google DeepMind).
- Woo et al. (2023). [*ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*](https://arxiv.org/abs/2301.00808). CVPR 2023.
- Tan & Le (2021). [*EfficientNetV2: Smaller Models and Faster Training*](http://proceedings.mlr.press/v139/tan21a/tan21a.pdf). ICML 2021.
- Daneshjou et al. (2022). [*Disparities in dermatology AI performance on a diverse, curated clinical image set*](https://www.science.org/doi/10.1126/sciadv.abq6147) (the DDI dataset). Science Advances 8:eabq6147.
- Groh et al. (2021). [*Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf). CVPR Workshops 2021 (ISIC).
- Combalia et al. (2019). [*BCN20000: Dermoscopic Lesions in the Wild*](https://arxiv.org/abs/1908.02288). arXiv 1908.02288 (the BCN20000 portion of ISIC 2019).
- Lin, Goyal, Girshick, He, Dollár (2017). [*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002). ICCV 2017.
- Platt (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. In *Advances in Large Margin Classifiers*, MIT Press. (Sigmoid calibration.)
- Wang et al. (2021). [*TENT: Fully Test-Time Adaptation by Entropy Minimization*](https://arxiv.org/abs/2006.10726). ICLR 2021.
- Bao, Dong, Piao, Wei (2022). [*BEiT: BERT Pre-Training of Image Transformers*](https://arxiv.org/abs/2106.08254). ICLR 2022. (Vision tower architecture in `RealPanDermClassifier`.)

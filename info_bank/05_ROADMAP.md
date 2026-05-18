# Next Steps — Skin Lesion v2 (archived 2026-05-15)

**Project status: research wrap-up.** All planned work paths have been tried, with both positive and negative results recorded. See:

- [04_FINDINGS_AND_RESUME.md](04_FINDINGS_AND_RESUME.md) — consolidated review of findings, results, publishability, and resume bullets.
- [../v2/README.md](../v2/README.md) — public-facing project page with full tables.
- [../v2/PHASES.md](../v2/PHASES.md) — phase-by-phase status (E, B, OA tracks).
- [archive/OVERNIGHT_RESULTS.md](archive/OVERNIGHT_RESULTS.md) — chronological log of every overnight run.

---

## Completed work tracks

| Track | Final outcome |
|---|---|
| **E** (HAM10000 in-distribution) | 4-model post-hoc ensemble, **test AUC 0.9733**, mel recall 0.566 |
| **B1** (ISIC 2019) | 8 models evaluated. Best: `pandem_real_probe` macro OvR **0.8845** |
| **B2** (DDI fairness) | 6 models + Fitzpatrick stratification. TENT + per-target SLSQP: **dark-skin Mal AUC 0.6571, L-D gap -0.0004** |
| **B3** (Fitz17k fairness) | 6 models, 3,730 usable images. Best: `efficientnetv2_m + TENT(n_steps=3)`, **dark-skin Mal AUC 0.6742** |
| **B4** (cross-dataset SLSQP) | Per-target ensemble weights computed; accuracy-fairness tradeoff explicit |
| **B5** (TENT test-time adaptation) | Working on 3/4 backbones for DDI, 2/4 for Fitz17k. Limitation: `pandem_real_probe` + Fitz17k fails |
| **E12** (sample efficiency) | `pandem_real_probe` at 25% data ≈ `efficientnetv2_m` at 100% (4× efficiency) |
| **OA1** (binary mel specialist) | Recovered ConvNeXt-V2 collapse; final test mel AUC 0.8942, but does NOT beat 7-way + F2 threshold |

## Negative results recorded

- 5-model ensemble (+ `siglip2_probe`): hurts (-0.0015 ensemble AUC)
- `pandem_real_ft` (full FT): hurts vs probe (-0.023 AUC)
- Binary mel specialist v2: does not beat 7-way + F2 threshold
- TENT on `pandem_real_probe` for Fitz17k: hurts under every hyperparam tried

## Open work — *deliberately* not pursued

The following are real follow-ups but were judged out of scope for this iteration. Listed for posterity:

- **OA2** Multimodal model — age + sex + anatomical site metadata fusion head. v1 had this; v2 port deferred.
- **OA3** Grad-CAM / attention visualisations — for clinical interpretability story; visual-only, no AUC impact.
- **OA4** Final write-up (workshop paper or arXiv preprint).
- **B6** GroupDRO / DANN training — research-grade domain-generalisation methods that go beyond test-time adaptation.
- **Real PanDerm-256 retraining** — A/B test 224 vs 256 for the actual encoder (not the DINOv2 fallback). Likely small effect; 224 is the documented native resolution.
- **Refit B4 ensemble using per-model best TENT config** — current B4 ensemble uses a single TENT config across all models. Mixing per-model best (e.g. lr=1e-3 for DDI, lr=1e-4 + n_steps=3 for Fitz17k where appropriate) might push Fitz17k dark-skin Mal AUC slightly past 0.67.

## How to reproduce the headline numbers

```powershell
cd D:\ML\skin-lesion-vit\v2
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"

# 1. In-distribution headline (0.9733)
python -m src.post_hoc --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe
# → experiments/post_hoc/results.txt

# 2. ISIC 2019 single-model winner (0.8845)
Get-Content experiments\cross_dataset\isic2019\pandem_real_probe\report.txt

# 3. DDI TENT fairness winner (dark 0.6571, gap -0.0004)
python -m src.cross_dataset_ensemble `
    --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe `
    --datasets ddi --probs_subdir cross_dataset_tent --out_suffix _tent_lr1e-3
# → experiments/cross_dataset_ensemble_tent_lr1e-3/report.txt

# 4. Sample efficiency curve
Get-Content experiments\sample_efficiency\pandem_real_probe_frac025\results.txt
Get-Content experiments\sample_efficiency\efficientnetv2_m_frac100\results.txt
```

## Documentation entry points

- New collaborator: start with [04_FINDINGS_AND_RESUME.md](04_FINDINGS_AND_RESUME.md) (10-min read) → [../v2/README.md](../v2/README.md) (depth) → [06_AI_AGENT_CONTEXT.md](06_AI_AGENT_CONTEXT.md) (pitfalls).
- Reviewer interested in the PanDerm load bug: [../v2/src/models/registry.py](../v2/src/models/registry.py) `RealPanDermClassifier` + the *PanDerm loader fix* section of the v2 README.
- Reviewer interested in fairness methodology: [../v2/src/tent.py](../v2/src/tent.py) + [../v2/src/cross_dataset_ensemble.py](../v2/src/cross_dataset_ensemble.py).

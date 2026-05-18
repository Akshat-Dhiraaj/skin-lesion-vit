# Phases — skin-lesion-vit-v2

Status legend:  ✅ done · 🟡 partial · 🔴 not started · ⛔ negative result (recorded, won't pursue)

---

## Direction E — HAM10000 benchmark (lesion-level)

- ✅ E1 · Patient-level splits + CI tests (`src/data/splits.py`, `tests/test_splits.py`)
- ✅ E2 · HAM10000 loader (`src/data/ham10000.py`)
- ✅ E3 · Augmentation + TTA pipelines (`src/data/transforms.py`, `src/tta.py`)
- ✅ E4 · Model registry (`src/models/registry.py`) — 11 entries (9 + pandem_real_probe + pandem_real_ft)
- ✅ E5 · Training engine (`src/engine.py`)
- ✅ E6 · DINOv2+Registers probe — test AUC 0.9232
- ✅ E7 · DINOv2+Registers full FT — test AUC 0.9358 (TTA test 0.9408)
- ✅ E8 · PanDerm probe (DINOv2 fallback ✗) — test AUC 0.9305
- ✅ E9 · PanDerm full FT (DINOv2 fallback ✗) — test AUC 0.9330 (TTA test 0.9359)
- ✅ E10 · ConvNeXt-V2 Large @ 384 — test AUC **0.9660** (TTA test 0.9677) [v2 best single]
- ✅ E10b · EfficientNetV2-M @ 384 — test AUC 0.9474 (TTA test 0.9555)
- ✅ E11 · SigLIP 2 SO400M probe — test AUC 0.9460 (TTA test 0.9490)
- ✅ E11b · Real PanDerm probe (`pandem_real_probe`, BEiT via open_clip) — test AUC **0.9527** (TTA test 0.9552)
- ✅ E11c · Real PanDerm full FT (`pandem_real_ft`) — test AUC 0.9295 (TTA test 0.9368). ⛔ FT < probe by 0.023 — pretrained features too task-relevant to disturb on 7k images.
- ✅ E12 · Sample-efficiency curves — `pandem_real_probe` at 25% data (0.9398) ≈ `efficientnetv2_m` at 100% (0.9500). 4× data efficiency.
- ✅ E13 · 4-model SLSQP ensemble — test AUC **0.9731** (with `pandem_real_probe`; was 0.9717 with the DINOv2-fallback `pandem_ft`)
- ✅ E14 · 3-stage post-hoc pipeline (per-class SLSQP + Platt sigmoid + F2 thresholds) — test AUC **0.9733**, mel recall 0.566
- ⛔ E15 · 5-model ensemble (+ `siglip2_probe`) — *negative*. SLSQP gives siglip2_probe the floor weight (0.06) because its errors correlate too tightly with `convnextv2_large`. Test AUC drops to 0.9703 (-0.0015). Artifact: `experiments/ensemble/results_5model.txt`.

---

## Direction B — Cross-dataset / fairness

- ✅ B1 · ISIC 2019 inference — 8 models. **`pandem_real_probe` macro OvR 0.8845** (+0.030 over `convnextv2_large` 0.8546).
- ✅ B2 · DDI Fitzpatrick-stratified — 656 images, universal-3 taxonomy. Best absolute Mal AUC: `pandem_real_probe` 0.6821. Smallest L-D gap: `dinov2_registers_ft` 0.068.
- ✅ B3 · Fitzpatrick17k Fitzpatrick-stratified — 3,730 usable images of 16,577 (URL liveness ~23%). Best dark-skin Mal AUC: `dinov2_registers_ft` 0.6071. Modality flip: dermatology-pretrained PanDerm trails ImageNet-pretrained CNNs on dark skin here.
- ✅ B4 · Cross-dataset SLSQP ensemble — per-target weights computed (`src/cross_dataset_ensemble.py`). Fitz17k accuracy-vs-fairness tradeoff: -0.010 macro AUC buys +0.044 dark-skin Mal AUC, closing the gap 0.167 → 0.070.
- ✅ B5 · TENT test-time adaptation (`src/tent.py`, `scripts/run_tent.py`) — `pandem_real_probe` on DDI dark-Mal AUC 0.5919 → **0.6931** (+0.10), L-D gap 0.175 → 0.044. On Fitz17k, `efficientnetv2_m + n_steps=3` reaches dark-Mal AUC **0.6742**. Per-target SLSQP ensemble on TENT-adapted probs closes the DDI L-D gap to **-0.0004** at dark-Mal 0.6571.
- ⛔ B5b · TENT on `pandem_real_probe` for Fitz17k — *negative*. Tried lr ∈ {1e-3, 1e-4}, n_steps ∈ {1, 3}; every config drops dark-Mal AUC below baseline. Bare entropy minimisation isn't enough when the backbone's pretraining distribution is far from clinical photos.
- 🔴 B6 · GroupDRO / DANN — research-grade follow-up, not started. Would need either Fitzpatrick labels on training data or domain-aligned training data (Fitz17k as a second domain).

---

## Open / aspirational

- ⛔ OA1 · Binary mel specialist v2 — *negative*. `mel_binary_convnextv2_large_v2` test mel AUC 0.8942, R@P=0.80 = 0.220. The 7-way model with F2-tuned threshold (mel recall 0.566 from post-hoc) beats it. v1 Phase 19's 0.83 mel recall was on image-level (leaky) splits and doesn't survive lesion-level evaluation with this recipe.
- 🔴 OA2 · Multimodal v2 (age, sex, anatomical_site metadata fusion) — not started; v1 had it.
- 🔴 OA3 · GradCAM / attention visualization — not started; useful for write-up only.
- 🔴 OA4 · Final paper / report — pending decision on whether to publish.

---

## Notes

- All AUCs are macro OvR on held-out test sets. HAM10000 splits: lesion-level `StratifiedGroupKFold` (N=1,431 / 1,067 lesions).
- TTA: 8 deterministic views (orig + 3 rots + 3 flips + zoom), softmax-averaged.
- ✗ marks `pandem_*` registry entries that silently fall back to `facebook/dinov2-with-registers-base`. Use `pandem_real_*` for the actual PanDerm encoder.
- `pandem_real_probe` is the strongest **single** cross-dataset model on ISIC 2019 and DDI; `dinov2_registers_ft` is the strongest standalone on Fitz17k dark skin.
- `efficientnetv2_m + TENT(n_steps=3)` is the strongest standalone on Fitz17k dark skin overall.
- Best in-distribution: 4-model post-hoc pipeline at HAM10000 test AUC **0.9733**.
- Best DDI fairness: per-target TENT-aware SLSQP ensemble — dark-skin Mal AUC 0.6571 with L-D gap **-0.0004** (full closure).
- Repository commit hash at this snapshot: see `git log --oneline | head -10` from inside `D:\ML\skin-lesion-vit-v2`.

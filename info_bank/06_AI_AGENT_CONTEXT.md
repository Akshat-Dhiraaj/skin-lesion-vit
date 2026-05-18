# AI Agent Context — D:\ML (Skin Lesion Classification)

> Single source of truth for any AI agent picking up work in this directory.
> **Verify before recommending.** This file documents the code state at 2026-05-13;
> always cross-check paths, file contents, and result numbers against the actual
> files before acting on them.
>
> **Model/technique deep-dives** (what each model is, how it works, why chosen,
> paper + link, result it produced) live in [`02_MODELS_AND_TECHNIQUES.md`](02_MODELS_AND_TECHNIQUES.md). This file
> stays terse and operational; `02_MODELS_AND_TECHNIQUES.md` is the explanatory companion.

---

## Top-level layout (`D:\ML`)

```
D:\ML\
├── .idea/                              PyCharm metadata — ignore
├── .venv/                              shared venv for both projects (Python 3.11+)
├── 03_COMPLETE_GUIDE.md   long-form project journal (~110KB) — narrative, not authoritative
├── 06_AI_AGENT_CONTEXT.md                       THIS FILE
├── main.py                             PyCharm boilerplate (`print_hi`) — unused, ignore
├── skin-lesion-vit/                    v1 — image-level splits, 22 phases
│   └── README.md                       authoritative for v1 — synced with experiments/*/results.txt
└── ../v2/                 v2 — lesion-level splits, model registry, cross-dataset eval
    └── README.md                       authoritative for v2 — synced with experiments/*/results.txt
```

The two projects are **independent git repos** (`.git/` in each subdir) but share the HAM10000 data folder — v1 owns the raw images, and v2 hardcodes v1's path as the default of `--data_dir` in `scripts/train.py:28` and `scripts/verify_data.py`. The `configs/data/ham10000.yaml` file exists with the same values but **no code reads it** (hydra-core is in deps but never imported).

---

## Project mission

Skin lesion classification on dermoscopy with bleeding-edge vision backbones, targeting clinical-relevant melanoma detection. Two versions:

- **v1** — exploratory: how far can ensembling and post-hoc tricks push HAM10000 macro AUC? **Val-selected headline:** 6-model stacked LogReg ensemble (Phase 21) at **0.9840 test AUC** (val AUC 0.9956). **Highest single-model test AUC:** Phase 18 ConvNeXt-V2 Large standalone at **0.9880** — actually beats the Phase 21 ensemble on test, but loses on val (0.9893 vs 0.9956), so was not the val-selected headline. All v1 numbers are on image-level splits — leaky, see caveat.
- **v2** — honest: re-evaluate the v1 winners on **lesion-level splits**, add 2025 dermatology foundation models (DermLIP-PanDerm, SigLIP 2), extend to **cross-dataset / Fitzpatrick fairness** evaluation. Final-so-far: ConvNeXt-V2 Large + TTA at **0.9816 val TTA AUC**, **0.9660 test AUC** single-model; 4-model SLSQP ensemble at 0.9717.

Same hardware budget throughout: **RTX 4060 Laptop 8 GB VRAM, 32 GB RAM, Windows 11 + PowerShell**.

---

## v1 — `D:\ML\skin-lesion-vit\`

### Purpose
Sequential 22-phase exploration: ResNet50 frozen baseline → individual backbone fine-tunes → TTA → grid-search ensembles → multimodal → focal loss → ConvNeXt-V2 → mel-binary specialist → meta-learning. Each new phase reuses earlier checkpoints.

### Authoritative entry points
- **`src/engine.py`** — unified phase runner for phases 1–7 and 10. `PHASE_CONFIGS` dict drives everything.
- **`src/advanced_phases.py`** — phases 11 (re-ensemble), 12 (multimodal), 13 (focal loss).
- **`src/tta_ensemble.py`** — phase 8 (TTA inference) + phase 9 (2-model grid ensemble).
- **`src/phase14_three_model_ensemble.py`** — combines eff+TTA, dino-FT+TTA, multimodal.
- **`src/phase16_unified_pipeline.py`** — unified TTA + SLSQP + mel threshold (50/50 tune/eval split).
- **`src/Phase17 maximum push.py`** — overnight 20-view TTA + 5-fold SLSQP.
- **`src/phase18_convnextv2_train.py`** — `convnextv2_large.fcmae_ft_in22k_in1k_384` @ 384px.
- **`src/phase19_mel_binary_train.py`** — binary mel-vs-rest specialist (focal binary loss).
- **`src/phase20_val_set_inference.py`** — TTA on val+test for 6 models, saves logits + probs.
- **`src/phase21_final_ensemble.py`** — LogReg / SLSQP / meta-meta stacking on Phase 20 artifacts.
- **`src/phase22_quick_checks.py`** — post-21 isotonic / cleaner-mel-binary / ConvNeXt+LogReg blend.
- **`run_all.bat`** — bulk runner for phases 1–7, 10, 11, 12, 13.

### Models trained (one folder under `experiments/` each)
`phase{1,2,3,4,5,6,7,10,12,13,18,19}` produce `best_model.pth` checkpoints. Phases 8, 9, 11, 14, 15, 16, 20, 21, 22 only produce derivative artifacts (`*.npy`, `results.txt`).

**Phases that ran partially or not at all:**
- **Phase 17** — `src/Phase17 maximum push.py` exists, **never completed**, no `experiments/phase17*` folder.
- **`experiments/phase7_medsiglip/`** — empty stub from an aborted MedSigLIP attempt (separate from the real `phase7_siglip/` SigLIP-SO400M run).
- **`experiments/phase16_optimizations/`** — `slsqp_probs.npy` + `xgb_probs.npy` from the older `opt_*.py` scripts (treat with caution; those scripts had test-set leakage in their original form).

### Best phase results (image-level splits, test N=1503)
- Phase 18 ConvNeXt-V2 Large standalone: Test AUC **0.9880** — the highest test AUC of any normally-evaluated phase in the project. Val AUC 0.9893, Test Acc 0.8862.
- Phase 21 (LogReg meta-learner on 6 models): Test AUC **0.9840**, mel recall 0.8323. Val AUC 0.9956 — this is the val-selected headline despite a lower test AUC than Phase 18.
- Phase 22 Check 3 (ConvNeXt 0.05 + LogReg 0.95): Test AUC 0.9887 — highest test number ever observed in this repo, but flagged as a post-hoc sanity check; loses to Phase 21 / Calibrated LogReg on val.

### v1 data pipeline gotcha (KNOW THIS)
- `src/dataset.py:split_dataset` is **image-level** `sklearn.train_test_split` stratified by `dx`. The same lesion has multiple near-duplicate images in HAM10000; they leak across splits. **All v1 numbers are optimistic.** v2 exists to fix this.
- `D:\ML\skin-lesion-vit\1.0.0` is **pip install output captured to a file**, not release notes. Ignore.
- `engine.py:361` sets `model.vit.encoder.gradient_checkpointing = True` only for the ViT-HF path; other backbones rely on framework defaults.
- `engine.py:300` accumulates gradients but **does not flush the last partial accumulation** at end-of-epoch — the check is `if (step + 1) % cfg["accum_steps"] == 0` with no `or step == len(loader) - 1` clause (v2 fixed this).
- `src/opt_slsqp_weights.py`, `src/opt_xgboost_stack.py`, `src/opt_learned_thresholds.py` **originally fit weights on the test set** — leakage. Each file's docstring documents the fix to a held-out val split, but read the file before reusing it.

### Key v1 files at a glance
| File | Role | Notable |
|---|---|---|
| `src/dataset.py` | HAM10000 loader, image-level splits, WeightedRandomSampler | `CLASS_NAMES = ['mel','nv','bcc','akiec','bkl','df','vasc']`; mel idx = 0 |
| `src/transforms.py` | Train + val Albumentations pipelines | Heavy aug train; resize+norm val. ImageNet stats |
| `src/model.py` | Legacy ViT/ResNet helpers | Pre-engine.py code; `train.py` still imports it |
| `src/train.py` | Legacy ViT-only loop | Superseded; `engine.py` is the real loop |
| `src/engine.py` | Unified phase runner | `PHASE_CONFIGS` dict; AMP, accum, OneCycleLR |
| `src/evaluate.py` | Per-class metrics + GradCAM helpers | Not used in main flow |
| `experiments/<phase>/results.txt` | Authoritative per-phase test metrics | Always reflect the latest run of that phase |

### Run examples
```powershell
python src/engine.py --phase 5            # one of {1,2,3,4,5,6,7,10}
python src/advanced_phases.py --phase 11  # one of {11,12,13}
python src/tta_ensemble.py --phase 8      # or 9
python src/phase18_convnextv2_train.py
python src/phase20_val_set_inference.py   # writes phase20_val_inference/*.npy
python src/phase21_final_ensemble.py      # consumes Phase 20 artifacts
.\run_all.bat                              # phases 1-7, 10-13
mlflow ui                                  # reads ./mlflow.db
```

---

## v2 — `D:\ML\skin-lesion-vit\v2\`

### Purpose
Rebuilt with three explicit goals:
1. **Honest splits** — `StratifiedGroupKFold(group='lesion_id')` enforced by pytest.
2. **Clean architecture** — single `ModelConfig` dataclass registry, uniform `forward(x) → logits`.
3. **Cross-dataset robustness** — ISIC 2019 / DDI / Fitzpatrick17k evaluation under universal-3 and HAM10K-7 taxonomies, Fitzpatrick-stratified malignant AUC.

### Authoritative entry points
- **`scripts/train.py`** — `--model <registry_key>` → calls `src/engine.py:run_phase`.
- **`src/engine.py:run_phase`** — single phase runner. Replaces v1's `PHASE_CONFIGS`/`engine.py` switch logic.
- **`src/models/registry.py`** — `REGISTRY: dict[str, ModelConfig]`. Adding a backbone = adding a config entry (and a class only if the architecture pattern is new).
- **`src/tta.py`** — 8-view TTA on val and/or test. Writes both `tta_val_*.npy` and `tta_test_*.npy` — both are needed by `post_hoc.py`. Test-side naming has a backward-compat alias (`tta_probs.npy` / `tta_labels.npy`) used by the older `src/ensemble.py`.
- **`src/ensemble.py`** — global weighted ensemble (grid or SLSQP). Reads `tta_probs.npy` per model.
- **`src/post_hoc.py`** — per-class SLSQP → Platt sigmoid calibration → F2 thresholds, fit on val, evaluated on test.
- **`src/cross_dataset.py`** — HAM10K-trained model → DDI / Fitz17k / ISIC2019 evaluation. Produces per-class AUC, malignant AUC, and (when available) Fitzpatrick-stratified results.
- **`scripts/leaderboard.py`** — scrape `experiments/*/results.txt` into a markdown table.
- **`scripts/verify_data.py`** — load HAM10000, print splits, smoke-test DataLoaders.
- **`tests/test_splits.py`** — pytest: split leakage, registry validity, engine train/eval smoke.

### Backbones in the registry (`src/models/registry.py`)
| Key | HF / timm ID | Type | Mode | Input |
|---|---|---|---|---|
| `convnextv2_large` | `convnextv2_large.fcmae_ft_in22k_in1k_384` | timm | full FT | 384 |
| `dinov2_registers_probe` | `facebook/dinov2-with-registers-base` | dinov2 | linear probe | 224 |
| `dinov2_registers_ft` | `facebook/dinov2-with-registers-base` | dinov2 | full FT | 224 |
| `pandem_probe` ✗ | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | pandem (DINOv2 fallback) | linear probe | 256 |
| `pandem_ft` ✗ | same | pandem (DINOv2 fallback) | full FT | 256 |
| `pandem_probe_224` ✗ | same | pandem (DINOv2 fallback) | linear probe | 224 |
| `pandem_ft_224` ✗ | same | pandem (DINOv2 fallback) | full FT, not trained | 224 |
| **`pandem_real_probe`** | same | **pandem_real** (BEiT via open_clip) | linear probe | 224 |
| **`pandem_real_ft`** | same | **pandem_real** | full FT | 224 |
| `siglip2_probe` | `google/siglip2-so400m-patch14-384` | siglip2 | linear probe | 384 |
| `efficientnetv2_m` | `tf_efficientnetv2_m` | timm | full FT | 384 |

✗ marks entries that *silently* use `facebook/dinov2-with-registers-base` because the HF repo lacks a `config.json` (it's an open_clip checkpoint). All `pandem_*` (no `_real`) checkpoints in `experiments/` are DINOv2+Registers, not PanDerm. Use `pandem_real_*` for the actual encoder.

`model_type` dispatch in `build_model()`:
- `timm` → `timm.create_model(..., pretrained=True, num_classes=7)` + optional head-only freezing.
- `dinov2` → `DINOv2Classifier` (HF AutoModel + CLS token + linear head).
- `pandem` → `PanDermClassifier` (HF AutoModel + CLS-or-pooled + linear head, **falls back to dinov2-with-registers-base silently** — see Pitfall #2).
- **`pandem_real`** → `RealPanDermClassifier`. Loads `open_clip_pytorch_model.bin` directly via `huggingface_hub.hf_hub_download`, filters to the `visual.*` sub-state-dict (strips `visual.` prefix, drops `head.*`), and builds a `timm.models.beit.Beit` with `img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_values=0.1, global_pool='token'`. Loads with **0 missing / 0 unexpected keys**. Uses OpenAI-CLIP normalisation `(0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758)` from `open_clip_config.json` — NOT ImageNet, NOT SigLIP.
- `siglip2` → `SigLIP2Classifier` (extracts `.vision_model` from full VLM, mean-pooled patch tokens via `pooler_output`, always frozen).

### Trained models (each has `experiments/<key>/best_model.pth` + `results.txt`)
- `convnextv2_large` (test AUC 0.9660), `dinov2_registers_ft` (0.9358), `dinov2_registers_probe` (0.9232), `pandem_ft` ✗ (0.9330), `pandem_probe` ✗ (0.9305), `pandem_probe_224` ✗ (0.9242), `efficientnetv2_m` (0.9474), `siglip2_probe` (0.9460).
- **`pandem_real_probe`** — test AUC **0.9527** / TTA 0.9552. Real PanDerm vision encoder. Best probe in v2; +0.022 over the DINOv2-fallback `pandem_probe`.
- **`pandem_real_ft`** — test AUC 0.9295 / TTA 0.9368. Full FT *under*performs the probe by 0.023 AUC — dermatology pretraining too task-relevant to disturb on 7k images. Probe is the correct operating mode for this backbone.
- **`pandem_ft_224`** — registered, NOT trained. Was kept for the original 224 A/B test against `pandem_ft`; superseded by `pandem_real_probe` and need not be trained.

### TTA + ensemble artifacts
For each trained model, after running `python -m src.tta --model <key>`:
```
experiments/<key>/
  best_model.pth
  results.txt
  tta_val_probs.npy   tta_val_labels.npy
  tta_test_probs.npy  tta_test_labels.npy
  tta_probs.npy       tta_labels.npy        ← test-split aliases for backward compat
  tta_results.txt
```
Plus shared aggregates:
```
experiments/
  ensemble/{ensemble_probs,labels}.npy + results.txt    ← from src.ensemble
  post_hoc/{per_class_weights,thresholds,test_probs_calibrated}.npy + results.txt   ← from src.post_hoc
  cross_dataset/<dataset>/<model>/{probs_*.npy, labels.npy, report.txt}             ← from src.cross_dataset
  overnight_logs/run_option1_*.log                                                  ← from scripts/run_option1.ps1
```

### Cross-dataset state
- **ISIC 2019** (`data/external/ISIC2019/`, 15,316 images, ham10k taxonomy): 8 models. Best macro OvR: **`pandem_real_probe` 0.8845** (linear probe). `convnextv2_large` 0.8546. `efficientnetv2_m` 0.8294. `pandem_probe_224` ✗ 0.8337. `pandem_real_ft` 0.8185. `pandem_ft` ✗ 0.8101. `dinov2_registers_ft` 0.8077. `siglip2_probe` 0.6610.
- **DDI** (`data/external/DDI/`, 656 images, universal-3 taxonomy + Fitzpatrick-stratified): 6 models. Best absolute Mal AUC: **`pandem_real_probe` 0.6821**. Best dark-skin Mal AUC standalone: `pandem_real_probe` 0.5919. Smallest light-to-dark gap: `dinov2_registers_ft` 0.068. `pandem_ft` ✗ at dark-skin Mal AUC 0.4396 — *worse than random*.
- **Fitzpatrick17k** (`data/external/Fitzpatrick17k/`, 3,730 of 16,577 usable; ~23% URL liveness): 6 models. Best dark-skin Mal AUC standalone: **`dinov2_registers_ft` 0.6071** — flip from DDI; clinical-photo modality favours ImageNet over dermatology pretraining on dark skin here.
- **TENT** (`src/tent.py` + `scripts/run_tent.py`) — adapts only normalisation affines at test time. DDI: dark-skin Mal AUC improves for 3 of 4 backbones, up to **+0.10 AUC** (`pandem_real_probe` 0.5919 → **0.6931**). Fitz17k: `efficientnetv2_m + n_steps=3` → dark-skin Mal AUC **0.6742** (new best on dark Fitz17k). `pandem_real_probe` + TENT fails on Fitz17k regardless of lr / n_steps.
- **Per-target SLSQP TENT ensemble** (`src/cross_dataset_ensemble.py` with `--probs_subdir cross_dataset_tent`) — DDI: dark-skin Mal AUC 0.6571 with L-D gap **-0.0004** (full closure). Reports at `experiments/cross_dataset_ensemble_tent_lr1e-{3,4}/`.

### v2 data pipeline
- `src/data/splits.py:split_dataset` — two-pass `StratifiedGroupKFold`. Pass 1 separates test, pass 2 separates val. Group disjoint by construction.
- `assert_no_group_leakage()` is called right after every split — cheap insurance, asserted in `tests/test_splits.py`.
- `src/data/ham10000.py:load_ham10000` — reads CSV, attaches paths, integer labels via `CLASS_TO_IDX`, splits via `split_ham10000` (group='lesion_id'), runs the leakage guard.
- `src/data/transforms.py` — exposes `get_norm_stats(model_type)` → SigLIP stats `(0.5,0.5,0.5)` when `model_type="siglip2"`, CLIP stats `(0.4815,0.4578,0.4082)` when `model_type="pandem_real"`, ImageNet otherwise; plus `get_train_transform` (heavy aug) and `get_val_transform` (resize+norm), both taking an optional `model_type` arg (default ImageNet, backward-compat). **The 8-view TTA pipeline lives in `src/tta.py:get_tta_transforms`, NOT here** — a dead duplicate `get_tta_transforms` in this file was removed in the 2026-05-16 behavior-preserving refactor (it differed from tta.py's view-8 augmentation; tta.py is the single source of truth, and produced every committed `tta_results.txt`). `src/engine.py` and `src/tta.py` pass `cfg.model_type` through.
- WeightedRandomSampler in `get_weighted_sampler` — `weights = 1 / count(class)`.

### Key behavioural details
- `evaluate()` in `src/engine.py` has a **NaN guard** for AMP overflow (EfficientNetV2 is prone). NaN rows → uniform 1/7 probabilities.
- Gradient clipping `max_norm=1.0` is on by default in `train_one_epoch`.
- `_enable_grad_checkpointing()` tries the HF API (`gradient_checkpointing_enable`) first, then the timm API (`set_grad_checkpointing`). Silently skips otherwise.
- `src/post_hoc.py` deliberately uses **Platt sigmoid** instead of isotonic — v1 Phase 22 showed isotonic destroys macro OvR AUC on small rare-class val populations (df=16, vasc=20). Platt is monotonic and preserves AUC.
- `apply_platt()` does **not** renormalise rows — renormalising kills per-class ranking and macro OvR AUC. Outputs are per-class probabilities in `[0,1]` that don't sum to 1.

### Run examples
```powershell
pip install -e ".[dev]"
pytest -v

python scripts/verify_data.py                                                 # smoke-test data
python scripts/train.py --model convnextv2_large                               # train
python -m src.tta --model convnextv2_large --data_dir D:\ML\skin-lesion-vit\data\raw  # TTA
python -m src.ensemble --method slsqp --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_ft
python -m src.post_hoc --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_ft
python -m src.cross_dataset --model convnextv2_large --dataset isic2019 --data_root data/external --label_mode ham10k
python scripts/leaderboard.py
mlflow ui
```

---

## Class registry (BOTH projects, in identical order)

```python
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
```
- `mel` = melanoma (index 0) — the clinically critical class.
- HAM10000 malignant classes for clinical metrics: `{mel, bcc, akiec}`.
- Universal-3 taxonomy (v2 cross-dataset): `[malignant, benign, non_neoplastic]`.

---

## Shared data location

```
D:\ML\skin-lesion-vit\data\raw\
  HAM10000_metadata.csv
  HAM10000_images_part_1/      5000 .jpg
  HAM10000_images_part_2/      5015 .jpg
  hmnist_*_RGB.csv             low-res preprocessed (unused by current code)
  hmnist_*_L.csv               grayscale (unused by current code)
```
- v1 reads this directly (`DATA_DIR = D:\ML\skin-lesion-vit\data\raw` hardcoded in scripts).
- v2 hardcodes the same path as the default of `--data_dir` in `scripts/train.py` and `scripts/verify_data.py`. The `configs/data/ham10000.yaml` file exists but **no code reads it** — hydra-core is in deps but never imported.

`D:\ML\skin-lesion-vit\v2\data\external\ISIC2019\` is the ISIC 2019 challenge data (15,316 images after HAM10000 exclusion). DDI and Fitz17k folders do not exist yet.

---

## Hardware constraints (affects every config decision)

- **RTX 4060 Laptop, 8 GB VRAM** — every config sized for this:
  - Mixed precision (`torch.cuda.amp` in v1, `torch.amp` in v2).
  - Gradient accumulation — effective batch 32–64, physical batch 4–16.
  - Gradient checkpointing on transformers and large CNNs.
  - 8-bit optimizers not used; AdamW default state is the limiting factor for ConvNeXt-V2 Large.
- **32 GB RAM** — sufficient for in-RAM image caching of HAM10000 (~2.4 GB for val+test combined, used by `phase20_val_set_inference.py`).
- **Windows + PowerShell** — use `num_workers=0` on DataLoaders if fork errors. v2 makes this a parameter.

---

## Common pitfalls when working in this directory

1. **Don't compare v1 and v2 numbers directly.** v1 splits leak; v2 doesn't. Same model on same data gives ~0.022 lower test AUC on v2 because the test set is genuinely harder.
2. **Use `pandem_real_*` for actual PanDerm, NOT `pandem_*`.** The DermLIP/PanDerm HF repo (`redlessone/DermLIP_PanDerm-base-w-PubMed-256`) lacks a `config.json` — it's an **open_clip** checkpoint, not a transformers model. `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` raises `ValueError: missing model_type key in config.json`. The `PanDermClassifier` `try/except` block then silently falls back to `facebook/dinov2-with-registers-base`. **Every `pandem_*` checkpoint in `experiments/` (probe/ft, 256/224) is therefore DINOv2+Registers, not PanDerm.**

   The fix lives in `RealPanDermClassifier` (model_type `pandem_real`). It downloads `open_clip_pytorch_model.bin`, filters to the `visual.*` sub-state-dict, and loads into a `timm.models.beit.Beit` with `use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, global_pool='token'`. Loads with 0 missing / 0 unexpected keys. Uses OpenAI-CLIP normalisation `(0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758)`, NOT ImageNet.

   The fix changes the model's accuracy meaningfully:
   - HAM10000 test AUC: 0.9305 (fallback) → 0.9527 (real, probe)
   - ISIC 2019 macro OvR: 0.8101 → 0.8845 (+0.074)

   References: PanDerm vision encoder [Yan et al., Nat. Med. 2025](https://www.nature.com/articles/s41591-025-03747-y); DermLIP / Derm1M text head [Yan et al., arXiv 2503.14911](https://arxiv.org/abs/2503.14911). Two separate Yan-first papers.

   **PanDerm full FT is worse than probe.** `pandem_real_ft` (test AUC 0.9295) < `pandem_real_probe` (0.9527). The 2M-image dermatology pretraining is too task-relevant to disturb on 7k images. Use the probe.
3. **Don't trust v1's `opt_*.py` raw outputs without reading the docstring.** They originally fit on test; the fix is documented in the file header but the older saved artifacts may predate it.
4. **Don't rerun v1 Phase 22 expecting better numbers.** It's a sanity-check script — the calibrated variant explicitly trades test AUC for honest val selection.
5. **Don't add backbones to v1's engine.py.** v1 is frozen; new work happens in v2's registry.
6. **Don't `os.makedirs` the data directory.** It exists. Just point `--data_dir` at `D:\ML\skin-lesion-vit\data\raw`.
7. **Don't skip `pytest tests/`** when touching split / registry / engine code in v2. The tests are fast (synthetic + tiny resnet18 fixtures) and catch leakage regressions.
8. **The `1.0.0` file in v1** is leftover pip install output, not a version manifest. Don't reference it.
9. **`main.py` at the D:\ML root** is PyCharm boilerplate; not part of either project.
10. **`PHASES.md` in v2** is now populated. See [PHASES.md](../v2/PHASES.md) for current phase status (E1–E14, B1–B5, OA1–OA4).
11. **`configs/data/ham10000.yaml` is dead config** — it isn't loaded by any code. To change the data path, edit the `--data_dir` default in `scripts/train.py` / `scripts/verify_data.py` or pass `--data_dir`.
12. **`experiments/<model>/tta_results.txt` in v2** now contains both `[VAL]` and `[TEST]` sections — re-run with `--split both` for all 4 FT models + siglip2_probe in this overnight run. TTA always reads both from `tta_results.txt` now; fallback to `post_hoc/results.txt` for legacy test AUCs is no longer needed.
13. **`siglip2_probe` hurts the ensemble.** Adding it to the canonical 4-model SLSQP ensemble drops test AUC from 0.9731 → 0.9703 because its error pattern is too correlated with `convnextv2_large`. SLSQP gives it the floor weight (0.06). Side-by-side at `experiments/ensemble/results_5model.txt`. Stick with the 4-model ensemble.
14. **Binary mel specialist (`mel_binary_*`) underperforms the 7-way + F2-threshold approach** on lesion-level splits. v1's Phase 19 0.83 mel recall was on image-level (leaky) splits. v2 numbers in `experiments/mel_binary/mel_binary_*_v2/results.txt`. Use the post-hoc pipeline's mel threshold instead.
15. **TENT hurts `pandem_real_probe` on Fitz17k regardless of hyperparams.** lr ∈ {1e-3, 1e-4} × n_steps ∈ {1, 3} all drop dark-skin Mal AUC below baseline. Use the unadapted baseline for that combination, or switch to `efficientnetv2_m + TENT(n_steps=3)` which works (dark Mal AUC 0.6742). TENT lr=1e-3 works *on DDI* for all four backbones.

---

## When asked to extend / fix / measure

- **New backbone:** add a `ModelConfig` to `src/models/registry.py:REGISTRY`. Add a new `*Classifier` class only if the existing four `model_type` paths don't fit. Add it to the expected-keys set in `tests/test_splits.py:test_registry_keys_exist`.
- **New evaluation split (e.g. new external dataset):** add a loader in `src/data/external/<name>.py`, mappings in `taxonomy.py`, a dispatch case in `cross_dataset.py:load_external_dataset`. Mirror DDI / ISIC2019 conventions.
- **Reproducing a v1 phase:** look up the phase in the v1 README leaderboard; the script name is in [Run examples](#run-examples-1). Always run from inside `skin-lesion-vit/` so `DATA_DIR` resolves correctly.
- **Cross-dataset run:** make sure both `best_model.pth` and the corresponding external dataset folder exist before invoking `src.cross_dataset`. Pass `--label_mode universal` for DDI/Fitz17k, `--label_mode ham10k` for ISIC 2019.
- **Performance regression:** check the four post-hoc stages in `experiments/post_hoc/results.txt`. The pipeline metrics line tells you which stage hurts. Calibration that drops test AUC is a known failure mode on small rare-class val populations — re-verify n_per_class on val before blaming Platt.

---

## Authoritative sources, in priority order

When information conflicts, trust:

1. The Python source files under `src/` and `scripts/`.
2. The actual files in `experiments/` (especially `results.txt`, `tta_results.txt`, `report.txt`).
3. The two `README.md` files.
4. `03_COMPLETE_GUIDE.md` — narrative, may lag.
5. This file (`06_AI_AGENT_CONTEXT.md`) — written 2026-05-13; verify before recommending anything time-sensitive.

Numerical results in v1/v2 READMEs were transcribed from `results.txt` files; if you observe a mismatch, the `results.txt` wins and the README needs a follow-up update.

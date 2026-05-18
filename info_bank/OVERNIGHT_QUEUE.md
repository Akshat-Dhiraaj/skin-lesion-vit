# Overnight Task Queue — Skin Lesion v2

> **Read [AI_context.md](AI_context.md) FIRST** for project structure, conventions, and known pitfalls. This file lists work to execute. The context file explains *why*.

You are picking this up cold. Do not assume any prior conversation. Treat the code as absolute truth; verify before editing.

---

## Ground rules

1. **Working directory:** `D:\ML\skin-lesion-vit\v2` for everything except T-DOC-1. Use absolute paths in commands.
2. **GPU:** RTX 4060 Laptop, 8 GB. Single GPU. Mind the VRAM.
3. **OS:** Windows 11 + PowerShell. If a Python subprocess deadlocks on multi-worker DataLoaders, drop to `--num_workers 0`.
4. **Commit cadence:** one commit per completed task with a clear message. Push only if the user has set an `origin` remote (check `git remote -v` first; otherwise just commit locally).
5. **Atomic execution:** finish a task's verification step before moving to the next. If verification fails, STOP and write a note in `OVERNIGHT_RESULTS.md` (create at repo root). Don't cascade fixes overnight.
6. **No destructive ops:** never delete existing `best_model.pth`, `results.txt`, or `*.npy` artifacts. If retraining, write to a *new* registry key, not the existing one.
7. **Progress log:** append a line to `D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md` after every task (create on first write). Format: `[ISO timestamp] T-XXX-N: <status> — <one-line note>`. This is the only deliverable the user reads first thing in the morning.

---

## Pre-flight (run before starting the queue)

```powershell
cd D:\ML\skin-lesion-vit\v2
.\.venv\Scripts\activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
pytest -v -x   # all 23+ tests must pass before any task starts
git status     # working tree should be clean
```

If any of the above fails, stop and report. Do not proceed.

---

## Task queue

Tasks are ordered by dependency. Within a tier, run sequentially. Estimated total wall-clock: **5–7 hours**.

---

### T-FIX-1 — SigLIP normalization (CODE, ~10 min)

**Why:** [src/data/transforms.py](../v2/src/data/transforms.py) and [src/tta.py](../v2/src/tta.py) hardcode ImageNet stats. SigLIP 2 expects `mean=std=(0.5, 0.5, 0.5)` per the [official model card](https://huggingface.co/google/siglip2-so400m-patch14-384). Training `siglip2_probe` with ImageNet stats silently underperforms.

**Change:** make normalization stats a function of `model_type`.

In `src/data/transforms.py`:
- Keep `IMAGENET_MEAN`, `IMAGENET_STD`.
- Add `SIGLIP_MEAN = (0.5, 0.5, 0.5)` and `SIGLIP_STD = (0.5, 0.5, 0.5)`.
- Add a helper `def get_norm_stats(model_type: str) -> tuple[tuple, tuple]:` returning SIGLIP stats for `"siglip2"` and ImageNet stats otherwise.
- Change `get_train_transform`, `get_val_transform`, `get_tta_transforms` to take an optional `model_type: str = None` arg; default behavior unchanged (ImageNet), `model_type="siglip2"` switches to SIGLIP stats.

In `src/engine.py:run_phase`:
- Pass `cfg.model_type` through to `get_train_transform` and `get_val_transform`.

In `src/tta.py`:
- Replace the module-level `IMAGENET_MEAN`/`IMAGENET_STD` constants with calls to `get_norm_stats(cfg.model_type)` inside `get_tta_transforms(image_size, model_type)`.
- Update `main()` to pass `cfg.model_type` when building transforms.

**Verification:**
```powershell
pytest -v -x
python -c "from src.data.transforms import get_norm_stats; print(get_norm_stats('siglip2'), get_norm_stats('dinov2'))"
# expect: ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
```

**Commit:** `fix(transforms): switch to SigLIP normalization for siglip2 backbones`

---

### T-FIX-2 — PanDerm input resolution: add a 224 variant (CODE, ~5 min)

**Why:** [registry.py:113,125](../v2/src/models/registry.py) sets `input_size=256` for `pandem_*`. The [HF model card](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256) says the model is **224×224**; the `-256` in the name is the PubmedBert text context length. **Do not touch the existing 256 entries** — they have trained checkpoints and we want to compare.

**Change:** in `src/models/registry.py`, add two new registry keys, copying the existing pandem configs but with `input_size=224`:

```python
"pandem_probe_224": ModelConfig(
    name="pandem_probe_224",
    model_type="pandem",
    model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
    freeze_backbone=True,
    input_size=224,
    batch_size=64,
    lr_head=1e-3,
    lr_backbone=0.0,
    epochs=20,
    accum_steps=1,
),
"pandem_ft_224": ModelConfig(
    name="pandem_ft_224",
    model_type="pandem",
    model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
    freeze_backbone=False,
    input_size=224,
    batch_size=8,
    lr_head=1e-3,
    lr_backbone=5e-6,
    epochs=25,
    accum_steps=8,
),
```

**Also update** `tests/test_splits.py:test_registry_keys_exist` to include `pandem_probe_224` and `pandem_ft_224` in the `expected` set.

**Verification:**
```powershell
pytest -v -x tests/test_splits.py
python -c "from src.models.registry import REGISTRY; print(sorted(REGISTRY.keys()))"
```

**Commit:** `feat(registry): add pandem_*_224 variants to verify HF-spec input resolution`

---

### T-FIX-3 — Re-run TTA with --split both for the 4 fine-tune models (GPU, ~3 hr)

**Why:** Existing `tta_results.txt` files contain only `[VAL]` because a previous run used `--split val`. The `tta_test_probs.npy` arrays exist (so `src/post_hoc.py` already works), but the human-readable reports are misleading.

**Run** (sequential — only one GPU):

```powershell
python -m src.tta --model convnextv2_large    --data_dir D:\ML\skin-lesion-vit\data\raw --split both --num_workers 0
python -m src.tta --model efficientnetv2_m    --data_dir D:\ML\skin-lesion-vit\data\raw --split both --num_workers 0
python -m src.tta --model dinov2_registers_ft --data_dir D:\ML\skin-lesion-vit\data\raw --split both --num_workers 0
python -m src.tta --model pandem_ft           --data_dir D:\ML\skin-lesion-vit\data\raw --split both --num_workers 0
```

**Verification (for each model):**
```powershell
Select-String -Pattern "^\[VAL\]|^\[TEST\]|^TTA AUC" .\experiments\<model>\tta_results.txt
# Expect BOTH [VAL] and [TEST] blocks with TTA AUC numbers.
```

Expected TTA test AUCs (must match within ±0.0005, from `experiments/post_hoc/results.txt`):
- `convnextv2_large`: 0.9676
- `efficientnetv2_m`: 0.9553
- `dinov2_registers_ft`: 0.9401
- `pandem_ft`: 0.9364

If the test number deviates by more than 0.001, the underlying probs changed (something is wrong) — STOP and log.

**Commit:** `chore(tta): re-run with --split both so test report blocks are persisted`

---

### T-TRAIN-1 — Train siglip2_probe (GPU, ~1.5–2 hr)

**Depends on:** T-FIX-1 (normalization).

**Why:** Registry entry exists; no `experiments/siglip2_probe/` folder. This is the cleanest win — linear probe, only the head trains, ImageNet-21k–scale features.

**Run:**

```powershell
python scripts/train.py --model siglip2_probe --num_workers 0
```

Estimated VRAM: ~5–6 GB at batch 8, 384 px, frozen backbone. If OOM, drop `accum_steps` from 8 to 4 (i.e., effective batch 32 instead of 64).

**Verification:**
```powershell
Test-Path .\experiments\siglip2_probe\best_model.pth
Get-Content .\experiments\siglip2_probe\results.txt
```

Expected: Val AUC > 0.93, Test AUC > 0.90 (rough — SigLIP 2 is strong but the probe ceiling on HAM10000 is bounded by linear separability). If Val AUC < 0.85, the normalization change in T-FIX-1 didn't take effect — investigate.

After training:
```powershell
python -m src.tta --model siglip2_probe --data_dir D:\ML\skin-lesion-vit\data\raw --split both --num_workers 0
```

**Commit:** `feat(experiments): train siglip2_probe + TTA`

---

### T-TRAIN-2 — Train pandem_probe_224 (GPU, ~30 min)

**Depends on:** T-FIX-2 (registry entries).

**Why:** Validates whether the 256-vs-224 resolution choice mattered. Probe-only because (a) it's fast, (b) if the 224 version beats the 256 version, that justifies retraining the FT variant.

**Run:**

```powershell
python scripts/train.py --model pandem_probe_224 --num_workers 0
```

**Verification:**
```powershell
Get-Content .\experiments\pandem_probe_224\results.txt
Get-Content .\experiments\pandem_probe\results.txt
# Compare Test AUC line.
```

**Decision logic** (record in `OVERNIGHT_RESULTS.md`):
- If `pandem_probe_224` test AUC > `pandem_probe` test AUC by ≥ 0.005 → recommend retraining `pandem_ft_224` (do NOT auto-trigger; flag for the user).
- If within ±0.005 → the 256 setting is harmless; recommend keeping existing checkpoints and just updating the registry comment.
- If `pandem_probe_224` is worse by ≥ 0.005 → flag as surprising; do nothing.

Do not retrain `pandem_ft_224` overnight regardless — that's a 2–3 hour run and the user should sign off first.

**Commit:** `feat(experiments): train pandem_probe_224 for resolution A/B test`

---

### T-EVAL-1 — ISIC 2019 cross-dataset for siglip2_probe (GPU, ~30 min)

**Depends on:** T-TRAIN-1.

**Run:**

```powershell
python -m src.cross_dataset --model siglip2_probe --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0
```

**Verification:**
```powershell
Get-Content .\experiments\cross_dataset\isic2019\siglip2_probe\report.txt
```

Append to `OVERNIGHT_RESULTS.md` the Macro OvR AUC, Malignant AUC, and Mel AUC for siglip2_probe on ISIC 2019, alongside the existing four models for comparison.

**Commit:** `feat(eval): siglip2_probe ISIC 2019 cross-dataset report`

---

### T-EVAL-2 — ISIC 2019 cross-dataset for pandem_probe_224 (GPU, ~20 min)

**Depends on:** T-TRAIN-2.

```powershell
python -m src.cross_dataset --model pandem_probe_224 --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0
```

Same verification + logging pattern as T-EVAL-1.

**Commit:** `feat(eval): pandem_probe_224 ISIC 2019 cross-dataset report`

---

### T-DATA-1 — Check for DDI and Fitzpatrick17k data (CHECK ONLY, ~1 min)

**Why:** Both datasets require human-gated downloads (Stanford AIMI registration for DDI; manual download script for Fitz17k). Don't attempt to download — just check whether the user pre-populated them.

```powershell
$ddiPath  = "D:\ML\skin-lesion-vit\v2\data\external\DDI\ddi_metadata.csv"
$fitzPath = "D:\ML\skin-lesion-vit\v2\data\external\Fitzpatrick17k\fitzpatrick17k.csv"

if (Test-Path $ddiPath) {
    "DDI present — running cross-dataset eval (T-EVAL-3)" | Tee-Object -Append D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md
} else {
    "DDI not present — skipping T-EVAL-3. To enable, place DDI under data/external/DDI/" | Tee-Object -Append D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md
}

if (Test-Path $fitzPath) {
    "Fitz17k present — running cross-dataset eval (T-EVAL-4)" | Tee-Object -Append D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md
} else {
    "Fitz17k not present — skipping T-EVAL-4. To enable, place Fitzpatrick17k under data/external/Fitzpatrick17k/" | Tee-Object -Append D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md
}
```

**No commit** — this is just a check that gates the next two tasks.

---

### T-EVAL-3 — DDI cross-dataset (GPU, ~10 min, CONDITIONAL on T-DATA-1)

Skip if `data/external/DDI/ddi_metadata.csv` doesn't exist.

For each of `convnextv2_large`, `efficientnetv2_m`, `dinov2_registers_ft`, `pandem_ft` (plus `siglip2_probe` if T-TRAIN-1 completed):

```powershell
python -m src.cross_dataset --model <model_key> --dataset ddi --data_root data/external --label_mode universal --num_workers 0
```

**Verification:**
```powershell
Get-Content .\experiments\cross_dataset\ddi\<model_key>\report.txt
# Must include: Macro OvR AUC, Malignant AUC, AND Fitzpatrick-stratified light/medium/dark malignant AUC.
```

Log light-to-dark AUC gap per model to `OVERNIGHT_RESULTS.md`. **This is the headline fairness deliverable.**

**Commit:** `feat(eval): DDI cross-dataset reports for all trained models`

---

### T-EVAL-4 — Fitzpatrick17k cross-dataset (GPU, ~30 min, CONDITIONAL on T-DATA-1)

Skip if `data/external/Fitzpatrick17k/fitzpatrick17k.csv` doesn't exist.

Same pattern as T-EVAL-3, swap `--dataset fitz17k`:

```powershell
python -m src.cross_dataset --model <model_key> --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0
```

**Commit:** `feat(eval): Fitzpatrick17k cross-dataset reports for all trained models`

---

### T-DOC-1 — Populate `PHASES.md` (DOC, ~15 min)

**File:** `D:\ML\skin-lesion-vit\v2\PHASES.md` (currently 0 bytes).

**Content:** transcribe the "Phase roadmap" from the **v2 README's** intent, marking each phase with a status badge based on `experiments/` contents. Do NOT invent phases the README doesn't already imply.

Structure:

```markdown
# Phases — v2

Status legend:  ✅ done · 🟡 partial · 🔴 not started

## Direction E — HAM10000 benchmark (lesion-level)

- ✅ E1 · Patient-level splits + CI tests (`src/data/splits.py`, `tests/test_splits.py`)
- ✅ E2 · HAM10000 loader (`src/data/ham10000.py`)
- ✅ E3 · Augmentation + TTA pipelines (`src/data/transforms.py`, `src/tta.py`)
- ✅ E4 · Model registry (`src/models/registry.py`) — 7 entries (+ 2 if T-FIX-2 ran)
- ✅ E5 · Training engine (`src/engine.py`)
- ✅ E6 · DINOv2+Registers probe — test AUC 0.9232
- ✅ E7 · DINOv2+Registers full FT — test AUC 0.9358 (TTA 0.9636/val, 0.9401/test)
- ✅ E8 · PanDerm probe — test AUC 0.9305
- ✅ E9 · PanDerm full FT — test AUC 0.9330 (TTA 0.9603/val, 0.9364/test)
- ✅ E10 · ConvNeXt-V2 Large @ 384 — test AUC 0.9660 (TTA 0.9816/val, 0.9676/test)
- ✅ E10b · EfficientNetV2-M — test AUC 0.9474 (TTA 0.9685/val, 0.9553/test)
- <STATUS-FROM-T-TRAIN-1> E11 · SigLIP 2 probe
- 🔴 E12 · Sample-efficiency curves (10/25/50/100% labels)
- ✅ E13 · 4-model SLSQP ensemble — test AUC 0.9717
- ✅ E14 · 3-stage post-hoc pipeline (per-class SLSQP + Platt + F2) — test AUC 0.9717, mel recall 0.566

## Direction B — Cross-dataset / fairness

- ✅ B1 · ISIC 2019 inference for 4 FT models
- <STATUS-FROM-T-DATA-1> B2 · DDI inference (Fitzpatrick-stratified)
- <STATUS-FROM-T-DATA-1> B3 · Fitzpatrick17k inference (Fitzpatrick-stratified)
- 🔴 B4 · Cross-dataset SLSQP ensemble (compare weightings across DDI / Fitz17k / ISIC)
- 🔴 B5 · Domain generalization (GroupDRO / DANN / TENT) — gated on B2–B3

## Open / aspirational

- 🔴 OA1 · Binary mel specialist v2 (v1 Phase 19 ported with lesion-level splits)
- 🔴 OA2 · Multimodal v2 (v1 Phase 12 ported — age/sex/localization features)
- 🔴 OA3 · GradCAM / attention visualization for clinical interpretability
- 🔴 OA4 · Final write-up / report
```

Replace `<STATUS-FROM-T-TRAIN-1>` and `<STATUS-FROM-T-DATA-1>` with ✅ or 🔴 based on whether those tasks completed in this run.

**Verification:** open the file, eyeball for ASCII-art correctness; commit only if the table renders cleanly in `Get-Content`.

**Commit:** `docs(phases): populate PHASES.md with current status`

---

### T-DOC-2 — Update the v2 README with new results (DOC, ~10 min)

After all training/eval tasks complete, update [../v2/README.md](../v2/README.md):

1. **Single-model table** (line ~17) — add a row for `siglip2_probe` if T-TRAIN-1 succeeded. Copy Val AUC / Test AUC / Test Acc from its `results.txt`.
2. **TTA table** (line ~32) — add a row for `siglip2_probe` if its TTA report exists.
3. **Cross-dataset ISIC 2019 table** (line ~73) — add a row for `siglip2_probe` and (if T-FIX-2 → T-TRAIN-2 → T-EVAL-2 chain succeeded) `pandem_probe_224`.
4. **Caveats section** (line ~264) — flip the `siglip2_probe is not yet trained` bullet to past tense, citing the new numbers. If T-EVAL-3/4 ran, flip the DDI/Fitz17k bullet similarly.
5. **Registry table** (line ~92) — add `pandem_probe_224` and `pandem_ft_224` rows if T-FIX-2 added them. Mark `pandem_ft_224` as "registered, untrained" in the Notes column.

**Commit:** `docs(readme): reflect overnight runs (siglip2_probe, pandem_*_224, new cross-dataset)`

---

### T-DOC-3 — Update `AI_context.md` (DOC, ~5 min)

In `D:\ML\skin-lesion-vit\info_bank\AI_context.md`:

1. **v2 trained models list** (around line 142) — add new entries that completed in this run.
2. **Cross-dataset state** (around line 165) — update DDI / Fitz17k status if T-EVAL-3/4 ran.
3. **Pitfall #2** (line ~245) — flip the "Probable bug in PanDerm input resolution" note: report the A/B result from T-TRAIN-2's comparison. If pandem_224 beat pandem_256 → upgrade from "Probable bug" to "Confirmed bug; pandem_probe_224 wins by X". If no significant difference → "256 is harmless; bug downgraded to cosmetic".
4. **Normalization caveat** (in v2 data pipeline section, line ~175) — flip "siglip2_probe is not yet trained" → cite the trained result.

**Commit:** `docs(context): update AI_context.md with overnight outcomes`

---

## Final verification

After all tasks complete (or are explicitly skipped):

```powershell
cd D:\ML\skin-lesion-vit\v2
pytest -v -x                     # everything still green
git log --oneline -15            # ~10 commits expected
Get-Content D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md   # human-readable summary for the morning
```

Append a final summary block to `OVERNIGHT_RESULTS.md`:

```
=== OVERNIGHT SUMMARY ===
Started:  <ISO timestamp>
Finished: <ISO timestamp>
Tasks completed: X / 12
Tasks skipped:   Y (reason)
Tasks failed:    Z (see notes above)

Headline new numbers:
  siglip2_probe        : Val AUC <…>  Test AUC <…>  (HAM10000)
  siglip2_probe ISIC   : Macro OvR <…>  Malignant <…>
  pandem_probe_224     : Test AUC <…>  (vs pandem_probe 0.9305)
  DDI light-dark gap   : <…>  (worst model: <…>)
  Fitz17k light-dark gap: <…>

Notes for the user:
  - <decisions deferred to user, e.g. "should we retrain pandem_ft_224?">
  - <any unexpected behavior worth flagging>
```

---

## What is explicitly NOT in this queue

These items from the project roadmap are deliberately deferred — too large for one overnight run, or require user sign-off first.

- **Retraining `pandem_ft_224`** — 2–3 hour run, only worth doing if T-TRAIN-2 shows pandem_224 is meaningfully better. Flag as a recommendation, don't execute.
- **Binary mel specialist v2** (OA1) — substantial new code (custom 1:1 sampler, binary focal loss head, BinaryFocalLoss class). Worth a dedicated session, not overnight.
- **Multimodal v2** (OA2) — same as above; new dataset class with metadata fusion head.
- **Sample-efficiency curves** (E12) — 4× the training time (10/25/50/100%). Plan separately.
- **Cross-dataset SLSQP ensemble** (B4) — only meaningful after DDI/Fitz17k data exists; queue for a follow-up.
- **Domain generalization** (B5) — research-grade, not overnight scope.

---

## If the world ends

If anything destroys the working tree (force-push, accidental `rm`, etc.):

```powershell
cd D:\ML\skin-lesion-vit\v2
git reflog                       # find pre-disaster HEAD
git reset --hard <commit>        # only if you're certain
```

But **prefer NOT to run destructive recovery overnight**. If you encounter a state you don't understand, stop, write a note in `OVERNIGHT_RESULTS.md`, and leave the rest of the queue undone. The user will sort it out in the morning.

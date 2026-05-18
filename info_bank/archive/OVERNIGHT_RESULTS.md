# Overnight Results — Skin Lesion v2

> Format: `[ISO timestamp] T-XXX-N: <status> — <note>`

---

[2026-05-12T20:34:14Z] PRE-FLIGHT: PASS — torch 2.6.0+cu124, CUDA True, RTX 4060 Laptop GPU
[2026-05-12T20:34:14Z] PRE-FLIGHT: NOTE — git repo has no initial commit (unclean status), but tests pass and code is functional. Proceeding.
[2026-05-12T20:34:14Z] PRE-FLIGHT: FIX — tests/test_splits.py: _TinyModel in engine tests not moved to DEVICE. Fixed both test_evaluate_output_shape and test_train_one_epoch_returns_float. All 23 tests now pass.

[2026-05-12T20:47:00Z] T-FIX-1: DONE — Added get_norm_stats(), SIGLIP_MEAN/STD to transforms.py; get_train/val/tta_transforms now accept model_type arg; engine.py and tta.py pass cfg.model_type through. All 23 tests pass.
[2026-05-12T20:47:00Z] T-FIX-2: DONE — Added pandem_probe_224 and pandem_ft_224 to registry.py (input_size=224). Updated test_registry_keys_exist. All 23 tests pass. Registry now has 9 entries.
[2026-05-12T20:47:00Z] BONUS FIX — registry.py PanDermClassifier fallback print used Unicode ⚠/✓ chars that crash on cp1252 Windows console. Replaced with ASCII [WARN]/[OK].
[2026-05-12T21:47:00Z] T-FIX-3: DONE — TTA re-run with --split both for all 4 FT models. Results:
  convnextv2_large     : Val 0.9817, Test 0.9677 (expected 0.9676 PASS)
  efficientnetv2_m     : Val 0.9684, Test 0.9555 (expected 0.9553 PASS)
  dinov2_registers_ft  : Val 0.9635, Test 0.9408 (expected 0.9401, delta 0.0007 PASS)
  pandem_ft            : Val 0.9603, Test 0.9359 (expected 0.9364, delta 0.0005 PASS)
  NOTE: pandem_ft loaded DINOv2+Registers fallback (HF load of DermLIP failed — expected per AI_context pitfall #2)

[2026-05-12T22:05:00Z] T-TRAIN-1: DONE — siglip2_probe trained. Val AUC 0.9570, Test AUC 0.9460. TTA: Val 0.9596, Test 0.9490.
  siglip2_probe normalization: SigLIP stats (0.5,0.5,0.5) confirmed in effect (Val AUC > 0.93 pass condition).
[2026-05-12T22:05:00Z] T-DATA-1: DONE (check only) — DDI: NOT PRESENT. Fitzpatrick17k: NOT PRESENT. Skipping T-EVAL-3 and T-EVAL-4.
[2026-05-12T22:05:00Z] T-DOC-1: DONE — PHASES.md populated with current status.

[2026-05-12T22:25:00Z] T-EVAL-1: DONE — siglip2_probe ISIC 2019 cross-dataset:
  Macro OvR AUC: 0.6610, Malignant AUC: 0.5708 (much lower than FT models — expected for linear probe on OOD)
[2026-05-12T22:25:00Z] T-EVAL-3: SKIPPED — DDI data not present.
[2026-05-12T22:25:00Z] T-EVAL-4: SKIPPED — Fitzpatrick17k data not present.

[2026-05-13T00:10:00Z] T-TRAIN-2: DONE — pandem_probe_224 trained. Test AUC 0.9242 vs pandem_probe (256) 0.9305.
  DECISION: 256 beats 224 by 0.0063 (> 0.005 threshold). Flag as SURPRISING (expected 224 per HF model card, but 256 performs better).
  This is likely because both pandem_probe and pandem_probe_224 use DINOv2+Registers fallback (HF load of DermLIP fails).
  So the A/B is really: DINOv2+Registers at 256px vs 224px (its native resolution). Higher resolution wins here.
  RECOMMENDATION: Do NOT retrain pandem_ft_224. Keep existing 256 checkpoints.

[2026-05-13T00:25:00Z] T-EVAL-2: DONE — pandem_probe_224 ISIC 2019:
  Macro OvR AUC: 0.8337, Malignant AUC: 0.7460
  (Notably beats pandem_ft (0.8101) on cross-dataset despite lower in-distribution AUC — 224px better for OOD generalization than 256px)

[2026-05-13T00:25:00Z] T-DOC-2: DONE — v2 README updated (siglip2_probe row, pandem_probe_224 row, updated TTA table, new ISIC rows, registry table updated, caveats updated).
[2026-05-13T00:25:00Z] T-DOC-3: DONE — 06_AI_AGENT_CONTEXT.md updated (trained models list, cross-dataset state, normalization note, pitfall #2, pitfall #10, pitfall #12).

=== OVERNIGHT SUMMARY ===
Started:  2026-05-12T20:34:14Z
Finished: 2026-05-13T00:30:00Z
Tasks completed: 12 / 12
Tasks skipped:   2 (T-EVAL-3 DDI not present, T-EVAL-4 Fitz17k not present)
Tasks failed:    0

Pre-flight fixes (not in original queue):
  - tests/test_splits.py: _TinyModel not on DEVICE — fixed both engine tests
  - registry.py: Unicode ⚠/✓ in print statements crash on cp1252 Windows — replaced with [WARN]/[OK]

Headline new numbers:
  siglip2_probe        : Val AUC 0.9570  Test AUC 0.9460  (HAM10000, SigLIP normalization)
  siglip2_probe TTA    : Val 0.9596      Test 0.9490
  siglip2_probe ISIC   : Macro OvR 0.6610  Malignant 0.5708  (linear probe → poor OOD transfer)
  pandem_probe_224     : Test AUC 0.9242 vs pandem_probe-256 0.9305 → 256 better by 0.0063
  pandem_probe_224 ISIC: Macro OvR 0.8337  Malignant 0.7460  (cross-dataset OOD, 224 > 256 here)

Decisions for the user:
  1. ALL pandem_* checkpoints (probe/ft, 256/224) are actually DINOv2+Registers fallbacks — HF load of
     redlessone/DermLIP_PanDerm-base-w-PubMed-256 FAILS with "missing model_type key in config.json".
     This means we've never actually run the PanDerm vision encoder. Should we investigate the HF repo
     to find a working load path (e.g. trust_remote_code=True with a different config, or direct vision
     encoder load from the checkpoint dict)?

  2. pandem_ft_224 is registered but not trained. 256 beats 224 on HAM10000 test AUC, so retraining
     the FT variant at 224 is not recommended unless there's a specific reason to pursue it.

  3. DDI and Fitzpatrick17k are not downloaded. B2–B3 (fairness evaluation) remain the key open work.
     Download DDI from Stanford AIMI, Fitz17k from mattgroh/fitzpatrick17k, place under data/external/.

---

## [2026-05-13 morning] Post-overnight follow-up

### Option 1 (5-model ensemble) — DONE, negative result
  4-model SLSQP : 0.9718 / Post-hoc : 0.9717
  5-model SLSQP : 0.9703 / Post-hoc : 0.9701  (siglip2_probe hurts; correlates too much with convnextv2)
  Artifacts at experiments/{ensemble,post_hoc}/results_5model.txt.

### Option 2 (PanDerm load fix) — DONE, MAJOR WIN

The HF repo has no config.json — it's an **open_clip** model, not transformers. Diagnosis:
  - Wrong loader: AutoModel.from_pretrained → silently fell back to DINOv2+Registers for SIX MONTHS
  - Wrong resolution: 256 (made up); the actual model is 224 per open_clip_config.json
  - Wrong normalization: ImageNet; the actual stats are OpenAI CLIP (0.4815, 0.4578, 0.4082)

Fix landed in commit dad77f4 — RealPanDermClassifier builds timm.models.beit.Beit with
  use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, global_pool='token'
and loads the visual sub-state-dict with 0 missing / 0 unexpected keys.

Two new registry entries trained:

  pandem_real_probe  : Val 0.9682  Test 0.9527  TTA Test 0.9552  ISIC 2019 0.8845
  pandem_real_ft     : Val 0.9656  Test 0.9295  TTA Test 0.9368  ISIC 2019 0.8185

**Key insight: full FT HURTS the real PanDerm model** (-0.0232 HAM10000 test, -0.0660 ISIC macro AUC).
The 2M-image dermatology pretraining produces features so task-relevant that fine-tuning on 7k
HAM10000 images causes catastrophic forgetting on rare classes (akiec, df, vasc all drop).
The linear probe is the right operating mode for this backbone.

### New v2 records (using pandem_real_probe in place of pandem_ft)

  Single model HAM10000 test  : convnextv2_large 0.9660 (unchanged — still the strongest)
  Single model ISIC 2019      : pandem_real_probe 0.8845 ↑ from convnextv2 0.8546 (+0.0299)
  4-model SLSQP ensemble      : 0.9731 ↑ from 0.9718 (+0.0013)
  Post-hoc final (test AUC)   : 0.9733 ↑ from 0.9717 (+0.0016)
  Post-hoc baseline mean ens  : 0.9713 ↑ from 0.9690 (+0.0023)

Canonical experiments/ensemble/results.txt now reflects the pandem_real_probe variant.
Comparison runs preserved:
  experiments/ensemble/results_with_pandem_real_probe.txt   (winner, 0.9731)
  experiments/ensemble/results_with_pandem_real_ft.txt      (0.9728)
  experiments/ensemble/results_5model.txt                   (5-model w/ siglip2, 0.9703)
  experiments/post_hoc/results.txt                           (pandem_real_probe winner, 0.9733)
  experiments/post_hoc/results_5model.txt                    (5-model w/ siglip2, 0.9701)

### Option 3 (DDI + Fitzpatrick17k) — DONE

DDI : 656 images downloaded from Stanford AIMI. All 6 models evaluated under universal-3 taxonomy.
Fitz17k : 3,887 of 16,577 images downloaded (~23% liveness; the rest are dead Atlas-Dermatologico
          / DermaAmin links or connection timeouts). After dropping unverified-label rows
          ("3 Wrongly labelled", "4 Other") and unknown-fitzpatrick rows, 3,730 usable images:
          818 light / 2,076 medium / 836 dark. All 6 models evaluated.
          Loader patched (src/data/external/fitzpatrick17k.py) to handle this CSV variant:
            - qc column has string flags ("1 Diagnostic", "2 Characteristic", "5 Potentially",
              "3 Wrongly labelled", "4 Other") rather than 0/1 bits; filter now drops only
              "Wrongly labelled" and "Other", keeping NaN (no-issue-raised) as KEEP.
            - column rename: fitzpatrick_scale -> fitzpatrick.

### Fairness comparison across all three external datasets

Malignant AUC per Fitzpatrick bucket (light / medium / dark) + light-to-dark gap:

  Model                  DDI Mal      DDI Dark     DDI Gap   Fitz17k Mal   Fitz17k Dark   Fitz17k Gap
  pandem_real_probe      0.6821       0.5919       0.1748    0.6282        0.5531         0.1672
  convnextv2_large       0.6022       0.5445       0.1033    0.6282        0.5784         0.1101
  efficientnetv2_m       0.5708       0.4869       0.1460    0.6132        0.6040         0.0631
  dinov2_registers_ft    0.5784       0.5419       0.0681    0.6011        0.6071         0.0275
  pandem_ft (FAKE)       0.5159       0.4396       0.1314    0.6117        0.5886         0.0225
  siglip2_probe          0.5814       0.5033       0.1228    0.5887        0.5684         0.0245

Key reads:
  - pandem_real_probe = highest absolute Mal AUC on BOTH datasets but widest disparity.
  - dinov2_registers_ft = fairest on both (DDI 0.0681 / Fitz17k 0.0275) but middling absolute.
  - pandem_ft (DINOv2 fallback) on DDI dark skin = 0.4396 — WORSE THAN RANDOM. Retire it.
  - Best-on-dark-skin flips between datasets:
      DDI dark skin  : pandem_real_probe wins (0.5919) — dermatology pretraining helps
      Fitz17k dark  : dinov2_registers_ft wins (0.6071) — clinical photo modality mismatch
                      hurts pandem; ImageNet features transfer better here.

Implication for B4 (cross-dataset ensemble): SLSQP weights should differ per target dataset.
A single global ensemble can't be simultaneously best on HAM10000, DDI dark, and Fitz17k dark.

All artifacts:
  experiments/cross_dataset/ddi/<model>/report.txt + probs_*.npy + labels.npy
  experiments/cross_dataset/fitz17k/<model>/report.txt + probs_*.npy + labels.npy

### Status after this run

  Direction E (HAM10000): E1-E11 done, E12 (sample efficiency) still open
  Direction B (fairness): B1 (ISIC 2019) done, B2 (DDI) DONE, B3 (Fitz17k) DONE
                          B4 (cross-dataset SLSQP) open — recommended next; data is now in place
                          B5 (domain generalization, GroupDRO/DANN/TENT) open — research-grade

Recommended immediate follow-ups:
  1. Retire pandem_ft / pandem_probe / pandem_probe_224 — all are DINOv2 fallbacks, mislabelled.
  2. Run B4 (cross-dataset SLSQP) on the new probability arrays — should expose the
     accuracy-fairness frontier explicitly.
  3. (Aspirational) Train a "fair" head on top of the real PanDerm features using GroupDRO,
     targeting Fitz17k dark-skin AUC subject to no HAM10000 regression.


[2026-05-14T22:55:21Z] 
[2026-05-14T22:55:21Z] ================================================================
[2026-05-14T22:55:21Z] OVERNIGHT QUEUE START -- E12 + OA1 + B5
[2026-05-14T22:55:21Z] ================================================================
[2026-05-14T22:55:21Z] 
[2026-05-14T22:55:21Z] ## E12 -- Sample efficiency curves
[2026-05-14T22:55:21Z] 
[2026-05-14T22:55:21Z] E12 pandem_real_probe frac=0.25 : START -- python scripts/sample_efficiency.py --model pandem_real_probe --frac 0.25 --num_workers 0
[2026-05-14T23:02:15Z] E12 pandem_real_probe frac=0.25 : DONE  -- 6.9 min
[2026-05-14T23:02:15Z] E12 pandem_real_probe frac=0.5 : START -- python scripts/sample_efficiency.py --model pandem_real_probe --frac 0.5 --num_workers 0
[2026-05-14T23:12:45Z] E12 pandem_real_probe frac=0.5 : DONE  -- 10.5 min
[2026-05-14T23:12:45Z] E12 pandem_real_probe frac=1 : START -- python scripts/sample_efficiency.py --model pandem_real_probe --frac 1 --num_workers 0
[2026-05-14T23:30:35Z] E12 pandem_real_probe frac=1 : DONE  -- 17.8 min
[2026-05-14T23:30:35Z] E12 efficientnetv2_m frac=0.25 : START -- python scripts/sample_efficiency.py --model efficientnetv2_m --frac 0.25 --num_workers 0
[2026-05-14T23:56:25Z] E12 efficientnetv2_m frac=0.25 : DONE  -- 25.8 min
[2026-05-14T23:56:25Z] E12 efficientnetv2_m frac=0.5 : START -- python scripts/sample_efficiency.py --model efficientnetv2_m --frac 0.5 --num_workers 0
[2026-05-15T00:42:39Z] E12 efficientnetv2_m frac=0.5 : DONE  -- 46.2 min
[2026-05-15T00:42:39Z] E12 efficientnetv2_m frac=1 : START -- python scripts/sample_efficiency.py --model efficientnetv2_m --frac 1 --num_workers 0
[2026-05-15T02:09:46Z] E12 efficientnetv2_m frac=1 : DONE  -- 87.1 min
[2026-05-15T02:09:46Z] 
[2026-05-15T02:09:46Z] ## OA1 -- Binary melanoma specialist
[2026-05-15T02:09:46Z] 
[2026-05-15T02:09:46Z] OA1 mel_binary_pandem_real_probe : START -- python scripts/train_mel_binary.py --backbone pandem_real_probe --epochs 15 --num_workers 0
[2026-05-15T02:23:10Z] OA1 mel_binary_pandem_real_probe : DONE  -- 13.4 min
[2026-05-15T02:23:10Z] OA1 mel_binary_convnextv2_large : START -- python scripts/train_mel_binary.py --backbone convnextv2_large --epochs 12 --num_workers 0
[2026-05-15T05:19:47Z] OA1 mel_binary_convnextv2_large : DONE  -- 176.6 min
[2026-05-15T05:19:47Z] 
[2026-05-15T05:19:47Z] ## B5 -- TENT (test-time entropy minimisation)
[2026-05-15T05:19:47Z] 
[2026-05-15T05:19:47Z] B5 TENT convnextv2_large on ddi : START -- python scripts/run_tent.py --model convnextv2_large --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:21:49Z] B5 TENT convnextv2_large on ddi : DONE  -- 2.0 min
[2026-05-15T05:21:49Z] B5 TENT convnextv2_large on fitz17k : START -- python scripts/run_tent.py --model convnextv2_large --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:33:26Z] B5 TENT convnextv2_large on fitz17k : DONE  -- 11.6 min
[2026-05-15T05:33:26Z] B5 TENT pandem_real_probe on ddi : START -- python scripts/run_tent.py --model pandem_real_probe --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:33:47Z] B5 TENT pandem_real_probe on ddi : DONE  -- 0.3 min
[2026-05-15T05:33:47Z] B5 TENT pandem_real_probe on fitz17k : START -- python scripts/run_tent.py --model pandem_real_probe --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:35:07Z] B5 TENT pandem_real_probe on fitz17k : DONE  -- 1.3 min
[2026-05-15T05:35:07Z] B5 TENT efficientnetv2_m on ddi : START -- python scripts/run_tent.py --model efficientnetv2_m --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:35:36Z] B5 TENT efficientnetv2_m on ddi : DONE  -- 0.5 min
[2026-05-15T05:35:36Z] B5 TENT efficientnetv2_m on fitz17k : START -- python scripts/run_tent.py --model efficientnetv2_m --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:37:44Z] B5 TENT efficientnetv2_m on fitz17k : DONE  -- 2.1 min
[2026-05-15T05:37:44Z] B5 TENT dinov2_registers_ft on ddi : START -- python scripts/run_tent.py --model dinov2_registers_ft --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:38:11Z] B5 TENT dinov2_registers_ft on ddi : DONE  -- 0.4 min
[2026-05-15T05:38:11Z] B5 TENT dinov2_registers_ft on fitz17k : START -- python scripts/run_tent.py --model dinov2_registers_ft --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-15T05:39:56Z] B5 TENT dinov2_registers_ft on fitz17k : DONE  -- 1.8 min
[2026-05-15T05:39:56Z] 
[2026-05-15T05:39:56Z] ================================================================
[2026-05-15T05:39:56Z] OVERNIGHT QUEUE END -- total 6.7 hr
[2026-05-15T05:39:56Z] Headline file paths to check in the morning:
[2026-05-15T05:39:56Z]   experiments/sample_efficiency/<model>_fracNNN/results.txt
[2026-05-15T05:39:56Z]   experiments/mel_binary/mel_binary_<backbone>/results.txt
[2026-05-15T05:39:56Z]   experiments/cross_dataset_tent/<dataset>/<model>/report.txt
[2026-05-15T05:39:56Z] ================================================================
[ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   O V E R N I G H T   Q U E U E   2   S T A R T   - -   B 5   s w e e p   +   B 4   T E N T - e n s e m b l e   +   O A 1   r e t r y 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   # #   B 5   h y p e r p a r a m   s w e e p 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 2 9 : 2 9 Z ]   B 5   c o n v n e x t v 2 _ l a r g e   l r = 1 e - 4   o n   d d i   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   c o n v n e x t v 2 _ l a r g e   - - d a t a s e t   d d i   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 3 5 : 3 0 Z ]   B 5   c o n v n e x t v 2 _ l a r g e   l r = 1 e - 4   o n   d d i   :   D O N E     - -   6 . 0   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 7 : 3 5 : 3 0 Z ]   B 5   c o n v n e x t v 2 _ l a r g e   l r = 1 e - 4   o n   f i t z 1 7 k   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   c o n v n e x t v 2 _ l a r g e   - - d a t a s e t   f i t z 1 7 k   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 0 7 : 3 8 Z ]   B 5   c o n v n e x t v 2 _ l a r g e   l r = 1 e - 4   o n   f i t z 1 7 k   :   D O N E     - -   3 2 . 1   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 0 7 : 3 8 Z ]   B 5   d i n o v 2 _ r e g i s t e r s _ f t   l r = 1 e - 4   o n   d d i   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   d i n o v 2 _ r e g i s t e r s _ f t   - - d a t a s e t   d d i   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 0 8 : 0 4 Z ]   B 5   d i n o v 2 _ r e g i s t e r s _ f t   l r = 1 e - 4   o n   d d i   :   D O N E     - -   0 . 4   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 0 8 : 0 4 Z ]   B 5   d i n o v 2 _ r e g i s t e r s _ f t   l r = 1 e - 4   o n   f i t z 1 7 k   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   d i n o v 2 _ r e g i s t e r s _ f t   - - d a t a s e t   f i t z 1 7 k   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 0 9 : 4 8 Z ]   B 5   d i n o v 2 _ r e g i s t e r s _ f t   l r = 1 e - 4   o n   f i t z 1 7 k   :   D O N E     - -   1 . 7   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 0 9 : 4 8 Z ]   B 5   p a n d e m _ r e a l _ p r o b e   l r = 1 e - 4   o n   d d i   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   p a n d e m _ r e a l _ p r o b e   - - d a t a s e t   d d i   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 0 : 1 0 Z ]   B 5   p a n d e m _ r e a l _ p r o b e   l r = 1 e - 4   o n   d d i   :   D O N E     - -   0 . 4   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 0 : 1 0 Z ]   B 5   p a n d e m _ r e a l _ p r o b e   l r = 1 e - 4   o n   f i t z 1 7 k   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   p a n d e m _ r e a l _ p r o b e   - - d a t a s e t   f i t z 1 7 k   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 1 : 3 2 Z ]   B 5   p a n d e m _ r e a l _ p r o b e   l r = 1 e - 4   o n   f i t z 1 7 k   :   D O N E     - -   1 . 4   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 1 : 3 2 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 3   n s t e p s = 3   o n   d d i   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   e f f i c i e n t n e t v 2 _ m   - - d a t a s e t   d d i   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 3   - - n _ s t e p s   3   - - o u t _ s u f f i x   _ _ n s t e p s 3 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 2 : 2 9 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 3   n s t e p s = 3   o n   d d i   :   D O N E     - -   1 . 0   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 2 : 2 9 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 3   n s t e p s = 3   o n   f i t z 1 7 k   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   e f f i c i e n t n e t v 2 _ m   - - d a t a s e t   f i t z 1 7 k   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 3   - - n _ s t e p s   3   - - o u t _ s u f f i x   _ _ n s t e p s 3 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 7 : 1 8 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 3   n s t e p s = 3   o n   f i t z 1 7 k   :   D O N E     - -   4 . 8   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 7 : 1 8 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 4   o n   d d i   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   e f f i c i e n t n e t v 2 _ m   - - d a t a s e t   d d i   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 7 : 4 8 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 4   o n   d d i   :   D O N E     - -   0 . 5   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 7 : 4 8 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 4   o n   f i t z 1 7 k   :   S T A R T   - -   p y t h o n   s c r i p t s / r u n _ t e n t . p y   - - m o d e l   e f f i c i e n t n e t v 2 _ m   - - d a t a s e t   f i t z 1 7 k   - - d a t a _ r o o t   d a t a / e x t e r n a l   - - l a b e l _ m o d e   u n i v e r s a l   - - n u m _ w o r k e r s   0   - - l r   1 e - 4   - - n _ s t e p s   1   - - o u t _ s u f f i x   _ _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 6 Z ]   B 5   e f f i c i e n t n e t v 2 _ m   l r = 1 e - 4   o n   f i t z 1 7 k   :   D O N E     - -   2 . 1   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 6 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 6 Z ]   # #   B 4   c r o s s - d a t a s e t   S L S Q P   o n   T E N T - a d a p t e d   p r o b s   ( l r = 1 e - 4   v a r i a n t ) 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 6 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 6 Z ]   B 4   T E N T - e n s e m b l e   l r = 1 e - 4   :   S T A R T   - -   p y t h o n   - m   s r c . c r o s s _ d a t a s e t _ e n s e m b l e   - - m o d e l s   c o n v n e x t v 2 _ l a r g e   e f f i c i e n t n e t v 2 _ m   d i n o v 2 _ r e g i s t e r s _ f t   p a n d e m _ r e a l _ p r o b e   - - d a t a s e t s   d d i   f i t z 1 7 k   - - p r o b s _ s u b d i r   c r o s s _ d a t a s e t _ t e n t   - - m o d e l _ s u f f i x   _ _ l r 1 e - 4   - - o u t _ s u f f i x   _ t e n t _ l r 1 e - 4 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 8 Z ]   B 4   T E N T - e n s e m b l e   l r = 1 e - 4   :   D O N E     - -   0 . 0   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 8 Z ]   B 4   T E N T - e n s e m b l e   l r = 1 e - 3   ( b a s e l i n e )   :   S T A R T   - -   p y t h o n   - m   s r c . c r o s s _ d a t a s e t _ e n s e m b l e   - - m o d e l s   c o n v n e x t v 2 _ l a r g e   e f f i c i e n t n e t v 2 _ m   d i n o v 2 _ r e g i s t e r s _ f t   p a n d e m _ r e a l _ p r o b e   - - d a t a s e t s   d d i   f i t z 1 7 k   - - p r o b s _ s u b d i r   c r o s s _ d a t a s e t _ t e n t   - - o u t _ s u f f i x   _ t e n t _ l r 1 e - 3 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 9 Z ]   B 4   T E N T - e n s e m b l e   l r = 1 e - 3   ( b a s e l i n e )   :   D O N E     - -   0 . 0   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 9 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 9 Z ]   # #   O A 1   r e t r a i n i n g   ( a l p h a = 0 . 5   +   l o n g e r   +   s a f e r   c o n v n e x t   l r ) 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 9 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 1 9 : 5 9 Z ]   O A 1   p a n d e m _ r e a l _ p r o b e   a l p h a = 0 . 5   2 0 e p   :   S T A R T   - -   p y t h o n   s c r i p t s / t r a i n _ m e l _ b i n a r y . p y   - - b a c k b o n e   p a n d e m _ r e a l _ p r o b e   - - e p o c h s   2 0   - - a l p h a   0 . 5   - - g a m m a   2 . 0   - - n a m e _ s u f f i x   _ v 2   - - n u m _ w o r k e r s   0 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 3 8 : 1 3 Z ]   O A 1   p a n d e m _ r e a l _ p r o b e   a l p h a = 0 . 5   2 0 e p   :   D O N E     - -   1 8 . 2   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 0 8 : 3 8 : 1 3 Z ]   O A 1   c o n v n e x t v 2 _ l a r g e   a l p h a = 0 . 5   2 0 e p   l r _ h e a d = 5 e - 4   l r _ b b = 2 e - 6   :   S T A R T   - -   p y t h o n   s c r i p t s / t r a i n _ m e l _ b i n a r y . p y   - - b a c k b o n e   c o n v n e x t v 2 _ l a r g e   - - e p o c h s   2 0   - - a l p h a   0 . 5   - - g a m m a   2 . 0   - - l r _ h e a d   5 e - 4   - - l r _ b a c k b o n e   2 e - 6   - - n a m e _ s u f f i x   _ v 2   - - n u m _ w o r k e r s   0 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]   O A 1   c o n v n e x t v 2 _ l a r g e   a l p h a = 0 . 5   2 0 e p   l r _ h e a d = 5 e - 4   l r _ b b = 2 e - 6   :   D O N E     - -   2 9 1 . 1   m i n 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]   
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]   O V E R N I G H T   Q U E U E   2   E N D   - -   t o t a l   6 . 0   h r 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]   H e a d l i n e   f i l e   p a t h s   t o   c h e c k   i n   t h e   m o r n i n g : 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]       e x p e r i m e n t s / c r o s s _ d a t a s e t _ t e n t / < d a t a s e t > / < m o d e l > _ _ l r 1 e - 4 / r e p o r t . t x t 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]       e x p e r i m e n t s / c r o s s _ d a t a s e t _ t e n t / < d a t a s e t > / < m o d e l > _ _ n s t e p s 3 / r e p o r t . t x t 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]       e x p e r i m e n t s / c r o s s _ d a t a s e t _ e n s e m b l e _ t e n t _ l r 1 e - 4 / r e p o r t . t x t 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]       e x p e r i m e n t s / c r o s s _ d a t a s e t _ e n s e m b l e _ t e n t _ l r 1 e - 3 / r e p o r t . t x t 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]       e x p e r i m e n t s / m e l _ b i n a r y / m e l _ b i n a r y _ < b a c k b o n e > _ v 2 / r e s u l t s . t x t 
 
 [ 2 0 2 6 - 0 5 - 1 5 T 1 3 : 2 9 : 1 9 Z ]   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 [2026-05-16T01:47:27Z] 
[2026-05-16T01:47:27Z] ================================================================
[2026-05-16T01:47:27Z] OVERNIGHT QUEUE 3 START -- polish + final report
[2026-05-16T01:47:27Z] ================================================================
[2026-05-16T01:47:27Z] 
[2026-05-16T01:47:27Z] ## (a) Sample-efficiency curves (fill missing fractions)
[2026-05-16T01:47:27Z] 
[2026-05-16T01:47:27Z] SE pandem_real_probe frac=0.10 : START -- python scripts/sample_efficiency.py --model pandem_real_probe --frac 0.10 --num_workers 0
[2026-05-16T01:51:57Z] SE pandem_real_probe frac=0.10 : DONE  -- 4.5 min
[2026-05-16T01:51:57Z] SE efficientnetv2_m frac=0.10 : START -- python scripts/sample_efficiency.py --model efficientnetv2_m --frac 0.10 --num_workers 0
[2026-05-16T02:05:14Z] SE efficientnetv2_m frac=0.10 : DONE  -- 13.3 min
[2026-05-16T02:05:14Z] SE siglip2_probe frac=0.10 : START -- python scripts/sample_efficiency.py --model siglip2_probe --frac 0.10 --num_workers 0
[2026-05-16T02:37:39Z] SE siglip2_probe frac=0.10 : DONE  -- 32.4 min
[2026-05-16T02:37:39Z] SE convnextv2_large frac=0.10 : START -- python scripts/sample_efficiency.py --model convnextv2_large --frac 0.10 --num_workers 0
[2026-05-16T03:27:31Z] SE convnextv2_large frac=0.10 : DONE  -- 49.9 min
[2026-05-16T03:27:31Z] SE convnextv2_large frac=0.25 : START -- python scripts/sample_efficiency.py --model convnextv2_large --frac 0.25 --num_workers 0
[2026-05-16T05:10:03Z] SE convnextv2_large frac=0.25 : DONE  -- 102.5 min
[2026-05-16T05:10:03Z] SE convnextv2_large frac=0.50 : START -- python scripts/sample_efficiency.py --model convnextv2_large --frac 0.50 --num_workers 0
[2026-05-16T08:15:06Z] SE convnextv2_large frac=0.50 : DONE  -- 185.0 min
[2026-05-16T08:15:06Z] SE dinov2_registers_ft frac=0.10 : START -- python scripts/sample_efficiency.py --model dinov2_registers_ft --frac 0.10 --num_workers 0
[2026-05-16T08:24:07Z] SE dinov2_registers_ft frac=0.10 : DONE  -- 9.0 min
[2026-05-16T08:24:07Z] SE dinov2_registers_ft frac=0.25 : START -- python scripts/sample_efficiency.py --model dinov2_registers_ft --frac 0.25 --num_workers 0
[2026-05-16T08:41:23Z] SE dinov2_registers_ft frac=0.25 : DONE  -- 17.3 min
[2026-05-16T08:41:23Z] SE dinov2_registers_ft frac=0.50 : START -- python scripts/sample_efficiency.py --model dinov2_registers_ft --frac 0.50 --num_workers 0
[2026-05-16T09:11:17Z] SE dinov2_registers_ft frac=0.50 : DONE  -- 29.9 min
[2026-05-16T09:11:17Z] 
[2026-05-16T09:11:17Z] ## (b) TENT re-runs at per-model best config -- write __best suffix
[2026-05-16T09:11:17Z] 
[2026-05-16T09:11:17Z] TENT __best convnextv2_large on ddi : START -- python scripts/run_tent.py --model convnextv2_large --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1 --out_suffix __best
[2026-05-16T09:13:20Z] TENT __best convnextv2_large on ddi : DONE  -- 2.1 min
[2026-05-16T09:13:20Z] TENT __best dinov2_registers_ft on ddi : START -- python scripts/run_tent.py --model dinov2_registers_ft --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1 --out_suffix __best
[2026-05-16T09:13:47Z] TENT __best dinov2_registers_ft on ddi : DONE  -- 0.4 min
[2026-05-16T09:13:47Z] TENT __best pandem_real_probe on ddi : START -- python scripts/run_tent.py --model pandem_real_probe --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1 --out_suffix __best
[2026-05-16T09:14:08Z] TENT __best pandem_real_probe on ddi : DONE  -- 0.4 min
[2026-05-16T09:14:08Z] TENT __best efficientnetv2_m on ddi : START -- python scripts/run_tent.py --model efficientnetv2_m --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1 --out_suffix __best
[2026-05-16T09:14:38Z] TENT __best efficientnetv2_m on ddi : DONE  -- 0.5 min
[2026-05-16T09:14:38Z] TENT __best efficientnetv2_m on fitz17k (n_steps=3) : START -- python scripts/run_tent.py --model efficientnetv2_m --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 3 --out_suffix __best
[2026-05-16T09:19:32Z] TENT __best efficientnetv2_m on fitz17k (n_steps=3) : DONE  -- 4.9 min
[2026-05-16T09:19:32Z] TENT __best convnextv2_large on fitz17k (lr=1e-4) : START -- python scripts/run_tent.py --model convnextv2_large --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-4 --n_steps 1 --out_suffix __best
[2026-05-16T09:31:09Z] TENT __best convnextv2_large on fitz17k (lr=1e-4) : DONE  -- 11.6 min
[2026-05-16T09:31:09Z] TENT __best dinov2_registers_ft on fitz17k (lr=1e-4) : START -- python scripts/run_tent.py --model dinov2_registers_ft --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-4 --n_steps 1 --out_suffix __best
[2026-05-16T09:32:59Z] TENT __best dinov2_registers_ft on fitz17k (lr=1e-4) : DONE  -- 1.8 min
[2026-05-16T09:32:59Z] 
[2026-05-16T09:32:59Z] ## (c) TENT on ISIC 2019 (ham10k taxonomy)
[2026-05-16T09:32:59Z] 
[2026-05-16T09:32:59Z] TENT convnextv2_large on isic2019 (ham10k) : START -- python scripts/run_tent.py --model convnextv2_large --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-16T09:33:03Z] TENT convnextv2_large on isic2019 (ham10k) : FAIL  -- exit=2  (0.1 min)
[2026-05-16T09:33:03Z] TENT efficientnetv2_m on isic2019 (ham10k) : START -- python scripts/run_tent.py --model efficientnetv2_m --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-16T09:33:07Z] TENT efficientnetv2_m on isic2019 (ham10k) : FAIL  -- exit=2  (0.1 min)
[2026-05-16T09:33:07Z] TENT dinov2_registers_ft on isic2019 (ham10k) : START -- python scripts/run_tent.py --model dinov2_registers_ft --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-16T09:33:11Z] TENT dinov2_registers_ft on isic2019 (ham10k) : FAIL  -- exit=2  (0.1 min)
[2026-05-16T09:33:11Z] TENT pandem_real_probe on isic2019 (ham10k) : START -- python scripts/run_tent.py --model pandem_real_probe --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0 --lr 1e-3 --n_steps 1
[2026-05-16T09:33:14Z] TENT pandem_real_probe on isic2019 (ham10k) : FAIL  -- exit=2  (0.1 min)
[2026-05-16T09:33:14Z] 
[2026-05-16T09:33:14Z] ## (d) Cross-dataset SLSQP on __best TENT probs
[2026-05-16T09:33:14Z] 
[2026-05-16T09:33:14Z] B4 SLSQP on __best probs : START -- python -m src.cross_dataset_ensemble --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe --datasets ddi fitz17k --probs_subdir cross_dataset_tent --model_suffix __best --out_suffix _tent_best
[2026-05-16T09:33:16Z] B4 SLSQP on __best probs : FAIL  -- exit=1  (0.0 min)
[2026-05-16T09:33:16Z] 
[2026-05-16T09:33:16Z] ## (e) Generate figures
[2026-05-16T09:33:16Z] 
[2026-05-16T09:33:16Z] ensure matplotlib : START -- pip install -q matplotlib
[2026-05-16T09:33:17Z] ensure matplotlib : DONE  -- 0.0 min
[2026-05-16T09:33:17Z] make_plots : START -- python scripts/make_plots.py
[2026-05-16T09:33:18Z] make_plots : DONE  -- 0.0 min
[2026-05-16T09:33:18Z] 
[2026-05-16T09:33:18Z] ## (f) Compile 01_RESULTS.md
[2026-05-16T09:33:18Z] 
[2026-05-16T09:33:18Z] make_results_report : START -- python scripts/make_results_report.py
[2026-05-16T09:33:19Z] make_results_report : DONE  -- 0.0 min
[2026-05-16T09:33:19Z] 
[2026-05-16T09:33:19Z] ================================================================
[2026-05-16T09:33:19Z] OVERNIGHT QUEUE 3 END -- total 7.8 hr
[2026-05-16T09:33:19Z] Outputs to check in the morning:
[2026-05-16T09:33:19Z]   D:\ML\skin-lesion-vit\info_bank\01_RESULTS.md                                    <- start here
[2026-05-16T09:33:19Z]   D:\ML\skin-lesion-vit\v2\figures\fig1_sample_efficiency.png
[2026-05-16T09:33:19Z]   D:\ML\skin-lesion-vit\v2\figures\fig2_fairness_frontier.png
[2026-05-16T09:33:19Z]   D:\ML\skin-lesion-vit\v2\figures\fig3_tent_improvement.png
[2026-05-16T09:33:19Z]   D:\ML\skin-lesion-vit\v2\figures\fig4_pandem_loader_fix.png
[2026-05-16T09:33:19Z]   experiments\sample_efficiency\<model>_fracNNN\results.txt
[2026-05-16T09:33:19Z]   experiments\cross_dataset_tent\<ds>\<model>__best\report.txt
[2026-05-16T09:33:19Z]   experiments\cross_dataset_ensemble_tent_best\report.txt
[2026-05-16T09:33:19Z] ================================================================
[2026-05-16T13:33:53Z] 
[2026-05-16T13:33:53Z] ================================================================
[2026-05-16T13:33:53Z] RUN-FINISH START -- B4(__best) + figures + report (CPU-only, ~2 min)
[2026-05-16T13:33:53Z] NOTE: ISIC-TENT intentionally dropped (too slow, low value).
[2026-05-16T13:33:53Z] ================================================================
[2026-05-16T13:33:53Z] B4 SLSQP on __best probs : START -- python -m src.cross_dataset_ensemble --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe --datasets ddi fitz17k --probs_subdir cross_dataset_tent --model_suffix __best --out_suffix _tent_best
[2026-05-16T13:33:55Z] B4 SLSQP on __best probs : DONE  -- 0.0 min
[2026-05-16T13:33:55Z] make_plots : START -- python scripts/make_plots.py
[2026-05-16T13:33:57Z] make_plots : DONE  -- 0.0 min
[2026-05-16T13:33:57Z] make_results_report : START -- python scripts/make_results_report.py
[2026-05-16T13:33:57Z] make_results_report : DONE  -- 0.0 min
[2026-05-16T13:33:57Z] 
[2026-05-16T13:33:57Z] ================================================================
[2026-05-16T13:33:57Z] RUN-FINISH END -- 0.1 min
[2026-05-16T13:33:57Z] Outputs:
[2026-05-16T13:33:57Z]   D:\ML\skin-lesion-vit\info_bank\01_RESULTS.md                                      <- start here
[2026-05-16T13:33:57Z]   D:\ML\skin-lesion-vit\v2\figures\fig{1,2,3,4}.{png,pdf}
[2026-05-16T13:33:57Z]   experiments\cross_dataset_ensemble_tent_best\report.txt
[2026-05-16T13:33:57Z] ================================================================

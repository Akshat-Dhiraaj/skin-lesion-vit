# scripts/run_overnight_2.ps1
# ---------------------------------------------------------------------------
# Overnight iteration 2: act on the findings from run_overnight.ps1.
#
#   1. B5 hyperparam sweep — per-architecture TENT lr.
#      LN-only models (convnextv2, dinov2, pandem_real) hurt at lr=1e-3 on
#      Fitz17k; try lr=1e-4. efficientnetv2_m (BN) was the win on Fitz17k;
#      try n_steps=3 to see if more iter improves DDI.
#
#   2. B4 — re-fit per-dataset SLSQP ensemble on the TENT-adapted probs
#      (lr=1e-4 variant). Should push DDI dark-skin Mal AUC past 0.70.
#
#   3. OA1 — retrain binary mel specialist with fixed hyperparams.
#      alpha=0.5 (50/50 sampler already balances; 0.25 was downweighting
#      positives). Lower lr_head for convnextv2 (was unstable, collapsed to
#      0 recall). More epochs.
#
# Estimated total: ~5 hr.
# ---------------------------------------------------------------------------

. "$PSScriptRoot\_lib.ps1"   # $WorkRoot, $LogFile, $StartTime, Log-Line, Run-Step, venv, UTF-8 (historical script; behavior unchanged)

Log-Line ""
Log-Line "================================================================"
Log-Line "OVERNIGHT QUEUE 2 START -- B5 sweep + B4 TENT-ensemble + OA1 retry"
Log-Line "================================================================"

# ───────────────────────────────────────────────────────────────────────────
# (1) B5 hyperparam sweep                                          ~1 hr
# ───────────────────────────────────────────────────────────────────────────
Log-Line ""
Log-Line "## B5 hyperparam sweep"
Log-Line ""

# LN-only models at lower lr (1e-4)
foreach ($model in 'convnextv2_large','dinov2_registers_ft','pandem_real_probe') {
    foreach ($ds in 'ddi','fitz17k') {
        Run-Step "B5 $model lr=1e-4 on $ds" `
            "python scripts/run_tent.py --model $model --dataset $ds --data_root data/external --label_mode universal --num_workers 0 --lr 1e-4 --n_steps 1 --out_suffix __lr1e-4"
    }
}

# BN model: lr=1e-3 stays (baseline already on disk), try n_steps=3
foreach ($ds in 'ddi','fitz17k') {
    Run-Step "B5 efficientnetv2_m lr=1e-3 nsteps=3 on $ds" `
        "python scripts/run_tent.py --model efficientnetv2_m --dataset $ds --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 3 --out_suffix __nsteps3"
}

# Optionally compare BN model at lr=1e-4 too, single step
foreach ($ds in 'ddi','fitz17k') {
    Run-Step "B5 efficientnetv2_m lr=1e-4 on $ds" `
        "python scripts/run_tent.py --model efficientnetv2_m --dataset $ds --data_root data/external --label_mode universal --num_workers 0 --lr 1e-4 --n_steps 1 --out_suffix __lr1e-4"
}

# ───────────────────────────────────────────────────────────────────────────
# (2) B4 — TENT-adapted ensemble                                    ~5 min
# ───────────────────────────────────────────────────────────────────────────
Log-Line ""
Log-Line "## B4 cross-dataset SLSQP on TENT-adapted probs (lr=1e-4 variant)"
Log-Line ""

Run-Step "B4 TENT-ensemble lr=1e-4" `
    "python -m src.cross_dataset_ensemble --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe --datasets ddi fitz17k --probs_subdir cross_dataset_tent --model_suffix __lr1e-4 --out_suffix _tent_lr1e-4"

# Also: ensemble on the ORIGINAL TENT probs (lr=1e-3) so we have a side-by-side
Run-Step "B4 TENT-ensemble lr=1e-3 (baseline)" `
    "python -m src.cross_dataset_ensemble --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe --datasets ddi fitz17k --probs_subdir cross_dataset_tent --out_suffix _tent_lr1e-3"

# ───────────────────────────────────────────────────────────────────────────
# (3) OA1 — binary mel specialist retraining                       ~3.5 hr
# ───────────────────────────────────────────────────────────────────────────
Log-Line ""
Log-Line "## OA1 retraining (alpha=0.5 + longer + safer convnext lr)"
Log-Line ""

# pandem_real_probe: only head trains, ~25 min
Run-Step "OA1 pandem_real_probe alpha=0.5 20ep" `
    "python scripts/train_mel_binary.py --backbone pandem_real_probe --epochs 20 --alpha 0.5 --gamma 2.0 --name_suffix _v2 --num_workers 0"

# convnextv2_large: full FT, halve both lrs, 20 epochs, ~3 hr
Run-Step "OA1 convnextv2_large alpha=0.5 20ep lr_head=5e-4 lr_bb=2e-6" `
    "python scripts/train_mel_binary.py --backbone convnextv2_large --epochs 20 --alpha 0.5 --gamma 2.0 --lr_head 5e-4 --lr_backbone 2e-6 --name_suffix _v2 --num_workers 0"

# ───────────────────────────────────────────────────────────────────────────
$totalDur = ((Get-Date) - $StartTime).TotalHours
Log-Line ""
Log-Line "================================================================"
Log-Line ("OVERNIGHT QUEUE 2 END -- total {0:N1} hr" -f $totalDur)
Log-Line "Headline file paths to check in the morning:"
Log-Line "  experiments/cross_dataset_tent/<dataset>/<model>__lr1e-4/report.txt"
Log-Line "  experiments/cross_dataset_tent/<dataset>/<model>__nsteps3/report.txt"
Log-Line "  experiments/cross_dataset_ensemble_tent_lr1e-4/report.txt"
Log-Line "  experiments/cross_dataset_ensemble_tent_lr1e-3/report.txt"
Log-Line "  experiments/mel_binary/mel_binary_<backbone>_v2/results.txt"
Log-Line "================================================================"

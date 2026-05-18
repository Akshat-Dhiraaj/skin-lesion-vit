# scripts/run_overnight_3.ps1
# ---------------------------------------------------------------------------
# Iteration 3: polish + final report.
#
# (a) Complete the sample-efficiency curves — fill in 10% for all key models
#     and the missing convnext / dinov2 fractions so the figure has 4 models
#     × 4 fractions.
# (b) Re-run TENT for each (model, dataset) at the BEST hyperparams from
#     iteration 2's sweep, write to a __best suffix.
# (c) Add TENT on ISIC 2019 for the 4 backbones (we never tried it).
# (d) Refit the per-target SLSQP cross-dataset ensemble on __best probs.
# (e) Generate publication-quality figures (4 PNG/PDF pairs).
# (f) Compile a single self-contained RESULTS_REPORT.md at D:\ML\.
#
# Estimated runtime: ~6-7 hr (conservative). Convnext is the slow part.
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Continue'
$WorkRoot = 'D:\ML\skin-lesion-vit-v2'
$LogFile  = 'D:\ML\OVERNIGHT_RESULTS.md'
$StartTime = Get-Date

Set-Location $WorkRoot
& "$WorkRoot\.venv\Scripts\Activate.ps1"
$env:PYTHONIOENCODING = 'utf-8'

function Log-Line { param([string]$L); $ts=(Get-Date).ToString('yyyy-MM-ddTHH:mm:ssZ'); "[$ts] $L" | Tee-Object -FilePath $LogFile -Append }
function Run-Step {
    param([string]$Tag, [string]$Cmd, [string]$SkipIfExists)
    if ($SkipIfExists -and (Test-Path $SkipIfExists)) {
        Log-Line "$Tag : SKIP -- already on disk ($SkipIfExists)"
        return
    }
    Log-Line "$Tag : START -- $Cmd"
    $s = Get-Date
    try {
        Invoke-Expression $Cmd
        $e = $LASTEXITCODE
        $d = ((Get-Date) - $s).TotalMinutes
        if ($e -eq 0 -or $e -eq $null) { Log-Line ("$Tag : DONE  -- {0:N1} min" -f $d) }
        else { Log-Line ("$Tag : FAIL  -- exit={0}  ({1:N1} min)" -f $e, $d) }
    } catch {
        $d = ((Get-Date) - $s).TotalMinutes
        Log-Line ("$Tag : ERROR -- $($_.Exception.Message)  ({0:N1} min)" -f $d)
    }
}

Log-Line ""
Log-Line "================================================================"
Log-Line "OVERNIGHT QUEUE 3 START -- polish + final report"
Log-Line "================================================================"

# ─── (a) Sample-efficiency completion ────────────────────────────────────────
Log-Line ""
Log-Line "## (a) Sample-efficiency curves (fill missing fractions)"
Log-Line ""

# 10% for the two probe models (fast)
foreach ($model in 'pandem_real_probe','efficientnetv2_m','siglip2_probe') {
    Run-Step "SE $model frac=0.10" `
        "python scripts/sample_efficiency.py --model $model --frac 0.10 --num_workers 0" `
        "experiments\sample_efficiency\${model}_frac010\results.txt"
}

# Missing convnextv2_large fractions (slow but worth it)
foreach ($f in '0.10','0.25','0.50') {
    $fi = [int]([double]$f * 100)
    $fpad = "{0:D3}" -f $fi
    Run-Step "SE convnextv2_large frac=$f" `
        "python scripts/sample_efficiency.py --model convnextv2_large --frac $f --num_workers 0" `
        "experiments\sample_efficiency\convnextv2_large_frac$fpad\results.txt"
}

# dinov2_registers_ft fractions to round out a 4th model
foreach ($f in '0.10','0.25','0.50') {
    $fi = [int]([double]$f * 100)
    $fpad = "{0:D3}" -f $fi
    Run-Step "SE dinov2_registers_ft frac=$f" `
        "python scripts/sample_efficiency.py --model dinov2_registers_ft --frac $f --num_workers 0" `
        "experiments\sample_efficiency\dinov2_registers_ft_frac$fpad\results.txt"
}

# ─── (b) Per-model best TENT configs ─────────────────────────────────────────
Log-Line ""
Log-Line "## (b) TENT re-runs at per-model best config -- write __best suffix"
Log-Line ""

# DDI: lr=1e-3, n_steps=1 was best for all four
foreach ($m in 'convnextv2_large','dinov2_registers_ft','pandem_real_probe','efficientnetv2_m') {
    Run-Step "TENT __best $m on ddi" `
        "python scripts/run_tent.py --model $m --dataset ddi --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1 --out_suffix __best" `
        "experiments\cross_dataset_tent\ddi\${m}__best\report.txt"
}

# Fitz17k: per-model best
Run-Step "TENT __best efficientnetv2_m on fitz17k (n_steps=3)" `
    "python scripts/run_tent.py --model efficientnetv2_m --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 3 --out_suffix __best" `
    "experiments\cross_dataset_tent\fitz17k\efficientnetv2_m__best\report.txt"
foreach ($m in 'convnextv2_large','dinov2_registers_ft') {
    Run-Step "TENT __best $m on fitz17k (lr=1e-4)" `
        "python scripts/run_tent.py --model $m --dataset fitz17k --data_root data/external --label_mode universal --num_workers 0 --lr 1e-4 --n_steps 1 --out_suffix __best" `
        "experiments\cross_dataset_tent\fitz17k\${m}__best\report.txt"
}
# pandem_real_probe on Fitz17k: every TENT config hurts. Skip — its
# baseline IS the best operating point. No __best file written intentionally.

# ─── (c) TENT on ISIC 2019 (new) ─────────────────────────────────────────────
Log-Line ""
Log-Line "## (c) TENT on ISIC 2019 (ham10k taxonomy)"
Log-Line ""
foreach ($m in 'convnextv2_large','efficientnetv2_m','dinov2_registers_ft','pandem_real_probe') {
    Run-Step "TENT $m on isic2019 (ham10k)" `
        "python scripts/run_tent.py --model $m --dataset isic2019 --data_root data/external --label_mode ham10k --num_workers 0 --lr 1e-3 --n_steps 1" `
        "experiments\cross_dataset_tent\isic2019\${m}\report.txt"
}

# ─── (d) Cross-dataset SLSQP on __best TENT probs ────────────────────────────
Log-Line ""
Log-Line "## (d) Cross-dataset SLSQP on __best TENT probs"
Log-Line ""
Run-Step "B4 SLSQP on __best probs" `
    "python -m src.cross_dataset_ensemble --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe --datasets ddi fitz17k --probs_subdir cross_dataset_tent --model_suffix __best --out_suffix _tent_best"

# ─── (e) Figures ─────────────────────────────────────────────────────────────
Log-Line ""
Log-Line "## (e) Generate figures"
Log-Line ""
# Install matplotlib if not present (harmless if already there)
Run-Step "ensure matplotlib" "pip install -q matplotlib"
Run-Step "make_plots" "python scripts/make_plots.py"

# ─── (f) Final report ────────────────────────────────────────────────────────
Log-Line ""
Log-Line "## (f) Compile RESULTS_REPORT.md"
Log-Line ""
Run-Step "make_results_report" "python scripts/make_results_report.py"

# ─── Summary ─────────────────────────────────────────────────────────────────
$totalDur = ((Get-Date) - $StartTime).TotalHours
Log-Line ""
Log-Line "================================================================"
Log-Line ("OVERNIGHT QUEUE 3 END -- total {0:N1} hr" -f $totalDur)
Log-Line "Outputs to check in the morning:"
Log-Line "  D:\ML\RESULTS_REPORT.md                                    <- start here"
Log-Line "  D:\ML\skin-lesion-vit-v2\figures\fig1_sample_efficiency.png"
Log-Line "  D:\ML\skin-lesion-vit-v2\figures\fig2_fairness_frontier.png"
Log-Line "  D:\ML\skin-lesion-vit-v2\figures\fig3_tent_improvement.png"
Log-Line "  D:\ML\skin-lesion-vit-v2\figures\fig4_pandem_loader_fix.png"
Log-Line "  experiments\sample_efficiency\<model>_fracNNN\results.txt"
Log-Line "  experiments\cross_dataset_tent\<ds>\<model>__best\report.txt"
Log-Line "  experiments\cross_dataset_ensemble_tent_best\report.txt"
Log-Line "================================================================"

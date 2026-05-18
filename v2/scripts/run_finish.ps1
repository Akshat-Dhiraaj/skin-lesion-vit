# scripts/run_finish.ps1
# ---------------------------------------------------------------------------
# Finish the iteration-3 polish. Picks up the 3 steps that failed in
# run_overnight_3.ps1, MINUS ISIC-TENT (dropped: ~3 hr/model x 4 models for
# a marginal macro-AUC delta on a dataset with no Fitzpatrick stratification
# — not worth the compute).
#
# Everything here is CPU-only and fast (~2 min total). Negligible VRAM —
# safe to run while the GPU is busy with something else.
#
#   1. B4 SLSQP cross-dataset ensemble on the __best TENT probs
#      (cross_dataset_ensemble.py now falls back to the no-TENT baseline
#       for any model lacking a __best variant — e.g. pandem_real_probe on
#       fitz17k, where TENT always hurt).
#   2. Regenerate the 4 figures from whatever is on disk.
#   3. Recompile D:\ML\RESULTS_REPORT.md.
#
# Run later with:
#   cd D:\ML\skin-lesion-vit-v2; .\scripts\run_finish.ps1 *>&1 | Tee-Object -FilePath D:\ML\finish_stdout.log
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
    param([string]$Tag, [string]$Cmd)
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
Log-Line "RUN-FINISH START -- B4(__best) + figures + report (CPU-only, ~2 min)"
Log-Line "NOTE: ISIC-TENT intentionally dropped (too slow, low value)."
Log-Line "================================================================"

# 1. B4 SLSQP on __best TENT probs (graceful fallback to baseline per model)
Run-Step "B4 SLSQP on __best probs" `
    "python -m src.cross_dataset_ensemble --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe --datasets ddi fitz17k --probs_subdir cross_dataset_tent --model_suffix __best --out_suffix _tent_best"

# 2. Figures (matplotlib, CPU)
Run-Step "make_plots" "python scripts/make_plots.py"

# 3. Final report (text parsing, CPU)
Run-Step "make_results_report" "python scripts/make_results_report.py"

$dur = ((Get-Date) - $StartTime).TotalMinutes
Log-Line ""
Log-Line "================================================================"
Log-Line ("RUN-FINISH END -- {0:N1} min" -f $dur)
Log-Line "Outputs:"
Log-Line "  D:\ML\RESULTS_REPORT.md                                      <- start here"
Log-Line "  D:\ML\skin-lesion-vit-v2\figures\fig{1,2,3,4}.{png,pdf}"
Log-Line "  experiments\cross_dataset_ensemble_tent_best\report.txt"
Log-Line "================================================================"

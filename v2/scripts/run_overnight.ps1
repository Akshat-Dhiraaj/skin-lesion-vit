# scripts/run_overnight.ps1
# ---------------------------------------------------------------------------
# Overnight runner for E12 (sample efficiency), OA1 (mel binary specialist),
# and B5 (TENT test-time adaptation).
#
# Usage (from D:\ML\skin-lesion-vit-v2 with venv NOT yet activated):
#   .\scripts\run_overnight.ps1
#
# Each task is independent; a failure in one is logged and the queue
# continues. Progress appended to D:\ML\OVERNIGHT_RESULTS.md.
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Continue'
$WorkRoot = 'D:\ML\skin-lesion-vit-v2'
$LogFile = 'D:\ML\OVERNIGHT_RESULTS.md'
$StartTime = Get-Date

Set-Location $WorkRoot
& "$WorkRoot\.venv\Scripts\Activate.ps1"
$env:PYTHONIOENCODING = 'utf-8'

function Log-Line {
    param([string]$Line)
    $ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssZ')
    "[$ts] $Line" | Tee-Object -FilePath $LogFile -Append
}

function Run-Step {
    param(
        [string]$Tag,
        [string]$Cmd
    )
    Log-Line "$Tag : START -- $Cmd"
    $stepStart = Get-Date
    try {
        Invoke-Expression $Cmd
        $exit = $LASTEXITCODE
        $dur = ((Get-Date) - $stepStart).TotalMinutes
        if ($exit -eq 0 -or $exit -eq $null) {
            Log-Line ("$Tag : DONE  -- {0:N1} min" -f $dur)
        } else {
            Log-Line ("$Tag : FAIL  -- exit={0}  ({1:N1} min)" -f $exit, $dur)
        }
    } catch {
        $dur = ((Get-Date) - $stepStart).TotalMinutes
        Log-Line ("$Tag : ERROR -- $($_.Exception.Message)  ({0:N1} min)" -f $dur)
    }
}

Log-Line ""
Log-Line "================================================================"
Log-Line "OVERNIGHT QUEUE START -- E12 + OA1 + B5"
Log-Line "================================================================"

# ---------- E12: Sample efficiency curves ---------- ~3.5 hr
Log-Line ""
Log-Line "## E12 -- Sample efficiency curves"
Log-Line ""

foreach ($frac in 0.25, 0.50, 1.00) {
    Run-Step "E12 pandem_real_probe frac=$frac" `
        "python scripts/sample_efficiency.py --model pandem_real_probe --frac $frac --num_workers 0"
}
foreach ($frac in 0.25, 0.50, 1.00) {
    Run-Step "E12 efficientnetv2_m frac=$frac" `
        "python scripts/sample_efficiency.py --model efficientnetv2_m --frac $frac --num_workers 0"
}

# ---------- OA1: Binary mel specialist ---------- ~2.5 hr
Log-Line ""
Log-Line "## OA1 -- Binary melanoma specialist"
Log-Line ""

# pandem_real_probe -- frozen backbone, only the binary head trains (~30 min)
Run-Step "OA1 mel_binary_pandem_real_probe" `
    "python scripts/train_mel_binary.py --backbone pandem_real_probe --epochs 15 --num_workers 0"

# convnextv2_large -- full FT, ~2 hr
Run-Step "OA1 mel_binary_convnextv2_large" `
    "python scripts/train_mel_binary.py --backbone convnextv2_large --epochs 12 --num_workers 0"

# ---------- B5: TENT test-time adaptation ---------- ~30 min
Log-Line ""
Log-Line "## B5 -- TENT (test-time entropy minimisation)"
Log-Line ""

foreach ($model in 'convnextv2_large', 'pandem_real_probe', 'efficientnetv2_m', 'dinov2_registers_ft') {
    foreach ($ds in 'ddi', 'fitz17k') {
        Run-Step "B5 TENT $model on $ds" `
            "python scripts/run_tent.py --model $model --dataset $ds --data_root data/external --label_mode universal --num_workers 0 --lr 1e-3 --n_steps 1"
    }
}

# ---------- Summary ----------
$totalDur = ((Get-Date) - $StartTime).TotalHours
Log-Line ""
Log-Line "================================================================"
Log-Line ("OVERNIGHT QUEUE END -- total {0:N1} hr" -f $totalDur)
Log-Line "Headline file paths to check in the morning:"
Log-Line "  experiments/sample_efficiency/<model>_fracNNN/results.txt"
Log-Line "  experiments/mel_binary/mel_binary_<backbone>/results.txt"
Log-Line "  experiments/cross_dataset_tent/<dataset>/<model>/report.txt"
Log-Line "================================================================"

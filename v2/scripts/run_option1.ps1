# =============================================================================
# run_option1.ps1
# -----------------------------------------------------------------------------
# Overnight runner for Option 1 - Post-hoc HAM10000 ceiling push.
#
# Each model runs in its OWN isolated block. If one fails, the others still
# run. Output file existence is the source of truth (not exit codes).
# Post-hoc step only uses models whose VAL TTA actually produced .npy files.
#
# Total runtime: ~25-30 minutes (mostly TTA inference).
#
# Run pre-flight check FIRST:
#     .\scripts\test_option1.ps1
#
# Then run this:
#     .\scripts\run_option1.ps1
# =============================================================================

# ── Settings ─────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Continue"   # Don't stop on errors
$ProgressPreference    = "SilentlyContinue"

# UTF-8 output
chcp 65001 | Out-Null
$OutputEncoding = [System.Text.Encoding]::UTF8

# ── Edit these if your paths differ ──────────────────────────────────────────
$DATA_DIR = "D:\ML\skin-lesion-vit\data\raw"
$LOG_DIR  = "experiments\overnight_logs"

# ── Setup ────────────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = Join-Path $LOG_DIR "run_option1_$timestamp.log"

Start-Transcript -Path $transcript -Force | Out-Null

Write-Output ""
Write-Output "============================================================"
Write-Output "  OPTION 1 OVERNIGHT RUN  -  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
Write-Output "Data dir : $DATA_DIR"
Write-Output "Log file : $transcript"
Write-Output ""

# Verify data dir exists (don't proceed if it doesn't)
if (-not (Test-Path $DATA_DIR)) {
    Write-Output "[FATAL] Data directory not found: $DATA_DIR"
    Write-Output "        Edit `$DATA_DIR at the top of this script."
    Stop-Transcript | Out-Null
    exit 1
}


# =============================================================================
#  STEP 1: VAL TTA - each model in its OWN isolated block
#  If one crashes, the others still run.
# =============================================================================

# ── Model 1: convnextv2_large ────────────────────────────────────────────────
Write-Output ""
Write-Output "============================================================"
Write-Output "  [1/4] convnextv2_large - VAL TTA"
Write-Output "============================================================"
$start = Get-Date
$ckpt  = "experiments\convnextv2_large\best_model.pth"
if (-not (Test-Path $ckpt)) {
    Write-Output "  [SKIP] Checkpoint missing: $ckpt"
} else {
    python -m src.tta --model convnextv2_large --data_dir "$DATA_DIR" --split val
    $exit = $LASTEXITCODE
    $elapsed = ((Get-Date) - $start).ToString('mm\:ss')
    if (Test-Path "experiments\convnextv2_large\tta_val_probs.npy") {
        Write-Output "  [OK] Completed in $elapsed (exit=$exit)"
    } else {
        Write-Output "  [FAIL] No tta_val_probs.npy produced (exit=$exit) - elapsed $elapsed"
    }
}

# ── Model 2: efficientnetv2_m ────────────────────────────────────────────────
Write-Output ""
Write-Output "============================================================"
Write-Output "  [2/4] efficientnetv2_m - VAL TTA"
Write-Output "============================================================"
$start = Get-Date
$ckpt  = "experiments\efficientnetv2_m\best_model.pth"
if (-not (Test-Path $ckpt)) {
    Write-Output "  [SKIP] Checkpoint missing: $ckpt"
} else {
    python -m src.tta --model efficientnetv2_m --data_dir "$DATA_DIR" --split val
    $exit = $LASTEXITCODE
    $elapsed = ((Get-Date) - $start).ToString('mm\:ss')
    if (Test-Path "experiments\efficientnetv2_m\tta_val_probs.npy") {
        Write-Output "  [OK] Completed in $elapsed (exit=$exit)"
    } else {
        Write-Output "  [FAIL] No tta_val_probs.npy produced (exit=$exit) - elapsed $elapsed"
    }
}

# ── Model 3: dinov2_registers_ft ─────────────────────────────────────────────
Write-Output ""
Write-Output "============================================================"
Write-Output "  [3/4] dinov2_registers_ft - VAL TTA"
Write-Output "============================================================"
$start = Get-Date
$ckpt  = "experiments\dinov2_registers_ft\best_model.pth"
if (-not (Test-Path $ckpt)) {
    Write-Output "  [SKIP] Checkpoint missing: $ckpt"
} else {
    python -m src.tta --model dinov2_registers_ft --data_dir "$DATA_DIR" --split val
    $exit = $LASTEXITCODE
    $elapsed = ((Get-Date) - $start).ToString('mm\:ss')
    if (Test-Path "experiments\dinov2_registers_ft\tta_val_probs.npy") {
        Write-Output "  [OK] Completed in $elapsed (exit=$exit)"
    } else {
        Write-Output "  [FAIL] No tta_val_probs.npy produced (exit=$exit) - elapsed $elapsed"
    }
}

# ── Model 4: pandem_ft ───────────────────────────────────────────────────────
Write-Output ""
Write-Output "============================================================"
Write-Output "  [4/4] pandem_ft - VAL TTA"
Write-Output "============================================================"
$start = Get-Date
$ckpt  = "experiments\pandem_ft\best_model.pth"
if (-not (Test-Path $ckpt)) {
    Write-Output "  [SKIP] Checkpoint missing: $ckpt"
} else {
    python -m src.tta --model pandem_ft --data_dir "$DATA_DIR" --split val
    $exit = $LASTEXITCODE
    $elapsed = ((Get-Date) - $start).ToString('mm\:ss')
    if (Test-Path "experiments\pandem_ft\tta_val_probs.npy") {
        Write-Output "  [OK] Completed in $elapsed (exit=$exit)"
    } else {
        Write-Output "  [FAIL] No tta_val_probs.npy produced (exit=$exit) - elapsed $elapsed"
    }
}


# =============================================================================
#  STEP 2: Verify and post-hoc analysis
# =============================================================================

Write-Output ""
Write-Output "============================================================"
Write-Output "  POST-FLIGHT VERIFICATION"
Write-Output "============================================================"

# Build the list of models that produced both VAL and TEST TTA probs
$ALL_MODELS = @("convnextv2_large", "efficientnetv2_m", "dinov2_registers_ft", "pandem_ft")
$ready = @()

foreach ($model in $ALL_MODELS) {
    $valOk  = Test-Path "experiments\$model\tta_val_probs.npy"
    $testOk = Test-Path "experiments\$model\tta_test_probs.npy"

    # Backward compat - older runs saved test as tta_probs.npy
    if (-not $testOk) {
        $testOk = Test-Path "experiments\$model\tta_probs.npy"
    }

    if ($valOk -and $testOk) {
        Write-Output "  [READY] $model    val=YES test=YES"
        $ready += $model
    } else {
        Write-Output "  [SKIP]  $model    val=$(if($valOk){'YES'}else{'NO '})  test=$(if($testOk){'YES'}else{'NO '})"
    }
}

Write-Output ""
Write-Output "$($ready.Count) of $($ALL_MODELS.Count) models ready for post-hoc"

# Need at least 2 for an ensemble
if ($ready.Count -lt 2) {
    Write-Output ""
    Write-Output "[ABORT] Need at least 2 models with both val + test TTA. Got $($ready.Count)."
    Write-Output "        Diagnose by checking the FAIL messages above."
    Stop-Transcript | Out-Null
    exit 2
}

# ── Run post-hoc on whichever models are ready ───────────────────────────────
Write-Output ""
Write-Output "============================================================"
Write-Output "  STEP 5: Post-hoc analysis on $($ready.Count) models"
Write-Output "============================================================"
Write-Output ""
$start = Get-Date

# Pass models as separate args (PowerShell will splat the array)
python -m src.post_hoc --models $ready

$exit = $LASTEXITCODE
$elapsed = ((Get-Date) - $start).ToString('mm\:ss')

if ((Test-Path "experiments\post_hoc\results.txt") -and ($exit -eq 0)) {
    Write-Output ""
    Write-Output "  [OK] Post-hoc complete in $elapsed"
} else {
    Write-Output ""
    Write-Output "  [FAIL] Post-hoc failed (exit=$exit, elapsed=$elapsed)"
}


# =============================================================================
#  FINAL SUMMARY
# =============================================================================
Write-Output ""
Write-Output "============================================================"
Write-Output "  DONE  -  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
Write-Output "Final report : experiments\post_hoc\results.txt"
Write-Output "Full log     : $transcript"
Write-Output ""
Write-Output "View results:    Get-Content experiments\post_hoc\results.txt"
Write-Output ""

Stop-Transcript | Out-Null
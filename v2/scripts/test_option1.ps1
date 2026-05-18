# =============================================================================
# test_option1.ps1
# -----------------------------------------------------------------------------
# PRE-FLIGHT CHECK for run_option1.ps1
#
# Verifies your setup is correct WITHOUT running any actual computation.
# Run this BEFORE run_option1.ps1 to catch issues quickly.
#
# Checks:
#   1. Python venv is active
#   2. Data directory exists with expected files
#   3. All 4 model checkpoints exist
#   4. Required Python modules importable
#   5. tta.py and post_hoc.py present and parseable
#   6. Disk space available
#
# Total runtime: ~30 seconds.
# =============================================================================

$ErrorActionPreference = "Continue"
chcp 65001 | Out-Null

$DATA_DIR = "D:\ML\skin-lesion-vit\data\raw"
$MODELS   = @("convnextv2_large", "efficientnetv2_m", "dinov2_registers_ft", "pandem_ft")

Write-Output ""
Write-Output "============================================================"
Write-Output "  PRE-FLIGHT CHECK  -  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
Write-Output ""

$failures = 0
$warnings = 0

# ── 1. Python venv ───────────────────────────────────────────────────────────
Write-Output "[1/6] Checking Python..."
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($null -eq $pythonExe) {
    Write-Output "  [FAIL] python not found in PATH"
    $failures++
} else {
    Write-Output "  Python: $pythonExe"
    if ($pythonExe -notmatch "\\.venv\\") {
        Write-Output "  [WARN] Not from .venv - activate with: .\.venv\Scripts\Activate.ps1"
        $warnings++
    } else {
        Write-Output "  [OK] Using project venv"
    }
}
Write-Output ""

# ── 2. Data directory ────────────────────────────────────────────────────────
Write-Output "[2/6] Checking data directory..."
if (-not (Test-Path $DATA_DIR)) {
    Write-Output "  [FAIL] Data directory not found: $DATA_DIR"
    Write-Output "         Edit `$DATA_DIR at the top of run_option1.ps1"
    $failures++
} else {
    Write-Output "  Found: $DATA_DIR"

    $csvPath = Join-Path $DATA_DIR "HAM10000_metadata.csv"
    if (Test-Path $csvPath) {
        Write-Output "  [OK] HAM10000_metadata.csv present"
    } else {
        Write-Output "  [FAIL] Missing: $csvPath"
        $failures++
    }

    # Check at least one image folder
    $imgFolders = Get-ChildItem $DATA_DIR -Directory -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -like "HAM10000_images*" }
    if ($imgFolders.Count -gt 0) {
        Write-Output "  [OK] Image folders present ($($imgFolders.Count) found)"
    } else {
        Write-Output "  [FAIL] No HAM10000_images_* folders found in data dir"
        $failures++
    }
}
Write-Output ""

# ── 3. Model checkpoints ─────────────────────────────────────────────────────
Write-Output "[3/6] Checking model checkpoints..."
$missingCkpts = @()
foreach ($model in $MODELS) {
    $ckptPath = "experiments\$model\best_model.pth"
    if (Test-Path $ckptPath) {
        $sizeMb = [math]::Round((Get-Item $ckptPath).Length / 1MB, 1)
        Write-Output "  [OK]   $model  ($sizeMb MB)"
    } else {
        Write-Output "  [FAIL] $model  - missing $ckptPath"
        $missingCkpts += $model
        $failures++
    }
}
if ($missingCkpts.Count -gt 0) {
    Write-Output ""
    Write-Output "  To rebuild missing models:"
    foreach ($m in $missingCkpts) {
        Write-Output "    python scripts\train.py --model $m --data_dir `"$DATA_DIR`""
    }
}
Write-Output ""

# ── 4. Pipeline scripts ──────────────────────────────────────────────────────
Write-Output "[4/6] Checking pipeline scripts..."
foreach ($script in @("src\tta.py", "src\post_hoc.py")) {
    if (Test-Path $script) {
        # Quick syntax check by compiling
        $null = python -c "import py_compile; py_compile.compile('$($script -replace '\\', '/')', doraise=True)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Output "  [OK] $script"
        } else {
            Write-Output "  [FAIL] $script - syntax error"
            $failures++
        }
    } else {
        Write-Output "  [FAIL] $script - missing"
        $failures++
    }
}
Write-Output ""

# ── 5. Python imports ────────────────────────────────────────────────────────
Write-Output "[5/6] Checking Python imports..."
$importCheck = @"
import sys
ok = True
for mod in ['torch', 'numpy', 'sklearn', 'scipy', 'albumentations', 'mlflow', 'tqdm', 'timm', 'transformers']:
    try:
        __import__(mod)
        print(f'  [OK] {mod}')
    except ImportError as e:
        print(f'  [FAIL] {mod}: {e}')
        ok = False
sys.exit(0 if ok else 1)
"@
$importCheck | python
if ($LASTEXITCODE -ne 0) {
    Write-Output "  [FAIL] Some imports failed - run: pip install -r requirements.txt"
    $failures++
} else {
    Write-Output "  [OK] All imports successful"
}
Write-Output ""

# ── 6. Disk space ────────────────────────────────────────────────────────────
Write-Output "[6/6] Checking disk space..."
$drive = Split-Path -Path (Get-Location) -Qualifier
$diskInfo = Get-PSDrive -Name $drive[0] -ErrorAction SilentlyContinue
if ($null -ne $diskInfo) {
    $freeGB = [math]::Round($diskInfo.Free / 1GB, 1)
    Write-Output "  Free space on $drive  : $freeGB GB"
    if ($freeGB -lt 2) {
        Write-Output "  [WARN] Less than 2 GB free - may run out of space"
        $warnings++
    } else {
        Write-Output "  [OK] Sufficient disk space"
    }
}
Write-Output ""

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Output "============================================================"
Write-Output "  SUMMARY"
Write-Output "============================================================"
if ($failures -eq 0 -and $warnings -eq 0) {
    Write-Output "  [PASS] All checks passed. Safe to run run_option1.ps1"
} elseif ($failures -eq 0) {
    Write-Output "  [PASS WITH $warnings WARNING(S)] Safe to run, but review warnings."
} else {
    Write-Output "  [FAIL] $failures failure(s), $warnings warning(s)"
    Write-Output "         Fix the failures before running run_option1.ps1"
}
Write-Output ""
exit $failures
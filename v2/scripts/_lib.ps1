# scripts/_lib.ps1
# ---------------------------------------------------------------------------
# Shared helpers for the run_*.ps1 batch scripts. Dot-source it:
#
#   . "$PSScriptRoot\_lib.ps1"
#
# Provides: $WorkRoot, $RepoRoot, $LogFile, $StartTime, Log-Line, Run-Step,
# and activates the venv + sets UTF-8. All paths are derived from this file's
# own location ($PSScriptRoot), so the scripts keep working no matter where
# the repository is checked out (this is what broke when v2 was moved from
# D:\ML\skin-lesion-vit-v2 into D:\ML\skin-lesion-vit\v2 — never again).
#
# NOTE: this file contains NO result-producing logic. The committed
# results.txt / report.txt / figures are produced by the Python entrypoints;
# these helpers only orchestrate + log. Refactoring them cannot change any
# numeric output.
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Continue'

# scripts/ -> v2/ (WorkRoot) -> repo root
$WorkRoot  = Split-Path -Parent $PSScriptRoot
$RepoRoot  = Split-Path -Parent $WorkRoot
$LogFile   = Join-Path $RepoRoot 'info_bank\OVERNIGHT_RESULTS.md'
$StartTime = Get-Date

Set-Location $WorkRoot
$activate = Join-Path $WorkRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate }
$env:PYTHONIOENCODING = 'utf-8'

function Log-Line {
    param([string]$L)
    $ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssZ')
    "[$ts] $L" | Tee-Object -FilePath $LogFile -Append
}

function Run-Step {
    # $SkipIfExists is optional & backward-compatible: callers that omit it
    # behave exactly like the original plain Run-Step (run_finish /
    # run_overnight / run_overnight_2); run_overnight_3 passes it to skip
    # steps whose artifact already exists on disk.
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
        if ($e -eq 0 -or $null -eq $e) { Log-Line ("$Tag : DONE  -- {0:N1} min" -f $d) }
        else { Log-Line ("$Tag : FAIL  -- exit={0}  ({1:N1} min)" -f $e, $d) }
    } catch {
        $d = ((Get-Date) - $s).TotalMinutes
        Log-Line ("$Tag : ERROR -- $($_.Exception.Message)  ({0:N1} min)" -f $d)
    }
}

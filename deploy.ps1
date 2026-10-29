<#
.SYNOPSIS
    Deploy SMC and MSS trading strategies to a target machine.

.DESCRIPTION
    Copies all strategy files, ML models, and helper scripts.
    Installs required Python packages and creates launch scripts.

.PARAMETER TargetDir
    Directory to deploy to (default: C:\tradingview)

.PARAMETER InstallDeps
    Install Python dependencies via pip

.EXAMPLE
    .\deploy.ps1 -TargetDir D:\trading -InstallDeps
#>
param(
    [string]$TargetDir = "C:\tradingview",
    [switch]$InstallDeps = $false
)

$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $SrcDir) { $SrcDir = Get-Location }

Write-Host "=== Trading Strategy Deployment ===" -F Cyan
Write-Host "Source: $SrcDir"
Write-Host "Target: $TargetDir"
Write-Host ""

# --- Create target directories ---
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    Write-Host "Created: $TargetDir" -F Green
}

# --- Files to deploy ---
$Files = @(
    # Strategy files
    "smc_strategy.py",
    "sweep_mss_zone.py",
    "trend.py",

    # ML models - sweep
    "sweep_ml_XAUUSD_h1.pkl",
    "sweep_ml_XAUUSD_m15.pkl",
    "sweep_ml_BTCUSD_h1.pkl",
    "sweep_ml_BTCUSD_m15.pkl",

    # ML models - SMC
    "smc_ml_XAUUSD_h1.pkl",
    "smc_ml_XAUUSD_m15.pkl",
    "smc_ml_BTCUSD_h1.pkl",
    "smc_ml_BTCUSD_m15.pkl",

    # ML model (optional)
    "lorentzian_ml_XAUUSD_h1.pkl",

    # Utility scripts
    "bias.py",
    "accumulation.py",
    "trend.py",
    "mt5_stats.py",
    "check_fvg.py",

    # Algo docs (reference)
    "DetectMarketTrendAlgo.txt",
    "AnalyseTimeframeAlgo.txt"
)

$Copied = 0; $Skipped = 0
foreach ($f in $Files) {
    $src = Join-Path $SrcDir $f
    $dst = Join-Path $TargetDir $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Force
        $Copied++
    } else {
        Write-Warning "Not found (skipped): $f"
        $Skipped++
    }
}

Write-Host "Copied $Copied files ($Skipped skipped)" -F Green

# --- Create launch scripts ---
@"
# Start MSS Strategy (sweep_mss_zone.py)
# Edit SYMBOL and TF as needed

python sweep_mss_zone.py XAUUSD --watch --trade --tf h1 --sl-atr 1.2 --tp-atr 0.5 --orderflow --of-mode ohlc --lookback 100 --market-only --max-signal-age 48
"@ | Set-Content -Path (Join-Path $TargetDir "run_mss_h1.ps1") -Force

@"
# Start MSS Strategy (M15)
python sweep_mss_zone.py XAUUSD --watch --trade --tf m15 --sl-atr 1.2 --tp-atr 0.5 --orderflow --of-mode ohlc --lookback 100 --market-only --max-signal-age 96
"@ | Set-Content -Path (Join-Path $TargetDir "run_mss_m15.ps1") -Force

@"
# Start SMC Strategy
python smc_strategy.py XAUUSD --watch --trade --tf h1 --sl-atr 1.2 --tp-atr 0.5 --orderflow --of-mode ohlc --lookback 200 --market-only --max-signal-age 48 --entry-mode all
"@ | Set-Content -Path (Join-Path $TargetDir "run_smc_h1.ps1") -Force

# --- Create combined launcher ---
@"
<#
.SYNOPSIS
    Launch both SMC and MSS simultaneously in separate windows.
    Edit the SYMBOL, TF, and parameters inside each section as needed.
#>
Write-Host "Launching MSS and SMC strategies..." -F Cyan

# MSS strategy
Start-Process powershell -ArgumentList "-NoExit -Command cd '$TargetDir'; .\run_mss_h1.ps1" -WindowStyle Normal

# SMC strategy
Start-Process powershell -ArgumentList "-NoExit -Command cd '$TargetDir'; .\run_smc_h1.ps1" -WindowStyle Normal

Write-Host "Both strategies launched in separate windows." -F Green
"@ | Set-Content -Path (Join-Path $TargetDir "start_both.ps1") -Force

Write-Host "Launch scripts created" -F Green

# --- Install dependencies ---
if ($InstallDeps) {
    Write-Host ""
    Write-Host "Installing Python dependencies..." -F Yellow
    $Deps = @(
        "pandas", "numpy", "ta", "scikit-learn", "MetaTrader5"
    )
    pip install $Deps
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dependencies installed" -F Green
    } else {
        Write-Host "pip install failed. Install manually:" -F Red
        Write-Host "  pip install pandas numpy ta scikit-learn MetaTrader5" -F Yellow
    }
}

Write-Host ""
Write-Host "=== Deployment Complete ===" -F Cyan
Write-Host ""
Write-Host "To launch both strategies:" -F White
Write-Host "  cd $TargetDir"
Write-Host "  .\start_both.ps1"
Write-Host ""
Write-Host "Or individually:" -F White
Write-Host "  cd $TargetDir; .\run_mss_h1.ps1"
Write-Host "  cd $TargetDir; .\run_smc_h1.ps1"
Write-Host ""
Write-Host "MT5 must be installed and running on the target machine." -F Yellow

<#
.SYNOPSIS
    Launch both SMC and MSS strategies simultaneously in separate windows.
    Edit parameters below before running.

.EXAMPLE
    .\start_both.ps1               # launch both strategies
    .\start_both.ps1 BTCUSD h1     # override symbol and timeframe
#>
param(
    [string]$Symbol = "XAUUSD",
    [string]$TF = "h1"
)

$Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Dir) { $Dir = Get-Location }

Write-Host "Starting MSS + SMC for $Symbol $TF ..." -F Cyan

# --- MSS (sweep_mss_zone.py) ---
Start-Process powershell -ArgumentList @"
-NoExit -Command cd '$Dir'; python sweep_mss_zone.py $Symbol --watch --trade --tf $TF --sl-atr 1.2 --tp-atr 0.5 --orderflow --of-mode ohlc --lookback 100 --market-only --max-signal-age 48
"@ -WindowStyle Normal

Start-Sleep 2

# --- SMC (smc_strategy.py) ---
Start-Process powershell -ArgumentList @"
-NoExit -Command cd '$Dir'; python smc_strategy.py $Symbol --watch --trade --tf $TF --sl-atr 1.2 --tp-atr 0.5 --orderflow --of-mode ohlc --lookback 200 --market-only --max-signal-age 48 --entry-mode all
"@ -WindowStyle Normal

Write-Host "Both launched in separate windows. Close each with Ctrl+C" -F Green

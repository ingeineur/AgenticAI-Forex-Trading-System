# extract_historical_data.py - Quick Fix Guide

## Problem Summary

You're not getting data before 2022 because of **3 issues**:

1. **MAX_YEARS = 1** → Creates 1-year chunks (was correct, but see #2)
2. **No error visibility** → Failed chunks are silently skipped
3. **Broker data retention** → Your MT5 broker may only provide 2022+ data

## Quick Diagnostic (DO THIS FIRST)

Run this to check what data your broker actually has:

```bash
python check_broker_data_availability.py XAUUSD
```

This will show you:
```
Year  Status   Bars       Coverage  Start      End
----  -------  -----      --------  ---------- ----------
2020  NO_DATA  0          0%        N/A        N/A
2021  PARTIAL  45,000     40%       2021-06-01 2021-12-31
2022  OK       96,000     95%       2022-01-01 2022-12-31
2023  OK       96,000     98%       2023-01-01 2023-12-31
2024  OK       96,000     97%       2024-01-01 2024-12-31
2025  PARTIAL  50,000     52%       2025-01-01 2025-06-15
```

**If data is only 2022+**: This is a **broker limitation**, not a script bug.

---

## Solution: Use Improved Script

Replace the original `extract_historical_data.py` with `extract_historical_data_v2.py`:

```bash
# Backup original
copy extract_historical_data.py extract_historical_data_BACKUP.py

# Use improved version
python extract_historical_data_v2.py XAUUSD
```

### What's Better in v2:

| Issue | Original | v2 |
|-------|----------|-----|
| Chunk size | 1 year | 6 months (smaller = more reliable) |
| Error handling | Silent skip | Shows error + retries |
| Visibility | 1 line per year | Detailed progress + coverage % |
| Retry logic | None | Automatic retry (2 attempts) |
| Validation | None | Checks for data gaps |

### Expected Output:

```
MT5 initialized
Symbol: XAUUSD
Requesting data range: 2018-01-01 00:00:00+00:00 -> 2026-01-01 00:00:00+00:00
Chunk size: 0.5 years

Starting download loop:

  [1/2] Fetching 2018-01-01 -> 2018-07-02
    OK: 28,541 bars (2018-01-01 -> 2018-07-02), 98.2% coverage
  [1/2] Fetching 2018-07-02 -> 2019-01-02
    OK: 29,584 bars (2018-07-02 -> 2019-01-02), 98.5% coverage
  ...
  [1/2] Fetching 2025-07-02 -> 2026-01-01
    OK: 18,250 bars (2025-07-02 -> 2025-12-31), 72.1% coverage

Download summary: 15 success, 0 failed

Combined: 458,234 total bars
After dedup: 458,234 bars

Final dataset:
  Range: 2022-01-04 09:15:00+00:00 -> 2025-12-31 22:45:00+00:00
  Duration: 1459 days (4.0 years)
  Rows: 458,234

Saved: XAUUSD_16388.csv
Done
```

---

## If Still Getting 2022+ Only

This means **your broker's MT5 server only has data from 2022 onward**. Options:

1. **Switch broker** - Use one with longer history (e.g., IC Markets, Pepperstone)
2. **Use alternative data source** - Download from Yahoo Finance, Alpha Vantage, etc.
3. **Work with what you have** - Train on 2022-2025 data (4 years is workable)

---

## Manual Data Download Alternative

If MT5 is limited, download from a public source:

```python
import yfinance as yf
import pandas as pd

# Download XAUUSD from Yahoo Finance
df = yf.download("GC=F", start="2015-01-01", end="2025-12-31", interval="15m")
df.to_csv("XAUUSD_historical.csv")
```

Requires: `pip install yfinance`

---

## Files Created

1. **check_broker_data_availability.py** → Diagnostic tool (run first!)
2. **extract_historical_data_v2.py** → Improved download script
3. **EXTRACT_DATA_ANALYSIS.md** → Full technical analysis

---

## Next Steps

1. Run: `python check_broker_data_availability.py XAUUSD`
2. Note the year range available
3. If <2022, try alternative data source
4. If 2022+, use v2 script to download


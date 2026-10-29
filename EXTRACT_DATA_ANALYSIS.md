# extract_historical_data.py - Issues & Fixes

## Problem Analysis

### Issue 1: MAX_YEARS = 1 is the PRIMARY CULPRIT
**Location**: Line 17

```python
MAX_YEARS = 1
```

This limits each chunk to 1 year of data. The download loop (lines 84-98) works correctly, but MetaTrader5 has a **limitation on the number of bars per request** (~300,000 bars), not years.

**Why you only get data from 2022 onward**:
- START = 2018, END = 2026
- Loop fetches: [2018-01-01 to 2019-01-01], [2019-01-01 to 2020-01-01], etc.
- The EARLIEST chunks (2018-2019, 2019-2020, etc.) may be **silently failing** or returning incomplete data
- By the time MT5 starts returning data reliably, you're already in 2022+

### Issue 2: No Error Handling for Failed Chunks
**Location**: Lines 52-59

```python
if rates is None:
    print("Warning:", mt5.last_error())
    return None
```

The code warns about failures but **silently ignores them**. If chunks fail, you never know.

### Issue 3: No Data Validation
The script doesn't check:
- Whether each chunk actually contains data
- If the timestamp range matches the request
- Whether there are gaps in the downloaded history

### Issue 4: Chunking Strategy is Wrong
M15 over 1 year = ~32,000 bars (well within MT5's limit).
**Better strategy**: Chunk by **smaller time blocks** (e.g., 3 months) to catch connection issues early.

---

## Recommended Fixes

### Fix 1: Reduce MAX_YEARS (Quick Fix)
```python
MAX_YEARS = 0.5  # 6 months per request instead of 1 year
```

This forces smaller, more reliable chunks. If the MT5 connection drops, you'll catch it earlier.

### Fix 2: Add Verbose Debugging
```python
def fetch_chunk(symbol, timeframe, date_from, date_to):
    print(f"Fetching: {date_from} -> {date_to}")
    
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    
    if rates is None:
        print(f"  ERROR: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    
    if df.empty:
        print(f"  WARNING: Empty response (0 bars)")
        return None
    
    print(f"  OK: Got {len(df)} bars")  # ADD THIS
    
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
    
    return df
```

### Fix 3: Validate Downloaded Range
```python
# After downloading each chunk, verify the range
if df_chunk is not None:
    chunk_start = df_chunk["timestamp"].min()
    chunk_end = df_chunk["timestamp"].max()
    print(f"  Chunk timestamp range: {chunk_start} -> {chunk_end}")
    
    # Check for large gaps (missing data)
    expected_bars = int((chunk_end - chunk_start).total_seconds() / (15 * 60))
    actual_bars = len(df_chunk)
    coverage = actual_bars / max(expected_bars, 1)
    
    if coverage < 0.80:
        print(f"  WARNING: Only {coverage:.1%} coverage (expected ~{expected_bars}, got {actual_bars})")
    
    all_data.append(df_chunk)
```

### Fix 4: Retry Failed Chunks
```python
def fetch_chunk_with_retry(symbol, timeframe, date_from, date_to, max_retries=3):
    for attempt in range(max_retries):
        print(f"Fetching: {date_from} -> {date_to} (attempt {attempt+1}/{max_retries})")
        
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        
        if rates is None:
            print(f"  Retry due to: {mt5.last_error()}")
            continue
        
        df = pd.DataFrame(rates)
        if df.empty:
            print(f"  Empty response, retrying...")
            continue
        
        print(f"  Success: {len(df)} bars")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
        return df
    
    print(f"  FAILED after {max_retries} attempts")
    return None
```

---

## Root Cause: MetaTrader5 API Behavior

MT5's `copy_rates_range` has quirks:
1. **May fail silently**: Returns `None` or empty DataFrame without clear error
2. **Server-side limits**: Brokers limit historical data availability (often 2-5 years)
3. **Connection drops**: First request after MT5 terminal starts may fail
4. **Data gaps**: Weekends/holidays create gaps; MT5 doesn't fill them

**Your 2022 cutoff suggests**: Your broker's MT5 server only provides reliable data from 2022 onward.

---

## Testing the Fix

After applying fixes, run:
```bash
py extract_historical_data.py XAUUSD
```

Look for:
1. Print statements for each chunk (verbose output)
2. Coverage warnings (if <80%, data is incomplete)
3. Final row count

If data is still only from 2022+, **the broker's data retention is the limit**, not the script.

---

## Alternative: Check Broker Data Availability

Add this diagnostic script:

```python
import MetaTrader5 as mt5

mt5.initialize()
symbol = "XAUUSD"

# Test different date ranges
ranges = [
    ("2024-01-01", "2024-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2020-01-01", "2020-12-31"),
]

for start_str, end_str in ranges:
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start, end)
    count = len(rates) if rates is not None else 0
    print(f"{start_str} to {end_str}: {count} bars")

mt5.shutdown()
```

Run this to identify exactly which years are available on your broker.

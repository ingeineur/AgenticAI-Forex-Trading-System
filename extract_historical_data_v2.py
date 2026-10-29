import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import sys
import time

# ============================================================
# CONFIG
# ============================================================

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
TF_STR = sys.argv[2] if len(sys.argv) > 2 else "M15"

TF_MAP = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
if TF_STR not in TF_MAP:
    print(f"ERROR: Unsupported timeframe '{TF_STR}'. Choose from: {list(TF_MAP.keys())}")
    sys.exit(1)

TIMEFRAME = TF_MAP[TF_STR]
TF_MINUTES = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}[TF_STR]

# Based on diagnostic, your broker only has data from 2022-03-03 onward.
START = datetime(2022, 3, 3, tzinfo=pytz.UTC)
END = datetime.now(tz=pytz.UTC)

# Smaller chunks for finer TFs, larger for coarser
MAX_YEARS = {15: 0.5, 60: 1, 240: 2, 1440: 3}[TF_MINUTES]

# Optional output filename override (3rd positional arg)
OUTPUT_FILE = sys.argv[3] if len(sys.argv) > 3 else f"{SYMBOL}_{TF_STR}_combined.csv"

# Retry config
MAX_RETRIES = 2
RETRY_DELAY = 1  # seconds


# ============================================================
# INIT MT5
# ============================================================

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

print("MT5 initialized")

symbol_info = mt5.symbol_info(SYMBOL)

if symbol_info is None:
    raise RuntimeError(f"Symbol not found: {SYMBOL}")

if not symbol_info.visible:
    mt5.symbol_select(SYMBOL, True)

print(f"Symbol: {SYMBOL}")
print(f"Requesting data range: {START} -> {END}")
print(f"Chunk size: {MAX_YEARS} years")
print()


# ============================================================
# HELPERS
# ============================================================

def add_years(dt, years):
    """Safe year increment (handles leap years and fractional years)."""
    return dt + timedelta(days=round(years * 365.25))


def fetch_chunk(symbol, timeframe, date_from, date_to, attempt=1):
    """Fetch a single chunk with retry logic and validation."""
    
    print(f"  [{attempt}/{MAX_RETRIES}] Fetching {date_from.date()} -> {date_to.date()}")
    
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    
    # Handle errors
    if rates is None:
        error_code = mt5.last_error()
        print(f"    ERROR: {error_code}")
        
        if attempt < MAX_RETRIES:
            print(f"    Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            return fetch_chunk(symbol, timeframe, date_from, date_to, attempt=attempt+1)
        else:
            print(f"    FAILED after {MAX_RETRIES} attempts")
            return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    if df.empty:
        print(f"    WARNING: Empty response (0 bars)")
        return None
    
    # Validate coverage
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    chunk_start = df["time"].min()
    chunk_end = df["time"].max()
    
    expected_bars = int((date_to - date_from).total_seconds() / (TF_MINUTES * 60))
    actual_bars = len(df)
    coverage = actual_bars / max(expected_bars, 1)
    
    print(f"    OK: {actual_bars} bars ({chunk_start.date()} -> {chunk_end.date()}), {coverage:.1%} coverage")
    
    if coverage < 0.70:
        print(f"    WARNING: Low coverage detected ({coverage:.1%})")
    
    # Rename columns
    df = df.rename(columns={
        "time": "timestamp",
        "tick_volume": "volume"
    })
    
    return df


# ============================================================
# BATCH DOWNLOAD LOOP
# ============================================================

print("Starting download loop:")
print()

all_data = []
chunks_failed = 0
chunks_success = 0

current_start = START

while current_start < END:
    
    current_end = min(add_years(current_start, MAX_YEARS), END)
    
    df_chunk = fetch_chunk(
        SYMBOL,
        TIMEFRAME,
        current_start,
        current_end
    )
    
    if df_chunk is not None:
        all_data.append(df_chunk)
        chunks_success += 1
    else:
        chunks_failed += 1
    
    current_start = current_end

print()
print(f"Download summary: {chunks_success} success, {chunks_failed} failed")
print()


# ============================================================
# COMBINE DATA
# ============================================================

if not all_data:
    mt5.shutdown()
    raise RuntimeError("No data downloaded. Check broker data availability.")

df = pd.concat(all_data, ignore_index=True)
print(f"Combined: {len(df)} total bars")


# ============================================================
# CLEANING
# ============================================================

# Remove duplicates
df = df.drop_duplicates(subset=["timestamp"])
print(f"After dedup: {len(df)} bars")

# Sort
df = df.sort_values("timestamp").reset_index(drop=True)

# Select columns
required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
available_cols = [col for col in required_cols if col in df.columns]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"WARNING: Missing columns: {missing_cols}")

df = df[available_cols]

# Add missing columns with defaults if needed
if "spread" not in df.columns:
    df["spread"] = 0.0
if "real_volume" not in df.columns:
    df["real_volume"] = 0.0

df = df[[
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spread",
    "real_volume"
]]


# ============================================================
# FINAL VALIDATION
# ============================================================

date_min = df["timestamp"].min()
date_max = df["timestamp"].max()
date_range = date_max - date_min

print()
print(f"Final dataset:")
print(f"  Range: {date_min} -> {date_max}")
print(f"  Duration: {date_range.days} days ({date_range.days / 365.25:.1f} years)")
print(f"  Rows: {len(df)}")
print()


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print("Done")


# ============================================================
# SHUTDOWN
# ============================================================

mt5.shutdown()

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import sys

# ============================================================
# CONFIG
# ============================================================

SYMBOL = sys.argv[1]
TIMEFRAME = mt5.TIMEFRAME_M15

START = datetime(2018, 1, 1, tzinfo=pytz.UTC)
END = datetime(2026, 1, 1, tzinfo=pytz.UTC)

MAX_YEARS = 1

OUTPUT_FILE = f"{SYMBOL}_{TIMEFRAME}.csv"


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


# ============================================================
# HELPERS
# ============================================================

def add_years(dt, years):
    """Safe year increment (handles leap years)."""
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        return dt.replace(month=2, day=28, year=dt.year + years)


def fetch_chunk(symbol, timeframe, date_from, date_to):
    print(f"Fetching: {date_from} -> {date_to}")

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None:
        print("Warning:", mt5.last_error())
        return None

    df = pd.DataFrame(rates)

    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df = df.rename(columns={
        "time": "timestamp",
        "tick_volume": "volume"
    })

    return df


# ============================================================
# BATCH DOWNLOAD LOOP (MAX 3 YEARS PER REQUEST)
# ============================================================

all_data = []

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

    current_start = current_end


# ============================================================
# COMBINE DATA
# ============================================================

if not all_data:
    mt5.shutdown()
    raise RuntimeError("No data downloaded")

df = pd.concat(all_data, ignore_index=True)


# ============================================================
# CLEANING
# ============================================================

df = df.drop_duplicates(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

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
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Total rows: {len(df)}")


# ============================================================
# SHUTDOWN
# ============================================================

mt5.shutdown()
print("Done")
import numpy as np
import pandas as pd
import sys

# ============================================================
# LOAD DATA
# ============================================================

# Expected columns:
# timestamp, open, high, low, close, volume

df = pd.read_csv(sys.argv[1])

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# ATR
# ============================================================

def compute_atr(dataframe, period=14):

    high = dataframe["high"]
    low = dataframe["low"]
    close = dataframe["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    return atr


df["atr"] = compute_atr(df)


# ============================================================
# SWING HIGHS / LOWS
# ============================================================

LOOKBACK = 10

df["swing_high"] = (
    df["high"]
    .rolling(LOOKBACK)
    .max()
)

df["swing_low"] = (
    df["low"]
    .rolling(LOOKBACK)
    .min()
)


# ============================================================
# LIQUIDITY SWEEP DETECTION
# ============================================================

# ------------------------------------------------------------
# SHORT SETUP (buy-side liquidity sweep)
#
# Candle sweeps ABOVE prior swing high
# then closes back BELOW it
#
# Expect reversal downward
# ------------------------------------------------------------

df["entry_short"] = (
    (df["high"] > df["swing_high"].shift(1)) &
    (df["close"] < df["swing_high"].shift(1))
)

# ------------------------------------------------------------
# LONG SETUP (sell-side liquidity sweep)
#
# Candle sweeps BELOW prior swing low
# then closes back ABOVE it
#
# Expect reversal upward
# ------------------------------------------------------------

df["entry_long"] = (
    (df["low"] < df["swing_low"].shift(1)) &
    (df["close"] > df["swing_low"].shift(1))
)


# ============================================================
# OPTIONAL FILTERS
# ============================================================

# Candle rejection strength
body = (df["close"] - df["open"]).abs()
candle_range = (df["high"] - df["low"]).replace(0, np.nan)

df["rejection_ratio"] = (
    candle_range / body.replace(0, np.nan)
)

# Volume spike
df["volume_ma"] = df["volume"].rolling(20).mean()

df["volume_spike"] = (
    df["volume"] > df["volume_ma"] * 1.5
)

# Enable filters if desired
USE_VOLUME_FILTER = False
USE_REJECTION_FILTER = False

if USE_VOLUME_FILTER:
    df["entry_long"] &= df["volume_spike"]
    df["entry_short"] &= df["volume_spike"]

if USE_REJECTION_FILTER:
    df["entry_long"] &= df["rejection_ratio"] > 2
    df["entry_short"] &= df["rejection_ratio"] > 2


# ============================================================
# TP / SL LABEL GENERATION
# ============================================================

TP_MULT = 2.0
SL_MULT = 1.0
MAX_HORIZON = 30

labels = np.full(len(df), np.nan)

high = df["high"].values
low = df["low"].values
close = df["close"].values
atr = df["atr"].values

entry_long = df["entry_long"].values
entry_short = df["entry_short"].values


for i in range(len(df) - MAX_HORIZON):

    if np.isnan(atr[i]) or atr[i] == 0:
        continue

    # ========================================================
    # LONG
    # ========================================================

    if entry_long[i]:

        entry_price = close[i]

        tp = entry_price + (TP_MULT * atr[i])
        sl = entry_price - (SL_MULT * atr[i])

        for j in range(i + 1, i + MAX_HORIZON):

            if high[j] >= tp:
                labels[i] = 1
                break

            if low[j] <= sl:
                labels[i] = 0
                break

    # ========================================================
    # SHORT
    # ========================================================

    elif entry_short[i]:

        entry_price = close[i]

        tp = entry_price - (TP_MULT * atr[i])
        sl = entry_price + (SL_MULT * atr[i])

        for j in range(i + 1, i + MAX_HORIZON):

            if low[j] <= tp:
                labels[i] = 1
                break

            if high[j] >= sl:
                labels[i] = 0
                break


# ============================================================
# FINAL DATASET
# ============================================================

df["label"] = labels

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr",
    "swing_high",
    "swing_low",
    "rejection_ratio",
    "volume_spike",
    "entry_long",
    "entry_short",
    "label"
]

df[output_columns].to_csv(
    "liquidity_sweep_dataset.csv",
    index=False
)

print("Saved dataset")
print("Rows:", len(df))

print(df["label"].value_counts(normalize=True))
import pandas as pd
import numpy as np
import sys


# ============================================================
# CONFIG
# ============================================================

TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0
MAX_HOLD_BARS = 20

EMA_FAST = 20
EMA_SLOW = 50
ATR_PERIOD = 14
RSI_PERIOD = 14
ROLLING_STD_PERIOD = 20

DROP_NA = True


# ============================================================
# LOAD DATA
# ============================================================

# Required columns:
# timestamp, open, high, low, close, volume

df = pd.read_csv(sys.argv[1])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ------------------------
# EMA Features
# ------------------------

df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

df["ema_distance"] = (
    (df["ema_fast"] - df["ema_slow"]) / df["close"]
)

df["ema_slope"] = df["ema_fast"].diff(5)


# ------------------------
# ATR
# ------------------------

high_low = df["high"] - df["low"]
high_close = np.abs(df["high"] - df["close"].shift())
low_close = np.abs(df["low"] - df["close"].shift())

tr = pd.concat(
    [high_low, high_close, low_close],
    axis=1
).max(axis=1)

df["atr"] = tr.rolling(ATR_PERIOD).mean()

df["atr_ratio"] = (
    df["atr"] / df["atr"].rolling(50).mean()
)


# ------------------------
# RSI
# ------------------------

delta = df["close"].diff()

gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

gain = pd.Series(gain).rolling(RSI_PERIOD).mean()
loss = pd.Series(loss).rolling(RSI_PERIOD).mean()

rs = gain / (loss + 1e-9)

df["rsi"] = 100 - (100 / (1 + rs))


# ------------------------
# Z-Score
# ------------------------

rolling_mean = df["close"].rolling(ROLLING_STD_PERIOD).mean()
rolling_std = df["close"].rolling(ROLLING_STD_PERIOD).std()

df["zscore"] = (
    (df["close"] - rolling_mean) /
    (rolling_std + 1e-9)
)


# ------------------------
# Wick Ratio
# ------------------------

body = np.abs(df["close"] - df["open"])

upper_wick = df["high"] - np.maximum(df["open"], df["close"])
lower_wick = np.minimum(df["open"], df["close"]) - df["low"]

df["wick_ratio"] = (
    (upper_wick + lower_wick) /
    (body + 1e-9)
)


# ------------------------
# Range Compression
# ------------------------

rolling_high = df["high"].rolling(20).max()
rolling_low = df["low"].rolling(20).min()

df["range_compression"] = (
    (rolling_high - rolling_low) /
    (df["atr"] + 1e-9)
)


# ------------------------
# Breakout Distance
# ------------------------

df["breakout_distance"] = (
    (df["close"] - rolling_high.shift(1)) /
    (df["atr"] + 1e-9)
)


# ------------------------
# Volume Impulse
# ------------------------

df["volume_impulse"] = (
    df["volume"] /
    (df["volume"].rolling(20).mean() + 1e-9)
)


# ============================================================
# LABEL GENERATION
# ============================================================

labels = []
entry_prices = []
tp_prices = []
sl_prices = []

for i in range(len(df)):

    if i + MAX_HOLD_BARS >= len(df):
        labels.append(np.nan)
        entry_prices.append(np.nan)
        tp_prices.append(np.nan)
        sl_prices.append(np.nan)
        continue

    entry = df.loc[i, "close"]
    atr = df.loc[i, "atr"]

    if np.isnan(atr):
        labels.append(np.nan)
        entry_prices.append(np.nan)
        tp_prices.append(np.nan)
        sl_prices.append(np.nan)
        continue

    tp = entry + TP_ATR_MULT * atr
    sl = entry - SL_ATR_MULT * atr

    label = np.nan

    future = df.iloc[i + 1:i + MAX_HOLD_BARS + 1]

    for _, row in future.iterrows():

        hit_tp = row["high"] >= tp
        hit_sl = row["low"] <= sl

        # Both touched same candle
        if hit_tp and hit_sl:
            label = np.nan
            break

        if hit_tp:
            label = 1
            break

        if hit_sl:
            label = 0
            break

    labels.append(label)
    entry_prices.append(entry)
    tp_prices.append(tp)
    sl_prices.append(sl)

df["label"] = labels
df["entry_price"] = entry_prices
df["tp_price"] = tp_prices
df["sl_price"] = sl_prices


# ============================================================
# CLEAN DATASET
# ============================================================

feature_columns = [
    "ema_distance",
    "ema_slope",
    "atr",
    "atr_ratio",
    "rsi",
    "zscore",
    "wick_ratio",
    "range_compression",
    "breakout_distance",
    "volume_impulse"
]

dataset = df[
    ["timestamp"] +
    feature_columns +
    ["label"]
].copy()

if DROP_NA:
    dataset = dataset.dropna()

dataset["label"] = dataset["label"].astype(int)


# ============================================================
# SAVE
# ============================================================

dataset.to_csv("tp_sl_dataset.csv", index=False)

print(dataset.head())
print()
print("Dataset shape:", dataset.shape)
print()
print(dataset["label"].value_counts(normalize=True))
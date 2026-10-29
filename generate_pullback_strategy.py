import pandas as pd
import numpy as np
import sys


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = sys.argv[1]

OUTPUT_CSV = "trend_pullback_dataset.csv"

EMA_PERIOD = 50
ATR_PERIOD = 14
RSI_PERIOD = 14

LOOKBACK_SWING = 10

RR = 2.0
SL_ATR_MULT = 1.0

MAX_HORIZON = 20


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# INDICATORS
# ============================================================

# ------------------------------------------------------------
# EMA
# ------------------------------------------------------------

df["ema"] = df["close"].ewm(
    span=EMA_PERIOD,
    adjust=False
).mean()

df["ema_slope"] = df["ema"].diff()

# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------

high_low = df["high"] - df["low"]

high_close = np.abs(
    df["high"] - df["close"].shift()
)

low_close = np.abs(
    df["low"] - df["close"].shift()
)

tr = pd.concat(
    [high_low, high_close, low_close],
    axis=1
).max(axis=1)

df["atr"] = tr.rolling(ATR_PERIOD).mean()

# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------

delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(RSI_PERIOD).mean()
avg_loss = loss.rolling(RSI_PERIOD).mean()

rs = avg_gain / avg_loss

df["rsi"] = 100 - (100 / (1 + rs))


# ============================================================
# SWING LEVELS
# ============================================================

df["swing_low"] = (
    df["low"]
    .rolling(LOOKBACK_SWING)
    .min()
)

df["swing_high"] = (
    df["high"]
    .rolling(LOOKBACK_SWING)
    .max()
)


# ============================================================
# PULLBACK FEATURES
# ============================================================

# Distance from EMA

df["ema_distance"] = (
    (df["close"] - df["ema"]) / df["atr"]
)

# Pullback depth

df["pullback_depth"] = (
    (df["ema"] - df["low"]) / df["atr"]
)

# Candle rejection

body = (
    df["close"] - df["open"]
).abs()

lower_wick = (
    df[["open", "close"]].min(axis=1)
    - df["low"]
)

upper_wick = (
    df["high"]
    - df[["open", "close"]].max(axis=1)
)

df["bullish_rejection"] = (
    lower_wick / (body + 1e-6)
)

df["bearish_rejection"] = (
    upper_wick / (body + 1e-6)
)

# Compression

df["range"] = (
    df["high"] - df["low"]
)

df["range_ma"] = (
    df["range"].rolling(20).mean()
)

df["compression"] = (
    df["range"] / df["range_ma"]
)

# Volume impulse

df["volume_ma"] = (
    df["volume"].rolling(20).mean()
)

df["volume_impulse"] = (
    df["volume"] / df["volume_ma"]
)


# ============================================================
# ENTRY CONDITIONS
# ============================================================

# ------------------------------------------------------------
# Long pullback
# ------------------------------------------------------------

trend_long = (
    (df["close"] > df["ema"]) &
    (df["ema_slope"] > 0)
)

pullback_long = (
    (df["low"] <= df["ema"]) &
    (df["rsi"] < 55)
)

trigger_long = (
    (df["close"] > df["high"].shift(1)) &
    (df["bullish_rejection"] > 1.5)
)

df["entry_long"] = (
    trend_long &
    pullback_long &
    trigger_long
)

# ------------------------------------------------------------
# Short pullback
# ------------------------------------------------------------

trend_short = (
    (df["close"] < df["ema"]) &
    (df["ema_slope"] < 0)
)

pullback_short = (
    (df["high"] >= df["ema"]) &
    (df["rsi"] > 45)
)

trigger_short = (
    (df["close"] < df["low"].shift(1)) &
    (df["bearish_rejection"] > 1.5)
)

df["entry_short"] = (
    trend_short &
    pullback_short &
    trigger_short
)


# ============================================================
# LABEL GENERATION
# ============================================================

labels = []

for i in range(len(df)):

    if not (
        df.loc[i, "entry_long"] or
        df.loc[i, "entry_short"]
    ):
        labels.append(np.nan)
        continue

    entry_price = df.loc[i, "close"]

    atr = df.loc[i, "atr"]

    if pd.isna(atr):
        labels.append(np.nan)
        continue

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if df.loc[i, "entry_long"]:

        sl = entry_price - atr * SL_ATR_MULT

        risk = entry_price - sl

        tp = entry_price + risk * RR

        label = 0

        for j in range(i + 1, min(i + 1 + MAX_HORIZON, len(df))):

            low = df.loc[j, "low"]
            high = df.loc[j, "high"]

            if low <= sl:
                label = 0
                break

            if high >= tp:
                label = 1
                break

        labels.append(label)

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    elif df.loc[i, "entry_short"]:

        sl = entry_price + atr * SL_ATR_MULT

        risk = sl - entry_price

        tp = entry_price - risk * RR

        label = 0

        for j in range(i + 1, min(i + 1 + MAX_HORIZON, len(df))):

            low = df.loc[j, "low"]
            high = df.loc[j, "high"]

            if high >= sl:
                label = 0
                break

            if low <= tp:
                label = 1
                break

        labels.append(label)

df["label"] = labels


# ============================================================
# CLEAN
# ============================================================

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(df["label"].value_counts(normalize=True))

print(f"\nSaved: {OUTPUT_CSV}")
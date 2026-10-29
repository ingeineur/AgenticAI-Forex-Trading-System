import pandas as pd
import numpy as np
import sys


# ============================================================
# CONFIG
# ============================================================

TP_MULT = 1.5   # take profit in ATR multiples
SL_MULT = 1.0   # stop loss in ATR multiples
LOOKAHEAD = 30  # bars to check TP/SL

EMA_FAST = 9
EMA_SLOW = 21


# ============================================================
# INDICATORS
# ============================================================

def compute_indicators(df):
    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(span=EMA_FAST).mean()
    df["ema21"] = df["close"].ewm(span=EMA_SLOW).mean()

    df["ema_distance"] = df["ema9"] - df["ema21"]
    df["ema_slope"] = df["ema9"].diff()

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["atr_ratio"] = df["atr"] / df["close"]

    # Volume features
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_impulse"] = df["volume"] / (df["volume_ma"] + 1e-9)

    return df


# ============================================================
# ENTRY SIGNALS (EMA CROSS)
# ============================================================

def generate_entries(df):
    df = df.copy()

    df["entry_long"] = (
        (df["ema9"] > df["ema21"]) &
        (df["ema9"].shift(1) <= df["ema21"].shift(1))
    ).astype(int)

    df["entry_short"] = (
        (df["ema9"] < df["ema21"]) &
        (df["ema9"].shift(1) >= df["ema21"].shift(1))
    ).astype(int)

    return df


# ============================================================
# LABEL GENERATION (TP BEFORE SL)
# ============================================================

def generate_labels(df):
    df = df.copy()
    labels = np.zeros(len(df))

    for i in range(len(df) - LOOKAHEAD):
        if df["entry_long"].iloc[i] == 1:
            entry = df["close"].iloc[i]
            atr = df["atr"].iloc[i]

            tp = entry + TP_MULT * atr
            sl = entry - SL_MULT * atr

            for j in range(1, LOOKAHEAD):
                if df["high"].iloc[i + j] >= tp:
                    labels[i] = 1
                    break
                if df["low"].iloc[i + j] <= sl:
                    labels[i] = 0
                    break

        elif df["entry_short"].iloc[i] == 1:
            entry = df["close"].iloc[i]
            atr = df["atr"].iloc[i]

            tp = entry - TP_MULT * atr
            sl = entry + SL_MULT * atr

            for j in range(1, LOOKAHEAD):
                if df["low"].iloc[i + j] <= tp:
                    labels[i] = 1
                    break
                if df["high"].iloc[i + j] >= sl:
                    labels[i] = 0
                    break

    df["label"] = labels
    return df


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(df):
    df = compute_indicators(df)
    df = generate_entries(df)
    df = generate_labels(df)

    df = df.dropna().reset_index(drop=True)

    return df

df = pd.read_csv(sys.argv[1])
df = build_dataset(df)
df.to_csv("ema_crossover_dataset.csv", index=False)
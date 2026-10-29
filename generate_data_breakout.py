import numpy as np
import pandas as pd
import sys


# ============================================================
# ATR CALCULATION
# ============================================================

def compute_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


# ============================================================
# BREAKOUT ENTRY LOGIC
# ============================================================

def add_breakout_entries(df, lookback=20, compression_threshold=0.8):
    df = df.copy()

    df["range_high"] = df["high"].rolling(lookback).max()
    df["range_low"] = df["low"].rolling(lookback).min()
    df["range_size"] = df["range_high"] - df["range_low"]

    df["range_ma"] = df["range_size"].rolling(lookback).mean()
    df["range_compression"] = df["range_size"] / (df["range_ma"] + 1e-9)

    df["entry_long"] = (
        (df["close"] > df["range_high"].shift(1)) &
        (df["range_compression"] < compression_threshold)
    )

    df["entry_short"] = (
        (df["close"] < df["range_low"].shift(1)) &
        (df["range_compression"] < compression_threshold)
    )

    return df


# ============================================================
# TP/SL LABEL GENERATOR
# ============================================================

def generate_breakout_tp_sl_labels(
    df,
    atr_period=14,
    tp_mult=2.0,
    sl_mult=1.0,
    max_horizon=50,
    require_entry=True
):
    df = df.copy()

    # -------------------------
    # ATR
    # -------------------------
    df["atr"] = compute_atr(df, atr_period)

    # -------------------------
    # Breakout entries
    # -------------------------
    df = add_breakout_entries(df)

    # -------------------------
    # Arrays for speed
    # -------------------------
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = df["atr"].values

    entry_long = df["entry_long"].values
    entry_short = df["entry_short"].values

    labels = np.full(len(df), np.nan)

    # ============================================================
    # SIMULATION LOOP
    # ============================================================
    for i in range(len(df) - max_horizon):

        if require_entry and not (entry_long[i] or entry_short[i]):
            continue

        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entry_price = close[i]

        # ============================
        # LONG TRADE
        # ============================
        if entry_long[i]:

            tp = entry_price + tp_mult * atr[i]
            sl = entry_price - sl_mult * atr[i]

            for j in range(i + 1, i + max_horizon):

                if high[j] >= tp:
                    labels[i] = 1
                    break

                if low[j] <= sl:
                    labels[i] = 0
                    break

        # ============================
        # SHORT TRADE
        # ============================
        elif entry_short[i]:

            tp = entry_price - tp_mult * atr[i]
            sl = entry_price + sl_mult * atr[i]

            for j in range(i + 1, i + max_horizon):

                if low[j] <= tp:
                    labels[i] = 1
                    break

                if high[j] >= sl:
                    labels[i] = 0
                    break

    # -------------------------
    # Final dataset
    # -------------------------
    df["label"] = labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df


df = pd.read_csv(sys.argv[1])

dataset = generate_breakout_tp_sl_labels(
    df,
    atr_period=14,
    tp_mult=2.0,
    sl_mult=1.0,
    max_horizon=50
)

dataset.to_csv("ml_breakout_dataset.csv", index=False)
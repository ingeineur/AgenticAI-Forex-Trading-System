import pandas as pd
import numpy as np
import sys


# ============================================================
# LOAD OHLCV
# ============================================================

df = pd.read_csv(sys.argv[1])


# ============================================================
# PARAMETERS
# ============================================================

TP_PCT = 0.006   # 0.6%
SL_PCT = 0.003   # 0.3%

LOOKAHEAD = 20

ENTRY_PROBABILITY = 0.02
# 2% of bars become random entries


# ============================================================
# RANDOM ENTRY GENERATION
# ============================================================

np.random.seed(42)

df["random_entry"] = (
    np.random.rand(len(df)) < ENTRY_PROBABILITY
)

# Random long/short direction
df["direction"] = np.where(
    np.random.rand(len(df)) > 0.5,
    1,   # long
    -1   # short
)


# ============================================================
# LABELING
# ============================================================

labels = []

for i in range(len(df)):

    # Skip non-entry bars
    if not df["random_entry"].iloc[i]:
        labels.append(np.nan)
        continue

    entry_price = df["close"].iloc[i]

    direction = df["direction"].iloc[i]

    future = df.iloc[i+1 : i+1+LOOKAHEAD]

    if len(future) == 0:
        labels.append(np.nan)
        continue

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == 1:

        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        label = 0

        for _, row in future.iterrows():

            hit_tp = row["high"] >= tp_price
            hit_sl = row["low"] <= sl_price

            if hit_tp and hit_sl:
                label = 0
                break

            if hit_tp:
                label = 1
                break

            if hit_sl:
                label = 0
                break

        labels.append(label)

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    else:

        tp_price = entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 + SL_PCT)

        label = 0

        for _, row in future.iterrows():

            hit_tp = row["low"] <= tp_price
            hit_sl = row["high"] >= sl_price

            if hit_tp and hit_sl:
                label = 0
                break

            if hit_tp:
                label = 1
                break

            if hit_sl:
                label = 0
                break

        labels.append(label)


# ============================================================
# SAVE LABELS
# ============================================================

df["label"] = labels

dataset = df.dropna(subset=["label"]).copy()

dataset["label"] = dataset["label"].astype(int)


# ============================================================
# STATS
# ============================================================

wins = (dataset["label"] == 1).sum()
losses = (dataset["label"] == 0).sum()

winrate = wins / len(dataset)

print("=" * 50)
print(f"Trades     : {len(dataset)}")
print(f"Wins       : {wins}")
print(f"Losses     : {losses}")
print(f"Win Rate   : {winrate:.2%}")
print("=" * 50)


# ============================================================
# SAVE
# ============================================================

dataset.to_csv(
    "random_entry_dataset.csv",
    index=False
)

print("Saved: random_entry_dataset.csv")
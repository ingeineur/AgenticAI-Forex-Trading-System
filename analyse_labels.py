import pandas as pd


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_FILE = "tp_sl_dataset.csv"

df = pd.read_csv(DATASET_FILE)


# ============================================================
# BASIC STATS
# ============================================================

total_trades = len(df)

wins = (df["label"] == 1).sum()
losses = (df["label"] == 0).sum()

winrate = wins / total_trades if total_trades > 0 else 0
lossrate = losses / total_trades if total_trades > 0 else 0

wl_ratio = wins / losses if losses > 0 else float("inf")


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 50)

print(f"Total Trades : {total_trades}")

print(f"Wins         : {wins}")
print(f"Losses       : {losses}")

print()

print(f"Win Rate     : {winrate:.2%}")
print(f"Loss Rate    : {lossrate:.2%}")

print(f"Win/Loss     : {wl_ratio:.2f}")

print("=" * 50)


# ============================================================
# OPTIONAL: CHECK CLASS BALANCE
# ============================================================

print()
print("Label Distribution:")
print(df["label"].value_counts(normalize=True))
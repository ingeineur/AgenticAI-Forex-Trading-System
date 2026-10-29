import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from eval_models import run_phase2

import joblib

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(sys.argv[1])

# Ensure time ordering (CRITICAL)
df = df.sort_values("timestamp").reset_index(drop=True)

# Drop invalid rows
df = df.dropna()


# ============================================================
# FEATURES (IMPORTANT: DO NOT USE LABEL OR FUTURE FLAGS)
# ============================================================

feature_cols = [
    # price structure
    "open", "high", "low", "close",

    # volatility / structure
    "atr",
    "range_high",
    "range_low",
    "range_size",
    "range_ma",
    "range_compression",

    # microstructure
    "spread",
    "volume",
    "real_volume",

    # entry context (optional but useful)
    "entry_long",
    "entry_short"
]

X = df[feature_cols]
y = df["label"].astype(int)


# ============================================================
# TIME-BASED SPLIT (NO SHUFFLE)
# ============================================================

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]


# ============================================================
# MODEL (GOOD BASELINE FOR TABULAR DATA)
# ============================================================

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    random_state=42
)

model.fit(X_train, y_train)

# save model
joblib.dump({"model": model, "features": feature_cols}, "bo_model.pkl")
print("Model saved to bo_model.pkl")

# ============================================================
# PREDICTIONS
# ============================================================

proba = model.predict_proba(X_test)[:, 1]
pred = (proba > 0.5).astype(int)

print("\nTrain:", len(X_train))
print("Test :", len(X_test))

# ============================================================
# METRICS
# ============================================================

print("\n=== RESULTS ===")
print("AUC:", roc_auc_score(y_test, proba))

print("\nClassification Report:")
print(classification_report(y_test, pred))


# ============================================================
# PROBABILITY BUCKET TEST (VERY IMPORTANT)
# ============================================================

test_df = df.iloc[split_idx:].copy()
test_df["prob"] = proba

results = run_phase2(test_df)

print("\n=== PROBABILITY BUCKET STATS ===")
print(
    test_df.groupby(pd.qcut(test_df["prob"], 5, duplicates="drop"))["label"].mean()
)

# ============================================================
# HIGH PROBABILITY TRADE ANALYSIS
# ============================================================

HIGH_PROB_THRESHOLD = 0.50  # adjust: 0.6 / 0.7 / 0.8

high_prob_trades = test_df[test_df["prob"] >= HIGH_PROB_THRESHOLD]

print("\n=== HIGH PROBABILITY TRADE STATS ===")

print("Threshold:", HIGH_PROB_THRESHOLD)
print("Total high-prob trades:", len(high_prob_trades))

if len(high_prob_trades) > 0:
    winrate = high_prob_trades["label"].mean()

    print("Winrate:", round(winrate * 100, 2), "%")
    print("Wins:", int(high_prob_trades["label"].sum()))
    print("Losses:", int(len(high_prob_trades) - high_prob_trades["label"].sum()))
else:
    print("No trades found at this threshold")
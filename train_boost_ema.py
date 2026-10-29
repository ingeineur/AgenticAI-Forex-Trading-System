import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from eval_models import run_phase2

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ema_crossover_dataset.csv")

df = df.sort_values("timestamp").reset_index(drop=True)
df = df.dropna()


# ============================================================
# FEATURES
# ============================================================

feature_cols = [
    "ema_distance",
    "ema_slope",
    "macd",
    "macd_hist",
    "rsi",
    "atr",
    "atr_ratio",
    "volume_impulse",
    "entry_long",
    "entry_short"
]

X = df[feature_cols]
y = df["label"].astype(int)


# ============================================================
# TIME SPLIT (NO SHUFFLE)
# ============================================================

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# ============================================================
# MODEL (XGBOOST BASELINE)
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

proba = model.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

print("\n=== RESULTS ===")
print("AUC:", roc_auc_score(y_test, proba))

pred = (proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, pred))


split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy()
test_df["prob"] = proba

results = run_phase2(test_df)
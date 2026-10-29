import pandas as pd
import numpy as np
from eval_models import run_phase2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("liquidity_sweep_dataset.csv")

print("Dataset size:", len(df))


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Distance from swing levels
# ------------------------------------------------------------

df["dist_to_swing_high"] = (
    (df["swing_high"] - df["close"]) / df["atr"]
)

df["dist_to_swing_low"] = (
    (df["close"] - df["swing_low"]) / df["atr"]
)

# ------------------------------------------------------------
# Candle body strength
# ------------------------------------------------------------

df["body_size"] = (
    (df["close"] - df["open"]).abs()
)

df["body_atr_ratio"] = (
    df["body_size"] / df["atr"]
)

# ------------------------------------------------------------
# Upper / lower wick
# ------------------------------------------------------------

df["upper_wick"] = (
    df["high"] - df[["open", "close"]].max(axis=1)
)

df["lower_wick"] = (
    df[["open", "close"]].min(axis=1) - df["low"]
)

df["upper_wick_ratio"] = (
    df["upper_wick"] / df["atr"]
)

df["lower_wick_ratio"] = (
    df["lower_wick"] / df["atr"]
)

# ------------------------------------------------------------
# Directional setup
# ------------------------------------------------------------

df["setup_direction"] = np.where(
    df["entry_long"] == True,
    1,
    -1
)

# ------------------------------------------------------------
# Volume normalization
# ------------------------------------------------------------

df["volume_ma"] = (
    df["volume"].rolling(20).mean()
)

df["volume_ratio"] = (
    df["volume"] / df["volume_ma"]
)


# ============================================================
# CLEANING
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna()

print("Clean dataset:", len(df))


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "atr",
    "dist_to_swing_high",
    "dist_to_swing_low",
    "rejection_ratio",
    "body_atr_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "volume_ratio",
    "setup_direction"
]

TARGET = "label"

X = df[FEATURES]

y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# shuffle=False preserves chronological order

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTING
# ============================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)


# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob > 0.5).astype(int)


# ============================================================
# METRICS
# ============================================================

auc = roc_auc_score(y_test, y_prob)

print("\nAUC:", auc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# ============================================================
# OPTIONAL:
# SAVE PREDICTIONS
# ============================================================

results = X_test.copy()

results["actual"] = y_test.values
results["prediction"] = y_pred
results["probability"] = y_prob

results.to_csv(
    "liquidity_sweep_predictions.csv",
    index=False
)

split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy()
test_df["prob"] = y_prob

results = run_phase2(test_df)

print("\nSaved predictions")
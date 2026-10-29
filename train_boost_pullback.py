import pandas as pd
import numpy as np
from eval_models import run_phase2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIG
# ============================================================

INPUT_DATASET = "trend_pullback_dataset.csv"
PROB_THRESHOLD = 0.55


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_DATASET)

print("Raw size:", len(df))


# ============================================================
# CLEAN
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Clean size:", len(df))


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ema_distance",
    "ema_slope",
    "atr",
    "rsi",
    "pullback_depth",
    "bullish_rejection",
    "bearish_rejection",
    "compression",
    "volume_impulse"
]

# Optional directional feature
df["direction"] = np.where(df["entry_long"] == True, 1, -1)
FEATURES.append("direction")

TARGET = "label"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# TIME SPLIT (NO SHUFFLE)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTrain:", len(X_train))
print("Test :", len(X_test))


# ============================================================
# CLASS IMBALANCE HANDLING
# ============================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos if pos > 0 else 1

print("\nscale_pos_weight:", scale_pos_weight)


# ============================================================
# MODEL (XGBOOST)
# ============================================================

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=5,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= PROB_THRESHOLD).astype(int)

# ============================================================
# METRICS
# ============================================================

auc = roc_auc_score(y_test, y_prob)

print("\n========================")
print("RESULTS")
print("========================")

print(f"AUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n========================")
print("FEATURE IMPORTANCE")
print("========================")

print(importance)


# ============================================================
# SAVE RESULTS
# ============================================================

results = X_test.copy()
results["actual"] = y_test.values
results["probability"] = y_prob
results["proba"] = y_prob
results["prediction"] = y_pred

results.to_csv("trend_pullback_xgb_predictions.csv", index=False)

print("\nSaved: trend_pullback_xgb_predictions.csv")


split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy()
test_df["prob"] = y_prob

results = run_phase2(test_df)

# ============================================================
# HIGH CONFIDENCE ANALYSIS
# ============================================================

high_conf = results[results["probability"] >= 0.70]

if len(high_conf) > 0:
    winrate = (high_conf["actual"] == high_conf["prediction"]).mean()

    print("\n========================")
    print("HIGH CONFIDENCE SET")
    print("========================")

    print("Trades :", len(high_conf))
    print(f"Winrate: {winrate:.2%}")
else:
    print("\nNo high-confidence trades found.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import sys

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(sys.argv[1])

df = df.dropna()


# ============================================================
# FEATURES
# ============================================================

features = [
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

X = df[features]
y = df["label"]


# ============================================================
# TIME-AWARE SPLIT (IMPORTANT)
# ============================================================

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# ============================================================
# MODEL
# ============================================================

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

proba = model.predict_proba(X_test)[:, 1]
pred = (proba > 0.5).astype(int)


# ============================================================
# EVALUATION
# ============================================================

print("AUC:", roc_auc_score(y_test, proba))

print("\nClassification Report:")
print(classification_report(y_test, pred))
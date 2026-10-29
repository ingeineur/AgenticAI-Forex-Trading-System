import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

TARGET_COL = "label"
PROB_COL = "prob"

TP_RR = 1.0   # assume TP = 1R for simplicity (you can refine later)
SL_RR = 1.0   # assume SL = -1R

THRESHOLDS = np.arange(0.50, 0.95, 0.05)


# ============================================================
# CORE METRICS
# ============================================================

def compute_metrics(df, threshold):
    d = df.copy()

    d["signal"] = (d[PROB_COL] >= threshold).astype(int)
    trades = d[d["signal"] == 1]

    n_trades = len(trades)

    if n_trades == 0:
        return {
            "threshold": threshold,
            "trades": 0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "avg_prob": np.nan
        }

    win_rate = trades[TARGET_COL].mean()

    losses = 1 - win_rate

    expectancy = (win_rate * TP_RR) - (losses * SL_RR)

    return {
        "threshold": threshold,
        "trades": n_trades,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "avg_prob": trades[PROB_COL].mean()
    }


# ============================================================
# THRESHOLD SWEEP
# ============================================================

def threshold_sweep(df):
    results = []

    for t in THRESHOLDS:
        results.append(compute_metrics(df, t))

    return pd.DataFrame(results)


# ============================================================
# PROBABILITY CALIBRATION (BINNING TEST)
# ============================================================

def probability_bins(df):
    bins = np.linspace(0.5, 1.0, 6)

    df["bin"] = pd.cut(df[PROB_COL], bins=bins)

    summary = df.groupby("bin")[TARGET_COL].agg(["mean", "count"])
    summary.columns = ["win_rate", "count"]

    return summary


# ============================================================
# REGIME ANALYSIS (OPTIONAL BUT CRITICAL)
# ============================================================

def regime_analysis(df, regime_col):
    results = []

    for regime in df[regime_col].unique():
        sub = df[df[regime_col] == regime]

        auc = roc_auc_score(sub[TARGET_COL], sub[PROB_COL]) if len(sub) > 10 else np.nan

        results.append({
            "regime": regime,
            "samples": len(sub),
            "auc": auc,
            "win_rate_0.7": sub[sub[PROB_COL] > 0.7][TARGET_COL].mean()
        })

    return pd.DataFrame(results)


# ============================================================
# WALK-FORWARD SPLIT
# ============================================================

def walk_forward_eval(df, n_splits=3):
    size = len(df)
    chunk = size // n_splits

    results = []

    for i in range(n_splits):
        start = i * chunk
        end = (i + 1) * chunk

        test = df.iloc[start:end]

        auc = roc_auc_score(test[TARGET_COL], test[PROB_COL])

        results.append({
            "fold": i,
            "samples": len(test),
            "auc": auc,
            "win_rate_0.7": test[test[PROB_COL] > 0.7][TARGET_COL].mean()
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN RUN
# ============================================================

def run_phase2(df, regime_col=None):

    print("\n==============================")
    print(" PHASE 2 EVALUATION ENGINE")
    print("==============================")

    print("\nAUC:", roc_auc_score(df[TARGET_COL], df[PROB_COL]))

    # 1. Threshold sweep
    sweep_df = threshold_sweep(df)
    print("\n=== THRESHOLD SWEEP ===")
    print(sweep_df)

    # 2. Best threshold by expectancy
    best = sweep_df.loc[sweep_df["expectancy"].idxmax()]
    print("\nBEST THRESHOLD:", best.to_dict())

    # 3. Probability calibration
    print("\n=== PROBABILITY CALIBRATION ===")
    print(probability_bins(df))

    # 4. Walk-forward stability
    print("\n=== WALK FORWARD ===")
    print(walk_forward_eval(df))

    # 5. Regime analysis (if available)
    if regime_col:
        print("\n=== REGIME ANALYSIS ===")
        print(regime_analysis(df, regime_col))

    # 6. Plot trade density vs quality
    plt.figure()
    plt.plot(sweep_df["threshold"], sweep_df["trades"], label="Trades")
    plt.plot(sweep_df["threshold"], sweep_df["win_rate"], label="Win Rate")
    plt.plot(sweep_df["threshold"], sweep_df["expectancy"], label="Expectancy")
    plt.legend()
    plt.title("Threshold Sensitivity Curve")
    plt.show()

    return sweep_df
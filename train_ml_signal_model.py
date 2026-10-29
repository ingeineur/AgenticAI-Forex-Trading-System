#!/usr/bin/env python3
"""
Machine Learning Signal Generator
Trains a RandomForest to predict TP/SL probability directly from OHLCV data.
This replaces hand-coded price action rules with data-driven signals.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_ohlcv(csv_path):
    """Load OHLCV data."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def atr(df, period=14):
    """Calculate ATR."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def create_features(df):
    """Create normalized OHLCV features for ML."""
    eps = 1e-9
    df = df.copy()
    
    # ATR
    df["atr"] = atr(df)
    
    # Candle metrics
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    
    # Normalized by ATR
    df["body_atr"] = df["body"] / (df["atr"] + eps)
    df["range_atr"] = df["range"] / (df["atr"] + eps)
    df["upper_wick_atr"] = df["upper_wick"] / (df["atr"] + eps)
    df["lower_wick_atr"] = df["lower_wick"] / (df["atr"] + eps)
    
    # EMA metrics
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["close_to_ema20_atr"] = (df["close"] - df["ema20"]) / (df["atr"] + eps)
    df["close_to_ema50_atr"] = (df["close"] - df["ema50"]) / (df["atr"] + eps)
    df["ema20_slope_atr"] = df["ema20"].diff(5) / (df["atr"] + eps)
    df["ema50_slope_atr"] = df["ema50"].diff(5) / (df["atr"] + eps)
    
    # Volume metrics
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + eps)
    
    # Volatility
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    
    # Returns
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["ret_5"] = df["close"].pct_change(5).fillna(0.0)
    
    # Direction from previous bars
    df["prev_close_gt_open"] = (df["close"].shift(1) > df["open"].shift(1)).astype(float)
    df["prev_2_close_gt_open"] = (df["close"].shift(2) > df["open"].shift(2)).astype(float)
    
    # Swing metrics
    df["recent_high"] = df["high"].rolling(5).max()
    df["recent_low"] = df["low"].rolling(5).min()
    df["close_above_recent_avg"] = (df["close"] > (df["recent_high"] + df["recent_low"]) / 2).astype(float)
    
    return df


def generate_labels(df, tp_mult=2.0, sl_mult=1.0, max_horizon=50):
    """Generate TP/SL labels for training."""
    labels = np.full(len(df), np.nan)
    
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr_vals = df["atr"].to_numpy()
    
    for i in range(len(df) - max_horizon):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        
        entry = close[i]
        tp = entry + tp_mult * atr_vals[i]
        sl = entry - sl_mult * atr_vals[i]
        
        for j in range(i + 1, i + max_horizon + 1):
            if low[j] <= sl:
                labels[i] = 0
                break
            if high[j] >= tp:
                labels[i] = 1
                break
    
    return labels


def main():
    """Train ML signal generator."""
    
    print("="*70)
    print("MACHINE LEARNING SIGNAL GENERATOR")
    print("="*70)
    print()
    
    # Load data
    print("Step 1: Loading OHLCV data...")
    df = load_ohlcv("XAUUSD_M15.csv")
    print(f"  Loaded {len(df):,} bars")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Create features
    print("Step 2: Creating features...")
    df = create_features(df)
    
    # Generate labels
    print("Step 3: Generating labels (TP vs SL)...")
    df["label"] = generate_labels(df, tp_mult=2.0, sl_mult=1.0, max_horizon=50)
    
    # Filter to labeled data
    labeled_mask = ~df["label"].isna()
    df_labeled = df[labeled_mask].reset_index(drop=True)
    
    print(f"  Generated {len(df_labeled):,} labeled samples")
    print(f"  Win rate in raw data: {df_labeled['label'].mean():.1%}")
    print()
    
    # Select features for ML
    feature_cols = [
        "body_atr", "range_atr", "upper_wick_atr", "lower_wick_atr",
        "close_to_ema20_atr", "close_to_ema50_atr", "ema20_slope_atr", "ema50_slope_atr",
        "volume_ratio", "volatility", "ret_1", "ret_5",
        "prev_close_gt_open", "prev_2_close_gt_open", "close_above_recent_avg"
    ]
    
    X = df_labeled[feature_cols].fillna(0.0).astype(np.float32)
    y = df_labeled["label"].astype(int)
    
    print(f"Step 4: Training RandomForest model...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,}")
    print()
    
    # Cross-validation
    print("  Running cross-validation (5 folds)...")
    model_cv = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    print()
    
    # Full training
    print("Step 5: Training final model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    print("  Done")
    print()
    
    # Get probabilities
    print("Step 6: Evaluating predictions...")
    y_prob = model.predict_proba(X)[:, 1]
    
    # Test at different thresholds
    print("\n  Testing at different thresholds:")
    print("  " + "-"*60)
    print(f"  {'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Exp Value':<12}")
    print("  " + "-"*60)
    
    results = []
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = y_prob >= threshold
        if mask.sum() > 0:
            wr = y[mask].mean()
            ev = wr * 2.0 - (1 - wr) * 1.0
            results.append({
                "threshold": threshold,
                "trades": mask.sum(),
                "win_rate": wr,
                "expected_value": ev
            })
            print(f"  {threshold:<12.2f} {mask.sum():<10,} {wr:<12.1%} {ev:<12.2f}R")
    
    print("  " + "-"*60)
    print()
    
    # Find best threshold
    best = max(results, key=lambda x: x["win_rate"])
    print(f"Best threshold: {best['threshold']:.2f} ({best['win_rate']:.1%} WR)")
    
    if best["win_rate"] >= 0.50:
        print(f"SUCCESS! Model achieves {best['win_rate']:.1%} win rate")
        print(f"This is {best['win_rate'] - 0.344:.1%} better than baseline (34.4%)")
    else:
        print(f"Warning: Best win rate {best['win_rate']:.1%} is below 50%")
    
    print()
    
    # Save model
    print("Step 7: Saving model...")
    model_artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "best_threshold": best["threshold"],
        "cv_accuracy": cv_scores.mean(),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
    }
    
    with open("ml_signal_model.pkl", "wb") as f:
        pickle.dump(model_artifact, f)
    
    print("  Saved to: ml_signal_model.pkl")
    print()
    
    # Feature importance
    print("Step 8: Feature importance (top 10)...")
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feat, imp) in enumerate(importances[:10], 1):
        bar = "#" * int(imp * 100)
        print(f"  {i:2d}. {feat:25s} {imp:.3f} {bar}")
    
    print()
    print("="*70)
    print("READY TO USE")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. python generate_ml_signals.py XAUUSD_M15.csv ml_signal_model.pkl")
    print("  2. This creates: price_action_mtf_dataset_ml.py")
    print("  3. python price_action_mtf_dataset_ml.py XAUUSD_M15.csv --output dataset_ml.npz")
    print("  4. python train_lstm_price_action.py dataset_ml.npz")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train ML model on ALL bars (no signal filtering) to predict TP/SL outcome.
Uses balanced TP/SL ratios for best label distribution."""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def atr(arr, period=14):
    """ATR from numpy arrays"""
    tr = np.maximum(np.maximum(
        arr['high'] - arr['low'],
        np.abs(arr['high'] - np.roll(arr['close'], 1))
    ), np.abs(arr['low'] - np.roll(arr['close'], 1)))
    return pd.Series(tr).rolling(period).mean().to_numpy()


def create_features_from_struct(arr):
    """Create ML features from structured numpy array"""
    n = len(arr)
    eps = 1e-9
    
    o, h, l, c, v = arr['open'], arr['high'], arr['low'], arr['close'], arr['volume']
    
    atr_vals = atr(arr, 14)
    
    features = np.zeros((n, 20), dtype=np.float32)
    
    # body/range normalized
    features[:, 0] = np.abs(c - o) / (atr_vals + eps)  # body_atr
    features[:, 1] = (h - l) / (atr_vals + eps)  # range_atr
    features[:, 2] = (h - np.maximum(o, c)) / (atr_vals + eps)  # upper_wick
    features[:, 3] = (np.minimum(o, c) - l) / (atr_vals + eps)  # lower_wick
    
    # EMA features
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    
    features[:, 4] = (c - ema20) / (atr_vals + eps)
    features[:, 5] = (c - ema50) / (atr_vals + eps)
    
    ema20_slope = np.diff(ema20, prepend=ema20[0])
    ema50_slope = np.diff(ema50, prepend=ema50[0])
    features[:, 6] = ema20_slope / (atr_vals + eps)
    features[:, 7] = ema50_slope / (atr_vals + eps)
    
    # Volume
    vol_ma = pd.Series(v).rolling(20).mean().to_numpy()
    features[:, 8] = v / (vol_ma + eps)
    
    # Volatility
    returns = pd.Series(c).pct_change()
    features[:, 9] = returns.rolling(20).std().to_numpy()
    
    # Returns
    features[:, 10] = returns.fillna(0.0).to_numpy()
    features[:, 11] = c / np.roll(c, 5) - 1.0  # ret_5
    
    # Prior direction
    features[:, 12] = (np.roll(c, 1) > np.roll(o, 1)).astype(float)
    features[:, 13] = (np.roll(c, 2) > np.roll(o, 2)).astype(float)
    
    # Swing
    features[:, 14] = c > (pd.Series(h).rolling(5).max() + pd.Series(l).rolling(5).min()).to_numpy() / 2.0
    
    features[:, 15] = atr_vals / (c + eps)  # atr_pct
    
    # Candle direction and size
    features[:, 16] = ((c - o) > 0).astype(float)  # bullish
    features[:, 17] = np.abs(c - o) / (h - l + eps)  # body_to_range
    
    # Price vs recent range
    recent_high = pd.Series(h).rolling(10).max().to_numpy()
    recent_low = pd.Series(l).rolling(10).min().to_numpy()
    features[:, 18] = (c - recent_low) / (recent_high - recent_low + eps)  # position in range
    
    features[:, 19] = v / (pd.Series(v).rolling(50).mean().to_numpy() + eps)  # vol_ratio_50
    
    return features, atr_vals


def generate_labels(c, h, l, atr_vals, tp_mult, sl_mult, max_horizon=50):
    """Generate TP/SL labels"""
    n = len(c)
    labels = np.full(n, -1, dtype=np.int32)
    
    for i in range(n - max_horizon):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        
        tp = c[i] + tp_mult * atr_vals[i]
        sl = c[i] - sl_mult * atr_vals[i]
        
        for j in range(i + 1, i + max_horizon + 1):
            if l[j] <= sl:
                labels[i] = 0
                break
            if h[j] >= tp:
                labels[i] = 1
                break
    
    return labels


def main():
    print("="*70)
    print("ML DIRECT PREDICTION - NO SIGNAL FILTERING")
    print("="*70)
    print()
    
    df = pd.read_csv("XAUUSD_M15.csv")
    print(f"Loaded {len(df):,} bars")
    print()
    
    # Convert to structured array
    dtype = [('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')]
    arr = np.zeros(len(df), dtype=dtype)
    arr['open'] = df['open'].to_numpy()
    arr['high'] = df['high'].to_numpy()
    arr['low'] = df['low'].to_numpy()
    arr['close'] = df['close'].to_numpy()
    arr['volume'] = df['volume'].to_numpy()
    
    c, h, l = arr['close'], arr['high'], arr['low']
    
    # Test multiple TP/SL ratios
    ratio_tests = [
        (1.0, 1.0, "1R/1R"),   # Balanced, 50% WR
        (2.0, 2.0, "2R/2R"),   # Balanced, 51% WR
        (3.0, 3.0, "3R/3R"),   # Balanced, 52% WR
        (2.0, 1.0, "2R/1R"),   # Current, 34% WR
        (1.5, 1.5, "1.5R/1.5R"),  # Balanced
    ]
    
    for tp_mult, sl_mult, label in ratio_tests:
        print(f"\n--- Ratio: {label} (TP={tp_mult}R, SL={sl_mult}R) ---")
        
        # Create features
        features, atr_vals = create_features_from_struct(arr)
        
        # Generate labels
        labels = generate_labels(c, h, l, atr_vals, tp_mult, sl_mult, max_horizon=50)
        
        # Filter
        valid = labels >= 0
        X = features[valid]
        y = labels[valid]
        
        print(f"  Total labeled: {len(y):,}")
        print(f"  Win rate: {y.mean():.1%}")
        
        # Remove NaN rows
        nan_mask = ~np.isnan(X).any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
        
        print(f"  Clean samples: {len(y):,}")
        
        if len(y) < 1000:
            print(f"  SKIP: too few samples")
            continue
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        
        # Time-based split
        split = int(len(y) * 0.7)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        model.fit(X_train, y_train)
        
        # Test
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Various thresholds
        print(f"  Test set: {len(y_test):,} samples ({y_test.mean():.1%} WR)")
        print(f"  {'Thresh':<8} {'Trades':<10} {'Win Rate':<12} {'EV':<10}")
        print(f"  {'-'*40}")
        
        for thresh in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
            mask = y_prob >= thresh
            if mask.sum() > 0:
                wr = y_test[mask].mean()
                ev = wr * tp_mult - (1 - wr) * sl_mult
                print(f"  {thresh:<8.2f} {mask.sum():<10,} {wr:<12.1%} {ev:<10.2f}R")
        
        print()


if __name__ == "__main__":
    main()

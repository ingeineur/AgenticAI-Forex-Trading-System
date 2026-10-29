#!/usr/bin/env python3
"""Full BTCUSD analysis: ratio test + ML pipeline"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BTCUSD M15 / DAILY ANALYSIS")
print("="*70)

# Try M15 from H4 resampling or daily
for label, csv_path in [("Daily (1D)", "BTCUSD_1d_yahoo.csv"), ("H4 (4H)", "BTCUSD_16388.csv")]:
    print(f"\n--- {label} ---")
    
    try:
        df = pd.read_csv(csv_path)
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.sort_values(time_col).reset_index(drop=True)
        
        print(f"  Bars: {len(df):,}")
        print(f"  Range: {df[time_col].iloc[0]} to {df[time_col].iloc[-1]}")
        
        o = df['open'].to_numpy()
        h = df['high'].to_numpy()
        l = df['low'].to_numpy()
        c = df['close'].to_numpy()
        
        atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()
        
        # Test 3R/3R ratio
        wins, total = 0, 0
        max_horizon = min(50, len(df) // 2)
        
        for i in range(len(df) - max_horizon):
            if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
                continue
            tp = c[i] + 3.0 * atr_vals[i]
            sl = c[i] - 3.0 * atr_vals[i]
            for j in range(i + 1, i + max_horizon + 1):
                if l[j] <= sl:
                    total += 1
                    break
                if h[j] >= tp:
                    wins += 1
                    total += 1
                    break
        
        wr = wins / total if total > 0 else 0
        ev = wr * 3.0 - (1 - wr) * 3.0
        print(f"  3R/3R baseline: {wins}/{total} = {wr:.1%} WR, {ev:.2f}R EV")
        print(f"  Forecast horizon: {max_horizon} bars")
        
        # Also test 2R/1R
        wins2, total2 = 0, 0
        for i in range(len(df) - max_horizon):
            if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
                continue
            tp = c[i] + 2.0 * atr_vals[i]
            sl = c[i] - 1.0 * atr_vals[i]
            for j in range(i + 1, i + max_horizon + 1):
                if l[j] <= sl:
                    total2 += 1
                    break
                if h[j] >= tp:
                    wins2 += 1
                    total2 += 1
                    break
        
        wr2 = wins2 / total2 if total2 > 0 else 0
        ev2 = wr2 * 2.0 - (1 - wr2) * 1.0
        print(f"  2R/1R baseline: {wins2}/{total2} = {wr2:.1%} WR, {ev2:.2f}R EV")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print()
print("="*70)
print("ML PIPELINE (3R/3R on daily data)")
print("="*70)

# Run ML pipeline on daily data
df = pd.read_csv("BTCUSD_1d_yahoo.csv")
time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
df[time_col] = pd.to_datetime(df[time_col], utc=True)
df = df.sort_values(time_col).reset_index(drop=True)

# Use H4 data if available for more bars
print("Using daily data (4183 bars)")

# Feature engineering
eps = 1e-9
o, h, l, c, v = df['open'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), df['volume'].to_numpy()
atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()

# Labels with 3R/3R
labels = np.full(len(c), -1, dtype=np.int32)
max_horizon = min(50, len(c) // 2)
for i in range(len(c) - max_horizon):
    if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
        continue
    tp = c[i] + 3.0 * atr_vals[i]
    sl = c[i] - 3.0 * atr_vals[i]
    for j in range(i + 1, i + max_horizon + 1):
        if l[j] <= sl:
            labels[i] = 0
            break
        if h[j] >= tp:
            labels[i] = 1
            break

valid = labels >= 0
print(f"  Labeled samples: {valid.sum():,}")
print(f"  Win rate: {labels[valid].mean():.1%}")

# ML features
n = len(df)
feat = np.zeros((n, 20), dtype=np.float32)

feat[:, 0] = np.abs(c - o) / (atr_vals + eps)
feat[:, 1] = (h - l) / (atr_vals + eps)
feat[:, 2] = (h - np.maximum(o, c)) / (atr_vals + eps)
feat[:, 3] = (np.minimum(o, c) - l) / (atr_vals + eps)
ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
feat[:, 4] = (c - ema20) / (atr_vals + eps)
feat[:, 5] = (c - ema50) / (atr_vals + eps)
feat[:, 6] = np.diff(ema20, prepend=ema20[0]) / (atr_vals + eps)
feat[:, 7] = np.diff(ema50, prepend=ema50[0]) / (atr_vals + eps)
vol_ma = pd.Series(v).rolling(20).mean().to_numpy()
feat[:, 8] = v / (vol_ma + eps)
feat[:, 9] = pd.Series(c).pct_change().rolling(20).std().to_numpy()
feat[:, 10] = pd.Series(c).pct_change().fillna(0.0).to_numpy()
feat[:, 11] = c / np.roll(c, 5) - 1.0
feat[:, 12] = (np.roll(c, 1) > np.roll(o, 1)).astype(float)
feat[:, 13] = (np.roll(c, 2) > np.roll(o, 2)).astype(float)
feat[:, 14] = c > (pd.Series(h).rolling(5).max() + pd.Series(l).rolling(5).min()).to_numpy() / 2.0
feat[:, 15] = atr_vals / (c + eps)
feat[:, 16] = ((c - o) > 0).astype(float)
feat[:, 17] = np.abs(c - o) / (h - l + eps)
recent_high = pd.Series(h).rolling(10).max().to_numpy()
recent_low = pd.Series(l).rolling(10).min().to_numpy()
feat[:, 18] = (c - recent_low) / (recent_high - recent_low + eps)
feat[:, 19] = v / (pd.Series(v).rolling(50).mean().to_numpy() + eps)

# Train/test split
X_all = feat[valid]
y_all = labels[valid]

nan_mask = ~np.isnan(X_all).any(axis=1)
X_all = X_all[nan_mask]
y_all = y_all[nan_mask]

split = int(len(X_all) * 0.7)
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_all[:split], y_all[:split])

# Test
y_prob = model.predict_proba(X_all[split:])[:, 1]
y_test = y_all[split:]

print(f"\nML model results on test set:")
print(f"  {'Thresh':<8} {'Trades':<10} {'Win Rate':<12} {'EV':<10}")
print(f"  {'-'*40}")

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = y_prob >= thresh
    if mask.sum() > 0:
        wr = y_test[mask].mean()
        ev = wr * 3.0 - (1 - wr) * 3.0
        print(f"  {thresh:<8.2f} {mask.sum():<10,} {wr:<12.1%} {ev:<10.2f}R")

# Generate sequences
print(f"\nGenerating LSTM sequences from ML signals...")
all_probs = model.predict_proba(X_all)[:, 1]
signal_indices = np.where(all_probs >= 0.55)[0]
print(f"  ML signals (>=0.55): {len(signal_indices):,}")

# Generate sequences
lookback = 5
X_seqs, y_seqs = [], []
for idx in signal_indices:
    if idx < lookback or idx >= len(c) - max_horizon:
        continue
    ws = idx - lookback
    we = idx
    seq_o = (o[ws:we] / c[idx] - 1.0) * 100
    seq_h = (h[ws:we] / c[idx] - 1.0) * 100
    seq_l = (l[ws:we] / c[idx] - 1.0) * 100
    seq_c = (c[ws:we] / c[idx] - 1.0) * 100
    seq_v = v[ws:we] / (np.mean(v[ws:we]) + eps)
    seq = np.column_stack([seq_o, seq_h, seq_l, seq_c, seq_v]).astype(np.float32)
    
    # Label with 3R/3R
    atr_val = atr_vals[idx]
    if np.isnan(atr_val) or atr_val <= 0:
        continue
    tp = c[idx] + 3.0 * atr_val
    sl = c[idx] - 3.0 * atr_val
    for j in range(idx + 1, min(idx + max_horizon + 1, len(c))):
        if l[j] <= sl:
            y_seqs.append(0)
            X_seqs.append(seq)
            break
        if h[j] >= tp:
            y_seqs.append(1)
            X_seqs.append(seq)
            break

if len(X_seqs) > 0:
    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)
    print(f"  Sequences: {len(X_seqs)}")
    print(f"  Win rate: {y_seqs.mean():.1%}")
    print(f"  Shape: {X_seqs.shape}")
    
    # Split
    sp = int(len(X_seqs) * 0.8)
    np.savez("dataset_btc.npz", X_train=X_seqs[:sp], y_train=y_seqs[:sp],
             X_test=X_seqs[sp:], y_test=y_seqs[sp:])
    print(f"  Saved: dataset_btc.npz")
else:
    print("  No sequences generated")

print()
print("="*70)
print("DONE")

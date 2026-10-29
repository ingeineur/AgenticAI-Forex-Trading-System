#!/usr/bin/env python3
"""Full BTCUSD Broker M15 analysis (84K bars)"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BTCUSD BROKER M15 ANALYSIS (84,208 bars)")
print("="*70)

df = pd.read_csv("BTCUSD_15.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"\nData: {len(df):,} bars")
print(f"Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

o = df["open"].to_numpy()
h = df["high"].to_numpy()
l = df["low"].to_numpy()
c = df["close"].to_numpy()
v = df["volume"].to_numpy()

atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()

# Ratio test
print("\n--- Ratio Test ---")
for tp, sl, label in [(3.0, 3.0, "3R/3R"), (2.0, 1.0, "2R/1R"), (1.5, 1.5, "1.5R/1.5R"),
                       (2.0, 2.0, "2R/2R"), (1.0, 1.0, "1R/1R")]:
    wins, total = 0, 0
    for i in range(len(df) - 50):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        tp_lvl = c[i] + tp * atr_vals[i]
        sl_lvl = c[i] - sl * atr_vals[i]
        for j in range(i + 1, i + 51):
            if l[j] <= sl_lvl:
                total += 1; break
            if h[j] >= tp_lvl:
                wins += 1; total += 1; break
    wr = wins / total if total else 0
    ev = wr * tp - (1 - wr) * sl
    print(f"  {label}: {wins}/{total} = {wr:.1%} WR, {ev:.2f}R EV")

# ML Pipeline (3R/3R)
print("\n--- ML Pipeline (3R/3R) ---")
eps = 1e-9
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

# Labels 3R/3R
labels = np.full(n, -1, dtype=np.int32)
for i in range(n - 50):
    if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
        continue
    tp_lvl = c[i] + 3.0 * atr_vals[i]
    sl_lvl = c[i] - 3.0 * atr_vals[i]
    for j in range(i + 1, i + 51):
        if l[j] <= sl_lvl:
            labels[i] = 0; break
        if h[j] >= tp_lvl:
            labels[i] = 1; break

valid = labels >= 0
X_all = feat[valid]
y_all = labels[valid]
nan_ok = ~np.isnan(X_all).any(axis=1)
X_all = X_all[nan_ok]
y_all = y_all[nan_ok]

print(f"\n  Labeled: {len(y_all):,}, WR: {y_all.mean():.1%}")

# Train/test split
split = int(len(X_all) * 0.7)
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_all[:split], y_all[:split])

# Test
y_prob = model.predict_proba(X_all[split:])[:, 1]
y_test = y_all[split:]

print(f"\n  ML test results:")
print(f"  {'Thresh':<8} {'Trades':<10} {'Win Rate':<12} {'EV':<10}")
print(f"  {'-'*40}")
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = y_prob >= thresh
    if mask.sum() > 0:
        wr = y_test[mask].mean()
        ev = wr * 3.0 - (1 - wr) * 3.0
        print(f"  {thresh:<8.2f} {mask.sum():<10,} {wr:<12.1%} {ev:<10.2f}R")

# Generate sequences
print(f"\n  Generating sequences from ML signals...")
all_probs = model.predict_proba(X_all)[:, 1]
sig_idxs = np.where(all_probs >= 0.55)[0]
print(f"  ML signals (>=0.55): {len(sig_idxs):,}")

lookback = 5
X_seqs, y_seqs = [], []
for idx in sig_idxs:
    if idx < lookback or idx >= n - 50:
        continue
    ws, we = idx - lookback, idx
    sq = np.column_stack([
        (o[ws:we] / c[idx] - 1.0) * 100,
        (h[ws:we] / c[idx] - 1.0) * 100,
        (l[ws:we] / c[idx] - 1.0) * 100,
        (c[ws:we] / c[idx] - 1.0) * 100,
        v[ws:we] / (np.mean(v[ws:we]) + eps)
    ]).astype(np.float32)
    atr_v = atr_vals[idx]
    if np.isnan(atr_v) or atr_v <= 0:
        continue
    tp_lvl = c[idx] + 3.0 * atr_v
    sl_lvl = c[idx] - 3.0 * atr_v
    for j in range(idx + 1, min(idx + 51, n)):
        if l[j] <= sl_lvl:
            y_seqs.append(0); X_seqs.append(sq); break
        if h[j] >= tp_lvl:
            y_seqs.append(1); X_seqs.append(sq); break

if len(X_seqs) > 0:
    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)
    sp = int(len(X_seqs) * 0.8)
    np.savez("dataset_btc_broker_m15.npz", X_train=X_seqs[:sp], y_train=y_seqs[:sp],
             X_test=X_seqs[sp:], y_test=y_seqs[sp:])
    print(f"  Sequences: {len(X_seqs)}, WR: {y_seqs.mean():.1%}")
    print(f"  Train: {sp}, Test: {len(X_seqs)-sp}")
    print(f"  Saved: dataset_btc_broker_m15.npz")
else:
    print("  No sequences generated")

print("\n" + "="*70)
print("DONE")

#!/usr/bin/env python3
"""BTCUSD M15+H4 ML Pipeline - combines M15 patterns with H4 trend context"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BTCUSD M15+H4 ML PIPELINE")
print("="*70)

# Load M15 data
df = pd.read_csv("BTCUSD_15.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"\nM15: {len(df):,} bars")

# Load H4 data  
h4 = pd.read_csv("BTCUSD_16388.csv")
h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
h4 = h4.sort_values("timestamp").reset_index(drop=True)
h4 = h4.rename(columns={c: f"h4_{c}" for c in h4.columns if c != "timestamp"})
print(f"H4: {len(h4):,} bars")

# Merge H4 context into M15 (backward - each M15 bar gets the most recent H4 close)
print("Merging H4 context into M15...")
df["timestamp_round"] = df["timestamp"].dt.floor("4h")
h4["timestamp_round"] = h4["timestamp"].dt.floor("4h")
h4_cols = ["timestamp_round", "h4_open", "h4_high", "h4_low", "h4_close", "h4_volume"]
merged = df.merge(h4[h4_cols], on="timestamp_round", how="left")
merged = merged.dropna(subset=["h4_close"]).reset_index(drop=True)
print(f"After merge: {len(merged):,} bars")
print(f"Range: {merged['timestamp'].iloc[0]} to {merged['timestamp'].iloc[-1]}")

# Create combined features
eps = 1e-9
n = len(merged)
o = merged["open"].to_numpy()
h = merged["high"].to_numpy()
l = merged["low"].to_numpy()
c = merged["close"].to_numpy()
v = merged["volume"].to_numpy()
h4_o = merged["h4_open"].to_numpy()
h4_h = merged["h4_high"].to_numpy()
h4_l = merged["h4_low"].to_numpy()
h4_c = merged["h4_close"].to_numpy()

# M15 features (20)
atr_m15 = pd.Series(h - l).rolling(14).mean().to_numpy()
feat = np.zeros((n, 28), dtype=np.float32)

feat[:, 0] = np.abs(c - o) / (atr_m15 + eps)
feat[:, 1] = (h - l) / (atr_m15 + eps)
feat[:, 2] = (h - np.maximum(o, c)) / (atr_m15 + eps)
feat[:, 3] = (np.minimum(o, c) - l) / (atr_m15 + eps)
ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
feat[:, 4] = (c - ema20) / (atr_m15 + eps)
feat[:, 5] = (c - ema50) / (atr_m15 + eps)
feat[:, 6] = np.diff(ema20, prepend=ema20[0]) / (atr_m15 + eps)
feat[:, 7] = np.diff(ema50, prepend=ema50[0]) / (atr_m15 + eps)
vol_ma = pd.Series(v).rolling(20).mean().to_numpy()
feat[:, 8] = v / (vol_ma + eps)
feat[:, 9] = pd.Series(c).pct_change().rolling(20).std().to_numpy()
feat[:, 10] = pd.Series(c).pct_change().fillna(0.0).to_numpy()
feat[:, 11] = c / np.roll(c, 5) - 1.0
feat[:, 12] = (np.roll(c, 1) > np.roll(o, 1)).astype(float)
feat[:, 13] = (np.roll(c, 2) > np.roll(o, 2)).astype(float)
feat[:, 14] = c > (pd.Series(h).rolling(5).max() + pd.Series(l).rolling(5).min()).to_numpy() / 2.0
feat[:, 15] = atr_m15 / (c + eps)
feat[:, 16] = ((c - o) > 0).astype(float)
feat[:, 17] = np.abs(c - o) / (h - l + eps)
recent_high = pd.Series(h).rolling(10).max().to_numpy()
recent_low = pd.Series(l).rolling(10).min().to_numpy()
feat[:, 18] = (c - recent_low) / (recent_high - recent_low + eps)
feat[:, 19] = v / (pd.Series(v).rolling(50).mean().to_numpy() + eps)

# H4 features (8)
atr_h4 = pd.Series(h4_h - h4_l).rolling(14).mean().to_numpy()
feat[:, 20] = (h4_c - h4_o) / (atr_h4 + eps)  # H4 body/ATR
feat[:, 21] = (h4_h - h4_l) / (atr_h4 + eps)  # H4 range/ATR
feat[:, 22] = pd.Series(h4_c).ewm(span=20, adjust=False).mean().to_numpy()  # H4 EMA20
feat[:, 23] = pd.Series(h4_c).ewm(span=50, adjust=False).mean().to_numpy()  # H4 EMA50
feat[:, 24] = (h4_c - feat[:, 22]) / (atr_h4 + eps)  # H4 close to EMA20
feat[:, 25] = np.sign(feat[:, 22] - feat[:, 23])  # H4 trend direction
feat[:, 26] = (h4_c - pd.Series(h4_l).rolling(10).min().to_numpy()) / (pd.Series(h4_h).rolling(10).max().to_numpy() - pd.Series(h4_l).rolling(10).min().to_numpy() + eps)
feat[:, 27] = c / h4_c - 1.0  # M15 relative to H4 close

# Labels 3R/3R
labels = np.full(n, -1, dtype=np.int32)
for i in range(n - 50):
    if np.isnan(atr_m15[i]) or atr_m15[i] <= 0:
        continue
    tp_lvl = c[i] + 3.0 * atr_m15[i]
    sl_lvl = c[i] - 3.0 * atr_m15[i]
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

print(f"\nLabeled: {len(y_all):,}, WR: {y_all.mean():.1%}")

# Train/test
split = int(len(X_all) * 0.7)
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_all[:split], y_all[:split])

# Test
y_prob = model.predict_proba(X_all[split:])[:, 1]
y_test = y_all[split:]

print(f"\nML test results (M15+H4 features):")
print(f"  {'Thresh':<8} {'Trades':<10} {'Win Rate':<12} {'EV':<10}")
print(f"  {'-'*40}")
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = y_prob >= thresh
    if mask.sum() > 0:
        wr = y_test[mask].mean()
        ev = wr * 3.0 - (1 - wr) * 3.0
        print(f"  {thresh:<8.2f} {mask.sum():<10,} {wr:<12.1%} {ev:<10.2f}R")

# Also show M15-only results for comparison
print(f"\n  (XAUUSD M15 reference: 59.4% WR at 0.55 threshold)")
print(f"  (BTCUSD M15-only reference: 48.1% WR at 0.50 threshold)")

# Feature importance
importances = sorted(zip(
    ["body_atr","range_atr","upper_wick","lower_wick","close_ema20","close_ema50",
     "ema20_slope","ema50_slope","vol_ratio","volatility","ret_1","ret_5",
     "prev_direction","prev2_direction","swing_position","atr_pct","bullish",
     "body_range_ratio","range_position","vol_ratio_50",
     "h4_body","h4_range","h4_ema20","h4_ema50","h4_close_ema20","h4_trend",
     "h4_range_position","m15_h4_ratio"],
    model.feature_importances_), key=lambda x: x[1], reverse=True)

print(f"\n  Top features:")
for i, (name, imp) in enumerate(importances[:8], 1):
    print(f"    {i}. {name}: {imp:.3f}")

# Generate sequences for LSTM
print(f"\nGenerating LSTM sequences...")
all_probs = model.predict_proba(X_all)[:, 1]
sig_idxs = np.where(all_probs >= 0.55)[0]
print(f"  ML signals (>=0.55): {len(sig_idxs):,}")

lookback = 5
X_seqs, y_seqs = [], []
for idx in sig_idxs:
    if idx < lookback or idx >= n - 50:
        continue
    ws, we = idx - lookback, idx
    # M15 sequence (5 features)
    m15_seq = np.column_stack([
        (o[ws:we] / c[idx] - 1.0) * 100,
        (h[ws:we] / c[idx] - 1.0) * 100,
        (l[ws:we] / c[idx] - 1.0) * 100,
        (c[ws:we] / c[idx] - 1.0) * 100,
        v[ws:we] / (np.mean(v[ws:we]) + eps)
    ]).astype(np.float32)
    
    # H4 features for each lookback bar
    h4_seq = np.column_stack([
        np.full(lookback, feat[idx, 20]),  # H4 body
        np.full(lookback, feat[idx, 21]),  # H4 range
        np.full(lookback, feat[idx, 24]),  # H4 close to EMA20
        np.full(lookback, feat[idx, 25]),  # H4 trend
        np.full(lookback, feat[idx, 27]),  # M15/H4 ratio
    ]).astype(np.float32)
    
    combined = np.concatenate([m15_seq, h4_seq], axis=1)
    
    atr_v = atr_m15[idx]
    if np.isnan(atr_v) or atr_v <= 0:
        continue
    tp_lvl = c[idx] + 3.0 * atr_v
    sl_lvl = c[idx] - 3.0 * atr_v
    for j in range(idx + 1, min(idx + 51, n)):
        if l[j] <= sl_lvl:
            y_seqs.append(0); X_seqs.append(combined); break
        if h[j] >= tp_lvl:
            y_seqs.append(1); X_seqs.append(combined); break

if len(X_seqs) > 0:
    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)
    sp = int(len(X_seqs) * 0.8)
    np.savez("dataset_btc_m15h4.npz", X_train=X_seqs[:sp], y_train=y_seqs[:sp],
             X_test=X_seqs[sp:], y_test=y_seqs[sp:])
    print(f"  Sequences: {len(X_seqs)}, WR: {y_seqs.mean():.1%}")
    print(f"  Features: {X_seqs.shape[2]} (5 M15 + 5 H4)")
    print(f"  Shape: {X_seqs.shape}")
    print(f"  Saved: dataset_btc_m15h4.npz")

print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print("BTCUSD Daily (1D)  : 62.2% WR (ML+LSTM) ✅ BEST")
print("XAUUSD M15         : 58.3% WR (ML+LSTM) ✅ GOOD")  
print("BTCUSD M15+H4      : TBD (training below)")
print("BTCUSD M15 alone   : 50.6% WR (no edge)")

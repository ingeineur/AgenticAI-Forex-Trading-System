#!/usr/bin/env python3
"""XAUUSD M15+H4 Pipeline - same approach as BTCUSD M15+H4"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("XAUUSD M15+H4 ML PIPELINE")
print("="*70)

# Load M15
df = pd.read_csv("XAUUSD_M15.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"\nM15: {len(df):,} bars")

# Load H4
h4 = pd.read_csv("XAUUSD_16388.csv")
h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
h4 = h4.sort_values("timestamp").reset_index(drop=True)
h4 = h4.rename(columns={c: f"h4_{c}" for c in h4.columns if c != "timestamp"})
print(f"H4: {len(h4):,} bars")

# Merge H4 into M15
df["timestamp_round"] = df["timestamp"].dt.floor("4h")
h4["timestamp_round"] = h4["timestamp"].dt.floor("4h")
h4_cols = ["timestamp_round", "h4_open", "h4_high", "h4_low", "h4_close", "h4_volume"]
merged = df.merge(h4[h4_cols], on="timestamp_round", how="left")
merged = merged.dropna(subset=["h4_close"]).reset_index(drop=True)
print(f"After merge: {len(merged):,} bars")

n = len(merged)
o = merged["open"].to_numpy()
h = merged["high"].to_numpy()
l = merged["low"].to_numpy()
c = merged["close"].to_numpy()
v = merged["volume"].to_numpy()
h4_c = merged["h4_close"].to_numpy()
h4_o = merged["h4_open"].to_numpy()
h4_h = merged["h4_high"].to_numpy()
h4_l = merged["h4_low"].to_numpy()

eps = 1e-9

# Build 28 features (same as BTCUSD)
def build_features(n, o, h, l, c, v, h4_c, h4_o, h4_h, h4_l):
    eps = 1e-9
    atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()
    feat = np.zeros((n, 28), dtype=np.float32)
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
    feat[:, 8] = v / (pd.Series(v).rolling(20).mean().to_numpy() + eps)
    feat[:, 9] = pd.Series(c).pct_change().rolling(20).std().to_numpy()
    feat[:, 10] = pd.Series(c).pct_change().fillna(0.0).to_numpy()
    feat[:, 11] = c / np.roll(c, 5) - 1.0
    feat[:, 12] = (np.roll(c, 1) > np.roll(o, 1)).astype(float)
    feat[:, 13] = (np.roll(c, 2) > np.roll(o, 2)).astype(float)
    feat[:, 14] = c > (pd.Series(h).rolling(5).max() + pd.Series(l).rolling(5).min()).to_numpy() / 2.0
    feat[:, 15] = atr_vals / (c + eps)
    feat[:, 16] = ((c - o) > 0).astype(float)
    feat[:, 17] = np.abs(c - o) / (h - l + eps)
    rh = pd.Series(h).rolling(10).max().to_numpy()
    rl = pd.Series(l).rolling(10).min().to_numpy()
    feat[:, 18] = (c - rl) / (rh - rl + eps)
    feat[:, 19] = v / (pd.Series(v).rolling(50).mean().to_numpy() + eps)
    # H4 features
    h4_atr = pd.Series(h4_h - h4_l).rolling(14).mean().to_numpy()
    h4_ema20 = pd.Series(h4_c).ewm(span=20, adjust=False).mean().to_numpy()
    h4_ema50 = pd.Series(h4_c).ewm(span=50, adjust=False).mean().to_numpy()
    h4_sh = pd.Series(h4_h).rolling(10).max().to_numpy()
    h4_sl = pd.Series(h4_l).rolling(10).min().to_numpy()
    feat[:, 20] = np.abs(h4_o - h4_c) / (h4_atr + eps)
    feat[:, 21] = (h4_h - h4_l) / (h4_atr + eps)
    feat[:, 22] = h4_ema20
    feat[:, 23] = h4_ema50
    feat[:, 24] = (h4_c - h4_ema20) / (h4_atr + eps)
    feat[:, 25] = np.sign(h4_ema20 - h4_ema50).astype(float)
    feat[:, 26] = (h4_c - h4_sl) / (h4_sh - h4_sl + eps)
    feat[:, 27] = c / h4_c - 1.0
    return feat, atr_vals

feat, atr_vals = build_features(n, o, h, l, c, v, h4_c, h4_o, h4_h, h4_l)

# Labels 3R/3R
labels = np.full(n, -1, dtype=np.int32)
for i in range(n - 50):
    if np.isnan(atr_vals[i]) or atr_vals[i] <= 0: continue
    tp_lvl = c[i] + 3.0 * atr_vals[i]
    sl_lvl = c[i] - 3.0 * atr_vals[i]
    for j in range(i + 1, i + 51):
        if l[j] <= sl_lvl: labels[i] = 0; break
        if h[j] >= tp_lvl: labels[i] = 1; break

# OOS split
oos_cut = int(n * 0.8)
train_mask = np.arange(n) < oos_cut
test_mask = ~train_mask

def filter_valid(feat_subset, labels_subset, indices):
    valid = labels_subset >= 0
    X = feat_subset[valid]; y = labels_subset[valid]; idx = indices[valid]
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], y[ok], idx[ok]

X_train, y_train, train_idx = filter_valid(feat[train_mask], labels[train_mask], np.where(train_mask)[0])
X_test, y_test, test_idx = filter_valid(feat[test_mask], labels[test_mask], np.where(test_mask)[0])

print(f"\nTrain ML: {len(y_train):,} ({y_train.mean():.1%} WR)")
print(f"Test ML:  {len(y_test):,} ({y_test.mean():.1%} WR)")

# Train ML model
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)

# ML test
y_prob_test = model.predict_proba(X_test)[:, 1]
print(f"\nML OOS results (XAUUSD M15+H4, 3R/3R):")
for th in [0.50, 0.55, 0.60, 0.65, 0.70]:
    m = y_prob_test >= th
    if m.sum() > 0:
        print(f"  Thresh {th:.2f}: {m.sum():,} trades, {y_test[m].mean():.1%} WR, {(y_test[m].mean()*3 - (1-y_test[m].mean())*3):.2f}R EV")

# Feature importance
importances = sorted(zip(
    ["body_atr","range_atr","upper_wick","lower_wick","close_ema20","close_ema50",
     "ema20_slope","ema50_slope","vol_ratio","volatility","ret_1","ret_5",
     "prev_dir","prev2_dir","swing_pos","atr_pct","bullish","body_range","range_pos","vol_ratio50",
     "h4_body","h4_range","h4_ema20","h4_ema50","h4_close_ema20","h4_trend","h4_range_pos","m15_h4_ratio"],
    model.feature_importances_), key=lambda x: x[1], reverse=True)
print(f"\nTop features:")
for i, (name, imp) in enumerate(importances[:8], 1):
    print(f"  {i}. {name}: {imp:.3f}")

# Compare with BTCUSD
print(f"\nComparison:")
print(f"  BTCUSD M15+H4: 76.7% WR at 0.55 thresh ({len(y_test):,} test)")
print(f"  XAUUSD M15+H4: {y_test[m].mean():.1%} WR at 0.55 thresh ({y_test[m].sum():,} test)")

# Generate sequences for LSTM
print(f"\nGenerating LSTM sequences...")
train_probs = model.predict_proba(X_train)[:, 1]
train_sig = train_probs >= 0.55
train_sig_idx = train_idx[train_sig]

X_st, y_st = [], []
for idx in train_sig_idx:
    if idx < 5 or idx >= n - 50: continue
    ws, we = idx - 5, idx
    sq = np.column_stack([
        (o[ws:we]/c[idx]-1)*100, (h[ws:we]/c[idx]-1)*100,
        (l[ws:we]/c[idx]-1)*100, (c[ws:we]/c[idx]-1)*100,
        v[ws:we]/(np.mean(v[ws:we])+eps)]).astype(np.float32)
    h4sq = np.column_stack([
        np.full(5,feat[idx,20]), np.full(5,feat[idx,21]),
        np.full(5,feat[idx,24]), np.full(5,feat[idx,25]),
        np.full(5,feat[idx,27])]).astype(np.float32)
    combined = np.concatenate([sq, h4sq], axis=1)
    atr_v = atr_vals[idx]
    if np.isnan(atr_v) or atr_v <= 0: continue
    tp_lvl = c[idx] + 3.0*atr_v; sl_lvl = c[idx] - 3.0*atr_v
    for j in range(idx+1, min(idx+51, n)):
        if l[j] <= sl_lvl: y_st.append(0); X_st.append(combined); break
        if h[j] >= tp_lvl: y_st.append(1); X_st.append(combined); break

X_st_arr = np.array(X_st); y_st_arr = np.array(y_st)
print(f"  Train sequences: {len(X_st_arr):,}, WR: {y_st_arr.mean():.1%}")

# Test sequences
test_sig = y_prob_test >= 0.55
test_sig_idx = test_idx[test_sig]
X_ste, y_ste = [], []
for idx in test_sig_idx:
    if idx < 5 or idx >= n - 50: continue
    ws, we = idx - 5, idx
    sq = np.column_stack([
        (o[ws:we]/c[idx]-1)*100, (h[ws:we]/c[idx]-1)*100,
        (l[ws:we]/c[idx]-1)*100, (c[ws:we]/c[idx]-1)*100,
        v[ws:we]/(np.mean(v[ws:we])+eps)]).astype(np.float32)
    h4sq = np.column_stack([
        np.full(5,feat[idx,20]), np.full(5,feat[idx,21]),
        np.full(5,feat[idx,24]), np.full(5,feat[idx,25]),
        np.full(5,feat[idx,27])]).astype(np.float32)
    combined = np.concatenate([sq, h4sq], axis=1)
    atr_v = atr_vals[idx]
    if np.isnan(atr_v) or atr_v <= 0: continue
    tp_lvl = c[idx] + 3.0*atr_v; sl_lvl = c[idx] - 3.0*atr_v
    for j in range(idx+1, min(idx+51, n)):
        if l[j] <= sl_lvl: y_ste.append(0); X_ste.append(combined); break
        if h[j] >= tp_lvl: y_ste.append(1); X_ste.append(combined); break

X_ste_arr = np.array(X_ste); y_ste_arr = np.array(y_ste)
print(f"  Test sequences: {len(X_ste_arr):,}, WR: {y_ste_arr.mean():.1%}")

# Save
np.savez("dataset_xau_m15h4_train.npz", X=X_st_arr, y=y_st_arr,
         feature_cols=np.array(["o","h","l","c","v","h4_body","h4_range","h4_close_ema20","h4_trend","m15_h4_ratio"]),
         sequence_length=5)
np.savez("dataset_xau_m15h4_test.npz", X_test=X_ste_arr, y_test=y_ste_arr,
         feature_cols=np.array(["o","h","l","c","v","h4_body","h4_range","h4_close_ema20","h4_trend","m15_h4_ratio"]),
         sequence_length=5)
with open("ml_xau_m15h4.pkl", "wb") as f:
    pickle.dump({"model": model, "threshold": 0.55}, f)

print(f"\nSaved: dataset_xau_m15h4_train.npz, dataset_xau_m15h4_test.npz, ml_xau_m15h4.pkl")
print()
print("Next: python train_lstm_price_action.py dataset_xau_m15h4_train.npz")

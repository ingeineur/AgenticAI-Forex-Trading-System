#!/usr/bin/env python3
"""BTCUSD M15+H4 Proper LSTM Training with clean OOS split"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BTCUSD M15+H4 PROPER LSTM TRAINING (clean OOS)")
print("="*70)

# Load and merge data
df = pd.read_csv("BTCUSD_15.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

h4 = pd.read_csv("BTCUSD_16388.csv")
h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
h4 = h4.sort_values("timestamp").reset_index(drop=True)
h4 = h4.rename(columns={c: f"h4_{c}" for c in h4.columns if c != "timestamp"})

df["timestamp_round"] = df["timestamp"].dt.floor("4h")
h4["timestamp_round"] = h4["timestamp"].dt.floor("4h")
h4_cols = ["timestamp_round", "h4_open", "h4_high", "h4_low", "h4_close", "h4_volume"]
merged = df.merge(h4[h4_cols], on="timestamp_round", how="left")
merged = merged.dropna(subset=["h4_close"]).reset_index(drop=True)

n = len(merged)
print(f"\nMerged data: {n:,} bars")
print(f"Range: {merged['timestamp'].iloc[0]} to {merged['timestamp'].iloc[-1]}")

# OOS split: first 80% train, last 20% test
oos_cut = int(n * 0.8)
print(f"Train: 0 to {oos_cut} ({oos_cut:,} bars)")
print(f"Test:  {oos_cut} to {n} ({n - oos_cut:,} bars)")

# Extract arrays
o, h, l, c, v = merged["open"].to_numpy(), merged["high"].to_numpy(), merged["low"].to_numpy(), merged["close"].to_numpy(), merged["volume"].to_numpy()
h4_c_arr = merged["h4_close"].to_numpy()
h4_o_arr = merged["h4_open"].to_numpy()
h4_h_arr = merged["h4_high"].to_numpy()
h4_l_arr = merged["h4_low"].to_numpy()

eps = 1e-9
atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()

# Build 28 features
def build_features(o, h, l, c, v, h4_c, h4_o_arr, h4_h_arr, h4_l_arr, atr_vals, n):
    eps = 1e-9
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
    h4_atr = pd.Series(h4_h_arr - h4_l_arr).rolling(14).mean().to_numpy()
    h4_ema20 = pd.Series(h4_c).ewm(span=20, adjust=False).mean().to_numpy()
    h4_ema50 = pd.Series(h4_c).ewm(span=50, adjust=False).mean().to_numpy()
    h4_sh = pd.Series(h4_h_arr).rolling(10).max().to_numpy()
    h4_sl = pd.Series(h4_l_arr).rolling(10).min().to_numpy()
    feat[:, 20] = np.abs(h4_o_arr - h4_c) / (h4_atr + eps)
    feat[:, 21] = (h4_h_arr - h4_l_arr) / (h4_atr + eps)
    feat[:, 22] = h4_ema20
    feat[:, 23] = h4_ema50
    feat[:, 24] = (h4_c - h4_ema20) / (h4_atr + eps)
    feat[:, 25] = np.sign(h4_ema20 - h4_ema50).astype(float)
    feat[:, 26] = (h4_c - h4_sl) / (h4_sh - h4_sl + eps)
    feat[:, 27] = c / h4_c - 1.0
    return feat

feat = build_features(o, h, l, c, v, h4_c_arr, h4_o_arr, h4_h_arr, h4_l_arr, atr_vals, n)

# Generate 3R/3R labels
labels = np.full(n, -1, dtype=np.int32)
for i in range(n - 50):
    if np.isnan(atr_vals[i]) or atr_vals[i] <= 0: continue
    tp_lvl = c[i] + 3.0 * atr_vals[i]
    sl_lvl = c[i] - 3.0 * atr_vals[i]
    for j in range(i + 1, i + 51):
        if l[j] <= sl_lvl: labels[i] = 0; break
        if h[j] >= tp_lvl: labels[i] = 1; break

# Separate train/test
train_mask = np.arange(n) < oos_cut
test_mask = ~train_mask

def filter_valid(feat_subset, labels_subset, indices):
    valid = labels_subset >= 0
    X = feat_subset[valid]
    y = labels_subset[valid]
    idx = indices[valid]
    nan_ok = ~np.isnan(X).any(axis=1)
    return X[nan_ok], y[nan_ok], idx[nan_ok]

X_train, y_train, train_idx = filter_valid(feat[train_mask], labels[train_mask], np.where(train_mask)[0])
X_test, y_test, test_idx = filter_valid(feat[test_mask], labels[test_mask], np.where(test_mask)[0])

print(f"\nTrain ML: {len(y_train):,} ({y_train.mean():.1%} WR)")
print(f"Test ML:  {len(y_test):,} ({y_test.mean():.1%} WR)")

# Train ML model
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ML test
y_prob_test = model.predict_proba(X_test)[:, 1]
print(f"\nML OOS results:")
for th in [0.50, 0.55, 0.60, 0.65, 0.70]:
    m = y_prob_test >= th
    if m.sum() > 0:
        print(f"  Thresh {th:.2f}: {m.sum():,} trades, {y_test[m].mean():.1%} WR")

# Generate LSTM sequences from TRAINING data only
print(f"\nGenerating LSTM training sequences (from training bars only)...")
train_probs = model.predict_proba(X_train)[:, 1]
train_sig = train_probs >= 0.55
train_sig_idx = train_idx[train_sig]

X_seqs_train, y_seqs_train = [], []
for idx in train_sig_idx:
    if idx < 5 or idx >= n - 50: continue
    ws, we = idx - 5, idx
    sq = np.column_stack([
        (o[ws:we] / c[idx] - 1.0) * 100, (h[ws:we] / c[idx] - 1.0) * 100,
        (l[ws:we] / c[idx] - 1.0) * 100, (c[ws:we] / c[idx] - 1.0) * 100,
        v[ws:we] / (np.mean(v[ws:we]) + eps)
    ]).astype(np.float32)
    h4sq = np.column_stack([
        np.full(5, feat[idx, 20]), np.full(5, feat[idx, 21]),
        np.full(5, feat[idx, 24]), np.full(5, feat[idx, 25]),
        np.full(5, feat[idx, 27]),
    ]).astype(np.float32)
    combined = np.concatenate([sq, h4sq], axis=1)
    atr_v = atr_vals[idx]
    if np.isnan(atr_v) or atr_v <= 0: continue
    tp_lvl = c[idx] + 3.0 * atr_v
    sl_lvl = c[idx] - 3.0 * atr_v
    for j in range(idx + 1, min(idx + 51, n)):
        if l[j] <= sl_lvl: y_seqs_train.append(0); X_seqs_train.append(combined); break
        if h[j] >= tp_lvl: y_seqs_train.append(1); X_seqs_train.append(combined); break

X_st = np.array(X_seqs_train)
y_st = np.array(y_seqs_train)
print(f"  Train sequences: {len(X_st):,}, WR: {y_st.mean():.1%}")

# Generate LSTM sequences from TEST data
print(f"Generating LSTM test sequences (from test bars only)...")
test_sig = y_prob_test >= 0.55
test_sig_idx = test_idx[test_sig]

X_seqs_test, y_seqs_test = [], []
for idx in test_sig_idx:
    if idx < 5 or idx >= n - 50: continue
    ws, we = idx - 5, idx
    sq = np.column_stack([
        (o[ws:we] / c[idx] - 1.0) * 100, (h[ws:we] / c[idx] - 1.0) * 100,
        (l[ws:we] / c[idx] - 1.0) * 100, (c[ws:we] / c[idx] - 1.0) * 100,
        v[ws:we] / (np.mean(v[ws:we]) + eps)
    ]).astype(np.float32)
    h4sq = np.column_stack([
        np.full(5, feat[idx, 20]), np.full(5, feat[idx, 21]),
        np.full(5, feat[idx, 24]), np.full(5, feat[idx, 25]),
        np.full(5, feat[idx, 27]),
    ]).astype(np.float32)
    combined = np.concatenate([sq, h4sq], axis=1)
    atr_v = atr_vals[idx]
    if np.isnan(atr_v) or atr_v <= 0: continue
    tp_lvl = c[idx] + 3.0 * atr_v
    sl_lvl = c[idx] - 3.0 * atr_v
    for j in range(idx + 1, min(idx + 51, n)):
        if l[j] <= sl_lvl: y_seqs_test.append(0); X_seqs_test.append(combined); break
        if h[j] >= tp_lvl: y_seqs_test.append(1); X_seqs_test.append(combined); break

X_ste = np.array(X_seqs_test)
y_ste = np.array(y_seqs_test)
print(f"  Test sequences: {len(X_ste):,}, WR: {y_ste.mean():.1%}")

# Save training in legacy format for LSTM trainer
# Combine train sequences into single X,y for LSTM trainer (it will split 70/15/15 internally)
X_all = X_st
y_all = y_st
print(f"\nSaving LSTM training dataset ({len(X_all):,} sequences)...")
np.savez("dataset_btc_m15h4_train.npz",
         X=X_all, y=y_all,
         feature_cols=np.array(["o","h","l","c","v","h4_body","h4_range","h4_close_ema20","h4_trend","m15_h4_ratio"]),
         sequence_length=5)
print("  Saved: dataset_btc_m15h4_train.npz")

# Save test separately for evaluation
np.savez("dataset_btc_m15h4_test.npz",
         X_test=X_ste, y_test=y_ste,
         feature_cols=np.array(["o","h","l","c","v","h4_body","h4_range","h4_close_ema20","h4_trend","m15_h4_ratio"]),
         sequence_length=5)
print("  Saved: dataset_btc_m15h4_test.npz")

# Save ML model
with open("ml_btc_m15h4.pkl", "wb") as f:
    pickle.dump({"model": model, "threshold": 0.55}, f)
print("  Saved: ml_btc_m15h4.pkl")

print()
print("="*70)
print("NEXT: python train_lstm_price_action.py dataset_btc_m15h4_train.npz")
print("="*70)

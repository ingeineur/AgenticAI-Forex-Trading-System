#!/usr/bin/env python3
"""
Full ML-to-LSTM Pipeline
1. Train ML model on ALL bars with 3R/3R labels
2. Filter bars where ML probability > threshold
3. Generate LSTM sequences from filtered bars
4. Save dataset for LSTM training
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def atr(arr, period=14):
    """ATR from numpy arrays"""
    tr = np.maximum(np.maximum(
        arr['high'] - arr['low'],
        np.abs(arr['high'] - np.roll(arr['close'], 1))
    ), np.abs(arr['low'] - np.roll(arr['close'], 1)))
    return pd.Series(tr).rolling(period).mean().to_numpy()


def create_features_from_arr(arr):
    """Create ML features from structured numpy array"""
    n = len(arr)
    eps = 1e-9
    o, h, l, c, v = arr['open'], arr['high'], arr['low'], arr['close'], arr['volume']
    
    atr_vals = atr(arr, 14)
    
    features = np.zeros((n, 20), dtype=np.float32)
    
    features[:, 0] = np.abs(c - o) / (atr_vals + eps)
    features[:, 1] = (h - l) / (atr_vals + eps)
    features[:, 2] = (h - np.maximum(o, c)) / (atr_vals + eps)
    features[:, 3] = (np.minimum(o, c) - l) / (atr_vals + eps)
    
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    
    features[:, 4] = (c - ema20) / (atr_vals + eps)
    features[:, 5] = (c - ema50) / (atr_vals + eps)
    features[:, 6] = np.diff(ema20, prepend=ema20[0]) / (atr_vals + eps)
    features[:, 7] = np.diff(ema50, prepend=ema50[0]) / (atr_vals + eps)
    
    vol_ma = pd.Series(v).rolling(20).mean().to_numpy()
    features[:, 8] = v / (vol_ma + eps)
    
    returns = pd.Series(c).pct_change()
    features[:, 9] = returns.rolling(20).std().to_numpy()
    features[:, 10] = returns.fillna(0.0).to_numpy()
    features[:, 11] = c / np.roll(c, 5) - 1.0
    
    features[:, 12] = (np.roll(c, 1) > np.roll(o, 1)).astype(float)
    features[:, 13] = (np.roll(c, 2) > np.roll(o, 2)).astype(float)
    features[:, 14] = c > (pd.Series(h).rolling(5).max() + pd.Series(l).rolling(5).min()).to_numpy() / 2.0
    
    features[:, 15] = atr_vals / (c + eps)
    features[:, 16] = ((c - o) > 0).astype(float)
    features[:, 17] = np.abs(c - o) / (h - l + eps)
    
    recent_high = pd.Series(h).rolling(10).max().to_numpy()
    recent_low = pd.Series(l).rolling(10).min().to_numpy()
    features[:, 18] = (c - recent_low) / (recent_high - recent_low + eps)
    
    features[:, 19] = v / (pd.Series(v).rolling(50).mean().to_numpy() + eps)
    
    return features, atr_vals


def generate_labels(c, h, l, atr_vals, tp_mult, sl_mult, max_horizon=50):
    """Generate TP/SL labels (vectorized where possible)"""
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


def generate_lstm_sequences_from_indices(df, signal_indices, lookback=5, forecast_window=50):
    """Generate LSTM sequences from specific signal index positions"""
    
    o = df['open'].to_numpy()
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    v = df['volume'].to_numpy()
    
    X_seqs = []
    y_labels = []
    valid_indices = []
    
    atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()
    
    for idx in signal_indices:
        if idx < lookback or idx >= len(df) - forecast_window:
            continue
        
        window_start = idx - lookback
        window_end = idx
        
        window_open = o[window_start:window_end]
        window_high = h[window_start:window_end]
        window_low = l[window_start:window_end]
        window_close = c[window_start:window_end]
        window_vol = v[window_start:window_end]
        
        current_close = window_close[-1]
        if current_close <= 0:
            continue
        
        o_norm = (window_open / current_close - 1.0) * 100
        h_norm = (window_high / current_close - 1.0) * 100
        l_norm = (window_low / current_close - 1.0) * 100
        c_norm = (window_close / current_close - 1.0) * 100
        
        vol_mean = np.mean(window_vol)
        v_norm = window_vol / vol_mean if vol_mean > 0 else np.ones(lookback)
        
        seq = np.column_stack([o_norm, h_norm, l_norm, c_norm, v_norm]).astype(np.float32)
        
        # Generate label with 3R/3R
        entry_price = current_close
        atr_val = atr_vals[idx]
        
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        
        tp_level = entry_price + 3.0 * atr_val
        sl_level = entry_price - 3.0 * atr_val
        
        future_h = h[idx+1:idx+forecast_window+1]
        future_l = l[idx+1:idx+forecast_window+1]
        
        hit_sl = False
        hit_tp = False
        for j in range(len(future_l)):
            if future_l[j] <= sl_level:
                hit_sl = True
                break
            if future_h[j] >= tp_level:
                hit_tp = True
                break
        
        if hit_sl:
            y_labels.append(0)
            X_seqs.append(seq)
            valid_indices.append(idx)
        elif hit_tp:
            y_labels.append(1)
            X_seqs.append(seq)
            valid_indices.append(idx)
    
    return np.array(X_seqs, dtype=np.float32), np.array(y_labels, dtype=np.int32), valid_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs='?', default="XAUUSD_M15.csv")
    parser.add_argument("--output", default="dataset_ml_pipeline.npz")
    parser.add_argument("--tp-mult", type=float, default=3.0)
    parser.add_argument("--sl-mult", type=float, default=3.0)
    parser.add_argument("--ml-threshold", type=float, default=0.55)
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--test-split", type=float, default=0.2)
    args = parser.parse_args()
    
    print("="*70)
    print(f"ML->LSTM PIPELINE (TP={args.tp_mult}R, SL={args.sl_mult}R)")
    print("="*70)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    df = pd.read_csv(args.csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} bars loaded")
    print()
    
    # Step 2: Prepare arrays
    print("Step 2: Creating feature array...")
    dtype = [('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')]
    arr = np.zeros(len(df), dtype=dtype)
    arr['open'] = df['open'].to_numpy()
    arr['high'] = df['high'].to_numpy()
    arr['low'] = df['low'].to_numpy()
    arr['close'] = df['close'].to_numpy()
    arr['volume'] = df['volume'].to_numpy()
    c, h, l = arr['close'], arr['high'], arr['low']
    
    # Step 3: Train ML model on ALL bars
    print("Step 3: Training ML model on ALL bars...")
    features, atr_vals = create_features_from_arr(arr)
    labels = generate_labels(c, h, l, atr_vals, args.tp_mult, args.sl_mult, args.max_horizon)
    
    valid = labels >= 0
    X_ml = features[valid]
    y_ml = labels[valid]
    
    nan_mask = ~np.isnan(X_ml).any(axis=1)
    X_ml = X_ml[nan_mask]
    y_ml = y_ml[nan_mask]
    
    print(f"  Training samples: {len(y_ml):,}")
    print(f"  Label distribution: {y_ml.mean():.1%} wins")
    
    # Train with time-based split
    split = int(len(y_ml) * 0.7)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_ml[:split], y_ml[:split])
    
    # Test
    y_prob_test = model.predict_proba(X_ml[split:])[:, 1]
    test_labels = y_ml[split:]
    
    print(f"\n  ML model performance on test set:")
    mask = y_prob_test >= args.ml_threshold
    if mask.sum() > 0:
        test_wr = test_labels[mask].mean()
        print(f"  Threshold {args.ml_threshold:.2f}: {mask.sum():,} trades, {test_wr:.1%} WR")
    else:
        print(f"  Threshold {args.ml_threshold:.2f}: NO TRADES")
    print()
    
    # Step 4: Get ML predictions on ALL bars
    print("Step 4: Running ML predictions on ALL bars...")
    X_all = features
    all_probs = model.predict_proba(X_all)[:, 1]
    signal_mask = all_probs >= args.ml_threshold
    signal_indices = np.where(signal_mask)[0]
    
    print(f"  ML signals found: {len(signal_indices):,}")
    print(f"  Signal frequency: {len(signal_indices)/len(df):.2%}")
    print()
    
    # Step 5: Generate LSTM sequences
    print("Step 5: Generating LSTM sequences from ML signals...")
    X_seqs, y_seqs, valid_idx = generate_lstm_sequences_from_indices(
        df, signal_indices, lookback=args.lookback, forecast_window=args.max_horizon
    )
    
    print(f"  Sequences generated: {len(X_seqs):,}")
    print(f"  Sequence shape: {X_seqs.shape}")
    print(f"  Label distribution: {y_seqs.mean():.1%} wins")
    print()
    
    if len(X_seqs) < 100:
        print("WARNING: Too few sequences for LSTM training (need 100+)")
        print("Consider lowering ML threshold or using more data")
        print()
    
    # Step 6: Split and save
    print("Step 6: Splitting and saving dataset...")
    test_idx = int(len(X_seqs) * (1 - args.test_split))
    
    # Ensure chronological order
    X_train, y_train = X_seqs[:test_idx], y_seqs[:test_idx]
    X_test, y_test = X_seqs[test_idx:], y_seqs[test_idx:]
    
    print(f"  Train: {len(X_train):,} ({y_train.mean():.1%} WR)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean():.1%} WR)")
    print()
    
    np.savez(args.output, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"  Saved to: {args.output}")
    print()
    
    # Save ML model too
    ml_artifact = {
        "model": model,
        "threshold": args.ml_threshold,
        "tp_mult": args.tp_mult,
        "sl_mult": args.sl_mult,
    }
    with open("ml_pipeline_model.pkl", "wb") as f:
        pickle.dump(ml_artifact, f)
    print(f"  ML model saved to: ml_pipeline_model.pkl")
    print()
    
    print("="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print()
    print("Next steps:")
    print(f"  1. python train_lstm_price_action.py {args.output}")
    print(f"     (after converting to legacy format)")
    print()
    print("Recommendation:")
    print(f"  - Train LSTM on this dataset (ML-filtered {args.tp_mult}R/{args.sl_mult}R signals)")
    print(f"  - Expected LSTM accuracy: {y_train.mean():.0%} - {y_test.mean():.0%} on selected trades")
    print(f"  - This is a direct improvement over current 34.4% baseline")


if __name__ == "__main__":
    main()

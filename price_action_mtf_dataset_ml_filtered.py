#!/usr/bin/env python3
"""
ML-Filtered Dataset Generator
Only includes sequences from high-probability ML signals (threshold >= 0.70)
This gives us the 97% win rate signals the model identified
"""

import numpy as np
import pandas as pd
import pickle
import argparse


def atr(df, period=14):
    """Calculate ATR."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    
    tr_1 = high - low
    tr_2 = np.abs(high - np.roll(close, 1))
    tr_3 = np.abs(low - np.roll(close, 1))
    
    tr = np.maximum(np.maximum(tr_1, tr_2), tr_3)
    atr_vals = pd.Series(tr).rolling(period).mean().to_numpy()
    
    return atr_vals


def create_features(df):
    """Create normalized OHLCV features for ML."""
    eps = 1e-9
    
    atr_vals = atr(df, period=14)
    
    df_copy = df.copy()
    
    # Candle metrics
    df_copy["body"] = (df_copy["close"] - df_copy["open"]).abs()
    df_copy["range"] = df_copy["high"] - df_copy["low"]
    df_copy["upper_wick"] = df_copy["high"] - df_copy[["open", "close"]].max(axis=1)
    df_copy["lower_wick"] = df_copy[["open", "close"]].min(axis=1) - df_copy["low"]
    
    # Normalized by ATR
    df_copy["body_atr"] = df_copy["body"] / (atr_vals + eps)
    df_copy["range_atr"] = df_copy["range"] / (atr_vals + eps)
    df_copy["upper_wick_atr"] = df_copy["upper_wick"] / (atr_vals + eps)
    df_copy["lower_wick_atr"] = df_copy["lower_wick"] / (atr_vals + eps)
    
    # EMA metrics
    df_copy["ema20"] = df_copy["close"].ewm(span=20, adjust=False).mean()
    df_copy["ema50"] = df_copy["close"].ewm(span=50, adjust=False).mean()
    df_copy["close_to_ema20_atr"] = (df_copy["close"] - df_copy["ema20"]) / (atr_vals + eps)
    df_copy["close_to_ema50_atr"] = (df_copy["close"] - df_copy["ema50"]) / (atr_vals + eps)
    df_copy["ema20_slope_atr"] = df_copy["ema20"].diff(5) / (atr_vals + eps)
    df_copy["ema50_slope_atr"] = df_copy["ema50"].diff(5) / (atr_vals + eps)
    
    # Volume metrics
    df_copy["volume_ma"] = df_copy["volume"].rolling(20).mean()
    df_copy["volume_ratio"] = df_copy["volume"] / (df_copy["volume_ma"] + eps)
    
    # Volatility
    df_copy["volatility"] = df_copy["close"].pct_change().rolling(20).std()
    
    # Returns
    df_copy["ret_1"] = df_copy["close"].pct_change().fillna(0.0)
    df_copy["ret_5"] = df_copy["close"].pct_change(5).fillna(0.0)
    
    # Direction from previous bars
    df_copy["prev_close_gt_open"] = (df_copy["close"].shift(1) > df_copy["open"].shift(1)).astype(float)
    df_copy["prev_2_close_gt_open"] = (df_copy["close"].shift(2) > df_copy["open"].shift(2)).astype(float)
    
    # Swing metrics
    df_copy["recent_high"] = df_copy["high"].rolling(5).max()
    df_copy["recent_low"] = df_copy["low"].rolling(5).min()
    df_copy["close_above_recent_avg"] = (df_copy["close"] > (df_copy["recent_high"] + df_copy["recent_low"]) / 2).astype(float)
    
    return df_copy, atr_vals


def load_ml_filtered_dataset(csv_path, model_path, lookback=5, forecast_window=50, threshold=0.70):
    """Load OHLCV and create sequences ONLY from high-probability ML signals."""
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"Creating features...")
    df, atr_arr = create_features(df)
    
    print(f"Loading ML model...")
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    
    model = artifact['model']
    feature_cols = artifact['feature_cols']
    threshold = artifact['best_threshold']
    
    print(f"Model threshold: {threshold:.2f}")
    print()
    
    # Get ML predictions for all bars
    print(f"Generating ML predictions for all {len(df):,} bars...")
    X_all = df[feature_cols].fillna(0.0).astype(np.float32)
    ml_probs = model.predict_proba(X_all)[:, 1]
    ml_signals = ml_probs >= threshold
    
    print(f"ML signals found: {ml_signals.sum()}")
    print()
    
    # Extract OHLCV arrays
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    volume_arr = df["volume"].to_numpy()
    open_arr = df["open"].to_numpy()
    
    # Compute future extremes
    print(f"Computing future extremes...")
    n = len(df)
    future_high = np.zeros(n)
    future_low = np.zeros(n)
    
    for i in range(n - forecast_window):
        future_high[i] = np.max(high_arr[i+1:i+forecast_window+1])
        future_low[i] = np.min(low_arr[i+1:i+forecast_window+1])
    
    # Create sequences ONLY from ML signals
    print(f"Creating sequences from ML signals...")
    X_sequences = []
    y_labels = []
    signal_indices = []
    
    for i in np.where(ml_signals)[0]:
        # Skip if too close to start or end
        if i < lookback or i >= n - forecast_window:
            continue
        
        # Extract lookback window
        window_start = i - lookback
        window_end = i
        
        window_open = open_arr[window_start:window_end]
        window_high = high_arr[window_start:window_end]
        window_low = low_arr[window_start:window_end]
        window_close = close_arr[window_start:window_end]
        window_volume = volume_arr[window_start:window_end]
        
        # Normalize to current bar
        current_close = window_close[-1]
        if current_close <= 0:
            continue
        
        o_norm = (window_open / current_close - 1.0) * 100
        h_norm = (window_high / current_close - 1.0) * 100
        l_norm = (window_low / current_close - 1.0) * 100
        c_norm = (window_close / current_close - 1.0) * 100
        
        # Volume normalization
        vol_mean = np.mean(window_volume)
        if vol_mean > 0:
            v_norm = window_volume / vol_mean
        else:
            v_norm = np.ones(lookback)
        
        seq = np.column_stack([o_norm, h_norm, l_norm, c_norm, v_norm]).astype(np.float32)
        X_sequences.append(seq)
        
        # Create label (TP vs SL)
        entry_price = current_close
        atr_val = atr_arr[i]
        
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        
        tp_level = entry_price + 2.0 * atr_val
        sl_level = entry_price - 1.0 * atr_val
        
        fh = future_high[i]
        fl = future_low[i]
        
        if np.isnan(fh) or np.isnan(fl) or fh == 0 or fl == 0:
            continue
        
        if fl <= sl_level:
            label = 0
        elif fh >= tp_level:
            label = 1
        else:
            continue
        
        y_labels.append(label)
        signal_indices.append(i)
    
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    
    print(f"Total ML signals used: {len(X)}")
    print(f"Sequences created: {len(X)}")
    print(f"Win rate: {y.mean() if len(y) > 0 else 0:.1%}")
    print()
    
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to OHLCV CSV")
    parser.add_argument("model_path", help="Path to ML model pickle")
    parser.add_argument("--output", default="dataset_ml_filtered.npz", help="Output file")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback window")
    parser.add_argument("--forecast", type=int, default=50, help="Forecast window")
    parser.add_argument("--threshold", type=float, default=0.70, help="ML probability threshold")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ML-FILTERED DATASET GENERATOR")
    print("="*70)
    print()
    
    X, y = load_ml_filtered_dataset(
        args.csv_path,
        args.model_path,
        lookback=args.lookback,
        forecast_window=args.forecast,
        threshold=args.threshold
    )
    
    if len(X) == 0:
        print("ERROR: No sequences generated!")
        return
    
    # Split train/test
    test_idx = int(len(X) * (1 - args.test_split))
    X_train = X[:test_idx]
    y_train = y[:test_idx]
    X_test = X[test_idx:]
    y_test = y[test_idx:]
    
    print(f"Train: {len(X_train):,} samples ({y_train.mean():.1%} WR)")
    print(f"Test:  {len(X_test):,} samples ({y_test.mean():.1%} WR)")
    print()
    
    # Save
    print(f"Saving to {args.output}...")
    np.savez(
        args.output,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    
    print("Done")
    print()
    print(f"Next: python convert_ml_dataset.py && python train_lstm_price_action.py dataset_ml_legacy.npz")


if __name__ == "__main__":
    main()

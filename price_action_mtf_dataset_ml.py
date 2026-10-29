#!/usr/bin/env python3
"""
ML-based Price Action Dataset Generator
Generated signals using trained RandomForest model
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path


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


def detect_swings(df, threshold=0.5):
    """Detect recent swing high/low."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    
    swing_high = np.full(len(df), np.nan)
    swing_low = np.full(len(df), np.nan)
    
    for i in range(5, len(df)):
        swing_high[i] = np.max(high[max(0, i-5):i])
        swing_low[i] = np.min(low[max(0, i-5):i])
    
    return swing_high, swing_low


def load_price_action_dataset(csv_path, lookback=5, forecast_window=50):
    """Load OHLCV and create price action sequences."""
    
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate ATR and swings
    df["atr"] = atr(df)
    df["swing_high"], df["swing_low"] = detect_swings(df)
    
    # Normalize OHLCV
    df["close_norm"] = df["close"] / (df["close"].iloc[0] + 1e-8)
    
    X_sequences = []
    y_labels = []
    indices = []
    
    lookback_idx = lookback
    
    for i in range(lookback_idx, len(df) - forecast_window):
        # Extract lookback window
        window = df.iloc[i - lookback_idx:i].copy()
        
        # Normalize to current bar
        current_close = window["close"].iloc[-1]
        window["o_norm"] = (window["open"] / current_close - 1.0) * 100
        window["h_norm"] = (window["high"] / current_close - 1.0) * 100
        window["l_norm"] = (window["low"] / current_close - 1.0) * 100
        window["c_norm"] = (window["close"] / current_close - 1.0) * 100
        window["v_norm"] = window["volume"] / (window["volume"].mean() + 1e-8)
        
        seq = np.column_stack([
            window["o_norm"],
            window["h_norm"],
            window["l_norm"],
            window["c_norm"],
            window["v_norm"],
        ]).astype(np.float32)
        
        X_sequences.append(seq)
        
        # Create label: TP vs SL (2R TP, 1R SL)
        entry_price = current_close
        atr_val = window["atr"].iloc[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        
        tp_level = entry_price + 2.0 * atr_val
        sl_level = entry_price - 1.0 * atr_val
        
        future_high = df["high"].iloc[i+1:i+forecast_window+1].max()
        future_low = df["low"].iloc[i+1:i+forecast_window+1].min()
        
        if future_low <= sl_level:
            label = 0
        elif future_high >= tp_level:
            label = 1
        else:
            continue
        
        y_labels.append(label)
        indices.append(i)
    
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    
    return X, y, indices, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to OHLCV CSV")
    parser.add_argument("--output", default="dataset_ml.npz", help="Output file")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback window")
    parser.add_argument("--forecast", type=int, default=50, help="Forecast window")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ML-BASED PRICE ACTION DATASET")
    print("="*70)
    print()
    
    X, y, indices, df = load_price_action_dataset(
        args.csv_path,
        lookback=args.lookback,
        forecast_window=args.forecast
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels: {len(y)}")
    print(f"Win rate: {y.mean():.1%}")
    print()
    
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


if __name__ == "__main__":
    main()

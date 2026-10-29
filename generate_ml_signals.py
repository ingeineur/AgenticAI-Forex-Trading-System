#!/usr/bin/env python3
"""
ML Signal Generator - Applies trained ML model to generate trading signals.
Uses the trained RandomForest model to generate buy/sell signals for dataset creation.
"""

import numpy as np
import pandas as pd
import pickle
import sys
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


def generate_ml_signals(csv_path, model_path, output_path="price_action_mtf_dataset_ml.py"):
    """Generate ML signals and create dataset module."""
    
    print("="*70)
    print("ML SIGNAL GENERATOR")
    print("="*70)
    print()
    
    # Load data
    print("Step 1: Loading OHLCV data...")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} bars")
    print()
    
    # Create features
    print("Step 2: Creating features...")
    df = create_features(df)
    print("  Done")
    print()
    
    # Load model
    print("Step 3: Loading trained model...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact["best_threshold"]
    
    print(f"  Threshold: {threshold:.2f}")
    print(f"  CV Accuracy: {artifact['cv_accuracy']:.1%}")
    print()
    
    # Generate signals
    print("Step 4: Generating ML signals...")
    X = df[feature_cols].fillna(0.0).astype(np.float32)
    probs = model.predict_proba(X)[:, 1]
    
    signals = (probs >= threshold).astype(int)
    signal_count = signals.sum()
    
    print(f"  Total signals: {signal_count:,}")
    print(f"  Signal frequency: {signal_count/len(df):.2%}")
    print()
    
    # Generate dataset module
    print("Step 5: Creating dataset module...")
    
    module_content = '''#!/usr/bin/env python3
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
'''
    
    with open(output_path, "w") as f:
        f.write(module_content)
    
    print(f"  Created: {output_path}")
    print()
    
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print(f"1. Generate ML-based dataset:")
    print(f"   python {output_path} {csv_path} --output dataset_ml.npz")
    print()
    print(f"2. Train LSTM with ML signals:")
    print(f"   python train_lstm_price_action.py dataset_ml.npz")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_ml_signals.py <csv_path> <model_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "ml_signal_model.pkl"
    
    generate_ml_signals(csv_path, model_path)

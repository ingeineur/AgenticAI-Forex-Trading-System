#!/usr/bin/env python3
"""
Quick Signal Filter Tester: Test signal filter thresholds without regenerating full datasets.
Directly analyzes how different filter combinations would affect win rates.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_raw_ohlcv(csv_path):
    """Load OHLCV data."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


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


def add_base_metrics(df):
    """Add ATR-normalized metrics."""
    eps = 1e-9
    df = df.copy()
    
    df["atr"] = atr(df)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    
    df["body_atr"] = df["body"] / (df["atr"] + eps)
    df["range_atr"] = df["range"] / (df["atr"] + eps)
    df["upper_wick_atr"] = df["upper_wick"] / (df["atr"] + eps)
    df["lower_wick_atr"] = df["lower_wick"] / (df["atr"] + eps)
    
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + eps)
    
    return df


def test_signals(df, sweep_body=0.0, breakout_comp=0.9, engulf_body=0.6, displacement_range=1.25):
    """Test a signal configuration."""
    eps = 1e-9
    
    # Sweeps
    swing_lookback = 20
    swing_high = df["high"].rolling(swing_lookback).max().shift(1)
    swing_low = df["low"].rolling(swing_lookback).min().shift(1)
    
    sweep_long = (
        (df["low"] < swing_low) & 
        (df["close"] > swing_low) &
        ((df["body_atr"] > sweep_body) | (sweep_body == 0))
    )
    
    # Breakouts
    range_lookback = 20
    range_high = df["high"].rolling(range_lookback).max().shift(1)
    range_low = df["low"].rolling(range_lookback).min().shift(1)
    range_size = range_high - range_low
    range_ma = range_size.rolling(range_lookback).mean()
    compression = range_size / (range_ma + eps)
    
    breakout_long = (
        (df["close"] > range_high) & 
        (compression < breakout_comp)
    )
    
    # Engulfing
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    
    engulf_long = (
        (df["close"] > df["open"]) &
        (prev_close < prev_open) &
        (df["close"] > prev_open) &
        (df["open"] < prev_close) &
        (df["body_atr"] > engulf_body)
    )
    
    # Displacement
    displacement_up = (df["close"] > df["open"]) & (df["range_atr"] > displacement_range)
    
    # Combine
    candidate_long = sweep_long | breakout_long | engulf_long | displacement_up
    
    return candidate_long.astype(float)


def generate_labels(df, tp_mult=2.0, sl_mult=1.0, max_horizon=50):
    """Generate TP/SL labels."""
    labels = np.full(len(df), np.nan)
    
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr_vals = df["atr"].to_numpy()
    
    for i in range(len(df) - max_horizon):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        
        entry = close[i]
        tp = entry + tp_mult * atr_vals[i]
        sl = entry - sl_mult * atr_vals[i]
        
        for j in range(i + 1, i + max_horizon + 1):
            if low[j] <= sl:
                labels[i] = 0
                break
            if high[j] >= tp:
                labels[i] = 1
                break
    
    return labels


def main():
    """Test different signal configurations."""
    
    print("="*70)
    print("QUICK SIGNAL FILTER TESTER")
    print("="*70)
    print()
    
    # Load data
    print("Loading XAUUSD_M15.csv...")
    df = load_raw_ohlcv("XAUUSD_M15.csv")
    print(f"Loaded {len(df):,} bars\n")
    
    # Add metrics
    print("Computing metrics...")
    df = add_base_metrics(df)
    
    # Add labels
    print("Generating labels...")
    df["label"] = generate_labels(df)
    labeled = df[~df["label"].isna()]
    print(f"Generated {len(labeled):,} labeled setups\n")
    
    # Test configurations
    configs = [
        ("Current (Baseline)", {"sweep_body": 0.0, "breakout_comp": 0.9, "engulf_body": 0.6, "displacement_range": 1.25}),
        ("Moderate", {"sweep_body": 0.3, "breakout_comp": 0.85, "engulf_body": 0.8, "displacement_range": 1.35}),
        ("Strict v1", {"sweep_body": 0.5, "breakout_comp": 0.8, "engulf_body": 0.9, "displacement_range": 1.4}),
        ("Strict v2", {"sweep_body": 0.6, "breakout_comp": 0.75, "engulf_body": 1.0, "displacement_range": 1.5}),
        ("Very Strict", {"sweep_body": 0.7, "breakout_comp": 0.7, "engulf_body": 1.1, "displacement_range": 1.6}),
    ]
    
    print("="*70)
    print("TESTING CONFIGURATIONS")
    print("="*70)
    print()
    
    results = []
    
    for name, params in configs:
        print(f"Testing: {name}")
        print(f"  Sweep body: {params['sweep_body']}, Breakout compression: {params['breakout_comp']}")
        print(f"  Engulf body: {params['engulf_body']}, Displacement range: {params['displacement_range']}")
        
        # Get candidates
        candidates = test_signals(
            df,
            sweep_body=params["sweep_body"],
            breakout_comp=params["breakout_comp"],
            engulf_body=params["engulf_body"],
            displacement_range=params["displacement_range"]
        )
        
        # Filter to labeled samples
        has_label = ~df["label"].isna()
        candidates_labeled = candidates[has_label]
        labels_labeled = df.loc[has_label, "label"]
        
        # Calculate win rate among candidates
        if candidates_labeled.sum() > 0:
            wins = labels_labeled[candidates_labeled > 0].sum()
            total_candidates = int(candidates_labeled.sum())
            win_rate = wins / total_candidates
            
            # Expected value
            ev = win_rate * 2.0 - (1 - win_rate) * 1.0
        else:
            total_candidates = 0
            win_rate = 0
            ev = 0
        
        status = "OK" if win_rate >= 0.50 else "LOW" if win_rate >= 0.40 else "BAD"
        
        print(f"  Result: {total_candidates:,} candidates, {win_rate:.1%} win rate, E[V]={ev:.2f}R [{status}]")
        print()
        
        results.append({
            "config": name,
            "sweep_body": params["sweep_body"],
            "breakout_comp": params["breakout_comp"],
            "engulf_body": params["engulf_body"],
            "displacement_range": params["displacement_range"],
            "candidates": total_candidates,
            "win_rate": win_rate,
            "expected_value": ev,
        })
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("win_rate", ascending=False)
    
    print(df_results[["config", "candidates", "win_rate", "expected_value"]].to_string(index=False))
    print()
    
    # Recommendation
    best = df_results.iloc[0]
    
    print("="*70)
    if best["win_rate"] >= 0.50:
        print(f"SUCCESS! {best['config']} achieves {best['win_rate']:.1%} win rate")
        print()
        print("Next steps:")
        print(f"1. Edit price_action_mtf_dataset.py with these parameters:")
        print(f"   - sweep_body >= {best['sweep_body']}")
        print(f"   - breakout_comp < {best['breakout_comp']}")
        print(f"   - engulf_body > {best['engulf_body']}")
        print(f"   - displacement_range > {best['displacement_range']}")
        print(f"2. Run: python price_action_mtf_dataset.py XAUUSD_15m_yahoo.csv --output dataset.npz")
        print(f"3. Run: python train_lstm_price_action.py dataset.npz")
    else:
        print(f"CHALLENGE: Best config {best['config']} only achieves {best['win_rate']:.1%}")
        print()
        print("Top 3 options:")
        for idx, row in df_results.head(3).iterrows():
            print(f"  {row['config']}: {row['win_rate']:.1%} win rate ({row['candidates']:,} setups)")
        print()
        print("Recommendation:")
        print("  Try combining elements from strict configs, or redesign signals entirely")
        print("  Consider: higher timeframe alignment, volume confirmation, volatility filters")


if __name__ == "__main__":
    main()

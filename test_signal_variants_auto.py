#!/usr/bin/env python3
"""
Signal Variant Tester: Generate and test multiple signal filter combinations.
Tests 5 different tightness levels to find the best performing signals.
"""

import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


VARIANTS = {
    "strict_v1": {
        "description": "Very tight: High body ratio + volume confirmation",
        "sweep_body": 0.6,
        "sweep_volume": 1.0,
        "breakout_compression": 0.7,
        "breakout_body": 0.8,
        "engulf_body": 1.0,
        "displacement_range": 1.5,
    },
    "strict_v2": {
        "description": "Tight: Higher body thresholds + compression filter",
        "sweep_body": 0.5,
        "sweep_volume": 0.9,
        "breakout_compression": 0.8,
        "breakout_body": 0.7,
        "engulf_body": 0.9,
        "displacement_range": 1.4,
    },
    "moderate": {
        "description": "Moderate: Balanced thresholds",
        "sweep_body": 0.4,
        "sweep_volume": 0.8,
        "breakout_compression": 0.85,
        "breakout_body": 0.6,
        "engulf_body": 0.8,
        "displacement_range": 1.3,
    },
    "loose": {
        "description": "Loose: Current defaults (baseline)",
        "sweep_body": 0.0,  # No filter
        "sweep_volume": 0.0,  # No filter
        "breakout_compression": 0.9,
        "breakout_body": 0.0,  # No filter
        "engulf_body": 0.6,
        "displacement_range": 1.25,
    },
    "very_strict": {
        "description": "Extreme: Very high selectivity",
        "sweep_body": 0.7,
        "sweep_volume": 1.1,
        "breakout_compression": 0.65,
        "breakout_body": 0.9,
        "engulf_body": 1.1,
        "displacement_range": 1.6,
    },
}


def create_variant_script(variant_name, params):
    """Generate a modified price_action_mtf_dataset.py with custom parameters."""
    
    # Read the original
    with open("price_action_mtf_dataset.py", "r") as f:
        original = f.read()
    
    # Extract the add_base_price_action function
    lines = original.split("\n")
    
    # Find the function start
    func_start = None
    for i, line in enumerate(lines):
        if "def add_base_price_action(df" in line:
            func_start = i
            break
    
    if func_start is None:
        print(f"ERROR: Could not find add_base_price_action function")
        return False
    
    # Create modified version
    modified = "\n".join(lines[:func_start]) + "\n"
    modified += f'''def add_base_price_action(df, swing_lookback=20, range_lookback=20):
    df = df.copy()
    eps = 1e-9

    df["atr"] = atr(df)
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_atr"] = df["body"] / (df["atr"] + eps)
    df["range_atr"] = df["range"] / (df["atr"] + eps)
    df["upper_wick_atr"] = df["upper_wick"] / (df["atr"] + eps)
    df["lower_wick_atr"] = df["lower_wick"] / (df["atr"] + eps)

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["close_to_ema20_atr"] = (df["close"] - df["ema20"]) / (df["atr"] + eps)
    df["close_to_ema50_atr"] = (df["close"] - df["ema50"]) / (df["atr"] + eps)
    df["ema20_slope_atr"] = df["ema20"].diff(5) / (df["atr"] + eps)
    df["ema50_slope_atr"] = df["ema50"].diff(5) / (df["atr"] + eps)

    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + eps)
    df["spread_atr"] = df["spread"] / (df["atr"] + eps)

    swing_high = df["high"].rolling(swing_lookback).max().shift(1)
    swing_low = df["low"].rolling(swing_lookback).min().shift(1)
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low

    # SWEEP: Modified with tighter filters
    sweep_body_threshold = {params["sweep_body"]}
    sweep_volume_threshold = {params["sweep_volume"]}
    
    df["sweep_short"] = (
        (df["high"] > swing_high) & 
        (df["close"] < swing_high) &
        (df["body_atr"] > sweep_body_threshold if sweep_body_threshold > 0 else True) &
        (df["volume_ratio"] > sweep_volume_threshold if sweep_volume_threshold > 0 else True)
    )
    df["sweep_long"] = (
        (df["low"] < swing_low) & 
        (df["close"] > swing_low) &
        (df["body_atr"] > sweep_body_threshold if sweep_body_threshold > 0 else True) &
        (df["volume_ratio"] > sweep_volume_threshold if sweep_volume_threshold > 0 else True)
    )

    range_high = df["high"].rolling(range_lookback).max().shift(1)
    range_low = df["low"].rolling(range_lookback).min().shift(1)
    range_size = range_high - range_low
    range_ma = range_size.rolling(range_lookback).mean()
    compression = range_size / (range_ma + eps)

    df["range_high"] = range_high
    df["range_low"] = range_low
    df["range_compression"] = compression
    
    # BREAKOUT: Modified with tighter compression and body requirements
    breakout_compression = {params["breakout_compression"]}
    breakout_body = {params["breakout_body"]}
    
    df["breakout_long"] = (
        (df["close"] > range_high) & 
        (compression < breakout_compression) &
        (df["body_atr"] > breakout_body if breakout_body > 0 else True)
    )
    df["breakout_short"] = (
        (df["close"] < range_low) & 
        (compression < breakout_compression) &
        (df["body_atr"] > breakout_body if breakout_body > 0 else True)
    )

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    
    # ENGULFING: Modified with tighter body requirement
    engulf_body = {params["engulf_body"]}
    
    df["engulf_long"] = (
        (df["close"] > df["open"])
        & (prev_close < prev_open)
        & (df["close"] > prev_open)
        & (df["open"] < prev_close)
        & (df["body_atr"] > engulf_body)
    )
    df["engulf_short"] = (
        (df["close"] < df["open"])
        & (prev_close > prev_open)
        & (df["close"] < prev_open)
        & (df["open"] > prev_close)
        & (df["body_atr"] > engulf_body)
    )

    # DISPLACEMENT: Modified with tighter range requirement
    displacement_range = {params["displacement_range"]}
    
    df["displacement_up"] = (df["close"] > df["open"]) & (df["range_atr"] > displacement_range)
    df["displacement_down"] = (df["close"] < df["open"]) & (df["range_atr"] > displacement_range)

    for col in [
        "sweep_long",
        "sweep_short",
        "breakout_long",
        "breakout_short",
        "engulf_long",
        "engulf_short",
        "displacement_up",
        "displacement_down",
    ]:
        df[col] = df[col].astype(float)

    df["candidate_long"] = (
        (df["sweep_long"] > 0)
        | (df["breakout_long"] > 0)
        | (df["engulf_long"] > 0)
        | (df["displacement_up"] > 0)
    )
    df["candidate_short"] = (
        (df["sweep_short"] > 0)
        | (df["breakout_short"] > 0)
        | (df["engulf_short"] > 0)
        | (df["displacement_down"] > 0)
    )
    df["setup_direction"] = np.select(
        [df["candidate_long"], df["candidate_short"]],
        [1, -1],
        default=0,
    )
    return df

'''
    
    # Find where to continue
    func_end = None
    for i in range(func_start + 1, len(lines)):
        if lines[i].startswith("def ") and i > func_start + 5:
            func_end = i
            break
    
    if func_end:
        modified += "\n".join(lines[func_end:])
    
    return modified


def run_variant(variant_name, params, csv_file="XAUUSD_15m_yahoo.csv"):
    """Test a signal variant."""
    
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant_name}")
    print(f"Description: {params['description']}")
    print(f"{'='*70}")
    print()
    
    # Create modified script
    print("Generating modified script...")
    modified_script = create_variant_script(variant_name, params)
    
    if not modified_script:
        print("ERROR: Could not create variant script")
        return None
    
    # Write to temp file
    temp_file = f"price_action_mtf_{variant_name}.py"
    with open(temp_file, "w") as f:
        f.write(modified_script)
    
    # Run dataset generation
    print(f"Generating dataset...")
    cmd = [
        sys.executable,
        temp_file,
        csv_file,
        "--output", f"dataset_{variant_name}.npz",
        "--meta-output", f"meta_{variant_name}.csv",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    print(result.stdout)
    
    # Analyze results
    print("Analyzing signal quality...")
    meta_file = f"meta_{variant_name}.csv"
    
    try:
        meta = pd.read_csv(meta_file)
        wr = meta['label'].mean()
        num_setups = len(meta)
        num_winners = (meta['label'] == 1).sum()
        num_losers = (meta['label'] == 0).sum()
        
        # Expected value
        ev = wr * 2.0 - (1 - wr) * 1.0
        
        print(f"Results:")
        print(f"  Setups: {num_setups:,}")
        print(f"  Winners: {num_winners:,} ({wr:.1%})")
        print(f"  Losers: {num_losers:,} ({(1-wr):.1%})")
        print(f"  Expected Value: {ev:.2f}R")
        
        if wr >= 0.50:
            status = "GOOD"
            icon = "[OK]"
        elif wr >= 0.45:
            status = "IMPROVING"
            icon = "[~]"
        else:
            status = "POOR"
            icon = "[X]"
        
        print(f"  Status: {icon} {status}")
        print()
        
        return {
            "variant": variant_name,
            "description": params['description'],
            "win_rate": wr,
            "num_setups": num_setups,
            "num_winners": num_winners,
            "expected_value": ev,
            "status": status,
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Test all variants and report results."""
    
    csv_file = "XAUUSD_15m_yahoo.csv"
    
    # Check if CSV exists
    if not Path(csv_file).exists():
        print(f"ERROR: {csv_file} not found")
        print("Please download Yahoo data first:")
        print("  python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m")
        sys.exit(1)
    
    print("="*70)
    print("SIGNAL VARIANT TESTER")
    print("="*70)
    print(f"Testing {len(VARIANTS)} signal filter combinations...")
    print(f"CSV: {csv_file}")
    print()
    
    results = []
    
    # Test each variant
    for variant_name in ["loose", "moderate", "strict_v2", "strict_v1", "very_strict"]:
        result = run_variant(variant_name, VARIANTS[variant_name], csv_file)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("win_rate", ascending=False)
    
    print(df_results[["variant", "win_rate", "num_setups", "expected_value", "status"]].to_string(index=False))
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    best = df_results.iloc[0]
    print(f"\nBest variant: {best['variant']}")
    print(f"  Win rate: {best['win_rate']:.1%}")
    print(f"  Setups: {best['num_setups']:,}")
    print(f"  Expected value: {best['expected_value']:.2f}R")
    
    if best['win_rate'] >= 0.50:
        print(f"\nVERDICT: {best['variant']} achieves >50% win rate!")
        print(f"Use: cp price_action_mtf_{best['variant']}.py price_action_mtf_dataset.py")
        print(f"Then retrain LSTM with: python train_lstm_price_action.py dataset_{best['variant']}.npz")
    else:
        top3 = df_results.head(3)
        print("\nNo variant achieved 50%+ win rate.")
        print("Top 3 options:")
        for idx, row in top3.iterrows():
            print(f"  {row['variant']}: {row['win_rate']:.1%} ({row['num_setups']:,} setups)")
        print("\nRecommendation: Combine best elements from top variants, or redesign signals.")


if __name__ == "__main__":
    main()

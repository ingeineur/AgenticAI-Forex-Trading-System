#!/usr/bin/env python3
"""
Test signal filter variants to find parameters that improve win rate.
"""

import sys
import subprocess
import pandas as pd
import numpy as np


def run_experiment(variant_name, additional_args):
    """Run dataset generation + analysis, return metrics."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant_name}")
    print(f"{'='*70}")
    
    # Generate dataset with variant parameters
    cmd = [
        sys.executable,
        "price_action_mtf_dataset.py",
        "XAUUSD_M15.csv",
        "--output", f"dataset_{variant_name}.npz",
        "--meta-output", f"meta_{variant_name}.csv",
    ] + additional_args
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    print(result.stdout)
    
    # Analyze results
    cmd_analyze = [sys.executable, "analyze_signal_quality.py", f"meta_{variant_name}.csv"]
    result_analyze = subprocess.run(cmd_analyze, capture_output=True, text=True)
    print(result_analyze.stdout)
    
    # Extract win rate
    meta = pd.read_csv(f"meta_{variant_name}.csv")
    wr = meta['label'].mean()
    
    return {
        "variant": variant_name,
        "win_rate": wr,
        "num_setups": len(meta),
        "num_winners": (meta['label'] == 1).sum(),
    }


if __name__ == "__main__":
    results = []
    
    # Baseline (current)
    results.append(run_experiment(
        "baseline",
        []  # Default parameters
    ))
    
    # Variant 1: Shorter sequence
    results.append(run_experiment(
        "seq32",
        ["--sequence-length", "32"]
    ))
    
    # Variant 2: Reduced horizon
    results.append(run_experiment(
        "horizon30",
        ["--max-horizon", "30"]
    ))
    
    # Variant 3: Higher TP multiplier (only label strong wins)
    results.append(run_experiment(
        "tp3r",
        ["--tp-mult", "3.0"]
    ))
    
    print(f"\n{'='*70}")
    print("SUMMARY OF VARIANTS")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame([r for r in results if r is not None])
    df_results = df_results.sort_values("win_rate", ascending=False)
    
    print(df_results.to_string(index=False))
    
    best = df_results.iloc[0]
    print(f"\nBest variant: {best['variant']} ({best['win_rate']:.1%} win rate)")

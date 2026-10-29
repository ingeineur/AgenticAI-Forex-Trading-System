#!/usr/bin/env python3
"""
Analyze price action signal quality: do they correlate with profitable outcomes?
"""

import numpy as np
import pandas as pd
import argparse


def analyze_signal_quality(meta_csv_path):
    """Load metadata and check signal effectiveness."""
    print("=" * 70)
    print(f"Analyzing signal quality from: {meta_csv_path}")
    print("=" * 70)

    meta = pd.read_csv(meta_csv_path)
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(meta):,}")
    print(f"  Positive (won): {(meta['label'] == 1).sum():,}")
    print(f"  Negative (lost): {(meta['label'] == 0).sum():,}")
    
    # Overall win rate
    overall_wr = meta['label'].mean()
    print(f"\nOverall Win Rate: {overall_wr:.1%}")
    
    if overall_wr <= 0.50:
        print("  [CRITICAL] Win rate is at or below 50%!")
        print("  -> Signals are NOT better than random entry.")
        print("  -> Reverting to random signals would break even; this is worse.")
    
    # Direction-based analysis
    print(f"\nWin Rate by Setup Direction:")
    for direction in [1, -1]:
        subset = meta[meta['setup_direction'] == direction]
        if len(subset) > 0:
            wr = subset['label'].mean()
            print(f"  Direction {direction:2d}: {len(subset):,} setups, {wr:.1%} win rate")
    
    # HTF alignment analysis
    print(f"\nWin Rate by HTF Alignment:")
    meta['alignment_bucket'] = pd.cut(
        meta['htf_alignment'], 
        bins=[-2, -0.5, 0.5, 2], 
        labels=['Opposed', 'Neutral', 'Aligned']
    )
    for bucket in ['Opposed', 'Neutral', 'Aligned']:
        subset = meta[meta['alignment_bucket'] == bucket]
        if len(subset) > 0:
            wr = subset['label'].mean()
            print(f"  {bucket:10s}: {len(subset):,} setups, {wr:.1%} win rate")
    
    # Risk/Reward analysis
    print(f"\nRisk Profile (ATR-based):")
    print(f"  Mean ATR: {meta['atr'].mean():.4f}")
    print(f"  Median ATR: {meta['atr'].median():.4f}")
    print(f"  Min ATR: {meta['atr'].min():.4f}")
    print(f"  Max ATR: {meta['atr'].max():.4f}")
    
    # Expected value
    tp_mult = 2.0
    sl_mult = 1.0
    expected_value = overall_wr * tp_mult - (1 - overall_wr) * sl_mult
    print(f"\nExpected Value per trade (2R TP, 1R SL):")
    print(f"  E[V] = {overall_wr:.1%} * 2R - {(1-overall_wr):.1%} * 1R = {expected_value:.2f}R")
    
    if expected_value <= 0:
        print(f"  [CRITICAL] Negative expectancy! Strategy is unprofitable.")
    elif expected_value < 0.1:
        print(f"  [WARNING] Very low expectancy. High risk of drawdown.")
    else:
        print(f"  [OK] Positive expectancy.")
    
    # Sequence of wins/losses
    print(f"\nWin/Loss Sequence (first 100 outcomes):")
    sequence = meta['label'].head(100).astype(int).astype(str).str.replace('1', 'W').str.replace('0', 'L')
    print(f"  {''.join(sequence)}")
    
    # Consecutive wins/losses
    runs = []
    current_run = (meta['label'].iloc[0], 1)
    for i in range(1, min(len(meta), 1000)):
        if meta['label'].iloc[i] == current_run[0]:
            current_run = (current_run[0], current_run[1] + 1)
        else:
            runs.append(current_run)
            current_run = (meta['label'].iloc[i], 1)
    runs.append(current_run)
    
    wins = [r[1] for r in runs if r[0] == 1]
    losses = [r[1] for r in runs if r[0] == 0]
    
    if wins:
        print(f"\nConsecutive Wins (first 1000 trades):")
        print(f"  Mean: {np.mean(wins):.1f}, Max: {max(wins)}, Min: {min(wins)}")
    if losses:
        print(f"\nConsecutive Losses (first 1000 trades):")
        print(f"  Mean: {np.mean(losses):.1f}, Max: {max(losses)}, Min: {min(losses)}")
    
    print("\n" + "=" * 70)
    print("SIGNAL QUALITY VERDICT:")
    print("=" * 70)
    
    if overall_wr < 0.50:
        print("[CRITICAL] Signals are worse than random!")
        print("  -> Do NOT train an LSTM on these signals.")
        print("  -> The model cannot learn from unprofitable bases.")
        print("  -> ACTION: Review price action rules, tighten entry filters.")
    elif overall_wr < 0.55:
        print("[WARNING] Low signal quality (< 55% win rate).")
        print("  -> LSTM may struggle to find discriminative patterns.")
        print("  -> Consider: increasing sequence length, adding more features, or filtering setups.")
    else:
        print("[OK] Signal quality is reasonable.")
        print("  -> LSTM should have something to learn from.")
        print("  -> If test metrics are still poor, check for overfitting or data scaling issues.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze price action signal quality.")
    parser.add_argument("meta_csv", help="Path to metadata .csv file")
    args = parser.parse_args()
    analyze_signal_quality(args.meta_csv)

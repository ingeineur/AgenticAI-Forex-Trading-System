#!/usr/bin/env python3
"""Test different TP/SL ratios to find optimal asymmetric structure for XAUUSD M15"""

import numpy as np
import pandas as pd
from itertools import product


def check_signals(df, tp_mult, sl_mult, max_horizon=50):
    """Calculate win rate for given TP/SL ratio"""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    
    atr_vals = pd.Series(high - low).rolling(14).mean().to_numpy()
    
    wins = 0
    total = 0
    
    for i in range(len(df) - max_horizon):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        
        entry = close[i]
        tp = entry + tp_mult * atr_vals[i]
        sl = entry - sl_mult * atr_vals[i]
        
        hit_sl = False
        hit_tp = False
        
        for j in range(i + 1, i + max_horizon + 1):
            if low[j] <= sl:
                hit_sl = True
                break
            if high[j] >= tp:
                hit_tp = True
                break
        
        if hit_sl or hit_tp:
            total += 1
            if hit_tp:
                wins += 1
    
    if total == 0:
        return 0.0, 0.0, 0
    
    wr = wins / total
    ev = wr * tp_mult - (1 - wr) * sl_mult
    return wr, ev, total


def main():
    df = pd.read_csv("XAUUSD_M15.csv")
    print("="*70)
    print("TP/SL RATIO OPTIMIZATION FOR XAUUSD M15")
    print("="*70)
    print(f"\nData: {len(df)} bars (2022-2025)")
    print("\nTesting asymmetric TP/SL ratios...\n")
    
    ratios = [
        (0.5, 1.0), (0.8, 1.0), (1.0, 1.0), (1.2, 1.0), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0),
        (1.0, 0.5), (1.0, 0.8), (1.0, 1.2), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0),
        (0.75, 1.0), (1.5, 0.75), (2.0, 1.5), (1.5, 1.5), (2.0, 2.0), (3.0, 3.0),
        (1.0, 0.75), (1.25, 1.0), (1.33, 1.0), (1.0, 1.33), (1.5, 1.2), (1.2, 1.5),
    ]
    
    results = []
    for tp_mult, sl_mult in ratios:
        wr, ev, total = check_signals(df, tp_mult, sl_mult, max_horizon=50)
        results.append({
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "ratio": f"{tp_mult:.2f}/{sl_mult:.2f}",
            "win_rate": wr,
            "expected_value": ev,
            "total_trades": total
        })
    
    results.sort(key=lambda x: x["expected_value"], reverse=True)
    
    print(f"{'Ratio':<16} {'TP/SL':<10} {'Win Rate':<12} {'Exp Value':<12} {'Trades':<10}")
    print("-"*60)
    
    for r in results:
        ratio_str = f"{r['tp_mult']:.2f}R/{r['sl_mult']:.2f}R"
        tp_sl_str = f"{r['tp_mult']/r['sl_mult']:.2f}"
        print(f"{ratio_str:<16} {tp_sl_str:<10} {r['win_rate']:<12.1%} {r['expected_value']:<12.2f}R {r['total_trades']:<10,}")
    
    print()
    
    best_ev = max(results, key=lambda x: x["expected_value"])
    best_wr = max(results, key=lambda x: x["win_rate"] if x["total_trades"] > 100 else -1)
    
    print(f"Best by Expected Value: {best_ev['ratio']} ({best_ev['expected_value']:.2f}R, {best_ev['win_rate']:.1%} WR)")
    print(f"Best by Win Rate: {best_wr['ratio']} ({best_wr['win_rate']:.1%} WR, {best_wr['expected_value']:.2f}R)")
    print()
    
    # Find best combinations with >0.5 EV
    good = [r for r in results if r["expected_value"] > 0.1 and r["win_rate"] > 0.3]
    if good:
        print("Recommended ratios (EV > 0.1R, WR > 30%):")
        for r in sorted(good, key=lambda x: x["expected_value"], reverse=True)[:5]:
            print(f"  {r['ratio']}: {r['win_rate']:.1%} WR, {r['expected_value']:.2f}R EV ({r['total_trades']:,} trades)")


if __name__ == "__main__":
    main()

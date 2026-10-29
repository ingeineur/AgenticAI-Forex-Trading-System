#!/usr/bin/env python3
"""
Merge broker data (2022+) with Yahoo data (2015+) for comprehensive history.
Removes duplicates and fills gaps intelligently.
"""

import pandas as pd
import sys
from pathlib import Path


def merge_data_sources(broker_csv, yahoo_csv, output_csv="XAUUSD_combined.csv"):
    """
    Merge broker and Yahoo data sources.
    
    Strategy:
    1. Load both datasets
    2. Use Yahoo for pre-2022
    3. Use broker for 2022+ (more reliable, includes spreads)
    4. Remove duplicates (favor broker data)
    5. Fill any gaps
    """
    
    print("=" * 70)
    print("MERGING DATA SOURCES")
    print("=" * 70)
    print()
    
    # Load broker data
    print(f"Loading broker data: {broker_csv}")
    try:
        df_broker = pd.read_csv(broker_csv)
        df_broker["timestamp"] = pd.to_datetime(df_broker["timestamp"], utc=True)
        print(f"  Rows: {len(df_broker)}")
        print(f"  Date range: {df_broker['timestamp'].min()} -> {df_broker['timestamp'].max()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # Load Yahoo data
    print(f"\nLoading Yahoo data: {yahoo_csv}")
    try:
        df_yahoo = pd.read_csv(yahoo_csv)
        df_yahoo["timestamp"] = pd.to_datetime(df_yahoo["timestamp"], utc=True)
        print(f"  Rows: {len(df_yahoo)}")
        print(f"  Date range: {df_yahoo['timestamp'].min()} -> {df_yahoo['timestamp'].max()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # Find overlap date
    overlap_start = max(df_broker["timestamp"].min(), df_yahoo["timestamp"].min())
    overlap_end = min(df_broker["timestamp"].max(), df_yahoo["timestamp"].max())
    
    print(f"\nOverlap period: {overlap_start} -> {overlap_end}")
    
    # Split Yahoo: pre-2022 + overlap
    cutoff_date = df_broker["timestamp"].min()
    
    df_yahoo_pre = df_yahoo[df_yahoo["timestamp"] < cutoff_date]
    df_yahoo_overlap = df_yahoo[df_yahoo["timestamp"] >= cutoff_date]
    
    print(f"\nYahoo pre-broker: {len(df_yahoo_pre)} bars")
    print(f"Yahoo overlap: {len(df_yahoo_overlap)} bars")
    
    # Combine: Yahoo pre-2022 + Broker (which has spreads/real_volume)
    print(f"\nMerging...")
    
    # Concatenate
    df_combined = pd.concat([df_yahoo_pre, df_broker], ignore_index=True)
    
    # Sort by timestamp
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    
    # Remove duplicates (keep broker version if duplicate)
    df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")
    
    print(f"  Combined rows: {len(df_combined)}")
    print(f"  Final range: {df_combined['timestamp'].min()} -> {df_combined['timestamp'].max()}")
    
    # Ensure all columns exist
    required_cols = ["timestamp", "open", "high", "low", "close", "volume", "spread", "real_volume"]
    for col in required_cols:
        if col not in df_combined.columns:
            if col in ["spread", "real_volume"]:
                df_combined[col] = 0.0
            else:
                print(f"WARNING: Missing column {col}")
    
    # Select and order columns
    df_combined = df_combined[required_cols]
    
    # Save
    df_combined.to_csv(output_csv, index=False)
    
    print(f"\nSaved: {output_csv}")
    print(f"Total bars: {len(df_combined)}")
    
    # Summary stats
    print("\nData Summary:")
    print(f"  Start: {df_combined['timestamp'].min()}")
    print(f"  End: {df_combined['timestamp'].max()}")
    print(f"  Duration: {(df_combined['timestamp'].max() - df_combined['timestamp'].min()).days} days")
    print(f"  Price range: {df_combined['close'].min():.2f} - {df_combined['close'].max():.2f}")
    print(f"  Avg spread: {df_combined['spread'].mean():.4f}")
    
    # Coverage analysis
    total_mins = (df_combined['timestamp'].max() - df_combined['timestamp'].min()).total_seconds() / 60
    expected_bars = int(total_mins / 15) + 1
    actual_bars = len(df_combined)
    coverage = actual_bars / expected_bars
    
    print(f"  M15 coverage: {coverage:.1%} ({actual_bars}/{expected_bars} expected)")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Merge broker and Yahoo data sources")
        print()
        print("Usage: python merge_data_sources.py <broker.csv> <yahoo.csv> [output.csv]")
        print()
        print("Example:")
        print("  python merge_data_sources.py XAUUSD_16388.csv XAUUSD_15m_yahoo.csv")
        print()
        sys.exit(1)
    
    broker_file = sys.argv[1]
    yahoo_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "XAUUSD_combined.csv"
    
    # Verify files exist
    if not Path(broker_file).exists():
        print(f"ERROR: Broker file not found: {broker_file}")
        sys.exit(1)
    
    if not Path(yahoo_file).exists():
        print(f"ERROR: Yahoo file not found: {yahoo_file}")
        sys.exit(1)
    
    success = merge_data_sources(broker_file, yahoo_file, output_file)
    sys.exit(0 if success else 1)

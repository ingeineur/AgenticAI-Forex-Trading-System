#!/usr/bin/env python3
"""
Diagnostic: Check which years of data are available on your MT5 broker.
Run this FIRST to understand data limitations before full download.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import sys

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15


def check_data_availability():
    """Test data availability for each year."""
    
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    
    print(f"Checking data availability for: {SYMBOL} M15")
    print("=" * 70)
    print()
    
    # Test each year from 2015 to 2025
    results = []
    
    for year in range(2015, 2026):
        start = datetime(year, 1, 1, tzinfo=pytz.UTC)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=pytz.UTC)
        
        print(f"Testing {year}...", end=" ", flush=True)
        
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start, end)
        
        if rates is None:
            print(f"FAILED ({mt5.last_error()})")
            results.append({"year": year, "status": "ERROR", "bars": 0})
        elif len(rates) == 0:
            print(f"NO DATA")
            results.append({"year": year, "status": "NO_DATA", "bars": 0})
        else:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            date_min = df["time"].min()
            date_max = df["time"].max()
            
            # Calculate expected bars (M15, ~250 trading days/year)
            expected = 250 * 24 * 4  # M15 bars
            actual = len(df)
            coverage = actual / expected
            
            status = "OK" if coverage > 0.95 else "PARTIAL"
            print(f"{status}: {actual:,} bars ({coverage:.1%})")
            
            results.append({
                "year": year,
                "status": status,
                "bars": actual,
                "coverage": f"{coverage:.1%}",
                "start": date_min.strftime("%Y-%m-%d"),
                "end": date_max.strftime("%Y-%m-%d")
            })
    
    mt5.shutdown()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    print()
    
    # Find continuous range
    ok_years = df_results[df_results["status"].isin(["OK", "PARTIAL"])]["year"].tolist()
    
    if ok_years:
        min_year = min(ok_years)
        max_year = max(ok_years)
        print(f"Data available from: {min_year} to {max_year}")
        print(f"Recommended download range: {min_year}-01-01 to {max_year+1}-01-01")
    else:
        print("No data found! Check:")
        print("  1. Is MT5 terminal running?")
        print("  2. Is the symbol name correct?")
        print("  3. Does your broker provide historical data?")


if __name__ == "__main__":
    check_data_availability()

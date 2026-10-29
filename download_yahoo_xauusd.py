#!/usr/bin/env python3
"""
Alternative data source: Download XAUUSD from Yahoo Finance
This provides data back to 2015 (or earlier) without broker limitations.

Requirements: pip install yfinance pandas
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    print("Install with: pip install yfinance")
    sys.exit(1)


def download_xauusd_yahoo(start_date, end_date, interval="15m"):
    """
    Download XAUUSD from Yahoo Finance.
    
    Args:
        start_date: datetime object (e.g., datetime(2015, 1, 1))
        end_date: datetime object
        interval: "1m", "5m", "15m", "1h", "1d", etc.
    
    Returns:
        DataFrame with OHLCV data
    """
    
    print(f"Downloading XAUUSD from Yahoo Finance")
    print(f"  Start: {start_date.date()}")
    print(f"  End: {end_date.date()}")
    print(f"  Interval: {interval}")
    print()
    
    # Yahoo Finance ticker for gold
    ticker = "GC=F"  # Comex Gold Futures
    
    print(f"Fetching {ticker} ({interval} bars)...")
    
    try:
        # Download data
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if df.empty:
            print("ERROR: No data returned from Yahoo Finance")
            return None
        
        # Rename columns to match MT5 format
        df = df.reset_index()
        df = df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Ensure timestamp is UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        
        # Add missing columns
        df["spread"] = 0.0
        df["real_volume"] = df["volume"]  # Yahoo doesn't distinguish
        
        # Select and order columns
        df = df[[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
            "real_volume"
        ]]
        
        # Remove any NaN rows
        df = df.dropna(subset=["open", "high", "low", "close"])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"Success: Got {len(df)} bars")
        print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        print()
        
        return df
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return None


def main():
    # Default to last 10 years of M15 data
    end_date = datetime.now(tz=pytz.UTC)
    start_date = end_date.replace(year=end_date.year - 10)
    
    # Allow command-line overrides
    if len(sys.argv) > 1:
        try:
            start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}")
            print("Usage: python download_yahoo_xauusd.py [START_DATE] [END_DATE] [INTERVAL]")
            print("Example: python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        except ValueError:
            print(f"Invalid date format: {sys.argv[2]}")
            sys.exit(1)
    
    interval = sys.argv[3] if len(sys.argv) > 3 else "15m"
    
    # Download data
    df = download_xauusd_yahoo(start_date, end_date, interval)
    
    if df is None or df.empty:
        print("FAILED: Could not download data")
        sys.exit(1)
    
    # Save to CSV
    output_file = f"XAUUSD_{interval}_yahoo.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    print(f"Rows: {len(df)}")
    print()
    
    # Summary
    print("Data Summary:")
    print(f"  Start: {df['timestamp'].min()}")
    print(f"  End: {df['timestamp'].max()}")
    print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
    print(f"  Open: {df['open'].min():.2f} - {df['open'].max():.2f}")
    print(f"  High: {df['high'].min():.2f} - {df['high'].max():.2f}")
    print(f"  Low: {df['low'].min():.2f} - {df['low'].max():.2f}")
    print(f"  Close: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    print("\nNOTE: This is commodity futures data (GC=F). Verify it matches your broker's quotes.")


if __name__ == "__main__":
    main()

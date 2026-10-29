#!/usr/bin/env python3
"""Download XAUUSD from Yahoo Finance, handling multi-level columns"""

import sys
import pandas as pd
from datetime import datetime
import pytz

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Install with: pip install yfinance")
    sys.exit(1)


def download_xauusd_yahoo(start_date, end_date, interval="1d"):
    """Download XAUUSD (GC=F) from Yahoo Finance."""
    print(f"Downloading XAUUSD (GC=F) from Yahoo Finance")
    print(f"  Range: {start_date.date()} to {end_date.date()}")
    print(f"  Interval: {interval}")
    print()
    
    ticker = "GC=F"
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            print("ERROR: No data returned")
            return None
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        rename_map = {
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
        
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        
        df["spread"] = 0.0
        df["real_volume"] = df["volume"]
        
        cols = ["timestamp", "open", "high", "low", "close", "volume", "spread", "real_volume"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"  Got {len(df)} bars")
        print(f"  Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        return df
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return None


def main():
    end_date = datetime.now(tz=pytz.UTC)
    start_date = end_date.replace(year=end_date.year - 10)
    
    if len(sys.argv) > 1:
        start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    if len(sys.argv) > 2:
        end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"
    
    df = download_xauusd_yahoo(start_date, end_date, interval)
    if df is None:
        sys.exit(1)
    
    output = f"XAUUSD_{interval}_yahoo.csv"
    df.to_csv(output, index=False)
    print(f"Saved: {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download BTCUSD data from Yahoo Finance"""
import sys
import pandas as pd
from datetime import datetime
import pytz

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)


def download_btcusd(start_date, end_date, interval="15m"):
    print(f"Downloading BTC-USD from Yahoo Finance")
    print(f"  Range: {start_date.date()} to {end_date.date()}")
    print(f"  Interval: {interval}")
    
    try:
        df = yf.download("BTC-USD", start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        rename_map = {"Date": "timestamp", "Datetime": "timestamp",
                       "Open": "open", "High": "high", "Low": "low",
                       "Close": "close", "Volume": "volume"}
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
        
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    end_date = datetime.now(tz=pytz.UTC)
    
    # Try 15m (limited to ~60 days)
    start_15m = end_date - pd.Timedelta(days=60)
    interval = "15m"
    
    df_15m = download_btcusd(start_15m, end_date, interval)
    if df_15m is not None and not df_15m.empty:
        out = f"BTCUSD_M15_yahoo.csv"
        df_15m.to_csv(out, index=False)
        print(f"Saved M15 data: {out} ({len(df_15m)} rows)")
        print(f"  Range: {df_15m['timestamp'].min()} to {df_15m['timestamp'].max()}")
    else:
        print("No M15 data available from Yahoo for BTCUSD")
        print("Trying daily data instead...")
        
        # Daily data for full history
        df_1d = download_btcusd(datetime(2015, 1, 1, tzinfo=pytz.UTC), end_date, "1d")
        if df_1d is not None and not df_1d.empty:
            out = f"BTCUSD_1d_yahoo.csv"
            df_1d.to_csv(out, index=False)
            print(f"Saved daily data: {out} ({len(df_1d)} rows)")
            print(f"  Range: {df_1d['timestamp'].min()} to {df_1d['timestamp'].max()}")
        else:
            print("ERROR: Could not download BTCUSD data")


if __name__ == "__main__":
    main()

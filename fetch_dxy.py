"""Fetch DXY data from MT5 broker (primary) or Yahoo Finance (fallback)."""
import pandas as pd
import sys
import os

M15_SCRIPT = "extract_historical_data_v2.py"

def fetch_mt5(tf="h1", bars=25000):
    """Fetch DXY via MT5 broker using extract_historical_data_v2.py."""
    tf_map = {"h1": "H1", "m15": "M15", "h4": "H4", "d1": "D1"}
    tf_str = tf_map.get(tf, "H1")
    output = f"USDX_{tf_str}_combined.csv"
    
    if os.path.exists(output):
        df = pd.read_csv(output)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        print(f"  MT5: {output} already exists ({len(df)} bars)")
    else:
        print(f"  MT5: extracting USDX {tf_str} via extract_historical_data_v2.py...")
        ret = os.system(f"{sys.executable} extract_historical_data_v2.py USDX {tf_str}")
        if ret != 0:
            print("  MT5 extraction failed")
            return None
        df = pd.read_csv(output)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    if len(df) > bars:
        df = df.tail(bars).reset_index(drop=True)
    
    return df

def fetch_yahoo(tf="h1", bars=25000):
    """Fetch DXY via Yahoo Finance as fallback."""
    import yfinance as yf
    interval_map = {"m15": "15m", "h1": "60m", "h4": "60m", "d1": "1d"}
    interval = interval_map.get(tf, "60m")
    period = "max"
    if bars <= 5000:
        period = "60d" if tf == "m15" else "2y"
    elif bars <= 10000:
        period = "120d" if tf == "m15" else "5y"
    
    print(f"  Yahoo: fetching {interval} up to {bars} bars...")
    ticker = yf.Ticker("DX-Y.NYB")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        print("  Yahoo: No data")
        return None
    
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['Datetime' if 'Datetime' in df else 'Date'], utc=True)
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df = df.dropna(subset=['close'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if len(df) > bars:
        df = df.tail(bars).reset_index(drop=True)
    
    return df

def fetch_dxy(tf="h1", bars=25000, output="DXY_H1.csv"):
    """Fetch DXY data — try MT5 first, fall back to Yahoo."""
    print(f"Fetching DXY ({tf}) up to {bars} bars...")
    
    df = fetch_mt5(tf, bars)
    if df is None:
        print("  MT5 unavailable, trying Yahoo Finance...")
        df = fetch_yahoo(tf, bars)
    
    if df is None:
        print("ERROR: No DXY data from any source")
        return False
    
    # Standardise output columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'volume'})
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} bars to {output}")
    print(f"  Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    return True

if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "h1"
    bars = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    valid = {"m15", "h1", "h4", "d1"}
    if tf not in valid:
        print(f"Usage: python fetch_dxy.py [tf] [bars]")
        print(f"  tf:    {'/'.join(valid)} (default: h1)")
        print(f"  bars:  max bars (default: 25000)")
        sys.exit(1)
    output = f"DXY_{tf.upper()}.csv"
    fetch_dxy(tf, bars, output)

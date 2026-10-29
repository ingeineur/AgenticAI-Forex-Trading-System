#!/usr/bin/env python3
"""Fetch MT5 data, combine with CSV, save combined datasets for retraining."""
import MetaTrader5 as mt5, pandas as pd, numpy as np

def fetch_mt5_bars(symbol, tf, name, count=50000):
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates)==0:
        print(f"  {symbol} {name}: failed ({mt5.last_error()})"); return None
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={c:c.lower() for c in df.columns if c!="time"}, inplace=True)
    if "tick_volume" in df.columns: df["volume"] = df["tick_volume"]
    df.sort_values("timestamp", inplace=True); df.reset_index(drop=True, inplace=True)
    print(f"  {symbol} {name}: {len(df):,} bars, {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")
    return df

def combine_with_csv(mt5_df, csv_path, symbol_label):
    try:
        csv_df = pd.read_csv(csv_path)
        csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"], utc=True)
        csv_df.sort_values("timestamp", inplace=True)
        print(f"  {symbol_label} CSV: {len(csv_df):,} bars, {csv_df['timestamp'].iloc[0].date()} to {csv_df['timestamp'].iloc[-1].date()}")
    except: csv_df = None
    if csv_df is None: return mt5_df
    # Combine, keep CSV cols that match MT5 cols
    mt5_cols = set(mt5_df.columns)
    csv_cols = [c for c in csv_df.columns if c in mt5_cols]
    combined = pd.concat([csv_df[csv_cols], mt5_df], ignore_index=True)
    combined.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"  Combined {symbol_label}: {len(combined):,} bars")
    return combined

if not mt5.initialize(): print("MT5 init failed:", mt5.last_error()); exit()

# XAUUSD
print("\nFetching XAUUSD...")
xau_m15_mt5 = fetch_mt5_bars("XAUUSD", mt5.TIMEFRAME_M15, "M15")
xau_h4_mt5 = fetch_mt5_bars("XAUUSD", mt5.TIMEFRAME_H4, "H4")

print("\nCombining XAUUSD M15...")
xau_m15 = combine_with_csv(xau_m15_mt5, "XAUUSD_M15.csv", "XAUUSD M15")
print("Combining XAUUSD H4...")
xau_h4 = combine_with_csv(xau_h4_mt5, "XAUUSD_16388.csv", "XAUUSD H4")

xau_m15.to_csv("XAUUSD_M15_combined.csv", index=False)
xau_h4.to_csv("XAUUSD_16388_combined.csv", index=False)
print("Saved: XAUUSD_M15_combined.csv, XAUUSD_16388_combined.csv")

# BTCUSD
print("\nFetching BTCUSD...")
btc_m15_mt5 = fetch_mt5_bars("BTCUSD", mt5.TIMEFRAME_M15, "M15")
btc_h4_mt5 = fetch_mt5_bars("BTCUSD", mt5.TIMEFRAME_H4, "H4")

print("\nCombining BTCUSD M15...")
btc_m15 = combine_with_csv(btc_m15_mt5, "BTCUSD_15.csv", "BTCUSD M15")
print("Combining BTCUSD H4...")
btc_h4 = combine_with_csv(btc_h4_mt5, "BTCUSD_16388.csv", "BTCUSD H4")

btc_m15.to_csv("BTCUSD_15_combined.csv", index=False)
btc_h4.to_csv("BTCUSD_16388_combined.csv", index=False)
print("Saved: BTCUSD_15_combined.csv, BTCUSD_16388_combined.csv")

mt5.shutdown()
print("\nDone!")

#!/usr/bin/env python3
"""Check BTCUSD broker data"""
import pandas as pd
df = pd.read_csv("BTCUSD_15.csv")
print("BTCUSD Broker M15:")
print("  Bars:", f"{len(df):,}")
print("  Columns:", list(df.columns))
print("  Range:", df["timestamp"].iloc[0], "to", df["timestamp"].iloc[-1])

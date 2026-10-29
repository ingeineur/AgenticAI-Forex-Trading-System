#!/usr/bin/env python3
"""Check existing CSV files"""
import pandas as pd

files = ['XAUUSD_15.csv', 'XAUUSD_16385.csv', 'XAUUSD_16388.csv', 'XAUUSD_M15.csv']
for f in files:
    df = pd.read_csv(f)
    time_col = [c for c in ['time', 'timestamp'] if c in df.columns]
    if time_col:
        tc = time_col[0]
        print(f'{f}: {len(df)} rows, {list(df.columns)}')
        print(f'  Time: {df[tc].iloc[0]} to {df[tc].iloc[-1]}')
    else:
        print(f'{f}: {len(df)} rows, {list(df.columns)}')
    print()

"""Check candles for FVG detection (gap only, no BOS check)."""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
TF = sys.argv[2].lower() if len(sys.argv) > 2 else "h1"
BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 50

TF_MAP = {"m15": mt5.TIMEFRAME_M15, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4}
tf_mt5 = TF_MAP.get(TF, mt5.TIMEFRAME_H1)

if not mt5.initialize():
    print('MT5 init failed'); exit()

ac, pw, sv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo'
mt5.login(ac, pw, sv)

rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 1, BARS)
if rates is None or len(rates) < 5:
    print('No data'); mt5.shutdown(); exit()

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')
h = df['high'].values; l = df['low'].values; c = df['close'].values; o = df['open'].values

print(f'\n{SYMBOL} {TF} — Last {min(5, BARS)} closed candles:')
for i in range(-5, 0):
    ts = str(df.iloc[i]['timestamp'])[:19]
    b = 'BULL' if c[i] > o[i] else 'BEAR'
    print(f'  {ts}  o={o[i]:.2f} h={h[i]:.2f} l={l[i]:.2f} c={c[i]:.2f} {b}')

print(f'\nFVG zones (no BOS check):')
found = 0
for j in range(len(df)-3, 0, -1):
    c0 = df.iloc[j]; c1 = df.iloc[j+1]; c2 = df.iloc[j+2]
    if c1['close'] > c1['open']:
        gap = c2['low'] - c0['high']
        if gap > 0:
            print(f'  BUY  FVG at bars {j}-{j+2}: [{c0["high"]:.2f} - {c2["low"]:.2f}] gap={gap:.2f}')
            found += 1
    if c1['close'] < c1['open']:
        gap = c0['low'] - c2['high']
        if gap > 0:
            print(f'  SELL FVG at bars {j}-{j+2}: [{c2["high"]:.2f} - {c0["low"]:.2f}] gap={gap:.2f}')
            found += 1
if found == 0:
    print('  No FVG zones found')

mt5.shutdown()

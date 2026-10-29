"""Check BTCUSD H1 swing structure."""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSD"
BARS = 50

TF_MAP = {"m15": mt5.TIMEFRAME_M15, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4}
tf_mt5 = TF_MAP.get("h1", mt5.TIMEFRAME_H1)
FRACTAL_STR = 2

def swings(highs, lows):
    n = len(highs); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(FRACTAL_STR, n - FRACTAL_STR):
        sh[i] = all(highs[i] >= highs[i-k] and highs[i] > highs[i+k] for k in range(1, FRACTAL_STR+1))
        sl[i] = all(lows[i] <= lows[i-k] and lows[i] < lows[i+k] for k in range(1, FRACTAL_STR+1))
    return sh, sl

if not mt5.initialize(): print('MT5 init failed'); exit()
mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo')
mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 1, BARS)
mt5.shutdown()
if rates is None: print('No data'); exit()

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')
h = df['high'].values; l = df['low'].values; c = df['close'].values

sh, sl = swings(h, l)

print(f'\n{SYMBOL} H1 — Last {BARS} bars (swing pts):')
for i in range(len(df)):
    if sh[i] or sl[i]:
        kind = "HIGH" if sh[i] else "LOW "
        ts = str(df.iloc[i]['timestamp'])[:19]
        print(f'  {i:3d} {ts}  {kind} @ {h[i] if sh[i] else l[i]:.2f}  (close: {c[i]:.2f})')

# Last 2 swings
si = np.where(sh | sl)[0]
if len(si) >= 2:
    i0, i1 = si[-2], si[-1]
    p0 = h[i0] if sh[i0] else l[i0]
    p1 = h[i1] if sh[i1] else l[i1]
    print(f'\nLast 2 swings:')
    print(f'  Prev: idx={i0} {"HIGH" if sh[i0] else "LOW"} @ {p0:.2f} close={c[i0]:.2f}')
    print(f'  Last: idx={i1} {"HIGH" if sh[i1] else "LOW"} @ {p1:.2f} close={c[i1]:.2f}')
    print(f'  Direction: {"BULLISH" if p1 > p0 else "BEARISH" if p1 < p0 else "FLAT"}')

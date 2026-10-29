"""
Quick bias check: bullish / bearish / sideways for any symbol and timeframe.

Usage:
    python bias.py XAUUSD h1
    python bias.py BTCUSD m15
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
TF = sys.argv[2].lower() if len(sys.argv) > 2 else "h1"
BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
STR = 2  # fractal strength

TF_MAP = {"m15": mt5.TIMEFRAME_M15, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4, "d1": mt5.TIMEFRAME_D1}

def swings(h, l):
    n = len(h); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(STR, n - STR):
        sh[i] = all(h[i] > h[i - k] and h[i] > h[i + k] for k in range(1, STR + 1))
        sl[i] = all(l[i] < l[i - k] and l[i] < l[i + k] for k in range(1, STR + 1))
    return sh, sl

if not mt5.initialize():
    print("MT5 init failed"); sys.exit(1)

ac, pw, sv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "Pepperstone-Demo"
if not mt5.login(ac, pw, sv):
    print("Login failed"); mt5.shutdown(); sys.exit(1)

mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, TF_MAP.get(TF, mt5.TIMEFRAME_H1), 1, BARS)
if rates is None:
    print("No data"); mt5.shutdown(); sys.exit(1)

df = pd.DataFrame(rates)
h = df['high'].values; l = df['low'].values
sh, sl = swings(h, l)
si = np.where(sh | sl)[0]

if len(si) < 2:
    print("SIDEWAYS (not enough swings)")
else:
    i0, i1 = si[-2], si[-1]
    p0 = h[i0] if sh[i0] else l[i0]
    p1 = h[i1] if sh[i1] else l[i1]
    if p1 > p0:
        print("BULLISH")
    elif p1 < p0:
        print("BEARISH")
    else:
        print("SIDEWAYS")

mt5.shutdown()

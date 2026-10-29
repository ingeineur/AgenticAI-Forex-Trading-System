"""
Test H4 bias detection (bullish / bearish / sideways) for any symbol and timeframe.

Usage:
    python test_h4_bias.py XAUUSD h1 200
    python test_h4_bias.py BTCUSD m15 300 --fractal 3
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
TF = sys.argv[2].lower() if len(sys.argv) > 2 else "h1"
BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
FRACTAL_STR = 2
if "--fractal" in sys.argv:
    idx = sys.argv.index("--fractal")
    FRACTAL_STR = int(sys.argv[idx + 1])

TF_MAP = {"m1": mt5.TIMEFRAME_M1, "m5": mt5.TIMEFRAME_M5, "m15": mt5.TIMEFRAME_M15,
          "m30": mt5.TIMEFRAME_M30, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4, "d1": mt5.TIMEFRAME_D1}
tf_mt5 = TF_MAP.get(TF, mt5.TIMEFRAME_H1)

def get_swing_points(highs, lows, strength=2):
    n = len(highs)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    s = strength
    for i in range(s, n - s):
        if all(highs[i] >= highs[i - k] and highs[i] > highs[i + k] for k in range(1, s + 1)):
            sh[i] = True
        if all(lows[i] <= lows[i - k] and lows[i] < lows[i + k] for k in range(1, s + 1)):
            sl[i] = True
    return sh, sl

def resample_h4(df):
    d = df.copy()
    d['time'] = d['timestamp']
    return d.resample('4h', on='time').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
    }).dropna().reset_index()

def get_h4_bias(df_h4):
    if df_h4 is None or len(df_h4) < 10:
        return "NEUTRAL (too few bars)", 0, 0
    highs = df_h4['high'].values
    lows = df_h4['low'].values
    sh, sl = get_swing_points(highs, lows, FRACTAL_STR)
    si = np.where(sh)[0]
    sli = np.where(sl)[0]
    all_swings = np.sort(np.concatenate([si, sli]))
    if len(all_swings) < 2:
        return "NEUTRAL (too few swings)", len(df_h4), len(all_swings)
    last_two = all_swings[-2:]  # last 2 swing indices
    i0, i1 = last_two[-2], last_two[-1]
    # Determine if high or low
    p0 = highs[i0] if sh[i0] else lows[i0]
    p1 = highs[i1] if sh[i1] else lows[i1]
    bias = "BULLISH" if p1 > p0 else "BEARISH" if p1 < p0 else "NEUTRAL"
    return bias, len(df_h4), len(all_swings)

# --- MAIN ---
print(f"Testing H4 bias: {SYMBOL} {TF} bars={BARS} fractal={FRACTAL_STR}")
print("=" * 60)

if not mt5.initialize():
    print("ERROR: MT5 init failed"); sys.exit(1)

acc, pwd, srv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "Pepperstone-Demo"
if not mt5.login(acc, pwd, srv):
    print("ERROR: Login failed"); mt5.shutdown(); sys.exit(1)

mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 0, BARS)
if rates is None or len(rates) < 50:
    print(f"ERROR: No data for {SYMBOL} {TF}"); mt5.shutdown(); sys.exit(1)

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')
print(f"Loaded {len(df)} bars: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

h4 = resample_h4(df)
bias, h4_bars, num_swings = get_h4_bias(h4)

print(f"\nH4 resample:   {len(h4)} bars ({h4_bars} relevant)")
print(f"H4 swing pts:  {num_swings}")
print(f"H4 bias:       {bias}")

# Show structure
print(f"\nH4 structure (last {min(30, len(h4))} bars):")
print(f"  {'idx':>4s}  {'time':<19s}  {'open':>8s}  {'high':>8s}  {'low':>8s}  {'close':>8s}  {'swing'}")
highs = h4['high'].values
lows = h4['low'].values
sh, sl = get_swing_points(highs, lows, FRACTAL_STR)
for i in range(max(0, len(h4) - 30), len(h4)):
    row = h4.iloc[i]
    t = str(row['time'])[:19]
    sw = ""
    if sh[i]: sw = "HIGH"
    elif sl[i]: sw = "LOW"
    print(f"  {i:4d}  {t:<19s}  {row['open']:>8.2f}  {row['high']:>8.2f}  {row['low']:>8.2f}  {row['close']:>8.2f}  {sw}")

# Show last few swings with direction
print(f"\nLast swing points:")
swings = np.where(sh | sl)[0]
for s in swings[-6:]:
    price = highs[s] if sh[s] else lows[s]
    kind = "HIGH" if sh[s] else "LOW"
    print(f"  idx={s:3d}  {kind:5s} @ {price:.2f}  ({h4.iloc[s]['time']})")

# Determine trend from last N swings
if len(swings) >= 4:
    recent = swings[-4:]
    kinds = ["HIGH" if sh[s] else "LOW" for s in recent]
    print(f"\nLast 4 swings: {list(zip(kinds, [highs[s] if sh[s] else lows[s] for s in recent]))}")
    print_swings = swings[-4:]
    for idx_h, s in enumerate(print_swings):
        if s in print_swings[-2:]:
            p = highs[s] if sh[s] else lows[s]
            direction = "UP" if (s == print_swings[-1] and p > (highs[print_swings[-2]] if sh[print_swings[-2]] else lows[print_swings[-2]])) else "DOWN"
    # HH = higher high, HL = higher low, LH = lower high, LL = lower low
    for j in range(len(print_swings) - 1):
        a, b = print_swings[j], print_swings[j+1]
        pa = highs[a] if sh[a] else lows[a]
        pb = highs[b] if sh[b] else lows[b]
        ka = "H" if sh[a] else "L"
        kb = "H" if sh[b] else "L"
        print(f"  {ka}{a.astype(int)} -> {kb}{b.astype(int)}: {pa:.2f} -> {pb:.2f}  ({'UP' if pb > pa else 'DOWN'})")

mt5.shutdown()

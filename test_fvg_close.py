"""Test close-based FVG detection for any symbol/timeframe.

Usage:
    python test_fvg_close.py XAUUSD h1
    python test_fvg_close.py BTCUSD m15
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
TF = sys.argv[2].lower() if len(sys.argv) > 2 else "h1"
BARS = 200

TF_MAP = {"m15": mt5.TIMEFRAME_M15, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4}
tf_mt5 = TF_MAP.get(TF, mt5.TIMEFRAME_H1)

FRACTAL_STR = 2

def get_swing_points(highs, lows, strength=2):
    n = len(highs); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(strength, n - strength):
        sh[i] = all(highs[i] > highs[i-k] and highs[i] > highs[i+k] for k in range(1, strength+1))
        sl[i] = all(lows[i] < lows[i-k] and lows[i] < lows[i+k] for k in range(1, strength+1))
    return sh, sl

def _last_swing_high_before(df, idx):
    sub = df.iloc[max(0, idx - 30):idx]
    if len(sub) < 3: return None
    c = sub['close'].values
    sh, _ = get_swing_points(c, c, FRACTAL_STR)
    si = np.where(sh)[0]
    return c[si[-1]] if len(si) > 0 else None

def _last_swing_low_before(df, idx):
    sub = df.iloc[max(0, idx - 30):idx]
    if len(sub) < 3: return None
    c = sub['close'].values
    _, sl = get_swing_points(c, c, FRACTAL_STR)
    si = np.where(sl)[0]
    return c[si[-1]] if len(si) > 0 else None

def find_fvg(df, i, direction, max_lookahead=5, min_gap_thresh=0.0):
    ref_atr = df.iloc[i]['atr'] if 'atr' in df.columns else 0
    min_gap = ref_atr * min_gap_thresh if ref_atr > 0 else 0
    n = len(df)
    for gap_start in range(i+1, min(i+max_lookahead, n-2)):
        gap_end = gap_start + 2
        if gap_end >= n: break
        c0 = df.iloc[gap_start]; c1 = df.iloc[gap_start+1]; c2 = df.iloc[gap_end]
        if direction == "BUY":
            if c1['close'] <= c1['open']: continue
            prev_sw = _last_swing_high_before(df, gap_start)
            if prev_sw is None or max(c0['close'], c1['close'], c2['close']) <= prev_sw: continue
            gap = c2['low'] - c0['high']
            if gap > min_gap and (gap_end+1 >= n or df['low'].iloc[gap_end+1:].min() > c0['high']):
                return c0['high'], c2['low']
        else:
            if c1['close'] >= c1['open']: continue
            prev_sw = _last_swing_low_before(df, gap_start)
            if prev_sw is None or min(c0['close'], c1['close'], c2['close']) >= prev_sw: continue
            gap = c0['low'] - c2['high']
            if gap > min_gap and (gap_end+1 >= n or df['high'].iloc[gap_end+1:].max() < c0['low']):
                return c2['high'], c0['low']
    return None, None

if not mt5.initialize():
    print('MT5 init failed'); exit()
mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo')
mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 1, BARS)
mt5.shutdown()
if rates is None: print('No data'); exit()

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')
n = len(df)

print(f'\n{SYMBOL} {TF} ({n} bars) — checking all break bars for BUY FVG...')
found = 0
for i in range(max(0, n-50), n-1):
    fvg_l, fvg_h = find_fvg(df, i, 'BUY')
    if fvg_l is not None:
        ts = str(df.iloc[i]['timestamp'])[:19]
        print(f'  BUY FVG at i={i} {ts}: [{fvg_l:.2f} - {fvg_h:.2f}]')
        found += 1
if found == 0:
    print('  No BUY FVG found.')
    
    # Show why: inspect last 5 bars
    print(f'\nDebug: last 5 bars')
    for i in range(n-5, n):
        ts = str(df.iloc[i]['timestamp'])[:19]
        o, h, l, c = df.iloc[i]['open'], df.iloc[i]['high'], df.iloc[i]['low'], df.iloc[i]['close']
        print(f'  {i:3d} {ts}  o={o:.2f} h={h:.2f} l={l:.2f} c={c:.2f}')
    
    # Check the 3-candle window ending at n-1
    print(f'\nCheck 3-candle window ending at n-1:')
    c0 = df.iloc[n-4]; c1 = df.iloc[n-3]; c2 = df.iloc[n-2]
    print(f'  c0: o={c0["open"]:.2f} h={c0["high"]:.2f} l={c0["low"]:.2f} c={c0["close"]:.2f}')
    print(f'  c1: o={c1["open"]:.2f} h={c1["high"]:.2f} l={c1["low"]:.2f} c={c1["close"]:.2f}')
    print(f'  c2: o={c2["open"]:.2f} h={c2["high"]:.2f} l={c2["low"]:.2f} c={c2["close"]:.2f}')
    print(f'  c1 bullish? {c1["close"] > c1["open"]}')
    prev_sw = _last_swing_high_before(df, n-4)
    print(f'  Prev swing high (close): {prev_sw}')
    print(f'  Max closes: {max(c0["close"], c1["close"], c2["close"]):.2f}')
    print(f'  Break above? {max(c0["close"], c1["close"], c2["close"]) > (prev_sw or 0)}')
    gap = c2['low'] - c0['high']
    print(f'  Gap: {gap:.2f}')
    fill_ok = n-2+1 >= n or df['low'].iloc[n-2+1:].min() > c0['high']
    print(f'  Filled? {not fill_ok}')

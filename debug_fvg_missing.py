"""Debug why BTCUSD H1 BUY FVG is not detected."""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

SYMBOL = "BTCUSD"
TF = "h1"
BARS = 200
FRACTAL_STR = 2

def swings(h, l):
    n = len(h); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(FRACTAL_STR, n - FRACTAL_STR):
        sh[i] = all(h[i] >= h[i-k] and h[i] > h[i+k] for k in range(1, FRACTAL_STR+1))
        sl[i] = all(l[i] <= l[i-k] and l[i] < l[i+k] for k in range(1, FRACTAL_STR+1))
    return sh, sl

def last_swing_high_before(df, idx):
    sub = df.iloc[max(0, idx-30):idx]
    if len(sub) < 3: return None
    h = sub['high'].values; l = sub['low'].values
    sh, _ = swings(h, l)
    si = np.where(sh)[0]
    return h[si[-1]] if len(si) > 0 else None

def last_swing_low_before(df, idx):
    sub = df.iloc[max(0, idx-30):idx]
    if len(sub) < 3: return None
    h = sub['high'].values; l = sub['low'].values
    _, sl = swings(h, l)
    si = np.where(sl)[0]
    return l[si[-1]] if len(si) > 0 else None

def find_fvg(df, i, direction):
    for gap_start in range(i+1, min(i+5, len(df)-2)):
        gap_end = gap_start + 2
        if gap_end >= len(df): break
        c0 = df.iloc[gap_start]; c1 = df.iloc[gap_start+1]; c2 = df.iloc[gap_end]
        if direction == "BUY":
            if c1['close'] <= c1['open']: continue
            prev = last_swing_high_before(df, gap_start)
            if prev is None or max(c0['high'], c1['high'], c2['high']) <= prev: continue
            gap = c2['low'] - c0['high']
            if gap > 0 and (gap_end+1 >= len(df) or df['low'].iloc[gap_end+1:].min() > c0['high']):
                return c0['high'], c2['low']
        else:
            if c1['close'] >= c1['open']: continue
            prev = last_swing_low_before(df, gap_start)
            if prev is None or min(c0['low'], c1['low'], c2['low']) >= prev: continue
            gap = c0['low'] - c2['high']
            if gap > 0 and (gap_end+1 >= len(df) or df['high'].iloc[gap_end+1:].max() < c0['low']):
                return c2['high'], c0['low']
    return None, None

if not mt5.initialize():
    print("MT5 init failed"); exit()

mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "Pepperstone-Demo")
mt5.symbol_select(SYMBOL, True)

rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, BARS)
mt5.shutdown()
if rates is None: print("No data"); exit()

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')

n = len(df)
N = min(100, n // 2)
h = df['high'].values; l = df['low'].values; c = df['close'].values
sh, sl = swings(h, l)

# Get active swing state
active_sh_idx = -1; active_sh_price = None; prev_sh_price = None
for i in range(FRACTAL_STR, n - FRACTAL_STR):
    if sh[i]:
        prev_sh_price = active_sh_price
        active_sh_price = h[i]; active_sh_idx = i
    if sl[i]:
        pass  # just need swing highs for BUY check

print(f"Active swing high: {active_sh_price:.2f} at index {active_sh_idx}")
print(f"Data range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
print(f"Scan range: i = {max(FRACTAL_STR+1, n-N)} to {n-1}")
print()

# Check each i in scan range for BUY break
for i in range(max(FRACTAL_STR+1, n-N), n-1):
    if active_sh_price and c[i] > active_sh_price:
        fvg_l, fvg_h = find_fvg(df, i, "BUY")
        ts = str(df.iloc[i]['timestamp'])[:19]
        dist = i - active_sh_idx
        if fvg_l is not None:
            print(f"  BUY entry at i={i} {ts}: close={c[i]:.2f} > swing={active_sh_price:.2f} (dist={dist})")
            print(f"    FVG zone: [{fvg_l:.2f} - {fvg_h:.2f}]")
        else:
            # Check why find_fvg fails - look at next 3 candles
            for gs in range(i+1, min(i+5, n-2)):
                ge = gs + 2
                if ge >= n: break
                c0 = df.iloc[gs]; c1 = df.iloc[gs+1]; c2 = df.iloc[ge]
                if c1['close'] > c1['open']:
                    gap = c2['low'] - c0['high']
                    prev_sw = last_swing_high_before(df, gs)
                    sw_fail = prev_sw is None or max(c0['high'], c1['high'], c2['high']) <= prev_sw
                    fill_fail = not (ge+1 >= n or df['low'].iloc[ge+1:].min() > c0['high'])
                    if gap > 0:
                        print(f"    BUY FVG candidate at gap_start={gs}: [{c0['high']:.2f}-{c2['low']:.2f}] gap={gap:.2f} "
                              f"sw_fail={sw_fail} fill_fail={fill_fail} "
                              f"prev_sw={prev_sw} max_high={max(c0['high'], c1['high'], c2['high']):.2f}"
                              f"c1_bull={c1['close'] > c1['open']}")

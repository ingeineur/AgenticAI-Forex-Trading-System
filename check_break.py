"""Check if any candle closed above swing high 62606.68"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

if not mt5.initialize():
    print('MT5 init failed'); exit()
mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo')
mt5.symbol_select('BTCUSD', True)

rates = mt5.copy_rates_from_pos('BTCUSD', mt5.TIMEFRAME_H1, 1, 200)
mt5.shutdown()
if rates is None: print('No data'); exit()

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')
n = len(df)
h = df['high'].values; c = df['close'].values

# Find the swing high
FRACTAL_STR = 2
def swings(highs, lows):
    n = len(highs); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(FRACTAL_STR, n - FRACTAL_STR):
        sh[i] = all(highs[i] >= highs[i-k] and highs[i] > highs[i+k] for k in range(1, FRACTAL_STR+1))
        sl[i] = all(lows[i] <= lows[i-k] and lows[i] < lows[i+k] for k in range(1, FRACTAL_STR+1))
    return sh, sl

sh, sl = swings(h, h)
si = np.where(sh)[0]
if len(si) > 0:
    last_swing_idx = si[-1]
    last_swing_high = h[last_swing_idx]
    last_swing_close = c[last_swing_idx]
    print(f'Last swing high: idx={last_swing_idx} high={last_swing_high:.2f} close={last_swing_close:.2f}')
    print(f'  at {str(df.iloc[last_swing_idx]["timestamp"])[:19]}')
    
    # Check all candles AFTER the swing high
    print(f'\nCandles after swing high (index {last_swing_idx}+):')
    for i in range(last_swing_idx+1, n):
        ts = str(df.iloc[i]['timestamp'])[:19]
        print(f'  {i:3d} {ts}  h={h[i]:.2f} c={c[i]:.2f}  above_high={h[i] > last_swing_high} above_close={c[i] > last_swing_close}')
else:
    print('No swing highs found')

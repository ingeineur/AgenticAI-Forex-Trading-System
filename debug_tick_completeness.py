"""Check tick completeness for recent bars."""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

if not mt5.initialize(): print('init fail'); exit()
if not mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo'):
    print('login fail:', mt5.last_error()); mt5.shutdown(); exit()

SYMBOL = 'XAUUSD'
mt5.symbol_select(SYMBOL, True)

rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 50)
df = pd.DataFrame(rates)
t0 = pd.to_datetime(df['time'].iloc[0], unit='s')
t1 = pd.to_datetime(df['time'].iloc[-1], unit='s')
ticks = mt5.copy_ticks_range(SYMBOL, t0, t1, mt5.COPY_TICKS_ALL)
if ticks is None: print('no ticks'); mt5.shutdown(); exit()

bar_times = rates['time']
n = len(bar_times)
tick_idx = 0
print(f"{'Bar time':<22} {'Ticks':>7} {'BidUp':>7} {'BidDn':>7} {'Delta':>7}")
print('-' * 55)
for bi in range(n - 1):
    bs = bar_times[bi]; be = bar_times[bi+1]
    cnt = 0; up = 0; dn = 0; prev = None
    while tick_idx < len(ticks) and ticks[tick_idx]['time'] < bs: tick_idx += 1
    while tick_idx < len(ticks) and ticks[tick_idx]['time'] < be:
        bid = ticks[tick_idx]['bid']
        if bid > 0 and prev is not None:
            if bid > prev: up += 1
            elif bid < prev: dn += 1
        prev = bid
        cnt += 1; tick_idx += 1
    ts = pd.to_datetime(bs, unit='s')
    vol = df.loc[bi, 'tick_volume']
    print(f'{ts:<22} {cnt:>7} {up:>7} {dn:>7} {up-dn:>7}  (vol={vol})')

# last bar
cnt = 0; up = 0; dn = 0; prev = None
while tick_idx < len(ticks):
    bid = ticks[tick_idx]['bid']
    if bid > 0 and prev is not None:
        if bid > prev: up += 1
        elif bid < prev: dn += 1
    prev = bid
    cnt += 1; tick_idx += 1
ts = pd.to_datetime(bar_times[-1], unit='s')
vol = df.loc[n-1, 'tick_volume']
print(f'{ts:<22} {cnt:>7} {up:>7} {dn:>7} {up-dn:>7}  (vol={vol})')
mt5.shutdown()

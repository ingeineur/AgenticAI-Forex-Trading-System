"""Debug tick coverage for last bars."""
import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

if not mt5.initialize(): exit()
if not mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo'):
    print('Login fail:', mt5.last_error()); mt5.shutdown(); exit()

SYMBOL = 'XAUUSD'
mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 100)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
t0 = df['time'].iloc[0].to_pydatetime()
t1 = df['time'].iloc[-1].to_pydatetime()

ticks = mt5.copy_ticks_range(SYMBOL, t0, t1, mt5.COPY_TICKS_ALL)
if ticks is None:
    print('No ticks:', mt5.last_error()); mt5.shutdown(); exit()

tick_times = pd.to_datetime(ticks['time'], unit='s')
print(f'Total ticks: {len(ticks)}')
print(f'Earliest tick: {tick_times.min()}')
print(f'Latest tick:   {tick_times.max()}')
print(f'First bar:     {df["time"].iloc[0]}')
print(f'Last bar:      {df["time"].iloc[-1]}')

bar_times = rates['time']
n = len(bar_times)
print(f'\nTicks per last 10 bars:')
start_idx = max(0, n - 11)
tick_idx = 0
for bar_idx in range(start_idx, n - 1):
    bs = bar_times[bar_idx]
    be = bar_times[bar_idx + 1]
    cnt = 0
    while tick_idx < len(ticks) and ticks[tick_idx]['time'] < bs:
        tick_idx += 1
    while tick_idx < len(ticks) and ticks[tick_idx]['time'] < be:
        cnt += 1
        tick_idx += 1
    tstr = f'{pd.to_datetime(bs, unit="s")} -> {pd.to_datetime(be, unit="s")}'
    print(f'  {tstr}: {cnt} ticks')

cnt = 0
while tick_idx < len(ticks):
    cnt += 1
    tick_idx += 1
print(f'  {pd.to_datetime(bar_times[-1], unit="s")} -> end: {cnt} ticks')

# bid changes in last 10 ticks
print(f'\nLast 15 ticks bid/flag:')
for t in ticks[-15:]:
    print(f'  time={pd.to_datetime(t["time"], unit="s")} bid={t["bid"]} flags={t["flags"]}')
mt5.shutdown()

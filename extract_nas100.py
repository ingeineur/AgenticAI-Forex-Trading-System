"""Extract NAS100 history from MT5 using paginated copy_rates_from_pos.

MT5's copy_rates_range fails on NAS100 (returns 0), but from_pos pagination works.
Usage: python extract_nas100.py [TF]   (TF = M15 or H1)
"""
import sys
import time
import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "NAS100"
TF = (sys.argv[1] if len(sys.argv) > 1 else "M15").upper()
TF_MAP = {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
if TF not in TF_MAP:
    print("choose M15 or H1"); sys.exit(1)
OUT = f"{SYMBOL}_{TF}_combined.csv"
CHUNK = 2000

if not mt5.initialize():
    raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
mt5.symbol_select(SYMBOL, True)

frames = []
pos = 0
while True:
    r = mt5.copy_rates_from_pos(SYMBOL, TF_MAP[TF], pos, CHUNK)
    if r is None or len(r) == 0:
        break
    df = pd.DataFrame(r)
    frames.append(df)
    pos += CHUNK
    if len(df) < CHUNK:
        break
    print(f"  fetched up to {pd.to_datetime(df['time'].iloc[-1], unit='s', utc=True)}  ({pos} bars)")
    time.sleep(0.05)

if not frames:
    print("no data"); mt5.shutdown(); sys.exit(1)

df = pd.concat(frames, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
df = df.rename(columns={'tick_volume': 'volume'})
for col in ('spread', 'real_volume'):
    if col not in df.columns:
        df[col] = 0.0
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume']]
df.to_csv(OUT, index=False)
print(f"\nNAS100 {TF}: {len(df)} bars  {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
print(f"Saved {OUT}")
mt5.shutdown()